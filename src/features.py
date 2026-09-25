"""
features.py — Pair-level similarity features for the LightGBM classifier.

For every (S1, S23) candidate pair we compute:

  Name features (6)
  -----------------
  name_ratio          — Levenshtein ratio (rapidfuzz)
  name_token_sort     — token-sort ratio   (handles word reorder)
  name_token_set      — token-set ratio    (handles subset / extra words)
  name_partial        — partial-string ratio (for abbreviations)
  name_jaro_winkler   — Jaro-Winkler similarity
  name_jaccard_token  — Jaccard of word-token sets

  Address features (5)
  --------------------
  addr_ratio          — Levenshtein ratio
  addr_token_sort     — token-sort ratio
  addr_token_set      — token-set ratio
  addr_jaccard_token  — Jaccard of word-token sets
  addr_num_overlap    — Jaccard of numeric-token sets (building/pin numbers)

  Country feature (1)
  -------------------
  country_match       — 1 if country_norm identical, 0 otherwise

  Blocking score (1)
  ------------------
  blocking_score      — cosine similarity from the TF-IDF blocking step

  Structural features (4)
  -----------------------
  name_len_s1         — len(name_norm) of S1 entity
  name_len_s23        — len(name_norm) of S23 entity
  name_len_ratio      — min/max of the above (0-1)
  addr_len_ratio      — min/max of addr_norm lengths

Total: 17 features.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import JaroWinkler
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    import difflib  # fallback (slower)

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "name_jaro_winkler",
    "name_jaccard_token",
    "addr_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_jaccard_token",
    "addr_num_overlap",
    "country_match",
    "blocking_score",
    "name_len_s1",
    "name_len_s23",
    "name_len_ratio",
    "addr_len_ratio",
]


# ── Low-level similarity helpers ──────────────────────────────────────────────

def _jaccard_tokens(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _jaccard_numbers(a: str, b: str) -> float:
    """Jaccard over digit-sequences extracted from address strings."""
    import re
    na = set(re.findall(r"\d+", a))
    nb = set(re.findall(r"\d+", b))
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


if _HAS_RAPIDFUZZ:
    def _ratio(a, b):           return rfuzz.ratio(a, b) / 100.0
    def _token_sort(a, b):      return rfuzz.token_sort_ratio(a, b) / 100.0
    def _token_set(a, b):       return rfuzz.token_set_ratio(a, b) / 100.0
    def _partial(a, b):         return rfuzz.partial_ratio(a, b) / 100.0
    def _jaro_winkler(a, b):    return JaroWinkler.similarity(a, b)
else:
    logger.warning("rapidfuzz not available — using slower difflib fallback.")
    def _ratio(a, b):           return difflib.SequenceMatcher(None, a, b).ratio()
    def _token_sort(a, b):
        ta = " ".join(sorted(a.split()))
        tb = " ".join(sorted(b.split()))
        return difflib.SequenceMatcher(None, ta, tb).ratio()
    def _token_set(a, b):
        sa, sb = set(a.split()), set(b.split())
        inter = " ".join(sorted(sa & sb))
        only_a = " ".join(sorted(sa - sb))
        only_b = " ".join(sorted(sb - sa))
        best = max(
            difflib.SequenceMatcher(None, inter, inter + " " + only_a).ratio(),
            difflib.SequenceMatcher(None, inter, inter + " " + only_b).ratio(),
            difflib.SequenceMatcher(None, inter + " " + only_a, inter + " " + only_b).ratio(),
        )
        return best
    def _partial(a, b):
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        best = 0.0
        for i in range(len(long) - len(short) + 1):
            r = difflib.SequenceMatcher(None, short, long[i:i+len(short)]).ratio()
            best = max(best, r)
        return best
    def _jaro_winkler(a, b):
        # Simple Jaro (not Jaro-Winkler) without external lib
        if a == b:
            return 1.0
        match_dist = max(len(a), len(b)) // 2 - 1
        if match_dist < 0:
            return 0.0
        a_matches = [False] * len(a)
        b_matches = [False] * len(b)
        matches = 0
        transpositions = 0
        for i, c in enumerate(a):
            start = max(0, i - match_dist)
            end   = min(i + match_dist + 1, len(b))
            for j in range(start, end):
                if b_matches[j] or c != b[j]:
                    continue
                a_matches[i] = b_matches[j] = True
                matches += 1
                break
        if matches == 0:
            return 0.0
        k = 0
        for i in range(len(a)):
            if not a_matches[i]:
                continue
            while not b_matches[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
        jaro = (matches/len(a) + matches/len(b) + (matches - transpositions/2)/matches) / 3
        prefix = 0
        for i in range(min(4, len(a), len(b))):
            if a[i] == b[i]:
                prefix += 1
            else:
                break
        return jaro + prefix * 0.1 * (1 - jaro)


# ── Main feature function ─────────────────────────────────────────────────────

def compute_features_for_pair(
    s1_name: str, s1_addr: str, s1_country: str,
    s23_name: str, s23_addr: str, s23_country: str,
    blocking_score: float = 0.0,
) -> List[float]:
    """
    Compute the 17 features for a single (S1, S23) pair.
    All string inputs should already be normalized (from preprocess.py).
    """
    # Name features
    f_name_ratio       = _ratio(s1_name, s23_name)
    f_name_token_sort  = _token_sort(s1_name, s23_name)
    f_name_token_set   = _token_set(s1_name, s23_name)
    f_name_partial     = _partial(s1_name, s23_name)
    f_name_jw          = _jaro_winkler(s1_name, s23_name)
    f_name_jaccard     = _jaccard_tokens(s1_name, s23_name)

    # Address features
    f_addr_ratio       = _ratio(s1_addr, s23_addr)
    f_addr_token_sort  = _token_sort(s1_addr, s23_addr)
    f_addr_token_set   = _token_set(s1_addr, s23_addr)
    f_addr_jaccard     = _jaccard_tokens(s1_addr, s23_addr)
    f_addr_num_overlap = _jaccard_numbers(s1_addr, s23_addr)

    # Country
    f_country_match    = float(s1_country == s23_country)

    # Blocking score (already a float)
    f_blocking         = blocking_score

    # Structural
    f_name_len_s1      = len(s1_name)
    f_name_len_s23     = len(s23_name)
    f_name_len_ratio   = _len_ratio(s1_name, s23_name)
    f_addr_len_ratio   = _len_ratio(s1_addr, s23_addr)

    return [
        f_name_ratio, f_name_token_sort, f_name_token_set,
        f_name_partial, f_name_jw, f_name_jaccard,
        f_addr_ratio, f_addr_token_sort, f_addr_token_set,
        f_addr_jaccard, f_addr_num_overlap,
        f_country_match,
        f_blocking,
        float(f_name_len_s1), float(f_name_len_s23),
        f_name_len_ratio, f_addr_len_ratio,
    ]


def build_feature_matrix(
    pairs: List[Tuple],
    s1_lookup: Dict,
    s23_lookup: Dict,
    blocking_scores: Dict[Tuple[str, str], float] = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Build feature matrix for a list of (s1_id, s23_id) pairs.

    Parameters
    ----------
    pairs         : list of (s1_id, s23_id) tuples
    s1_lookup     : dict {entity_id: {name_norm, addr_norm, country_norm}}
    s23_lookup    : dict {entity_id: {name_norm, addr_norm, country_norm}}
    blocking_scores: optional dict {(s1_id, s23_id): cosine_score}
    n_jobs        : number of parallel workers (-1 = all cores)

    Returns
    -------
    np.ndarray of shape (n_pairs, 17)
    """
    if blocking_scores is None:
        blocking_scores = {}

    if n_jobs == 1 or len(pairs) < 1000:
        # Single-threaded path
        rows = []
        for s1_id, s23_id in pairs:
            s1  = s1_lookup[s1_id]
            s23 = s23_lookup[s23_id]
            score = blocking_scores.get((s1_id, s23_id), 0.0)
            rows.append(compute_features_for_pair(
                s1["name_norm"],  s1["addr_norm"],  s1["country_norm"],
                s23["name_norm"], s23["addr_norm"], s23["country_norm"],
                blocking_score=score,
            ))
        return np.array(rows, dtype=np.float32)

    # Multi-threaded path via joblib
    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.warning("joblib not found — falling back to single-threaded feature computation")
        return build_feature_matrix(pairs, s1_lookup, s23_lookup, blocking_scores, n_jobs=1)

    def _compute_chunk(chunk):
        rows = []
        for s1_id, s23_id in chunk:
            s1  = s1_lookup[s1_id]
            s23 = s23_lookup[s23_id]
            score = blocking_scores.get((s1_id, s23_id), 0.0)
            rows.append(compute_features_for_pair(
                s1["name_norm"],  s1["addr_norm"],  s1["country_norm"],
                s23["name_norm"], s23["addr_norm"], s23["country_norm"],
                blocking_score=score,
            ))
        return rows

    n_workers = n_jobs if n_jobs > 0 else None
    chunk_size = max(1000, len(pairs) // (4 * (n_workers or 4)))
    chunks = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]

    all_rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_compute_chunk)(chunk) for chunk in chunks
    )
    flat = [row for chunk in all_rows for row in chunk]
    return np.array(flat, dtype=np.float32)


def build_lookup(df: pd.DataFrame) -> Dict:
    """Convert preprocessed DataFrame to {entity_id: dict} for fast access."""
    return {
        row["entity_id"]: {
            "name_norm":    row["name_norm"],
            "addr_norm":    row["addr_norm"],
            "country_norm": row["country_norm"],
        }
        for _, row in df.iterrows()
    }
