"""
features.py — Pair-level similarity & domain interaction features for LightGBM.

Computes 28 similarity, domain interaction, digit conflict, and structural features per pair:

Base Similarity Features (17)
-----------------------------
1. name_ratio          — Levenshtein ratio (rapidfuzz)
2. name_token_sort     — token-sort ratio   (handles word reorder)
3. name_token_set      — token-set ratio    (handles subset / extra words)
4. name_partial        — partial-string ratio (for abbreviations)
5. name_jaro_winkler   — Jaro-Winkler similarity
6. name_jaccard_token  — Jaccard of word-token sets
7. addr_ratio          — Levenshtein ratio
8. addr_token_sort     — token-sort ratio
9. addr_token_set      — token-set ratio
10. addr_jaccard_token — Jaccard of word-token sets
11. addr_num_overlap   — Jaccard of numeric-token sets (building/pin numbers)
12. country_match      — 1 if country_norm identical, 0 otherwise
13. blocking_score     — cosine similarity from TF-IDF blocking step
14. name_len_s1        — len(name_norm) of S1 entity
15. name_len_s23       — len(name_norm) of S23 entity
16. name_len_ratio     — min/max of name lengths (0-1)
17. addr_len_ratio     — min/max of addr_norm lengths (0-1)

Advanced Interaction & Domain Features (11)
--------------------------------------------
18. name_x_address              — name_jw * addr_token_set
19. min_name_address            — min(name_token_set, addr_token_set)
20. max_name_address            — max(name_token_set, addr_token_set)
21. has_both_addresses          — 1 if both S1 and S23 addresses >= 5 chars
22. missing_address_asymmetry   — 1 if one address >= 10 chars while other < 3 chars
23. exact_name_match            — 1 if normalized names are identical
24. conflicting_digits          — 1 if both have numbers but d1 ∩ d2 is empty
25. strong_name_and_number_match — 1 if name_token_set >= 0.85 and addr_num_overlap >= 0.5
26. strong_name_but_number_conflict — 1 if name_token_set >= 0.85 and conflicting_digits == 1
27. candidate_is_s3             — 1 if candidate starts with S3-
28. candidate_rank              — 1-based rank of candidate within S1 candidate list

Total: 28 features.
"""

import logging
import re
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import JaroWinkler
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    import difflib

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
    # Full Document Context Features
    "doc_ratio",
    "doc_token_set",
    "doc_jw",
    # Advanced 11
    "name_x_address",
    "min_name_address",
    "max_name_address",
    "has_both_addresses",
    "missing_address_asymmetry",
    "exact_name_match",
    "conflicting_digits",
    "strong_name_and_number_match",
    "strong_name_but_number_conflict",
    "candidate_is_s3",
    "candidate_rank",
    "name_num_overlap",
    "name_conflicting_digits",
    "addr_jaro_winkler",
    "name_length_diff",
    "addr_length_diff",
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


def _extract_digit_set(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


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
        return difflib.SequenceMatcher(None, a, b).ratio()


# ── Feature Computation ───────────────────────────────────────────────────────

def compute_features_for_pair(
    s1_name: str, s1_addr: str, s1_country: str,
    s23_name: str, s23_addr: str, s23_country: str,
    s23_id: str = "",
    cand_rank: int = 1,
    blocking_score: float = 0.0,
) -> List[float]:
    """Compute the 28 features for a single (S1, S23) pair."""
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

    # Country & Blocking
    f_country_match    = float(s1_country == s23_country)
    f_blocking         = blocking_score

    # Structural
    f_name_len_s1      = float(len(s1_name))
    f_name_len_s23     = float(len(s23_name))
    f_name_len_ratio   = _len_ratio(s1_name, s23_name)
    f_addr_len_ratio   = _len_ratio(s1_addr, s23_addr)

    # Full Document Context (name + addr + country)
    doc1 = f"{s1_name} {s1_addr} {s1_country}".strip()
    doc2 = f"{s23_name} {s23_addr} {s23_country}".strip()
    f_doc_ratio = _ratio(doc1, doc2)
    f_doc_token_set = _token_set(doc1, doc2)
    f_doc_jw = _jaro_winkler(doc1, doc2)

    # Advanced Interactions & Domain features
    f_name_x_addr      = f_name_jw * f_addr_token_set
    f_min_name_addr    = min(f_name_token_set, f_addr_token_set)
    f_max_name_addr    = max(f_name_token_set, f_addr_token_set)

    l1, l2             = len(s1_addr), len(s23_addr)
    f_both_addr        = 1.0 if (l1 >= 5 and l2 >= 5) else 0.0
    f_missing_asym     = 1.0 if ((l1 >= 10 and l2 < 3) or (l2 >= 10 and l1 < 3)) else 0.0
    f_exact_name       = 1.0 if (s1_name and s1_name == s23_name) else 0.0

    d1, d2             = _extract_digit_set(s1_addr), _extract_digit_set(s23_addr)
    f_conflict_digits  = 1.0 if (d1 and d2 and not (d1 & d2)) else 0.0

    f_strong_name_num   = 1.0 if (f_name_token_set >= 0.85 and f_addr_num_overlap >= 0.5) else 0.0
    f_name_num_conflict = 1.0 if (f_name_token_set >= 0.85 and f_conflict_digits == 1.0) else 0.0

    f_is_s3            = 1.0 if s23_id.startswith("S3-") else 0.0
    f_cand_rank        = float(cand_rank)

    # Hard negative handling features
    nd1, nd2             = _extract_digit_set(s1_name), _extract_digit_set(s23_name)
    f_name_num_overlap   = _jaccard_numbers(s1_name, s23_name)
    f_name_conflict_digits = 1.0 if (nd1 and nd2 and not (nd1 & nd2)) else 0.0
    f_addr_jw            = _jaro_winkler(s1_addr, s23_addr)
    f_name_len_diff      = float(abs(len(s1_name) - len(s23_name)))
    f_addr_len_diff      = float(abs(len(s1_addr) - len(s23_addr)))

    return [
        f_name_ratio, f_name_token_sort, f_name_token_set,
        f_name_partial, f_name_jw, f_name_jaccard,
        f_addr_ratio, f_addr_token_sort, f_addr_token_set,
        f_addr_jaccard, f_addr_num_overlap,
        f_country_match,
        f_blocking,
        f_name_len_s1, f_name_len_s23,
        f_name_len_ratio, f_addr_len_ratio,
        # Full Document Context
        f_doc_ratio, f_doc_token_set, f_doc_jw,
        # Advanced 11
        f_name_x_addr, f_min_name_addr, f_max_name_addr,
        f_both_addr, f_missing_asym, f_exact_name,
        f_conflict_digits, f_strong_name_num, f_name_num_conflict,
        f_is_s3, f_cand_rank,
        # Hard Negatives 5
        f_name_num_overlap, f_name_conflict_digits,
        f_addr_jw, f_name_len_diff, f_addr_len_diff
    ]


def build_feature_matrix(
    pairs: List[Tuple[str, str]],
    s1_lookup: Dict,
    s23_lookup: Dict,
    blocking_scores: Dict[Tuple[str, str], float] = None,
    candidate_ranks: Dict[Tuple[str, str], int] = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """Build feature matrix (N, 28) for pairs."""
    if blocking_scores is None:
        blocking_scores = {}
    if candidate_ranks is None:
        candidate_ranks = {}

    rows = []
    for s1_id, s23_id in pairs:
        s1  = s1_lookup[s1_id]
        s23 = s23_lookup[s23_id]
        b_score = blocking_scores.get((s1_id, s23_id), 0.0)
        c_rank  = candidate_ranks.get((s1_id, s23_id), 1)
        rows.append(compute_features_for_pair(
            s1["name_norm"],  s1["addr_norm"],  s1["country_norm"],
            s23["name_norm"], s23["addr_norm"], s23["country_norm"],
            s23_id=s23_id,
            cand_rank=c_rank,
            blocking_score=b_score,
        ))
    return np.array(rows, dtype=np.float32)


def build_lookup(df: pd.DataFrame) -> Dict:
    return {
        row["entity_id"]: {
            "name_norm":    row["name_norm"],
            "addr_norm":    row["addr_norm"],
            "country_norm": row["country_norm"],
        }
        for _, row in df.iterrows()
    }

