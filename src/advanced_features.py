"""
advanced_features.py — High-value interaction features, digit conflict indicators,
and entity-level candidate context features.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from features import FEATURE_NAMES, compute_features_for_pair

ADVANCED_FEATURE_NAMES = FEATURE_NAMES + [
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
]


def extract_digit_set(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


def compute_advanced_features_for_pair(
    s1_name: str, s1_addr: str, s1_country: str,
    s23_name: str, s23_addr: str, s23_country: str,
    s23_id: str,
    cand_rank: int = 1,
    blocking_score: float = 0.0,
) -> List[float]:
    """
    Computes base 17 features + 11 advanced interaction & domain features.
    """
    base_feats = compute_features_for_pair(
        s1_name, s1_addr, s1_country,
        s23_name, s23_addr, s23_country,
        blocking_score=blocking_score,
    )

    # Base feature references for convenience
    f_name_tok_set = base_feats[2]   # name_token_set
    f_name_jw      = base_feats[4]   # name_jaro_winkler
    f_addr_tok_set = base_feats[8]   # addr_token_set
    f_addr_num_ov  = base_feats[10]  # addr_num_overlap

    # Advanced 1: Interaction terms
    f_name_x_addr   = f_name_jw * f_addr_tok_set
    f_min_name_addr = min(f_name_tok_set, f_addr_tok_set)
    f_max_name_addr = max(f_name_tok_set, f_addr_tok_set)

    # Advanced 2: Address missingness asymmetry
    len_a1 = len(s1_addr)
    len_a2 = len(s23_addr)
    f_both_addr = 1.0 if (len_a1 >= 5 and len_a2 >= 5) else 0.0
    f_missing_asym = 1.0 if ((len_a1 >= 10 and len_a2 < 3) or (len_a2 >= 10 and len_a1 < 3)) else 0.0

    # Advanced 3: Exact name match
    f_exact_name = 1.0 if (s1_name and s1_name == s23_name) else 0.0

    # Advanced 4: Conflicting digit analysis
    d1 = extract_digit_set(s1_addr)
    d2 = extract_digit_set(s23_addr)
    if d1 and d2 and not (d1 & d2):
        f_conflict_digits = 1.0
    else:
        f_conflict_digits = 0.0

    # Advanced 5: Evidence combinations
    f_strong_name_num = 1.0 if (f_name_tok_set >= 0.85 and f_addr_num_ov >= 0.5) else 0.0
    f_name_num_conflict = 1.0 if (f_name_tok_set >= 0.85 and f_conflict_digits == 1.0) else 0.0

    # Advanced 6: Candidate Source (S2 vs S3)
    f_is_s3 = 1.0 if s23_id.startswith("S3-") else 0.0

    # Advanced 7: Candidate Rank
    f_cand_rank = float(cand_rank)

    advanced_list = [
        f_name_x_addr,
        f_min_name_addr,
        f_max_name_addr,
        f_both_addr,
        f_missing_asym,
        f_exact_name,
        f_conflict_digits,
        f_strong_name_num,
        f_name_num_conflict,
        f_is_s3,
        f_cand_rank,
    ]

    return base_feats + advanced_list


def build_advanced_feature_matrix(
    pairs: List[Tuple[str, str]],
    s1_lookup: Dict,
    s23_lookup: Dict,
    candidate_ranks: Dict[Tuple[str, str], int] = None,
    n_jobs: int = -1,
) -> np.ndarray:
    if candidate_ranks is None:
        candidate_ranks = {}

    rows = []
    for s1_id, s23_id in pairs:
        s1  = s1_lookup[s1_id]
        s23 = s23_lookup[s23_id]
        rank = candidate_ranks.get((s1_id, s23_id), 1)
        rows.append(compute_advanced_features_for_pair(
            s1["name_norm"],  s1["addr_norm"],  s1["country_norm"],
            s23["name_norm"], s23["addr_norm"], s23["country_norm"],
            s23_id=s23_id,
            cand_rank=rank,
            blocking_score=0.0,
        ))
    return np.array(rows, dtype=np.float32)
