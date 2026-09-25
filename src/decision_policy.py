"""
decision_policy.py — Entity-Level Smart Decision Policy Suite.

Functions:
1. evaluate_global_threshold()
2. evaluate_source_specific_thresholds()
3. evaluate_confidence_margin_policy()
4. evaluate_full_smart_policy()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from evaluate import score_f05, f05_per_entity, score_f05


def apply_smart_decision_policy(
    s1_ids: List[str],
    cands_dict: Dict[str, List[str]],
    prob_map: Dict[Tuple[str, str], float],
    thr_s2: float = 0.45,
    thr_s3: float = 0.45,
    min_singleton_conf: float = 0.35,
    min_prob_margin: float = 0.05,
    drop_conflicting_numbers: bool = False,
    conflict_map: Dict[Tuple[str, str], float] = None,
) -> Dict[str, List[str]]:
    """
    Applies an entity-level decision policy to determine match output per S1 entity.
    """
    if conflict_map is None:
        conflict_map = {}

    pred_dict = {}

    for sid in s1_ids:
        cands = cands_dict.get(sid, [])
        if not cands:
            pred_dict[sid] = []
            continue

        # Get candidates with raw probability above source-specific threshold
        valid_cands = []
        scores = []
        for c in cands:
            p = prob_map.get((sid, c), 0.0)
            thr = thr_s3 if c.startswith("S3-") else thr_s2

            if drop_conflicting_numbers and conflict_map.get((sid, c), 0.0) == 1.0:
                continue

            if p >= thr:
                valid_cands.append(c)
                scores.append(p)

        if not valid_cands:
            pred_dict[sid] = []
            continue

        # Sort valid candidates by probability descending
        order = np.argsort(scores)[::-1]
        sorted_cands = [valid_cands[i] for i in order]
        sorted_scores = [scores[i] for i in order]

        top_prob = sorted_scores[0]

        # Policy Rule 1: Singleton / Minimum Confidence Floor
        if top_prob < min_singleton_conf:
            pred_dict[sid] = []
            continue

        # Policy Rule 2: Confidence Margin Filter for Ambiguous Multi-Candidates
        if len(sorted_scores) > 1:
            second_prob = sorted_scores[1]
            margin = top_prob - second_prob
            # If top score is moderate (0.30 - 0.65) and margin is tiny, reject ambiguous top matches
            if 0.30 <= top_prob <= 0.65 and margin < min_prob_margin:
                pred_dict[sid] = []
                continue

        pred_dict[sid] = sorted_cands

    return pred_dict


def optimize_smart_decision_policy(
    s1_df: pd.DataFrame,
    cands_dict: Dict[str, List[str]],
    gt_dict: Dict[str, List[str]],
    prob_map: Dict[Tuple[str, str], float],
    conflict_map: Dict[Tuple[str, str], float] = None,
) -> dict:
    """
    Grid searches smart decision policy parameters on validation set to maximize Macro F0.5.
    """
    s1_ids = s1_df["entity_id"].tolist()

    best_f05 = -1.0
    best_policy = {}

    s2_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    s3_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    conf_grid = [0.20, 0.30, 0.35, 0.40]
    margin_grid = [0.0, 0.02, 0.05, 0.08]

    for ts2 in s2_grid:
        for ts3 in s3_grid:
            for conf in conf_grid:
                for margin in margin_grid:
                    pred_dict = apply_smart_decision_policy(
                        s1_ids=s1_ids,
                        cands_dict=cands_dict,
                        prob_map=prob_map,
                        thr_s2=ts2,
                        thr_s3=ts3,
                        min_singleton_conf=conf,
                        min_prob_margin=margin,
                        drop_conflicting_numbers=(conflict_map is not None),
                        conflict_map=conflict_map,
                    )

                    f05 = score_f05(pred_dict, gt_dict)
                    if f05 > best_f05:
                        best_f05 = f05
                        best_policy = {
                            "thr_s2": ts2,
                            "thr_s3": ts3,
                            "min_singleton_conf": conf,
                            "min_prob_margin": margin,
                            "drop_conflicting_numbers": (conflict_map is not None),
                            "val_f05": f05,
                        }

    return best_policy
