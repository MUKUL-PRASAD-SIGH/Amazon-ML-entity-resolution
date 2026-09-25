"""
evaluate.py — Local F₀.₅ scorer (matches the official leaderboard metric).

Usage
-----
    from evaluate import score_f05, f05_macro

    # from a results dict
    f05 = score_f05(pred_dict, gt_dict)

    # from TSV files
    f05 = score_from_files("output/matching_results.tsv",
                           "dataset/train/train_ground_truth.tsv")
"""

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


# ── Per-entity F₀.₅ ──────────────────────────────────────────────────────────

def f05_per_entity(
    predicted: Set[str],
    ground_truth: Set[str],
) -> float:
    """
    Compute F₀.₅ for a single Source-1 entity.

    • Singleton entity (no true matches):
        - Correct prediction (empty predicted) → 1.0
        - Any false merge → 0.0

    • Non-singleton entity:
        Standard precision-recall formula with β=0.5.
    """
    # Singleton case
    if not ground_truth:
        return 1.0 if not predicted else 0.0

    if not predicted:
        # missed all true matches
        precision, recall = 0.0, 0.0
    else:
        tp = len(predicted & ground_truth)
        precision = tp / len(predicted)
        recall    = tp / len(ground_truth)

    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_sq = 0.25  # β=0.5 → β²=0.25
    f05 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return f05


# ── Macro-averaged F₀.₅ ──────────────────────────────────────────────────────

def score_f05(
    pred_dict: Dict[str, List[str]],
    gt_dict:   Dict[str, List[str]],
    verbose:   bool = False,
) -> float:
    """
    Macro-average F₀.₅ across all S1 entities present in gt_dict.

    Parameters
    ----------
    pred_dict : {s1_id: [matched_s23_ids]} — your predictions
    gt_dict   : {s1_id: [matched_s23_ids]} — ground truth
    verbose   : print per-bucket stats

    Returns
    -------
    Macro-averaged F₀.₅ (float, 0-1)
    """
    scores = []
    singletons_correct = 0
    singletons_total   = 0
    non_single_scores  = []

    for s1_id, gt_ids in gt_dict.items():
        gt_set   = set(gt_ids) if gt_ids else set()
        pred_set = set(pred_dict.get(s1_id, []))
        f = f05_per_entity(pred_set, gt_set)
        scores.append(f)

        if not gt_set:
            singletons_total += 1
            if not pred_set:
                singletons_correct += 1
        else:
            non_single_scores.append(f)

    macro = sum(scores) / len(scores) if scores else 0.0

    if verbose:
        ns = len(non_single_scores)
        ns_avg = sum(non_single_scores) / ns if ns else 0.0
        print(f"  Entities evaluated : {len(scores)}")
        print(f"  Singletons         : {singletons_total} "
              f"(correctly empty: {singletons_correct})")
        print(f"  Non-singleton avg  : {ns_avg:.4f}  ({ns} entities)")
        print(f"  Macro F₀.₅         : {macro:.4f}")

    return macro


# ── From files ────────────────────────────────────────────────────────────────

def _load_results_tsv(path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    result = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        ids_str = row.get("matched_entity_ids", "").strip()
        result[s1_id] = [x for x in ids_str.split(",") if x] if ids_str else []
    return result


def _load_gt_tsv(path: Path) -> Dict[str, List[str]]:
    return _load_results_tsv(path)


def score_from_files(
    pred_path: Path,
    gt_path:   Path,
    verbose:   bool = True,
) -> float:
    """Convenience wrapper: load TSV files and compute macro F₀.₅."""
    pred = _load_results_tsv(pred_path)
    gt   = _load_gt_tsv(gt_path)

    # Only score entities in the GT
    print(f"Scoring {len(gt)} GT entities vs {len(pred)} predicted...")
    return score_f05(pred, gt, verbose=verbose)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute local F₀.₅ score")
    parser.add_argument("--pred", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--gt",   required=True, help="Path to ground truth TSV")
    args = parser.parse_args()

    f05 = score_from_files(Path(args.pred), Path(args.gt), verbose=True)
    print(f"\n>>> Macro F₀.₅ = {f05:.6f}")
