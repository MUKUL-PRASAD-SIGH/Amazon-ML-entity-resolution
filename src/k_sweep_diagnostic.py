"""
k_sweep_diagnostic.py — Measure blocking candidate recall vs K.

Runs blocking at K = [10, 20, 30, 50, 100] on the 5K dev sample,
then for each K reports:
  - Candidate recall   (what % of GT pairs are in the candidate set)
  - Avg candidates/S1  (reduction ratio indicator)
  - LightGBM F0.5      (end-to-end model quality at that K)
  - Runtime

Usage:
    python src/k_sweep_diagnostic.py [--sample 5000] [--ks 10 20 30 50 100]

Outputs a results table to stdout AND writes to:
    models/k_sweep_results.tsv
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS, MODEL_DIR, TRAIN_DIR,
    NEG_PER_POS, RANDOM_STATE, THRESHOLD_GRID, VAL_FRACTION,
)
from preprocess import preprocess_df
from blocking import generate_candidates
from features import build_feature_matrix, build_lookup, FEATURE_NAMES
from evaluate import score_f05
from train_model import load_train_data, split_s1, load_ground_truth, build_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_k_experiment(
    s1_train, s1_val,
    s2, s3,
    gt_train, gt_val,
    s1_lookup_train, s1_lookup_val,
    s23_lookup,
    k: int,
) -> dict:
    """Run a single K experiment and return all metrics."""
    t_start = time.time()
    result = {"K": k}

    # ── Blocking ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"K = {k}")
    logger.info(f"{'='*60}")

    cands_train = generate_candidates(s1_train, s2, s3, k=k)
    cands_val   = generate_candidates(s1_val,   s2, s3, k=k)

    # ── Candidate recall ──────────────────────────────────────────────────
    def candidate_recall(s1_df, gt_dict, cands):
        total_gt = sum(len(v) for v in gt_dict.values() if v)
        found    = 0
        for s1_id, gt_ids in gt_dict.items():
            cand_set = set(cands.get(s1_id, []))
            found += sum(1 for g in gt_ids if g in cand_set)
        return found, total_gt, found / max(total_gt, 1)

    tr_found, tr_total, tr_recall = candidate_recall(s1_train, gt_train, cands_train)
    vl_found, vl_total, vl_recall = candidate_recall(s1_val,   gt_val,   cands_val)

    avg_cands_train = np.mean([len(v) for v in cands_train.values()])
    avg_cands_val   = np.mean([len(v) for v in cands_val.values()])

    result["train_cand_recall"] = round(tr_recall, 4)
    result["val_cand_recall"]   = round(vl_recall, 4)
    result["avg_cands_per_s1"]  = round(avg_cands_val, 1)
    result["total_gt_train"]    = tr_total
    result["found_gt_train"]    = tr_found

    logger.info(f"  Train cand recall: {tr_found}/{tr_total} = {tr_recall:.4f}")
    logger.info(f"  Val   cand recall: {vl_found}/{vl_total} = {vl_recall:.4f}")
    logger.info(f"  Avg candidates/S1 (val): {avg_cands_val:.1f}")

    # ── Build labeled pairs ───────────────────────────────────────────────
    train_pairs, train_labels = build_pairs(s1_train, cands_train, gt_train)
    val_pairs,   val_labels   = build_pairs(s1_val,   cands_val,   gt_val)

    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    logger.info(f"  Train pairs: {len(train_pairs)} ({pos_count} pos, {neg_count} neg)")

    # ── Features ──────────────────────────────────────────────────────────
    X_train = build_feature_matrix(train_pairs, s1_lookup_train, s23_lookup, n_jobs=-1)
    X_val   = build_feature_matrix(val_pairs,   s1_lookup_val,   s23_lookup, n_jobs=-1)
    y_train = np.array(train_labels, dtype=np.int32)
    y_val   = np.array(val_labels,   dtype=np.int32)

    # ── LightGBM ──────────────────────────────────────────────────────────
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed")
        return result

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=FEATURE_NAMES, reference=dtrain)

    params = dict(LGBM_PARAMS)
    params.pop("n_estimators", None)
    try:
        import cupy as cp
        params["device_type"] = "gpu"
    except ImportError:
        pass

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=LGBM_PARAMS["n_estimators"],
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(200),
        ],
    )

    val_probs = model.predict(X_val)

    # ── Threshold sweep → F0.5 ────────────────────────────────────────────
    prob_map = {(s1_id, s23_id): prob
                for (s1_id, s23_id), prob in zip(val_pairs, val_probs)}

    best_f05, best_thr = -1.0, 0.5
    thr_results = {}
    for thr in THRESHOLD_GRID:
        pred_dict = {}
        for s1_id in s1_val["entity_id"]:
            cands = cands_val.get(s1_id, [])
            pred_dict[s1_id] = [c for c in cands
                                 if prob_map.get((s1_id, c), 0.0) >= thr]
        f05 = score_f05(pred_dict, gt_val)
        thr_results[thr] = f05
        if f05 > best_f05:
            best_f05, best_thr = f05, thr

    result["best_f05"]       = round(best_f05, 4)
    result["best_threshold"] = best_thr
    result["auc"]            = round(float(model.best_score.get("valid_0", {}).get("auc", 0)), 4)
    result["runtime_s"]      = round(time.time() - t_start, 1)

    logger.info(f"  Best threshold: {best_thr}  |  Val F₀.₅: {best_f05:.4f}")
    logger.info(f"  Val AUC: {result['auc']:.4f}  |  Runtime: {result['runtime_s']}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="K-sweep blocking diagnostic")
    parser.add_argument("--sample", type=int, default=50000,
                        help="Number of S1 entities for dev run")
    parser.add_argument("--ks", nargs="+", type=int,
                        default=[10, 20, 30, 50, 100],
                        help="K values to test")
    args = parser.parse_args()

    logger.info(f"K-sweep diagnostic | sample={args.sample} | ks={args.ks}")

    # Load once, reuse for all K experiments
    logger.info("Loading data (once)...")
    s1_raw, s2_raw, s3_raw, gt_raw = load_train_data(sample=args.sample)

    logger.info("Preprocessing...")
    s1 = preprocess_df(s1_raw)
    s2 = preprocess_df(s2_raw)
    s3 = preprocess_df(s3_raw)

    s1_train, s1_val = split_s1(s1)
    all_ids = set(s1["entity_id"])
    gt_dict  = load_ground_truth(gt_raw, all_ids)
    gt_train = {k: v for k, v in gt_dict.items() if k in set(s1_train["entity_id"])}
    gt_val   = {k: v for k, v in gt_dict.items() if k in set(s1_val["entity_id"])}

    s1_lookup_train = build_lookup(s1_train)
    s1_lookup_val   = build_lookup(s1_val)
    s23_lookup      = build_lookup(pd.concat([s2, s3], ignore_index=True))

    # Run experiments
    all_results = []
    for k in args.ks:
        res = run_k_experiment(
            s1_train, s1_val, s2, s3,
            gt_train, gt_val,
            s1_lookup_train, s1_lookup_val, s23_lookup,
            k=k,
        )
        all_results.append(res)

    # Summary table
    cols = ["K", "val_cand_recall", "avg_cands_per_s1", "best_f05",
            "best_threshold", "auc", "runtime_s"]
    df = pd.DataFrame(all_results)[cols]

    print("\n" + "="*70)
    print("K-SWEEP RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    out_path = MODEL_DIR / "k_sweep_results.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
