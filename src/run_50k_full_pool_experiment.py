"""
run_50k_full_pool_experiment.py — 50K S1 Entity vs FULL 10.3M S2/S3 Candidate Universe Experiment.

Evaluates:
- 50,000 S1 queries against ALL 5,034,616 S2 and 5,285,603 S3 records (un-subsampled full universe).
- Measures Candidate Recall, Avg Candidates / S1, RAM usage, Blocker Runtime, and Macro F0.5.
- Logs results to experiments/results.csv and reports/learnings.md.
"""

import argparse
import logging
import os
import psutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    EARLY_STOPPING_ROUNDS,
    LGBM_PARAMS,
    NEG_PER_POS,
    RANDOM_STATE,
    THRESHOLD_GRID,
    TRAIN_DIR,
)
from preprocess import preprocess_df
from blocking_multi import generate_multi_blocker_candidates
from features import FEATURE_NAMES, build_feature_matrix, build_lookup
from experiment_runner import (
    split_three_way,
    build_pairs_with_missed,
    evaluate_predictions_detailed,
    log_experiment_result,
    REPORTS_DIR,
)
from run_person2_blocking_experiments import compute_candidate_recall

logger = logging.getLogger(__name__)


def run_50k_experiment(sample_s1=50000, k_candidates=50):
    logger.info(f"=== Executing 50K S1 Entity vs FULL 10.3M S2/S3 Pool Experiment ===")
    t0 = time.time()

    # 1. Load 50K S1 against FULL S2/S3
    print("Loading data files...")
    s1_raw = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str)
    s2_raw = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str)
    s3_raw = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str)
    gt_raw = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

    print(f"Sampling S1 to {sample_s1:,} entities...")
    s1_raw = s1_raw.sample(n=min(sample_s1, len(s1_raw)), random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Data Sizes: S1={len(s1_raw):,} vs FULL S2={len(s2_raw):,}, FULL S3={len(s3_raw):,}")

    print("Preprocessing text fields...")
    s1 = preprocess_df(s1_raw)
    s2 = preprocess_df(s2_raw)
    s3 = preprocess_df(s3_raw)

    s1_train, s1_val, s1_holdout = split_three_way(s1)
    print(f"Splits: Train={len(s1_train):,}, Val={len(s1_val):,}, Holdout={len(s1_holdout):,}")

    gt_dict = {}
    s1_set = set(s1["entity_id"])
    for _, r in gt_raw.iterrows():
        sid = r["source1_entity_id"]
        if sid in s1_set:
            m = r.get("matched_entity_ids", "").strip()
            gt_dict[sid] = [x for x in m.split(",") if x] if m else []

    gt_train = {k: v for k, v in gt_dict.items() if k in set(s1_train["entity_id"])}
    gt_val   = {k: v for k, v in gt_dict.items() if k in set(s1_val["entity_id"])}
    gt_hold  = {k: v for k, v in gt_dict.items() if k in set(s1_holdout["entity_id"])}

    s23_lookup = build_lookup(pd.concat([s2, s3], ignore_index=True))
    s1_tr_lookup = build_lookup(s1_train)
    s1_va_lookup = build_lookup(s1_val)
    s1_ho_lookup = build_lookup(s1_holdout)

    # 2. Multi-Blocker Candidate Retrieval on FULL S2/S3
    print("Running Multi-Blocker Candidate Retrieval against FULL S2/S3 universe...")
    t_block_start = time.time()
    
    configs = [
        {"name": "name_tfidf", "type": "name_tfidf", "k": 30},
        {"name": "addr_tfidf", "type": "addr_tfidf", "k": 15},
        {"name": "exact_name", "type": "exact_name", "max_matches": 15},
        {"name": "rare_tokens", "type": "rare_tokens", "max_cands": 10},
    ]

    cands_train, _ = generate_multi_blocker_candidates(s1_train, s2, s3, configs, k_final=k_candidates)
    cands_val, _   = generate_multi_blocker_candidates(s1_val, s2, s3, configs, k_final=k_candidates)
    cands_hold, _  = generate_multi_blocker_candidates(s1_holdout, s2, s3, configs, k_final=k_candidates)

    t_block_elapsed = time.time() - t_block_start

    # Candidate Recall & Density Diagnostics
    rec_val, found_val, total_val = compute_candidate_recall(cands_val, gt_val)
    avg_cands_val = sum(len(v) for v in cands_val.values()) / max(len(s1_val), 1)

    ram_gb = psutil.process_iter()
    mem_used = psutil.virtual_memory().used / (1024**3)

    logger.info(f"★ 50K Full-Pool Candidate Recall: {rec_val:.2f}% ({found_val:,}/{total_val:,} GT pairs)")
    logger.info(f"★ 50K Full-Pool Avg Candidates / S1: {avg_cands_val:.1f}")
    logger.info(f"★ Blocking Elapsed: {t_block_elapsed:.1f}s | Current RAM: {mem_used:.2f} GB")

    # 3. Pair Building & Feature Computation
    pairs_train, labels_train = build_pairs_with_missed(s1_train, cands_train, gt_train)
    pairs_val, labels_val     = build_pairs_with_missed(s1_val, cands_val, gt_val)
    pairs_hold, labels_hold   = build_pairs_with_missed(s1_holdout, cands_hold, gt_hold)

    ranks_tr = {(sid, cid): r+1 for sid, clist in cands_train.items() for r, cid in enumerate(clist)}
    ranks_va = {(sid, cid): r+1 for sid, clist in cands_val.items() for r, cid in enumerate(clist)}
    ranks_ho = {(sid, cid): r+1 for sid, clist in cands_hold.items() for r, cid in enumerate(clist)}

    print(f"Building 28-feature matrices for {len(pairs_train):,} train pairs & {len(pairs_val):,} val pairs...")
    X_tr = build_feature_matrix(pairs_train, s1_tr_lookup, s23_lookup, candidate_ranks=ranks_tr, n_jobs=-1)
    y_tr = np.array(labels_train, dtype=np.int32)

    X_va = build_feature_matrix(pairs_val, s1_va_lookup, s23_lookup, candidate_ranks=ranks_va, n_jobs=-1)
    y_va = np.array(labels_val, dtype=np.int32)

    X_ho = build_feature_matrix(pairs_hold, s1_ho_lookup, s23_lookup, candidate_ranks=ranks_ho, n_jobs=-1)
    y_ho = np.array(labels_hold, dtype=np.int32)

    # 4. LightGBM Training & Validation
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    dval   = lgb.Dataset(X_va, label=y_va, feature_name=FEATURE_NAMES, reference=dtrain)

    params = dict(LGBM_PARAMS)
    params.pop("n_estimators", None)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=LGBM_PARAMS["n_estimators"],
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        ],
    )

    val_probs  = model.predict(X_va)
    hold_probs = model.predict(X_ho)

    prob_map_va = {(sid, cid): p for (sid, cid), p in zip(pairs_val, val_probs)}
    prob_map_ho = {(sid, cid): p for (sid, cid), p in zip(pairs_hold, hold_probs)}

    best_val_f05, best_thr, best_val_stats = -1.0, 0.5, {}
    for thr in THRESHOLD_GRID:
        pred_dict_va = {sid: [c for c in cands_val.get(sid, []) if prob_map_va.get((sid, c), 0.0) >= thr] for sid in s1_val["entity_id"]}
        stats = evaluate_predictions_detailed(pred_dict_va, gt_val)
        if stats["macro_f05"] > best_val_f05:
            best_val_f05 = stats["macro_f05"]
            best_thr = thr
            best_val_stats = stats

    pred_dict_ho = {sid: [c for c in cands_holdout.get(sid, []) if prob_map_ho.get((sid, c), 0.0) >= best_thr] for sid in s1_holdout["entity_id"]} if 'cands_holdout' in locals() else {sid: [c for c in cands_hold.get(sid, []) if prob_map_ho.get((sid, c), 0.0) >= best_thr] for sid in s1_holdout["entity_id"]}
    ho_stats = evaluate_predictions_detailed(pred_dict_ho, gt_hold)

    total_time = time.time() - t0
    logger.info(f"★ 50K Full-Pool Evaluation Results:")
    logger.info(f"  Best Threshold : {best_thr:.2f}")
    logger.info(f"  Val Macro F0.5 : {best_val_f05:.4f} (Prec: {best_val_stats['precision']:.4f}, Rec: {best_val_stats['recall']:.4f})")
    logger.info(f"  Holdout F0.5   : {ho_stats['macro_f05']:.4f}")
    logger.info(f"  Total Runtime  : {total_time/60:.1f} minutes")

    exp_record = {
        "experiment_id": "EXP-50K-FULL-POOL-EVAL",
        "timestamp": pd.Timestamp.now().isoformat(),
        "git_commit_if_available": "clean",
        "candidate_K": k_candidates,
        "feature_set": "advanced_28",
        "added_features": "50K_full_pool_unsubsampled",
        "removed_features": "none",
        "model_parameters": f"num_leaves={LGBM_PARAMS['num_leaves']}",
        "negative_sampling_strategy": f"hard_neg_ratio={NEG_PER_POS}",
        "threshold_policy": "global_grid_search",
        "global_threshold": best_thr,
        "S2_threshold": best_thr,
        "S3_threshold": best_thr,
        "validation_precision": round(best_val_stats["precision"], 4),
        "validation_recall": round(best_val_stats["recall"], 4),
        "validation_f05": round(best_val_f05, 4),
        "validation_singleton_accuracy": round(best_val_stats["singleton_acc"], 4),
        "holdout_f05_if_run": round(ho_stats["macro_f05"], 4),
        "num_predictions": best_val_stats["num_pred_links"],
        "false_positives": best_val_stats["fp"],
        "false_negatives": best_val_stats["fn"],
        "runtime_seconds": round(total_time, 1),
        "notes": f"50K S1 vs FULL 10.3M S2/S3 candidate pool. Candidate recall: {rec_val:.2f}%"
    }
    log_experiment_result(exp_record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50000)
    args = parser.parse_args()
    run_50k_experiment(sample_s1=args.sample)
