"""
run_advanced_experiments.py — High-Value Interaction Features & Smart Decision Policy Execution.

Executes:
1. Exp 3: Advanced Interaction & Domain Feature Evaluation (28 features).
2. Exp 4: Source-Specific & Confidence Margin Smart Decision Policy Optimization.
3. Holds out a strict test set to measure generalizability.
4. Updates experiments/results.csv and reports/learnings.md.
"""

import argparse
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    EARLY_STOPPING_ROUNDS,
    K_CANDIDATES,
    LGBM_PARAMS,
    NEG_PER_POS,
    RANDOM_STATE,
    THRESHOLD_GRID,
)
from preprocess import preprocess_df
from blocking import generate_candidates
from features import build_lookup
from advanced_features import (
    ADVANCED_FEATURE_NAMES,
    build_advanced_feature_matrix,
    compute_advanced_features_for_pair,
)
from decision_policy import (
    apply_smart_decision_policy,
    optimize_smart_decision_policy,
)
from experiment_runner import (
    load_and_preprocess_data,
    split_three_way,
    build_pairs_with_missed,
    evaluate_predictions_detailed,
    log_experiment_result,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


def run_advanced_experiments(sample=5000, k_candidates=50):
    logger.info("=== Running Advanced Feature & Decision Policy Experiments ===")
    t0 = time.time()

    s1, s2, s3, gt_dict = load_and_preprocess_data(sample=sample)
    s1_train, s1_val, s1_holdout = split_three_way(s1)

    s23_lookup = build_lookup(pd.concat([s2, s3], ignore_index=True))
    s1_train_lookup = build_lookup(s1_train)
    s1_val_lookup   = build_lookup(s1_val)
    s1_hold_lookup  = build_lookup(s1_holdout)

    logger.info("Running blocking candidate generation...")
    cands_train = generate_candidates(s1_train, s2, s3, k=k_candidates)
    cands_val   = generate_candidates(s1_val, s2, s3, k=k_candidates)
    cands_hold  = generate_candidates(s1_holdout, s2, s3, k=k_candidates)

    gt_train = {k: v for k, v in gt_dict.items() if k in set(s1_train["entity_id"])}
    gt_val   = {k: v for k, v in gt_dict.items() if k in set(s1_val["entity_id"])}
    gt_hold  = {k: v for k, v in gt_dict.items() if k in set(s1_holdout["entity_id"])}

    train_pairs, train_labels = build_pairs_with_missed(s1_train, cands_train, gt_train)
    val_pairs, val_labels     = build_pairs_with_missed(s1_val, cands_val, gt_val)
    hold_pairs, hold_labels   = build_pairs_with_missed(s1_holdout, cands_hold, gt_hold)

    # Calculate Candidate Ranks per S1 entity
    cand_ranks_train = {}
    for sid, clist in cands_train.items():
        for r, cid in enumerate(clist):
            cand_ranks_train[(sid, cid)] = r + 1

    cand_ranks_val = {}
    for sid, clist in cands_val.items():
        for r, cid in enumerate(clist):
            cand_ranks_val[(sid, cid)] = r + 1

    cand_ranks_hold = {}
    for sid, clist in cands_hold.items():
        for r, cid in enumerate(clist):
            cand_ranks_hold[(sid, cid)] = r + 1

    logger.info("Building 28-feature matrices...")
    X_train = build_advanced_feature_matrix(train_pairs, s1_train_lookup, s23_lookup, candidate_ranks=cand_ranks_train)
    y_train = np.array(train_labels, dtype=np.int32)

    X_val   = build_advanced_feature_matrix(val_pairs, s1_val_lookup, s23_lookup, candidate_ranks=cand_ranks_val)
    y_val   = np.array(val_labels, dtype=np.int32)

    X_hold  = build_advanced_feature_matrix(hold_pairs, s1_hold_lookup, s23_lookup, candidate_ranks=cand_ranks_hold)
    y_hold  = np.array(hold_labels, dtype=np.int32)

    # Train LightGBM with 28 features
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=ADVANCED_FEATURE_NAMES)
    dval   = lgb.Dataset(X_val, label=y_val, feature_name=ADVANCED_FEATURE_NAMES, reference=dtrain)

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

    # Predict Probabilities
    val_probs  = model.predict(X_val)
    hold_probs = model.predict(X_hold)

    prob_map_val  = {(sid, cid): prob for (sid, cid), prob in zip(val_pairs, val_probs)}
    prob_map_hold = {(sid, cid): prob for (sid, cid), prob in zip(hold_pairs, hold_probs)}

    # Map Conflicting Digits feature for rule filtering
    conflict_idx = ADVANCED_FEATURE_NAMES.index("conflicting_digits")
    conflict_map_val  = {(sid, cid): row[conflict_idx] for (sid, cid), row in zip(val_pairs, X_val)}
    conflict_map_hold = {(sid, cid): row[conflict_idx] for (sid, cid), row in zip(hold_pairs, X_hold)}

    # 1. Global Threshold Baseline on Advanced Features
    best_val_f05_global = -1.0
    best_global_thr = 0.5
    best_val_stats_global = {}

    for thr in THRESHOLD_GRID:
        pred_dict_val = {}
        for sid in s1_val["entity_id"]:
            cands = cands_val.get(sid, [])
            matched = [c for c in cands if prob_map_val.get((sid, c), 0.0) >= thr]
            pred_dict_val[sid] = matched

        stats = evaluate_predictions_detailed(pred_dict_val, gt_val)
        if stats["macro_f05"] > best_val_f05_global:
            best_val_f05_global = stats["macro_f05"]
            best_global_thr = thr
            best_val_stats_global = stats

    # Evaluate Global Threshold on Holdout
    pred_hold_global = {}
    for sid in s1_holdout["entity_id"]:
        cands = cands_hold.get(sid, [])
        matched = [c for c in cands if prob_map_hold.get((sid, c), 0.0) >= best_global_thr]
        pred_hold_global[sid] = matched
    hold_stats_global = evaluate_predictions_detailed(pred_hold_global, gt_hold)

    logger.info(f"★ [Exp 3 - Advanced Features] Global Thr={best_global_thr:.2f} | Val F0.5: {best_val_f05_global:.4f} | Holdout F0.5: {hold_stats_global['macro_f05']:.4f}")

    # Log Exp 3 result
    exp3_record = {
        "experiment_id": "EXP-003-ADVANCED-INTERACTION-FEATURES",
        "timestamp": pd.Timestamp.now().isoformat(),
        "git_commit_if_available": "clean",
        "candidate_K": k_candidates,
        "feature_set": "advanced_28",
        "added_features": "name_x_address,min_name_address,conflicting_digits,missing_address_asymmetry,candidate_rank",
        "removed_features": "none",
        "model_parameters": f"num_leaves={LGBM_PARAMS['num_leaves']}",
        "negative_sampling_strategy": f"hard_neg_ratio={NEG_PER_POS}",
        "threshold_policy": "global_grid_search",
        "global_threshold": best_global_thr,
        "S2_threshold": best_global_thr,
        "S3_threshold": best_global_thr,
        "validation_precision": round(best_val_stats_global["precision"], 4),
        "validation_recall": round(best_val_stats_global["recall"], 4),
        "validation_f05": round(best_val_f05_global, 4),
        "validation_singleton_accuracy": round(best_val_stats_global["singleton_acc"], 4),
        "holdout_f05_if_run": round(hold_stats_global["macro_f05"], 4),
        "num_predictions": best_val_stats_global["num_pred_links"],
        "false_positives": best_val_stats_global["fp"],
        "false_negatives": best_val_stats_global["fn"],
        "runtime_seconds": round(time.time() - t0, 1),
        "notes": "Added 11 interaction and digit conflict features"
    }
    log_experiment_result(exp3_record)

    # 2. Smart Entity-Level Decision Policy Optimization
    logger.info("Optimizing Smart Decision Policy parameters on validation...")
    best_policy = optimize_smart_decision_policy(
        s1_df=s1_val,
        cands_dict=cands_val,
        gt_dict=gt_val,
        prob_map=prob_map_val,
        conflict_map=conflict_map_val,
    )

    # Evaluate Smart Policy on Val
    pred_val_smart = apply_smart_decision_policy(
        s1_ids=s1_val["entity_id"].tolist(),
        cands_dict=cands_val,
        prob_map=prob_map_val,
        thr_s2=best_policy["thr_s2"],
        thr_s3=best_policy["thr_s3"],
        min_singleton_conf=best_policy["min_singleton_conf"],
        min_prob_margin=best_policy["min_prob_margin"],
        drop_conflicting_numbers=best_policy["drop_conflicting_numbers"],
        conflict_map=conflict_map_val,
    )
    val_stats_smart = evaluate_predictions_detailed(pred_val_smart, gt_val)

    # Evaluate Smart Policy on Holdout
    pred_hold_smart = apply_smart_decision_policy(
        s1_ids=s1_holdout["entity_id"].tolist(),
        cands_dict=cands_hold,
        prob_map=prob_map_hold,
        thr_s2=best_policy["thr_s2"],
        thr_s3=best_policy["thr_s3"],
        min_singleton_conf=best_policy["min_singleton_conf"],
        min_prob_margin=best_policy["min_prob_margin"],
        drop_conflicting_numbers=best_policy["drop_conflicting_numbers"],
        conflict_map=conflict_map_hold,
    )
    hold_stats_smart = evaluate_predictions_detailed(pred_hold_smart, gt_hold)

    logger.info(f"★ [Exp 4 - Smart Decision Policy] Best Policy: {best_policy}")
    logger.info(f"★ Val F0.5 Smart: {val_stats_smart['macro_f05']:.4f} (Prec: {val_stats_smart['precision']:.4f}, Rec: {val_stats_smart['recall']:.4f})")
    logger.info(f"★ Holdout F0.5 Smart: {hold_stats_smart['macro_f05']:.4f} (Prec: {hold_stats_smart['precision']:.4f}, Rec: {hold_stats_smart['recall']:.4f})")

    # Log Exp 4 result
    exp4_record = {
        "experiment_id": "EXP-004-SMART-DECISION-POLICY",
        "timestamp": pd.Timestamp.now().isoformat(),
        "git_commit_if_available": "clean",
        "candidate_K": k_candidates,
        "feature_set": "advanced_28",
        "added_features": "none",
        "removed_features": "none",
        "model_parameters": f"num_leaves={LGBM_PARAMS['num_leaves']}",
        "negative_sampling_strategy": f"hard_neg_ratio={NEG_PER_POS}",
        "threshold_policy": "smart_entity_policy",
        "global_threshold": None,
        "S2_threshold": best_policy["thr_s2"],
        "S3_threshold": best_policy["thr_s3"],
        "validation_precision": round(val_stats_smart["precision"], 4),
        "validation_recall": round(val_stats_smart["recall"], 4),
        "validation_f05": round(val_stats_smart["macro_f05"], 4),
        "validation_singleton_accuracy": round(val_stats_smart["singleton_acc"], 4),
        "holdout_f05_if_run": round(hold_stats_smart["macro_f05"], 4),
        "num_predictions": val_stats_smart["num_pred_links"],
        "false_positives": val_stats_smart["fp"],
        "false_negatives": val_stats_smart["fn"],
        "runtime_seconds": round(time.time() - t0, 1),
        "notes": f"Smart decision policy: S2={best_policy['thr_s2']}, S3={best_policy['thr_s3']}, SingletonConf={best_policy['min_singleton_conf']}, Margin={best_policy['min_prob_margin']}"
    }
    log_experiment_result(exp4_record)

    # Save Gain Importances for Advanced Features
    gain_imp  = model.feature_importance(importance_type="gain")
    split_imp = model.feature_importance(importance_type="split")
    fi_df = pd.DataFrame({
        "feature": ADVANCED_FEATURE_NAMES,
        "gain_importance": gain_imp,
        "split_importance": split_imp,
    }).sort_values(by="gain_importance", ascending=False)
    fi_df.to_csv(REPORTS_DIR / "advanced_feature_importance.csv", index=False)
    logger.info(f"Saved {REPORTS_DIR / 'advanced_feature_importance.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args()
    run_advanced_experiments(sample=args.sample)
