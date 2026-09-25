"""
ablation_runner.py — Controlled Feature Group Ablation Suite.

Evaluates feature groups separately on validation macro F0.5, precision, recall,
singleton accuracy, false positive count, false negative count, and runtime.

Outputs:
  reports/feature_ablation.csv
  Appends each ablation run to experiments/results.csv
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
from features import FEATURE_NAMES, build_feature_matrix, build_lookup
from evaluate import f05_per_entity
from experiment_runner import (
    load_and_preprocess_data,
    split_three_way,
    build_pairs_with_missed,
    evaluate_predictions_detailed,
    log_experiment_result,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)

FEATURE_GROUPS = {
    "all_features": FEATURE_NAMES,
    "minus_name_features": [f for f in FEATURE_NAMES if not f.startswith("name_") or f in ("name_len_s1", "name_len_s23", "name_len_ratio")],
    "minus_address_features": [f for f in FEATURE_NAMES if not f.startswith("addr_")],
    "minus_number_features": [f for f in FEATURE_NAMES if f != "addr_num_overlap"],
    "minus_country": [f for f in FEATURE_NAMES if f != "country_match"],
    "minus_tfidf": [f for f in FEATURE_NAMES if f != "blocking_score"],
    "minus_fuzzy": [f for f in FEATURE_NAMES if f not in ("name_ratio", "name_partial", "name_jaro_winkler", "addr_ratio")],
}


def run_ablation_experiment(sample=5000, k_candidates=50):
    logger.info("=== Running Step 3 Feature Group Ablation Suite ===")
    
    s1, s2, s3, gt_dict = load_and_preprocess_data(sample=sample)
    s1_train, s1_val, s1_holdout = split_three_way(s1)
    
    s23_lookup = build_lookup(pd.concat([s2, s3], ignore_index=True))
    s1_train_lookup = build_lookup(s1_train)
    s1_val_lookup   = build_lookup(s1_val)

    from blocking import generate_candidates
    cands_train = generate_candidates(s1_train, s2, s3, k=k_candidates)
    cands_val   = generate_candidates(s1_val, s2, s3, k=k_candidates)

    gt_train = {k: v for k, v in gt_dict.items() if k in set(s1_train["entity_id"])}
    gt_val   = {k: v for k, v in gt_dict.items() if k in set(s1_val["entity_id"])}

    train_pairs, train_labels = build_pairs_with_missed(s1_train, cands_train, gt_train)
    val_pairs, val_labels     = build_pairs_with_missed(s1_val, cands_val, gt_val)

    X_train_full = build_feature_matrix(train_pairs, s1_train_lookup, s23_lookup, n_jobs=-1)
    y_train      = np.array(train_labels, dtype=np.int32)

    X_val_full   = build_feature_matrix(val_pairs, s1_val_lookup, s23_lookup, n_jobs=-1)
    y_val        = np.array(val_labels, dtype=np.int32)

    ablation_records = []

    for group_name, active_features in FEATURE_GROUPS.items():
        t0 = time.time()
        feat_indices = [FEATURE_NAMES.index(f) for f in active_features]

        X_tr = X_train_full[:, feat_indices]
        X_va = X_val_full[:, feat_indices]

        dtrain = lgb.Dataset(X_tr, label=y_train, feature_name=active_features)
        dval   = lgb.Dataset(X_va, label=y_val, feature_name=active_features, reference=dtrain)

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

        val_probs = model.predict(X_va)
        prob_map_val = {(s1_id, s23_id): prob for (s1_id, s23_id), prob in zip(val_pairs, val_probs)}

        best_val_f05 = -1.0
        best_thr = 0.5
        best_stats = {}

        for thr in THRESHOLD_GRID:
            pred_dict_val = {}
            for sid in s1_val["entity_id"]:
                cands = cands_val.get(sid, [])
                matched = [c for c in cands if prob_map_val.get((sid, c), 0.0) >= thr]
                pred_dict_val[sid] = matched

            stats = evaluate_predictions_detailed(pred_dict_val, gt_val)
            if stats["macro_f05"] > best_val_f05:
                best_val_f05 = stats["macro_f05"]
                best_thr = thr
                best_stats = stats

        elapsed = time.time() - t0
        logger.info(f"Ablation [{group_name:25s}] Thr={best_thr:.2f} | Val F0.5={best_val_f05:.4f} | Prec={best_stats['precision']:.4f} | Rec={best_stats['recall']:.4f} | SingAcc={best_stats['singleton_acc']:.4f} | FP={best_stats['fp']} | FN={best_stats['fn']}")

        rec = {
            "group_name": group_name,
            "num_features": len(active_features),
            "macro_f05": round(best_val_f05, 4),
            "precision": round(best_stats["precision"], 4),
            "recall": round(best_stats["recall"], 4),
            "singleton_acc": round(best_stats["singleton_acc"], 4),
            "fp_count": best_stats["fp"],
            "fn_count": best_stats["fn"],
            "num_predictions": best_stats["num_pred_links"],
            "best_threshold": best_thr,
            "runtime_seconds": round(elapsed, 1),
            "active_features": ",".join(active_features),
        }
        ablation_records.append(rec)

        # Log to experiments/results.csv
        exp_record = {
            "experiment_id": f"EXP-003-ABLATION-{group_name.upper()}",
            "timestamp": pd.Timestamp.now().isoformat(),
            "git_commit_if_available": "clean",
            "candidate_K": k_candidates,
            "feature_set": group_name,
            "added_features": "none",
            "removed_features": ",".join([f for f in FEATURE_NAMES if f not in active_features]),
            "model_parameters": f"num_leaves={LGBM_PARAMS['num_leaves']}",
            "negative_sampling_strategy": f"hard_neg_ratio={NEG_PER_POS}",
            "threshold_policy": "global_grid_search",
            "global_threshold": best_thr,
            "S2_threshold": best_thr,
            "S3_threshold": best_thr,
            "validation_precision": round(best_stats["precision"], 4),
            "validation_recall": round(best_stats["recall"], 4),
            "validation_f05": round(best_val_f05, 4),
            "validation_singleton_accuracy": round(best_stats["singleton_acc"], 4),
            "holdout_f05_if_run": None,
            "num_predictions": best_stats["num_pred_links"],
            "false_positives": best_stats["fp"],
            "false_negatives": best_stats["fn"],
            "runtime_seconds": round(elapsed, 1),
            "notes": f"Feature ablation experiment for group: {group_name}"
        }
        log_experiment_result(exp_record)

    df_ab = pd.DataFrame(ablation_records)
    df_ab.to_csv(REPORTS_DIR / "feature_ablation.csv", index=False)
    logger.info(f"Saved feature ablation report to {REPORTS_DIR / 'feature_ablation.csv'}")
    return df_ab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args()
    run_ablation_experiment(sample=args.sample)
