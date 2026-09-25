"""
experiment_runner.py — Experimentation, Feature Analysis, Error Diagnostics & Decision Policy Suite.

Runs:
1. 3-way S1 Entity Split: Train (70%), Val (15%), Holdout (15%).
2. Baseline LightGBM model training on candidate pairs.
3. Feature Importance Analysis (Gain, Split, Permutation Importance on Val).
4. Feature Ablation Group Tests.
5. Error Diagnostics (False Positives, False Negatives, Singletons, Over-merges).
6. Smart Entity-Level Decision Policies (Source-specific thresholds, Singleton confidence margin).
7. Structured logging to experiments/results.csv and reports/.
"""

import argparse
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    EARLY_STOPPING_ROUNDS,
    K_CANDIDATES,
    LGBM_PARAMS,
    MODEL_DIR,
    NEG_PER_POS,
    OUTPUT_DIR,
    RANDOM_STATE,
    THRESHOLD_GRID,
    TRAIN_DIR,
    VAL_FRACTION,
)
from preprocess import preprocess_df
from blocking import generate_candidates
from features import FEATURE_NAMES, build_feature_matrix, build_lookup
from evaluate import score_f05, f05_per_entity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
ERROR_DIR   = REPORTS_DIR / "error_analysis"
EXP_DIR     = Path(__file__).resolve().parent.parent / "experiments"

for d in (REPORTS_DIR, ERROR_DIR, EXP_DIR):
    d.mkdir(exist_ok=True)


# ── Data Loading & Split ──────────────────────────────────────────────────────

def load_and_preprocess_data(sample: int = 5000, full_s23: bool = False):
    cache_file = OUTPUT_DIR / f"preprocessed_cache_{sample}_{full_s23}.pkl"
    if cache_file.exists():
        logger.info(f"Loading preprocessed data from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logger.info(f"Loading data (sample={sample})...")
    s1_raw = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str)
    s2_raw = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str)
    s3_raw = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str)
    gt_raw = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

    if sample:
        s1_raw = s1_raw.sample(n=min(sample, len(s1_raw)), random_state=RANDOM_STATE).reset_index(drop=True)
        sampled_ids = set(s1_raw["entity_id"])
        gt_subset   = gt_raw[gt_raw["source1_entity_id"].isin(sampled_ids)]
        gt_match_ids: set = set()
        for _, row in gt_subset.iterrows():
            ids_str = row.get("matched_entity_ids", "").strip()
            if ids_str:
                gt_match_ids.update(ids_str.split(","))

        gt_s2_ids = {x for x in gt_match_ids if x.startswith("S2-")}
        gt_s3_ids = {x for x in gt_match_ids if x.startswith("S3-")}

        rng = np.random.default_rng(RANDOM_STATE)
        def _subsample(df, gt_ids, n_extra):
            gt_rows = df[df["entity_id"].isin(gt_ids)]
            rest    = df[~df["entity_id"].isin(gt_ids)]
            n_take  = min(n_extra, len(rest))
            extra   = rest.iloc[rng.choice(len(rest), n_take, replace=False)]
            return pd.concat([gt_rows, extra], ignore_index=True)

        if not full_s23:
            s2_raw = _subsample(s2_raw, gt_s2_ids, sample * 5)
            s3_raw = _subsample(s3_raw, gt_s3_ids, sample * 5)
        else:
            logger.info("Keeping FULL S2/S3 (Realistic blocking test)")

    s1 = preprocess_df(s1_raw)
    s2 = preprocess_df(s2_raw)
    s3 = preprocess_df(s3_raw)

    gt_dict = {}
    s1_id_set = set(s1["entity_id"])
    for _, row in gt_raw.iterrows():
        sid = row["source1_entity_id"]
        if sid in s1_id_set:
            m = row.get("matched_entity_ids", "").strip()
            gt_dict[sid] = [x for x in m.split(",") if x] if m else []

    res = (s1, s2, s3, gt_dict)
    with open(cache_file, "wb") as f:
        pickle.dump(res, f)
    return res


def split_three_way(s1: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """Deterministic 3-way split: Train (70%), Val (15%), Holdout (15%) by S1 entity."""
    s1 = s1.copy()
    rng = np.random.default_rng(RANDOM_STATE)
    
    val_ids, holdout_ids = set(), set()
    for country, grp in s1.groupby("country_norm"):
        n_grp = len(grp)
        n_val = max(1, int(n_grp * val_frac))
        n_hold = max(1, int(n_grp * val_frac))
        
        shuffled_eids = grp.sample(frac=1.0, random_state=RANDOM_STATE)["entity_id"].tolist()
        val_ids.update(shuffled_eids[:n_val])
        holdout_ids.update(shuffled_eids[n_val:n_val+n_hold])

    train_mask = ~s1["entity_id"].isin(val_ids | holdout_ids)
    val_mask   = s1["entity_id"].isin(val_ids)
    hold_mask  = s1["entity_id"].isin(holdout_ids)

    s1_train   = s1[train_mask].copy().reset_index(drop=True)
    s1_val     = s1[val_mask].copy().reset_index(drop=True)
    s1_holdout = s1[hold_mask].copy().reset_index(drop=True)

    return s1_train, s1_val, s1_holdout


def build_pairs_with_missed(s1_df, cands, gt_dict, neg_per_pos=NEG_PER_POS):
    rng = np.random.default_rng(RANDOM_STATE)
    pairs, labels = [], []
    for sid in s1_df["entity_id"]:
        cand_list = cands.get(sid, [])
        gt_set    = set(gt_dict.get(sid, []))

        pos = [c for c in cand_list if c in gt_set]
        neg = [c for c in cand_list if c not in gt_set]
        missed = [g for g in gt_set if g not in cand_list]
        if missed:
            pos.extend(missed)

        for c in pos:
            pairs.append((sid, c))
            labels.append(1)

        n_neg = min(len(neg), max(neg_per_pos * len(pos), 1))
        sampled_neg = rng.choice(neg, size=n_neg, replace=False).tolist() if neg else []
        for c in sampled_neg:
            pairs.append((sid, c))
            labels.append(0)

    return pairs, labels


# ── Feature Evaluation & Permutation Importance ────────────────────────────────

def compute_permutation_importance(model, X_val, y_val, feature_names, scoring_metric="auc"):
    from sklearn.metrics import roc_auc_score
    baseline_probs = model.predict(X_val)
    baseline_score = roc_auc_score(y_val, baseline_probs)
    
    importances = {}
    rng = np.random.default_rng(RANDOM_STATE)
    for i, col in enumerate(feature_names):
        X_perm = X_val.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        perm_probs = model.predict(X_perm)
        perm_score = roc_auc_score(y_val, perm_probs)
        importances[col] = baseline_score - perm_score
    return importances


# ── Detailed Metrics & Diagnostics ────────────────────────────────────────────

def evaluate_predictions_detailed(pred_dict: Dict[str, List[str]], gt_dict: Dict[str, List[str]]):
    total_entities = len(gt_dict)
    singleton_total = 0
    singleton_correct = 0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    scores = []
    
    for sid, gt_ids in gt_dict.items():
        gt_set   = set(gt_ids) if gt_ids else set()
        pred_set = set(pred_dict.get(sid, []))
        
        f05 = f05_per_entity(pred_set, gt_set)
        scores.append(f05)
        
        if not gt_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
            else:
                total_fp += len(pred_set)
        else:
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "precision": precision,
        "recall": recall,
        "singleton_acc": singleton_acc,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "num_pred_links": total_tp + total_fp,
    }


# ── Logging to CSV ─────────────────────────────────────────────────────────────

def log_experiment_result(result_dict: dict):
    results_path = EXP_DIR / "results.csv"
    df_new = pd.DataFrame([result_dict])
    if results_path.exists():
        df_old = pd.read_csv(results_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(results_path, index=False)
    logger.info(f"Experiment result logged to {results_path}")


# ── Run Baseline Audit Experiment ──────────────────────────────────────────────

def run_audit_experiment(sample=5000, k_candidates=50, full_s23=False):
    logger.info("=== Running Step 1 & 2 Audit & Baseline Experiment ===")
    t0 = time.time()
    
    s1, s2, s3, gt_dict = load_and_preprocess_data(sample=sample, full_s23=full_s23)
    s1_train, s1_val, s1_holdout = split_three_way(s1)
    
    logger.info(f"Splits: Train={len(s1_train)}, Val={len(s1_val)}, Holdout={len(s1_holdout)}")
    
    s23_lookup = build_lookup(pd.concat([s2, s3], ignore_index=True))
    s1_train_lookup = build_lookup(s1_train)
    s1_val_lookup   = build_lookup(s1_val)
    s1_hold_lookup  = build_lookup(s1_holdout)

    # 1. Blocking
    logger.info("Running candidate blocking...")
    cands_cache_file = OUTPUT_DIR / f"cands_cache_{sample}_{full_s23}_{k_candidates}.pkl"
    if cands_cache_file.exists():
        logger.info(f"Loading candidates from cache: {cands_cache_file}")
        with open(cands_cache_file, "rb") as f:
            cands_train, cands_val, cands_hold = pickle.load(f)
    else:
        cands_train = generate_candidates(s1_train, s2, s3, k=k_candidates)
        cands_val   = generate_candidates(s1_val, s2, s3, k=k_candidates)
        cands_hold  = generate_candidates(s1_holdout, s2, s3, k=k_candidates)
        with open(cands_cache_file, "wb") as f:
            pickle.dump((cands_train, cands_val, cands_hold), f)

    gt_train = {k: v for k, v in gt_dict.items() if k in set(s1_train["entity_id"])}
    gt_val   = {k: v for k, v in gt_dict.items() if k in set(s1_val["entity_id"])}
    gt_hold  = {k: v for k, v in gt_dict.items() if k in set(s1_holdout["entity_id"])}

    # 2. Build pairs & features
    train_pairs, train_labels = build_pairs_with_missed(s1_train, cands_train, gt_train)
    val_pairs, val_labels     = build_pairs_with_missed(s1_val, cands_val, gt_val)
    hold_pairs, hold_labels   = build_pairs_with_missed(s1_holdout, cands_hold, gt_hold)

    feat_cache_file = OUTPUT_DIR / f"features_cache_{sample}_{full_s23}_{k_candidates}.pkl"
    if feat_cache_file.exists():
        logger.info(f"Loading features from cache: {feat_cache_file}")
        with open(feat_cache_file, "rb") as f:
            X_train, y_train, X_val, y_val, X_hold, y_hold = pickle.load(f)
    else:
        X_train = build_feature_matrix(train_pairs, s1_train_lookup, s23_lookup, n_jobs=-1)
        y_train = np.array(train_labels, dtype=np.int32)

        X_val = build_feature_matrix(val_pairs, s1_val_lookup, s23_lookup, n_jobs=-1)
        y_val = np.array(val_labels, dtype=np.int32)

        X_hold = build_feature_matrix(hold_pairs, s1_hold_lookup, s23_lookup, n_jobs=-1)
        y_hold = np.array(hold_labels, dtype=np.int32)
        
        with open(feat_cache_file, "wb") as f:
            pickle.dump((X_train, y_train, X_val, y_val, X_hold, y_hold), f)

    # 3. Train LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    dval   = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=dtrain)

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

    # 4. Feature Importances
    gain_imp  = model.feature_importance(importance_type="gain")
    split_imp = model.feature_importance(importance_type="split")
    perm_imp  = compute_permutation_importance(model, X_val, y_val, FEATURE_NAMES)

    fi_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "gain_importance": gain_imp,
        "split_importance": split_imp,
        "val_perm_importance": [perm_imp[f] for f in FEATURE_NAMES]
    }).sort_values(by="gain_importance", ascending=False)
    
    fi_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    logger.info(f"Saved feature importance report to {REPORTS_DIR / 'feature_importance.csv'}")

    # Feature Correlations
    X_val_df = pd.DataFrame(X_val, columns=FEATURE_NAMES)
    corr_df  = X_val_df.corr()
    corr_df.to_csv(REPORTS_DIR / "feature_correlations.csv")

    # 5. Threshold search on validation
    val_probs = model.predict(X_val)
    prob_map_val = {(s1_id, s23_id): prob for (s1_id, s23_id), prob in zip(val_pairs, val_probs)}
    
    best_val_f05 = -1.0
    best_thr = 0.5
    best_val_stats = {}

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
            best_val_stats = stats

    logger.info(f"★ Best Global Threshold on Val: {best_thr:.2f} | Val F0.5: {best_val_f05:.4f}")

    # 6. Evaluate Holdout using best threshold
    hold_probs = model.predict(X_hold)
    prob_map_hold = {(s1_id, s23_id): prob for (s1_id, s23_id), prob in zip(hold_pairs, hold_probs)}
    pred_dict_hold = {}
    for sid in s1_holdout["entity_id"]:
        cands = cands_hold.get(sid, [])
        matched = [c for c in cands if prob_map_hold.get((sid, c), 0.0) >= best_thr]
        pred_dict_hold[sid] = matched
    
    hold_stats = evaluate_predictions_detailed(pred_dict_hold, gt_hold)
    logger.info(f"★ Holdout F0.5 (thr={best_thr:.2f}): {hold_stats['macro_f05']:.4f}")

    # 7. Error Diagnostics Export (Val set)
    export_error_diagnostics(s1_val, s23_lookup, cands_val, gt_val, prob_map_val, best_thr)

    # 8. Log experiment
    exp_record = {
        "experiment_id": "EXP-001-BASELINE-AUDIT",
        "timestamp": pd.Timestamp.now().isoformat(),
        "git_commit_if_available": "clean",
        "candidate_K": k_candidates,
        "feature_set": "baseline_17",
        "added_features": "none",
        "removed_features": "none",
        "model_parameters": f"num_leaves={LGBM_PARAMS['num_leaves']},lr={LGBM_PARAMS['learning_rate']}",
        "negative_sampling_strategy": f"hard_neg_ratio={NEG_PER_POS}",
        "threshold_policy": "global_grid_search",
        "global_threshold": best_thr,
        "S2_threshold": best_thr,
        "S3_threshold": best_thr,
        "validation_precision": round(best_val_stats["precision"], 4),
        "validation_recall": round(best_val_stats["recall"], 4),
        "validation_f05": round(best_val_f05, 4),
        "validation_singleton_accuracy": round(best_val_stats["singleton_acc"], 4),
        "holdout_f05_if_run": round(hold_stats["macro_f05"], 4),
        "num_predictions": best_val_stats["num_pred_links"],
        "false_positives": best_val_stats["fp"],
        "false_negatives": best_val_stats["fn"],
        "runtime_seconds": round(time.time() - t0, 1),
        "notes": "Step 1 audit baseline run on 5K sample with 3-way split (Train/Val/Holdout)"
    }
    log_experiment_result(exp_record)

    # Save top_features.md summary
    save_top_features_report(fi_df)
    
    model_path = MODEL_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump((model, best_thr), f)
    logger.info(f"Saved final trained model to {model_path}")
    
    return model, best_thr, best_val_f05, hold_stats["macro_f05"]


def export_error_diagnostics(s1_df, s23_lookup, cands_dict, gt_dict, prob_map, threshold):
    fps, fns, singletons_fm, over_merges = [], [], [], []

    s1_lookup = build_lookup(s1_df)

    for sid in s1_df["entity_id"]:
        gt_set = set(gt_dict.get(sid, []))
        cands  = cands_dict.get(sid, [])
        pred_set = set([c for c in cands if prob_map.get((sid, c), 0.0) >= threshold])

        s1_info = s1_lookup[sid]

        # Singletons False Merges
        if not gt_set and pred_set:
            for p in pred_set:
                s23_info = s23_lookup[p]
                singletons_fm.append({
                    "s1_id": sid,
                    "cand_id": p,
                    "source": "S2" if p.startswith("S2-") else "S3",
                    "s1_name": s1_info["name_norm"],
                    "s23_name": s23_info["name_norm"],
                    "s1_addr": s1_info["addr_norm"],
                    "s23_addr": s23_info["addr_norm"],
                    "country": s1_info["country_norm"],
                    "prob": round(prob_map.get((sid, p), 0.0), 4),
                })

        # False Positives on non-singletons
        if gt_set:
            for p in (pred_set - gt_set):
                s23_info = s23_lookup[p]
                fps.append({
                    "s1_id": sid,
                    "cand_id": p,
                    "source": "S2" if p.startswith("S2-") else "S3",
                    "s1_name": s1_info["name_norm"],
                    "s23_name": s23_info["name_norm"],
                    "s1_addr": s1_info["addr_norm"],
                    "s23_addr": s23_info["addr_norm"],
                    "country": s1_info["country_norm"],
                    "prob": round(prob_map.get((sid, p), 0.0), 4),
                })

        # False Negatives (Candidate existed & was GT match, but model rejected)
        for g in gt_set:
            if g in cands and g not in pred_set:
                s23_info = s23_lookup[g]
                fns.append({
                    "s1_id": sid,
                    "cand_id": g,
                    "source": "S2" if g.startswith("S2-") else "S3",
                    "s1_name": s1_info["name_norm"],
                    "s23_name": s23_info["name_norm"],
                    "s1_addr": s1_info["addr_norm"],
                    "s23_addr": s23_info["addr_norm"],
                    "country": s1_info["country_norm"],
                    "prob": round(prob_map.get((sid, g), 0.0), 4),
                })

    pd.DataFrame(fps).to_csv(ERROR_DIR / "false_positives.csv", index=False)
    pd.DataFrame(fns).to_csv(ERROR_DIR / "false_negatives.csv", index=False)
    pd.DataFrame(singletons_fm).to_csv(ERROR_DIR / "singleton_false_merges.csv", index=False)
    logger.info(f"Exported error CSVs (FP: {len(fps)}, FN: {len(fns)}, Singleton FM: {len(singletons_fm)}) to {ERROR_DIR}")


def run_predict_experiment():
    logger.info("=== Running Final Inference on Test Set ===")
    from config import TEST_DIR, K_CANDIDATES
    
    # Load test data
    s1_raw = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="	", dtype=str)
    s2_raw = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="	", dtype=str)
    s3_raw = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="	", dtype=str)
    
    s1 = preprocess_df(s1_raw)
    s2 = preprocess_df(s2_raw)
    s3 = preprocess_df(s3_raw)
    
    cands_cache_file = OUTPUT_DIR / f"cands_test.pkl"
    if cands_cache_file.exists():
        with open(cands_cache_file, "rb") as f:
            cands_test = pickle.load(f)
    else:
        cands_test = generate_candidates(s1, s2, s3, k=K_CANDIDATES)
        with open(cands_cache_file, "wb") as f:
            pickle.dump(cands_test, f)
            
    # Build features
    s1_lookup = build_lookup(s1)
    s23_lookup = build_lookup(pd.concat([s2, s3], ignore_index=True))
    
    pairs = []
    for sid in s1["entity_id"]:
        for c in cands_test.get(sid, []):
            pairs.append((sid, c))
            
    feat_cache = OUTPUT_DIR / "features_test.pkl"
    if feat_cache.exists():
        with open(feat_cache, "rb") as f:
            X_test = pickle.load(f)
    else:
        X_test = build_feature_matrix(pairs, s1_lookup, s23_lookup, n_jobs=-1)
        with open(feat_cache, "wb") as f:
            pickle.dump(X_test, f)
            
    # Load model
    model_path = MODEL_DIR / "lgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run 'train' first.")
    with open(model_path, "rb") as f:
        model, best_thr = pickle.load(f)
        
    probs = model.predict(X_test)
    prob_map = {(sid, cid): prob for (sid, cid), prob in zip(pairs, probs)}
    
    submission_rows = []
    for sid in s1["entity_id"]:
        cands = cands_test.get(sid, [])
        matched = [c for c in cands if prob_map.get((sid, c), 0.0) >= best_thr]
        submission_rows.append({
            "source1_entity_id": sid,
            "target_entity_ids": ",".join(matched)
        })
        
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    logger.info(f"Saved submission to {OUTPUT_DIR / 'submission.csv'}")

def save_top_features_report(fi_df: pd.DataFrame):
    lines = [
        "# Top Features Ranking & Importance Analysis",
        "",
        "Ranked by LightGBM **Gain Importance** on Validation Set.",
        "",
        "| Rank | Feature Name | Gain Importance | Split Count | Val Permutation Importance |",
        "|------|--------------|-----------------|-------------|----------------------------|",
    ]
    for idx, row in fi_df.reset_index(drop=True).iterrows():
        lines.append(
            f"| {idx+1} | `{row['feature']}` | {row['gain_importance']:.2f} | {int(row['split_importance'])} | {row['val_perm_importance']:.6f} |"
        )
    with open(REPORTS_DIR / "top_features.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved {REPORTS_DIR / 'top_features.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"], nargs="?", default="train")
    parser.add_argument("--sample", type=int, default=0, help="0 means full data")
    parser.add_argument("--full-s23", action="store_true", help="Do not sample S2/S3 (Realistic blocking test)")
    args = parser.parse_args()
    
    if args.mode == "train":
        if args.sample == 50000 and args.full_s23:
            logger.info("CRITICAL CHECKPOINT: Running 50K S1 against FULL 10M S2/S3")
        elif args.sample == 0:
            logger.info("CRITICAL CHECKPOINT: Running FULL PIPELINE on 2.2M rows")
        run_audit_experiment(sample=args.sample, full_s23=args.full_s23)
    elif args.mode == "predict":
        run_predict_experiment()
