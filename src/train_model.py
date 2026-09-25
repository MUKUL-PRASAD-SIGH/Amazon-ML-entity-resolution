"""
train_model.py — LightGBM pair-classifier training.

Pipeline
--------
1.  Load preprocessed train S1/S2/S3 DataFrames.
2.  Split S1 entities 80/20 into dev-train / dev-val (stratified by country).
3.  Run blocking on dev-train S1 → candidate pairs.
4.  Build (s1_id, s23_id, label) training frame:
        label=1  if s23_id in ground-truth matches for s1_id
        label=0  for hard negatives (blocking candidates not in GT)
5.  Compute 17 similarity features per pair (parallel via joblib).
6.  Train LightGBM with early stopping on dev-val AUC.
7.  Sweep threshold on dev-val F₀.₅ → pick best.
8.  Save model + threshold to models/ directory.

Usage
-----
    python train_model.py [--sample N]

    --sample N   only use N S1 entities for fast development runs
"""

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    EARLY_STOPPING_ROUNDS,
    K_CANDIDATES,
    LGBM_PARAMS,
    MODEL_DIR,
    NEG_PER_POS,
    RANDOM_STATE,
    THRESHOLD_GRID,
    TRAIN_DIR,
    VAL_FRACTION,
)
from preprocess import preprocess_df
from blocking import generate_candidates
from features import FEATURE_NAMES, build_feature_matrix, build_lookup
from evaluate import score_f05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_train_data(sample: int = None):
    """
    Load training data.  When `sample` is set:
      - S1 is randomly subsampled to `sample` entities.
      - S2 / S3 are reduced to (all GT matches for sampled S1) ∪ (5×sample
        random records) so that:
          a) blocking recall is not artificially broken, and
          b) the TF-IDF index is tiny → blocking runs in seconds for dev.
    Full training (sample=None) uses all data.
    """
    logger.info("Loading training data...")
    s1 = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str)
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False)

    if sample:
        s1 = s1.sample(n=min(sample, len(s1)), random_state=RANDOM_STATE)
        logger.info(f"  Sampled S1 to {len(s1)} entities (dev mode)")

        # Find all GT match IDs for the sampled S1 entities
        sampled_ids = set(s1["entity_id"])
        gt_subset   = gt[gt["source1_entity_id"].isin(sampled_ids)]
        gt_match_ids: set = set()
        for _, row in gt_subset.iterrows():
            ids_str = row.get("matched_entity_ids", "").strip()
            if ids_str:
                gt_match_ids.update(ids_str.split(","))

        gt_s2_ids = {x for x in gt_match_ids if x.startswith("S2-")}
        gt_s3_ids = {x for x in gt_match_ids if x.startswith("S3-")}

        rng = np.random.default_rng(RANDOM_STATE)

        def _subsample(df, gt_ids, n_extra):
            gt_rows   = df[df["entity_id"].isin(gt_ids)]
            rest      = df[~df["entity_id"].isin(gt_ids)]
            n_take    = min(n_extra, len(rest))
            extra     = rest.iloc[rng.choice(len(rest), n_take, replace=False)]
            return pd.concat([gt_rows, extra], ignore_index=True)

        n_extra = sample * 5
        s2 = _subsample(s2, gt_s2_ids, n_extra)
        s3 = _subsample(s3, gt_s3_ids, n_extra)
        logger.info(
            f"  Dev S2={len(s2)} (GT={len(gt_s2_ids)} + extra), "
            f"S3={len(s3)} (GT={len(gt_s3_ids)} + extra)"
        )

    logger.info(f"  S1={len(s1)}, S2={len(s2)}, S3={len(s3)}, GT={len(gt)}")
    return s1, s2, s3, gt


def load_ground_truth(gt_df: pd.DataFrame, s1_ids: set) -> Dict[str, List[str]]:
    """Parse GT TSV into {s1_id: [s23_id, ...]} for the requested S1 ids."""
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        if s1_id not in s1_ids:
            continue
        ids_str = row.get("matched_entity_ids", "").strip()
        gt_dict[s1_id] = [x for x in ids_str.split(",") if x] if ids_str else []
    return gt_dict


# ── Train/val split ───────────────────────────────────────────────────────────

def split_s1(s1_df: pd.DataFrame, val_frac: float = VAL_FRACTION) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by country_norm so val has the same country distribution.
    """
    s1_df = s1_df.copy()
    val_ids = set()
    for country, grp in s1_df.groupby("country_norm"):
        n_val = max(1, int(len(grp) * val_frac))
        val_ids |= set(grp.sample(n=n_val, random_state=RANDOM_STATE)["entity_id"])

    val_mask = s1_df["entity_id"].isin(val_ids)
    return s1_df[~val_mask].copy(), s1_df[val_mask].copy()


# ── Pair building ─────────────────────────────────────────────────────────────

def build_pairs(
    s1_df: pd.DataFrame,
    candidates: Dict[str, List[str]],
    gt_dict: Dict[str, List[str]],
    neg_per_pos: int = NEG_PER_POS,
    rng: np.random.Generator = None,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Construct labeled (s1_id, s23_id, label) pairs.

    Positives : all GT matches that appear in the candidate list.
    Negatives : up to neg_per_pos blocking candidates that are NOT GT matches,
                sampled to keep the dataset balanced.

    Returns
    -------
    pairs  : list of (s1_id, s23_id)
    labels : list of int (0 or 1)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    pairs, labels = [], []
    for s1_id in s1_df["entity_id"]:
        cands = candidates.get(s1_id, [])
        gt_set = set(gt_dict.get(s1_id, []))

        pos = [c for c in cands if c in gt_set]
        neg = [c for c in cands if c not in gt_set]

        # Also add GT matches that might not be in candidates (recall tracking)
        missed_gt = [g for g in gt_set if g not in cands]
        if missed_gt:
            pos.extend(missed_gt)   # still label them 1 so model sees true positives

        for c in pos:
            pairs.append((s1_id, c))
            labels.append(1)

        # Sample negatives
        n_neg = min(len(neg), max(neg_per_pos * len(pos), 1))
        sampled_neg = rng.choice(neg, size=n_neg, replace=False).tolist() if neg else []
        for c in sampled_neg:
            pairs.append((s1_id, c))
            labels.append(0)

    return pairs, labels


# ── Main training routine ─────────────────────────────────────────────────────

def train(sample: int = None):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is required: pip install lightgbm")

    t0 = time.time()

    # 1. Load data
    s1_raw, s2_raw, s3_raw, gt_raw = load_train_data(sample=sample)

    logger.info("Preprocessing...")
    s1 = preprocess_df(s1_raw)
    s2 = preprocess_df(s2_raw)
    s3 = preprocess_df(s3_raw)

    # 2. Train / val split
    s1_train, s1_val = split_s1(s1)
    logger.info(f"Split: {len(s1_train)} train, {len(s1_val)} val S1 entities")

    # Ground truth dicts
    all_s1_ids = set(s1["entity_id"])
    gt_dict    = load_ground_truth(gt_raw, all_s1_ids)
    gt_train   = {k: v for k, v in gt_dict.items() if k in set(s1_train["entity_id"])}
    gt_val     = {k: v for k, v in gt_dict.items() if k in set(s1_val["entity_id"])}

    # 3. Build lookups
    s23_lookup = build_lookup(pd.concat([s2, s3], ignore_index=True))

    # 4. Blocking on train split
    logger.info("Running blocking on train split...")
    cands_train = generate_candidates(s1_train, s2, s3, k=K_CANDIDATES)

    # 5. Build labeled pairs
    logger.info("Building training pairs...")
    train_pairs, train_labels = build_pairs(
        s1_train, cands_train, gt_train, neg_per_pos=NEG_PER_POS
    )
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    logger.info(f"  Training pairs: {len(train_pairs)} total "
                f"({pos_count} pos, {neg_count} neg)")

    # 6. Blocking recall check
    total_gt_pairs = sum(len(v) for v in gt_train.values())
    found_in_cands = sum(
        1 for s1_id, gt_ids in gt_train.items()
        for g in gt_ids if g in set(cands_train.get(s1_id, []))
    )
    logger.info(
        f"  Blocking recall: {found_in_cands}/{total_gt_pairs} "
        f"({100*found_in_cands/max(total_gt_pairs,1):.1f}%)"
    )

    # 7. Compute features (train)
    logger.info("Computing training features...")
    s1_lookup = build_lookup(s1_train)
    X_train = build_feature_matrix(
        train_pairs, s1_lookup, s23_lookup, n_jobs=-1
    )
    y_train = np.array(train_labels, dtype=np.int32)
    logger.info(f"  Feature matrix: {X_train.shape}")

    # 8. Blocking + pairs for val
    logger.info("Running blocking on val split...")
    s1_val_lookup = build_lookup(s1_val)
    cands_val = generate_candidates(s1_val, s2, s3, k=K_CANDIDATES)
    val_pairs, val_labels = build_pairs(
        s1_val, cands_val, gt_val, neg_per_pos=NEG_PER_POS
    )
    X_val = build_feature_matrix(val_pairs, s1_val_lookup, s23_lookup, n_jobs=-1)
    y_val = np.array(val_labels, dtype=np.int32)

    # 9. Train LightGBM
    logger.info("Training LightGBM...")
    dtrain = lgb.Dataset(X_train, label=y_train,
                         feature_name=FEATURE_NAMES)
    dval   = lgb.Dataset(X_val,   label=y_val,
                         feature_name=FEATURE_NAMES, reference=dtrain)

    params = dict(LGBM_PARAMS)
    params.pop("n_estimators", None)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=LGBM_PARAMS["n_estimators"],
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(50),
        ],
    )

    # 10. Sweep threshold on val F₀.₅
    logger.info("Sweeping threshold on validation F₀.₅...")
    val_probs = model.predict(X_val)

    # Reconstruct per-entity prediction dicts for each threshold
    # (re-run inference over cands_val properly)
    best_f05  = -1.0
    best_thr  = 0.5

    # Score all val candidate pairs
    # Build (s1_id, s23_id) → prob mapping
    prob_map = {}
    for (s1_id, s23_id), prob in zip(val_pairs, val_probs):
        prob_map[(s1_id, s23_id)] = prob

    for thr in THRESHOLD_GRID:
        pred_dict = {}
        for s1_id in s1_val["entity_id"]:
            cands = cands_val.get(s1_id, [])
            matched = [c for c in cands if prob_map.get((s1_id, c), 0.0) >= thr]
            pred_dict[s1_id] = matched

        f05 = score_f05(pred_dict, gt_val)
        logger.info(f"  Threshold={thr:.2f}  F₀.₅={f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thr = thr

    logger.info(f"\n★ Best threshold: {best_thr}  |  Val F₀.₅: {best_f05:.4f}")

    # 11. Save model + threshold
    model_path = MODEL_DIR / "lgbm_model.pkl"
    meta_path  = MODEL_DIR / "meta.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(meta_path, "wb") as f:
        pickle.dump({"threshold": best_thr, "val_f05": best_f05}, f)

    logger.info(f"Model saved → {model_path}")
    logger.info(f"Meta  saved → {meta_path}")
    logger.info(f"Total training time: {(time.time()-t0)/60:.1f} min")
    return model, best_thr, best_f05


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to N S1 entities (dev speed-up)")
    args = parser.parse_args()
    train(sample=args.sample)
