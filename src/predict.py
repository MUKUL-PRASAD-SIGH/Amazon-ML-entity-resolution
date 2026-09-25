"""
predict.py — Generate output TSVs for the test set.

Steps
-----
1.  Load test S1, S2, S3 and preprocess them.
2.  Load the trained LightGBM model + threshold from models/.
3.  Run blocking (same pipeline as training) → candidate_pairs.tsv.
4.  Compute features for all candidate pairs.
5.  Score with LightGBM + apply threshold → matching_results.tsv.
6.  Validate that every test S1 entity has exactly one row.

Usage
-----
    python predict.py [--candidates path/to/candidate_pairs.tsv]

    --candidates   skip blocking and reuse a previously generated file
                   (saves time during iteration)
"""

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    K_CANDIDATES,
    MODEL_DIR,
    OUTPUT_DIR,
    TEST_DIR,
)
from preprocess import preprocess_df
from blocking import generate_candidates, save_candidate_pairs
from features import FEATURE_NAMES, build_feature_matrix, build_lookup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_candidates_from_file(path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    result = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        ids_str = row.get("candidate_entity_ids", "").strip()
        result[s1_id] = [x for x in ids_str.split(",") if x] if ids_str else []
    return result


def save_matching_results(
    predictions: Dict[str, List[str]],
    s1_ids: List[str],
    output_path: Path,
) -> None:
    rows = [
        {
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(predictions.get(s1_id, [])),
        }
        for s1_id in s1_ids
    ]
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
    n_matched   = sum(1 for r in rows if r["matched_entity_ids"])
    n_singleton = len(rows) - n_matched
    logger.info(
        f"Saved {len(rows)} rows → {output_path}  "
        f"(matched={n_matched}, singletons={n_singleton})"
    )


# ── Main prediction routine ───────────────────────────────────────────────────

def predict(candidates_file: Path = None):
    t0 = time.time()

    # 1. Load and preprocess test data
    logger.info("Loading test data...")
    s1_raw = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", dtype=str)
    s2_raw = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t", dtype=str)
    s3_raw = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t", dtype=str)
    logger.info(f"  S1={len(s1_raw)}, S2={len(s2_raw)}, S3={len(s3_raw)}")

    logger.info("Preprocessing...")
    s1 = preprocess_df(s1_raw)
    s2 = preprocess_df(s2_raw)
    s3 = preprocess_df(s3_raw)

    # 2. Load model + threshold
    model_path = MODEL_DIR / "lgbm_model.pkl"
    meta_path  = MODEL_DIR / "meta.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    threshold = meta["threshold"]
    logger.info(
        f"Loaded model | Val F₀.₅={meta['val_f05']:.4f} | "
        f"Threshold={threshold}"
    )

    # 3. Candidate generation (blocking)
    if candidates_file and Path(candidates_file).exists():
        logger.info(f"Loading pre-computed candidates from {candidates_file}...")
        candidates = load_candidates_from_file(Path(candidates_file))
        # Ensure all S1 entities are represented
        for s1_id in s1["entity_id"]:
            if s1_id not in candidates:
                candidates[s1_id] = []
    else:
        logger.info("Running blocking on test data...")
        candidates = generate_candidates(s1, s2, s3, k=K_CANDIDATES)

    # 4. Save candidate_pairs.tsv (required for submission)
    cands_path = OUTPUT_DIR / "candidate_pairs.tsv"
    save_candidate_pairs(candidates, s1["entity_id"].tolist(), cands_path)

    # 5. Build feature matrix for all candidate pairs
    logger.info("Building lookups and computing features...")
    s1_lookup  = build_lookup(s1)
    s23_lookup = build_lookup(pd.concat([s2, s3], ignore_index=True))

    all_pairs = []
    for s1_id, cands in candidates.items():
        for c in cands:
            all_pairs.append((s1_id, c))

    logger.info(f"  Total candidate pairs: {len(all_pairs):,}")

    if not all_pairs:
        logger.warning("No candidate pairs found — all predictions will be empty!")
        predictions = {s1_id: [] for s1_id in s1["entity_id"]}
        save_matching_results(predictions, s1["entity_id"].tolist(),
                              OUTPUT_DIR / "matching_results.tsv")
        return predictions

    # Process in batches to avoid OOM
    INFERENCE_BATCH = 500_000
    all_probs = []
    for start in range(0, len(all_pairs), INFERENCE_BATCH):
        end   = min(start + INFERENCE_BATCH, len(all_pairs))
        batch = all_pairs[start:end]
        X     = build_feature_matrix(batch, s1_lookup, s23_lookup, n_jobs=-1)
        probs = model.predict(X)
        all_probs.extend(probs.tolist())
        logger.info(f"  Scored {end:,}/{len(all_pairs):,} pairs")

    # 6. Apply threshold → predictions
    logger.info(f"Applying threshold={threshold}...")
    predictions: Dict[str, List[str]] = {s1_id: [] for s1_id in s1["entity_id"]}

    for (s1_id, s23_id), prob in zip(all_pairs, all_probs):
        if prob >= threshold:
            predictions[s1_id].append(s23_id)

    # Deduplicate (should already be clean but be safe)
    for s1_id in predictions:
        predictions[s1_id] = list(dict.fromkeys(predictions[s1_id]))

    # 7. Save matching_results.tsv
    results_path = OUTPUT_DIR / "matching_results.tsv"
    save_matching_results(predictions, s1["entity_id"].tolist(), results_path)

    n_matched = sum(1 for v in predictions.values() if v)
    logger.info(
        f"\n✓ Prediction complete in {(time.time()-t0)/60:.1f} min\n"
        f"  Entities with ≥1 match : {n_matched:,} / {len(s1):,}\n"
        f"  Singletons predicted   : {len(s1)-n_matched:,}\n"
        f"  Output → {results_path}"
    )
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Path to pre-computed candidate_pairs.tsv (skips blocking step)",
    )
    args = parser.parse_args()
    predict(candidates_file=args.candidates)
