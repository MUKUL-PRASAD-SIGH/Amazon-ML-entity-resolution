"""
run_pipeline.py — End-to-end pipeline runner with stage checkpointing.

Stages
------
  eda       Quick dataset statistics (always fast)
  train     Blocking + feature engineering + LightGBM training
  predict   Blocking on test + inference → output TSVs
  validate  Run utils/validate_submission.py
  score     Compute local F₀.₅ from a matching_results.tsv + GT file

Usage examples
--------------
  # Full run with 50k-entity dev sample
  python run_pipeline.py train --sample 50000

  # Full run on all training data
  python run_pipeline.py train

  # Generate test predictions (after training)
  python run_pipeline.py predict

  # Re-use pre-computed candidates (faster iteration)
  python run_pipeline.py predict --candidates output/candidate_pairs.tsv

  # Validate submission
  python run_pipeline.py validate

  # Score a specific prediction file against held-out GT
  python run_pipeline.py score --pred output/matching_results.tsv --gt dataset/train/train_ground_truth.tsv
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"


# ── EDA ───────────────────────────────────────────────────────────────────────

def run_eda():
    """Print basic dataset statistics."""
    import pandas as pd
    from config import TRAIN_DIR, TEST_DIR

    logger.info("=== EDA ===")
    for label, path in [
        ("train_source1", TRAIN_DIR / "train_source1.tsv"),
        ("train_source2", TRAIN_DIR / "train_source2.tsv"),
        ("train_source3", TRAIN_DIR / "train_source3.tsv"),
        ("train_gt",      TRAIN_DIR / "train_ground_truth.tsv"),
        ("test_source1",  TEST_DIR  / "test_source1.tsv"),
        ("test_source2",  TEST_DIR  / "test_source2.tsv"),
        ("test_source3",  TEST_DIR  / "test_source3.tsv"),
    ]:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        print(f"\n{label}: {len(df):,} rows × {df.shape[1]} cols")
        if "country" in df.columns:
            print("  Country dist:", df["country"].value_counts().to_dict())
        if "matched_entity_ids" in df.columns:
            has_match = df["matched_entity_ids"].str.len() > 0
            print(f"  Entities with matches  : {has_match.sum():,}")
            print(f"  Singleton entities     : {(~has_match).sum():,}")
            counts = df["matched_entity_ids"].apply(
                lambda x: len(x.split(",")) if x else 0
            )
            print(f"  Avg matches per entity : {counts.mean():.2f}")
        print(f"  NaN in business_name   : {df.get('business_name', pd.Series()).isna().sum()}")
        print(f"  NaN in business_address: {df.get('business_address', pd.Series()).isna().sum()}")


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_train(sample: int = None):
    sys.path.insert(0, str(SRC))
    from train_model import train
    model, threshold, f05 = train(sample=sample)
    logger.info(f"\n✓ Training complete | Val F₀.₅={f05:.4f} | Threshold={threshold}")


def run_predict(candidates_file: str = None):
    sys.path.insert(0, str(SRC))
    from predict import predict
    predict(candidates_file=candidates_file)


def run_validate():
    script = ROOT / "utils" / "validate_submission.py"
    matching  = ROOT / "output" / "matching_results.tsv"
    candidate = ROOT / "output" / "candidate_pairs.tsv"
    test_dir  = ROOT / "dataset" / "test"

    if not matching.exists():
        logger.error(f"matching_results.tsv not found at {matching}")
        return

    cmd = [
        sys.executable, str(script),
        "--matching",  str(matching),
        "--candidate", str(candidate),
        "--test-dir",  str(test_dir),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        logger.info("✓ Validation PASSED")
    else:
        logger.error("✗ Validation FAILED — fix issues before submitting")


def run_score(pred_path: str, gt_path: str):
    sys.path.insert(0, str(SRC))
    from evaluate import score_from_files
    f05 = score_from_files(Path(pred_path), Path(gt_path), verbose=True)
    print(f"\n>>> Macro F₀.₅ = {f05:.6f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    # eda
    sub.add_parser("eda", help="Print dataset statistics")

    # train
    p_train = sub.add_parser("train", help="Train the LightGBM classifier")
    p_train.add_argument("--sample", type=int, default=None,
                         help="Limit to N S1 entities (for fast dev runs)")

    # predict
    p_pred = sub.add_parser("predict", help="Generate test-set predictions")
    p_pred.add_argument("--candidates", type=str, default=None,
                        help="Reuse pre-computed candidate_pairs.tsv")

    # validate
    sub.add_parser("validate", help="Run submission validator")

    # score
    p_score = sub.add_parser("score", help="Score predictions against GT")
    p_score.add_argument("--pred", required=True,
                         help="Path to matching_results.tsv")
    p_score.add_argument("--gt", required=True,
                         help="Path to ground_truth.tsv")

    args = parser.parse_args()

    if args.stage == "eda":
        run_eda()
    elif args.stage == "train":
        run_train(sample=args.sample)
    elif args.stage == "predict":
        run_predict(candidates_file=args.candidates)
    elif args.stage == "validate":
        run_validate()
    elif args.stage == "score":
        run_score(args.pred, args.gt)


if __name__ == "__main__":
    main()
