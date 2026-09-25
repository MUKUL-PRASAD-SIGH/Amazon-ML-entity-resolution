"""
config.py — Central configuration for the entity resolution pipeline.
All paths, hyperparameters, and constants live here.
"""
from pathlib import Path

# ── Project layout ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR  = ROOT / "dataset" / "train"
TEST_DIR   = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR  = ROOT / "models"
SRC_DIR    = ROOT / "src"

for _d in (OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(exist_ok=True)

# ── Blocking ──────────────────────────────────────────────────────────────────
# TF-IDF vectoriser (character n-grams, l2-normalised → dot == cosine)
TFIDF_ANALYZER     = "char_wb"   # pad with word-boundary spaces
TFIDF_NGRAM_RANGE  = (2, 4)      # 2- to 4-character n-grams
TFIDF_MAX_FEATURES = 50_000      # vocabulary cap
TFIDF_MIN_DF       = 2           # ignore singletons

# Candidate retrieval
# K-sweep results (5K dev run):
#   K=10 → recall=86.1%, F₀.₅=0.9238  K=30 → recall=88.7%, F₀.₅=0.9411
#   K=50 → recall=89.5%, F₀.₅=0.9472  K=100 → recall=90.2%, F₀.₅=0.9534
# Recommendation: K=50 for 50K run, K=100 for full training.
# Bottleneck is blocking recall (AUC=1.0 at all K), not the model.
K_CANDIDATES   = 50   # top-K per (S1, source) pair  → up to 100 per S1
MIN_SIM_SCORE  = 0.05 # discard near-zero cosine hits

# Memory-safe batching during cosine search
S1_BATCH_SIZE  = 300   # S1 rows per iteration
S23_CHUNK_SIZE = 400_000  # S23 rows per chunk (controls peak RAM)

# ── Train / validation split ──────────────────────────────────────────────────
VAL_FRACTION = 0.20
RANDOM_STATE = 42

# Negative sampling for classifier training
NEG_PER_POS = 5   # hard negatives drawn from blocking candidates

# ── LightGBM ──────────────────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    objective        = "binary",
    metric           = "auc",
    num_leaves       = 127,
    learning_rate    = 0.05,
    n_estimators     = 1000,
    min_child_samples= 20,
    subsample        = 0.80,
    colsample_bytree = 0.80,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
    verbose          = -1,
)
EARLY_STOPPING_ROUNDS = 50

# Threshold grid — sweep on validation F₀.₅
THRESHOLD_GRID = [round(t, 2) for t in [x * 0.05 for x in range(4, 20)]]
# [0.20, 0.25, 0.30, ..., 0.95]
