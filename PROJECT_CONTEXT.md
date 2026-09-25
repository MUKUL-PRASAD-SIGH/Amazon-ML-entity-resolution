# Amazon ML 2026 — Business Entity Resolution
## Full Project Context Document
### (For LLM context handoff — last updated after 5K dev run)

---

## 1. Problem Statement

Match business records across 3 independent noisy data sources. **Source 1 is the deduplicated reference**; find all S2/S3 records referring to the same real-world business as each S1 entity.

**Metric**: Macro-averaged **F₀.₅** (precision-weighted, β=0.5):
```
F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```
- Per-entity: empty prediction on a singleton S1 = **1.0**; any false merge on singleton = **0.0**
- Singletons **count in the macro average** — don't ignore them

**Key constraint**: Final model must be MIT/Apache 2.0 licensed and ≤8B parameters.

---

## 2. Dataset Facts (from EDA)

### Scale
| File | Rows | Countries |
|------|------|-----------|
| train_source1.tsv | 2,206,821 | US: 1,323,633 \| India: 883,188 |
| train_source2.tsv | 5,034,616 | US: 3,016,817 \| India: 2,017,799 |
| train_source3.tsv | 5,285,603 | US: 3,170,056 \| India: 2,115,547 |
| train_ground_truth.tsv | 2,206,821 | — |
| test_source1.tsv | 1,732,544 | India: 809,986 \| US: 663,106 \| **France: 259,452** |
| test_source2.tsv | 4,887,273 | India: 2,312,565 \| US: 1,871,330 \| France: 703,378 |
| test_source3.tsv | 5,082,316 | India: 2,405,000 \| US: 1,945,701 \| France: 731,615 |

### Ground Truth
- **Entities with ≥1 match**: 2,083,574 (94.4%)
- **Singleton entities** (no match): 123,247 (5.6%)
- **Average matches per non-singleton**: 3.46
- France appears **only in test** — country is open-set, never hard-code

### File Format
- All files: **tab-separated** (.tsv) — read with `sep="\t"`
- Columns: `entity_id`, `business_name`, `business_address`, `country`
- GT columns: `source1_entity_id`, `matched_entity_ids` (comma-separated)

### Noise Patterns
- **Name**: Corp/Corporation, Ltd/Limited, Pvt/Private, DBA names, word reordering, typos, transliterations
- **Address**: Rd/Road, St/Street, Ave/Avenue, missing PIN/state, landmark references
- **Script**: S2 has Devanagari (Hindi), S3 has Kannada — Unicode NFKC normalization required
- **NaN**: Some S3 records have NaN addresses (string "nan" when read with dtype=str)

---

## 3. Repository Structure

```
Amazon-ML-entity-resolution/
├── dataset/
│   ├── train/   ← train_source1/2/3.tsv, train_ground_truth.tsv
│   └── test/    ← test_source1/2/3.tsv (no GT)
├── src/
│   ├── config.py              ← ALL hyperparameters in one place
│   ├── preprocess.py          ← Text normalization
│   ├── blocking.py            ← TF-IDF candidate generation
│   ├── features.py            ← 17 similarity features per pair
│   ├── train_model.py         ← LightGBM training
│   ├── predict.py             ← Test inference → output TSVs
│   ├── evaluate.py            ← Local F₀.₅ scorer
│   ├── run_pipeline.py        ← CLI orchestrator
│   └── k_sweep_diagnostic.py  ← Blocking K experiment
├── output/    ← matching_results.tsv, candidate_pairs.tsv
├── models/    ← lgbm_model.pkl, meta.pkl, k_sweep_results.tsv
├── utils/
│   └── validate_submission.py ← Official format validator (stdlib only)
├── requirements.txt
└── Documentation_template.md
```

---

## 4. Pipeline Architecture

```
PREPROCESSING
  Unicode NFKC + abbrev expansion + fillna("") on all text columns
        ↓
BLOCKING (country-bucketed TF-IDF cosine, top-K per source)
  → candidate_pairs.tsv
        ↓
FEATURE ENGINEERING (17 similarity features per pair via rapidfuzz)
        ↓
LGBM CLASSIFIER (binary: match vs non-match, early stop on AUC)
  Threshold swept on val F₀.₅
        ↓
OUTPUT
  matching_results.tsv + candidate_pairs.tsv
```

### 4.1 Preprocessing (src/preprocess.py)
- Unicode NFKC normalization (Devanagari, Kannada, accented chars)
- Explicit `fillna("")` — catches NaN floats AND string literal "nan"/"NaN"
- Lowercase + punctuation removal
- Abbreviation expansion: 30+ name abbrevs (Corp→Corporation, Ltd→Limited, Pvt→Private...), 20+ address abbrevs (Rd→Road, Ave→Avenue...), French (SAS, SARL, SA)
- Produces: `name_norm`, `addr_norm`, `country_norm`, `addr_nums`, `blocking_text`

### 4.2 Blocking (src/blocking.py)
Algorithm: Country-bucketed TF-IDF cosine similarity (l2-normalized → dot product = cosine)
- Char 2-4 n-gram TF-IDF, max_features=50000, sublinear_tf=True
- Build one TF-IDF index per country on [S2 ∪ S3]
- Query S1 against country's index; fallback to global index if country missing in S23
- Memory-safe batching: S1 in batches of 300, S23 in chunks of 400K (~480MB peak RAM)
- Top-K candidates per source (S2 and S3 queried separately, then merged)

### 4.3 Features (src/features.py) — 17 total

| # | Feature | Description |
|---|---------|-------------|
| 1 | name_ratio | Levenshtein similarity |
| 2 | name_token_sort | Token-sort ratio (handles word reordering) |
| 3 | name_token_set | Token-set ratio (handles extra/missing words) |
| 4 | name_partial | Partial ratio (catches abbreviations) |
| 5 | name_jaro_winkler | Jaro-Winkler (prefix-weighted) |
| 6 | name_jaccard_token | Jaccard of word sets |
| 7 | addr_ratio | Address Levenshtein |
| 8 | addr_token_sort | Address token-sort ratio |
| 9 | addr_token_set | Address token-set ratio |
| 10 | addr_jaccard_token | Address word Jaccard |
| 11 | addr_num_overlap | Jaccard of numeric tokens (building #, PIN, ZIP) |
| 12 | country_match | 1 if countries identical, else 0 |
| 13 | blocking_score | TF-IDF cosine from blocking step |
| 14 | name_len_s1 | Length of S1 normalized name |
| 15 | name_len_s23 | Length of S23 normalized name |
| 16 | name_len_ratio | min/max of name lengths |
| 17 | addr_len_ratio | min/max of address lengths |

Library: rapidfuzz (C extension, ~30x faster than pure Python). Falls back to difflib.

### 4.4 Model (src/train_model.py)
- LightGBM binary classifier (MIT licensed, <1B params)
- 80/20 stratified split by country_norm
- Dev mode (--sample N): S1 subsampled to N; S2/S3 reduced to (GT-matches ∪ 5×N extras)
- Negative sampling: hard negatives from blocking candidates only (NEG_PER_POS=5)
- Early stopping on validation AUC (patience=50)
- Threshold sweep: 0.20–0.95 step 0.05, pick threshold maximizing val F₀.₅

### 4.5 Evaluation (src/evaluate.py)
Exact leaderboard formula:
- Singleton S1 + empty prediction → 1.0
- Singleton S1 + any prediction → 0.0
- Non-singleton: F₀.₅ = (1.25 × P × R) / (0.25 × P + R)
- Macro-averaged across all S1 entities in GT

---

## 5. First Dev Run Results (5K S1 sample, 3.4 minutes total)

| Metric | Value |
|--------|-------|
| **Val F₀.₅** | **0.9411** |
| Best threshold | 0.45 |
| Val AUC | **0.99992** |
| LightGBM rounds | 123 (early stopped) |
| **Blocking recall (train)** | **89.3%** (12,404/13,893 GT pairs found) |
| Training pairs | 83,586 (13,893 pos + 69,693 neg) |
| Total runtime | 3.4 min |

> ⚠️ These results use the DEV MODE subsampled S23 (33K S2 + 34K S3 instead of 5M each).
> Blocking recall on full 5M S23 will likely differ. K-sweep on full data is essential.

### Threshold Analysis (dev run)
```
Thr   F₀.₅     Notes
0.20  0.9374   too many false merges
0.25  0.9389
0.30  0.9400
0.35  0.9402
0.40  0.9409
0.45  0.9411  ← BEST
0.50  0.9409
0.55  0.9411  (tied)
0.60  0.9409
0.65  0.9399
0.70  0.9398
0.80  0.9386
0.95  0.9348   too many missed matches
```

---

## 6. Commands Reference

```bash
# Fast dev run (5K entities, ~4 min total)
python src/run_pipeline.py train --sample 5000

# Medium run (50K entities, ~30-60 min)
python src/run_pipeline.py train --sample 50000

# Full training (all 2.2M S1, ~4-8 hrs, run only after pipeline validated)
python src/run_pipeline.py train

# K-sweep experiment
python src/k_sweep_diagnostic.py --sample 5000 --ks 10 20 30 50 100

# Generate test predictions
python src/run_pipeline.py predict

# Re-use existing candidates (skip blocking step)
python src/run_pipeline.py predict --candidates output/candidate_pairs.tsv

# Validate submission format (must PASS before uploading)
python src/run_pipeline.py validate

# Score against GT locally
python src/run_pipeline.py score \
  --pred output/matching_results.tsv \
  --gt dataset/train/train_ground_truth.tsv
```

---

## 7. Configuration (src/config.py) — Tune These

```python
K_CANDIDATES       = 30        # top-K candidates per source — KEY parameter
TFIDF_MAX_FEATURES = 50_000    # TF-IDF vocab cap
TFIDF_NGRAM_RANGE  = (2, 4)    # char n-gram range
MIN_SIM_SCORE      = 0.05      # discard near-zero cosine hits
S1_BATCH_SIZE      = 300       # S1 rows per cosine batch
S23_CHUNK_SIZE     = 400_000   # S23 rows per chunk (~480MB RAM)
VAL_FRACTION       = 0.20      # 80/20 split
NEG_PER_POS        = 5         # hard negatives per positive
LGBM_PARAMS = { num_leaves=127, learning_rate=0.05, n_estimators=1000, ... }
EARLY_STOPPING_ROUNDS = 50
THRESHOLD_GRID     = [0.20, 0.25, ..., 0.95]
```

---

## 8. Known Issues & Fixes

| Issue | Status | Fix Applied |
|-------|--------|-------------|
| NaN addresses (S3) | ✅ Fixed | `_safe_str()` catches float NaN + string "nan"; `preprocess_df` has explicit `fillna("")` |
| France not in training | ✅ Handled | Same-country TF-IDF bucket; fallback to global index if country missing |
| Memory OOM on cosine | ✅ Handled | Nested S1-batch×S23-chunk loops, peak ~480MB |
| Country hard-coding | ✅ Avoided | Always open-set string, never OneHotEncoded |

---

## 9. Experiment Roadmap

### Phase 1: Pipeline Validation — DONE ✅
- 5K dev run: F₀.₅=0.9411, AUC=0.99992, blocking recall=89.3% (K=30)

### Phase 2: K-Sweep — DONE ✅

**Full results** (`models/k_sweep_results.tsv`):

```
K   | Val Cand Recall | Avg Cands/S1 | Val F₀.₅ | Best Threshold | AUC
----|-----------------|--------------|---------- |----------------|------
10  | 86.1%           | 20           | 0.9238    | 0.70           | 0.9998
20  | 87.6%           | 40           | 0.9329    | 0.50           | 0.9999
30  | 88.7%           | 60           | 0.9411    | 0.45           | 0.9999
50  | 89.5%           | 100          | 0.9472    | 0.60           | 1.0000  ← chosen
100 | 90.2%           | 200          | 0.9534    | 0.65           | 1.0000
```

**Key diagnosis**: AUC is near-perfect (0.9999–1.0) at ALL K values.
The LightGBM classifier is essentially perfect on pairs it sees.
The F₀.₅ ceiling is set entirely by blocking recall.
→ **Priority: improve blocking, not the model.**

**Marginal gains**:
- K=10→30: +2.6% recall, +0.0173 F₀.₅ (large — worth it)
- K=30→50: +0.8% recall, +0.0061 F₀.₅ (moderate — worth it)
- K=50→100: +0.7% recall, +0.0062 F₀.₅ (moderate — worth it for full run)

**Config updated**: `K_CANDIDATES = 50` (balance of recall vs compute).
For full training, consider K=100 if compute allows.

**Threshold drift**: Best threshold increases with K (0.70→0.50→0.45→0.60→0.65).
With more candidates, model needs higher threshold to stay precision-heavy (F₀.₅ metric).

### Phase 3: Scale to 50K
- Full S23 (no subsampling), optimal K from Phase 2
- First reliable estimate of real-world performance

### Phase 4: Four Branches (parallel)
```
Branch A: TF-IDF char ngram blocking → LightGBM        (CURRENT)
Branch B: Token inverted index blocking → LightGBM
Branch C: TF-IDF + address TF-IDF + token index → LightGBM
Branch D: TF-IDF cosine as direct match score (no ML)
```
Compare on same 50K validation set.

### Phase 5: Feature Ablation
After best branch chosen:
- Address TF-IDF cosine feature
- Phonetic encoding (Soundex/Metaphone) blocking + feature
- Country-specific number overlap (US ZIP, India PIN)
- Raw shared-token count

### Phase 6: Full Training
- After Phases 3-5 prove stable
- All 2.2M S1, all 10.3M S23 records
- Est. 4-8 hrs blocking + 1-2 hrs training

---

## 10. Output Format Requirements

### matching_results.tsv
```
source1_entity_id[TAB]matched_entity_ids
S1-00001[TAB]S2-00047,S2-00193,S3-00812
S1-00002[TAB]S3-00004
S1-00003[TAB]
```
Rules: Every test S1 has exactly one row; empty for singletons; S2/S3 IDs only; no duplicates.

### candidate_pairs.tsv
Same format with `candidate_entity_ids` column.
Every ID in matching_results must appear in candidate_pairs.

### Validation (run before every official submission)
```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
# PASS = exit 0, issues = exit 1 with numbered list
```

---

## 11. Dependencies
```
lightgbm>=4.3.0   # MIT
scikit-learn>=1.4.0
numpy>=1.26.0
scipy>=1.13.0
pandas>=2.2.0
rapidfuzz>=3.8.0  # MIT — fast string similarity (C extension)
joblib>=1.4.0
```

---

## 12. Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-09-25 | LightGBM not Transformer | Scale (2.2M), license (≤8B), speed |
| 2026-09-25 | Country-bucketed blocking | Prevents cross-country FPs; 2-3x search space reduction |
| 2026-09-25 | Char 2-4 ngram TF-IDF | Handles abbreviations, typos without exact tokenization |
| 2026-09-25 | Hard negatives from blocking | Forces fine-grained discrimination |
| 2026-09-25 | F₀.₅ threshold sweep | Precision-heavy; optimal threshold often > 0.5 |
| 2026-09-25 | Dev mode S23 subsampling | GT-match guarantee preserves blocking recall; fast iteration |

---

## 13. What NOT to Do (contest rules)
- External API lookups (geocoding, business registrations)
- Hard-coding countries — France exists in test only
- Model > 8B parameters or non-MIT/Apache 2.0 license
- OneHotEncoder(categories=["US", "India"]) — breaks on France
