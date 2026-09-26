# Amazon ML Entity Resolution — Final Optimization Pipeline
## Colab T4 GPU Edition

> **Runtime target:** Google Colab · T4 GPU (16 GB VRAM) · ~13 GB system RAM · 2 CPU cores

---

## Hardware Constraints & Ground Rules

| Resource | Colab T4 Limit | Implication |
|---|---|---|
| GPU VRAM | 16 GB | Keep dense S23 matrix < 12 GB; use chunked CSRMM |
| System RAM | ~13 GB | Load one S23 chunk at a time; use `float32` everywhere |
| Disk (local SSD) | ~100 GB | Copy dataset from Drive once at session start |
| Session timeout | ~12 h | Use `joblib` checkpoints after every major step |
| CUDA | 12.x | Install `cupy-cuda12x`; fall back to NumPy if import fails |

### T4 OOM Rules (non-negotiable)
1. **Never** do Sparse × Sparse (`spgemm`) — always Sparse × Dense (`csrmm` / `@`).
2. Keep `S23_CHUNK_SIZE = 400_000` rows per chunk during cosine search.
3. Use `float32`, never `float64`.
4. Call `cupy.get_default_memory_pool().free_all_blocks()` + `gc.collect()` between blockers.
5. If OOM occurs, halve `S23_CHUNK_SIZE` and restart the cell.

---

## Core Philosophy
1. **Candidate Recall is the ceiling.** AUC on artificially injected pairs is diagnostic only. The competition score is determined by blocking recall first, matcher second.
2. **K=50 is the baseline.** Do not blindly push K higher. Analyse *what* K=50 misses, then build a targeted blocker for those specific failure modes.
3. **RAM and runtime are gates.** Validate 50K S1 vs full 10.3M S2/S3 before launching the 2.2M run. If peak RAM > 11 GB, reduce chunk size.
4. **Checkpoint everything.** Colab sessions die. Every preprocessing result, candidate set, and feature matrix must be saved to `/content/drive/MyDrive/` or `/content/checkpoints/`.

---

## Colab Session Setup (run every session)

```python
# Cell 1 — Always run first
from google.colab import drive
drive.mount('/content/drive')

import shutil
from pathlib import Path

# Copy dataset from Drive to local SSD (~1-2 min, much faster I/O)
DRIVE_DATASET = Path('/content/drive/MyDrive/Amazon-ML-entity-resolution/dataset')
LOCAL_DATASET = Path('/content/dataset')
if not LOCAL_DATASET.exists():
    shutil.copytree(DRIVE_DATASET, LOCAL_DATASET)

# Install GPU-accelerated deps
!pip install -q lightgbm>=4.3.0 rapidfuzz>=3.8.0 joblib>=1.4.0 cupy-cuda12x psutil
```

### GPU Verification
```python
import cupy as cp
print(cp.cuda.runtime.getDeviceProperties(0)['name'])  # should print 'Tesla T4'
print(f'VRAM free: {cp.cuda.runtime.memGetInfo()[0]/1e9:.1f} GB')
```

---

## Phase 1 — Freeze the experimental protocol

- **Split:** 70% Train · 15% Validation · 15% Holdout (split by S1 entity ID, not random rows).
- **Evaluation table** (log every experiment):

| Experiment | K | Blocker Config | Cand Recall | AUC | Precision | Recall | F0.5 | Singleton Acc | Runtime (s) | Peak RAM (GB) |
|---|---|---|---|---|---|---|---|---|---|---|

- **Rule:** No experiment is "better" unless evaluated on the **identical** validation/holdout partition.

---

## Phase 2 — T4 Gate: 50K S1 vs Full 10.3M S2/S3

**This is the mandatory gate before the 2.2M full run.**

In `MASTER_PIPELINE.ipynb` Cell 2, set:
```python
SAMPLE_S1  = 50_000
FULL_S23   = True    # ← keeps all 5M S2 + 5.3M S3
K_CANDIDATES = 50
```

Run Cell 8. Record:
- Candidate Recall (target: ≥97%)
- Peak RAM (target: < 11 GB)
- Runtime (target: < 20 min on T4)
- Avg candidates / S1 (target: ~50)

**If any gate fails:** do not proceed to Phase 12. Reduce `S23_CHUNK_SIZE` or redesign the blocking index.

---

## Phase 3 — Missed-Match Analysis

After Phase 2, run Cell 9 (`MASTER_PIPELINE.ipynb`). It outputs:
- `reports/error_analysis/missed_blocking_matches.csv`
- Auto-categorised failure types

**Failure categories to track:**

| Category | Description |
|---|---|
| NAME_TYPO | Edit distance > 5% but < 20% |
| WORD_ORDER | Token set match > 90%, ratio < 70% |
| ABBREVIATION | One name is prefix/suffix of other |
| ADDRESS_VARIATION | Identical name, different address |
| MISSING_FIELD | One or both fields empty |
| TRANSLITERATION | Script variation (Hindi → Roman) |
| DBA_TRADE_NAME | Completely different name, same address |
| OTHER | Catch-all |

**Rule:** Do not add a new blocker until the frequency table is generated.

---

## Phase 4 — Independent Blockers (T4-optimised)

| Blocker | Method | GPU? | Notes |
|---|---|---|---|
| A | Name char-TF-IDF cosine | ✅ CuPy CSRMM | Primary, handles most pairs |
| B | Address char-TF-IDF cosine | ✅ CuPy CSRMM | Catches different-name same-address |
| C | Exact normalised name | CPU dict | Free, O(N), no GPU needed |
| D | Rare-token inverted index | CPU dict | Unique brand/product names |
| E | Numeric / PIN overlap | CPU dict | Street numbers, postal codes |
| F | Full-doc TF-IDF (name+addr) | ✅ CuPy CSRMM | Holistic match |

**T4 memory budget for TF-IDF blocking:**
- S23 dense matrix (400K rows × 50K vocab, float32) ≈ **80 GB** — far too large.
- **Solution:** chunk S23 into 400K-row slices. Process each slice as a CPU→GPU dense block (400K × 50K float32 ≈ 80 GB → use 50K vocab, 400K rows × 50K = 20 GB → still too large).
- **Correct approach:** use **50K max features**, S1 batch size 300 rows. Dense S23 slice: 400K × 50K float32 = 80 GB... still too large.
- **Actual implementation:** keep S23 as **sparse CSR on CPU**, convert to dense in **small S23 chunks** (100K rows × 50K vocab × 4 bytes = 20 GB → still large).
- **Safe T4 approach:** S1 query in dense batches (300 × 50K = 60 MB), S23 sparse on CPU, multiply as `(dense_S1) @ (sparse_S23.T)` via scipy — this keeps GPU free for LightGBM training.

---

## Phase 5 — Union the Blockers

```text
S1 query
   │
   ├── Blocker A (name TF-IDF, top-K)
   ├── Blocker B (addr TF-IDF, top-K/2)
   ├── Blocker C (exact name dict)
   ├── Blocker D (rare token dict)
   ├── Blocker E (numeric dict)
   └── Blocker F (full-doc TF-IDF, top-K/3)
          │
       UNION + deduplicate by max score
          │
       Sort descending → keep top-K
```

---

## Phase 6 — Evaluate Blocking Before LightGBM

Report candidate recall after each blocker is added to the union. Example table to fill in:

| Config | Cand Recall | Avg Cands | Runtime |
|---|---|---|---|
| A only (K=50) | 97.70% | 50.0 | ? |
| A+C | ? | ? | ? |
| A+C+D | ? | ? | ? |
| A+B+C+D+E | ? | ? | ? |

**Do not touch LightGBM until this table has at least 3 rows.**

---

## Phase 7 — Matcher (LightGBM on T4)

- Train on **actual candidate pairs only** (honest training — no injected GT pairs).
- LightGBM runs entirely on CPU (T4 GPU not used here; LightGBM GPU mode has inconsistent behaviour on Colab).
- 36 features already implemented in Cell 6 of `MASTER_PIPELINE.ipynb`.
- Scale of training pairs at 50K S1, K=50: ~2.5M pairs — fits in ~1.2 GB RAM as float32.

---

## Phase 8 — Hard-Negative Mining

Hard negatives = candidate pairs where:
- `name_token_set >= 0.85` (very similar name) AND
- `country_match == 1` AND
- Ground truth label = 0 (not a real match)

These are the pairs where the model currently fails. The 5 hard-negative features in Cell 6 (`name_num_overlap`, `name_conflict_digits`, `addr_jw`, `name_len_diff`, `addr_len_diff`) directly target these cases.

---

## Phase 9 — Threshold Optimization

- Always sweep `[0.20, 0.25, … 0.95]` on the **validation set** after every change to K, features, or blockers.
- Never carry a threshold from one experiment to another.
- The sweep is already implemented in Cell 8.

---

## Phase 10 — Holdout Discipline

- **TRAIN** → model development.
- **VALIDATION** → all hyperparameter / threshold choices.
- **HOLDOUT** → one final check before freezing the model.
- Rule: once the holdout is checked, the configuration is **frozen**. Do not look at holdout again until the very end.

---

## Phase 11 — Sample-Size Ablation

Before committing to the full 2.2M S1 run (which takes ~8–12 hours on Colab T4), run:

| S1 Sample | Est. Runtime | Expected F0.5 |
|---|---|---|
| 50K | ~20 min | current baseline |
| 200K | ~1.5 h | +? |
| 500K | ~4 h | +? |
| 2.2M | ~10 h | +? |

If F0.5 plateaus at 200K, skip the 2.2M run.

---

## Phase 12 — Full-Data Model

```python
# In Cell 2:
SAMPLE_S1  = None   # use all 2.2M
FULL_S23   = False  # use co-sampled S2/S3 for training
```

Run Cells 1–8. This is the **only** time you use all training data.

**Colab tip:** Enable "Background execution" so the session survives browser disconnects.

---

## Phase 13 — Full Test Inference

```python
# In Cell 2:
SAMPLE_S1  = None
FULL_S23   = True   # must search full test S2/S3 universe
```

Run Cell 10. Features are computed in 500K-pair batches to avoid OOM.

---

## Phase 14 — Generate Submission Files

- `output/candidate_pairs.tsv` — actual candidates fed to the model (Cell 10).
- `output/matching_results.tsv` — every test S1 appears exactly once (Cell 10).
- Copy both to Drive immediately: `shutil.copy(OUTPUT_DIR/'matching_results.tsv', DRIVE_OUTPUT_DIR)`.

---

## Phase 15 — Validation

```python
# Cell 11 in MASTER_PIPELINE.ipynb
!python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Expected output: `PASS`. If it fails, check:
1. Every test S1 entity ID is present in `matching_results.tsv`.
2. All matched IDs in `matching_results.tsv` are a subset of IDs in `candidate_pairs.tsv`.
3. No duplicate S1 rows in either file.

---

## Go / No-Go Checklist

```
[ ] Session setup: Drive mounted, dataset on local SSD, GPU detected
[ ] Cell 2 config matches the intended experiment (SAMPLE_S1, FULL_S23, K)
[ ] Phase 2 gate passed (50K vs full S2/S3: recall ≥97%, RAM < 11 GB)
[ ] Missed-match analysis done (Cell 9 ran, CSV saved to Drive)
[ ] Blocker union defined from error analysis
[ ] Candidate recall measured per blocker config (Phase 6 table filled)
[ ] LightGBM trained on honest pairs only
[ ] Threshold swept and frozen from validation set
[ ] Holdout checked exactly once
[ ] Sample-size ablation shows diminishing returns
[ ] Full model trained (or skipped if ablation says so)
[ ] Test inference batched, both TSV files saved to Drive
[ ] Validator returns PASS
```

---

## Quick Colab Commands Reference

```python
# Check GPU VRAM
import cupy as cp
free, total = cp.cuda.runtime.memGetInfo()
print(f'VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total')

# Free GPU memory between cells
cp.get_default_memory_pool().free_all_blocks()
import gc; gc.collect()

# Save checkpoint to Drive
import joblib, shutil
joblib.dump(obj, '/content/checkpoints/my_checkpoint.pkl')
shutil.copy('/content/checkpoints/my_checkpoint.pkl',
            '/content/drive/MyDrive/Amazon-ML-entity-resolution/checkpoints/')

# Check system RAM
import psutil
print(f'RAM: {psutil.virtual_memory().available/1e9:.1f} GB free')
```
