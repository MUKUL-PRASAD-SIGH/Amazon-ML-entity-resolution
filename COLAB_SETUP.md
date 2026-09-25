# Google Colab Setup Guide
## Amazon ML 2026 — Business Entity Resolution

---

## Step 1: Upload Code to Google Drive

Create this folder structure in your Google Drive:

```
My Drive/
└── AmazonML/
    ├── student_resource.zip     ← competition data ZIP
    └── code/
        ├── src/                 ← copy your entire src/ folder here
        │   ├── config.py
        │   ├── preprocess.py
        │   ├── blocking.py
        │   ├── features.py
        │   ├── train_model.py
        │   ├── predict.py
        │   ├── evaluate.py
        │   ├── run_pipeline.py
        │   └── k_sweep_diagnostic.py
        └── utils/
            └── validate_submission.py
```

**How to upload:**
1. Go to [drive.google.com](https://drive.google.com)
2. Create folder: `AmazonML` → inside it, create `code` → inside that, create `src` and `utils`
3. Upload each `.py` file from your `src/` folder into `AmazonML/code/src/`
4. Upload `utils/validate_submission.py` into `AmazonML/code/utils/`
5. Upload `student_resource.zip` directly into `AmazonML/`

> **Tip**: You can zip the `src/` folder and extract in Colab, but uploading individual files is simpler.

---

## Step 2: Upload the Notebook to Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → select `COLAB_PIPELINE_GPU.ipynb` (or `COLAB_PIPELINE.ipynb` for the CPU fallback)
3. Save a copy to Drive: File → Save a copy in Drive
   - Suggested location: `My Drive/AmazonML/`

---

## Step 3: Configure the Runtime

1. Runtime → Change runtime type
2. Hardware accelerator: **T4 GPU**
   *(The GPU notebook uses CuPy to accelerate blocking matrix multiplication and LightGBM for training)*
3. Runtime shape:
   - **Standard** (free): 12 GB RAM + T4 GPU — fast and sufficient for the 50K experiment
   - **High-RAM** (Colab Pro): 25 GB RAM + T4 GPU — recommended for the full 2.2M dataset run

---

## Step 4: Run the Notebook

Run **all cells in order** (top to bottom). Never skip a cell.

The notebook will:
1. Mount your Drive
2. Install `lightgbm` and `rapidfuzz` (all others pre-installed)
3. Extract the data ZIP from Drive → copy to fast local `/content/workspace/`
4. Copy your `src/` code from Drive
5. Auto-detect RAM and set optimal batch sizes
6. Run the full pipeline with checkpointing

---

## Step 5: Handle Session Disconnects

Colab free sessions disconnect after **~12 hours** (Pro after 24h).

The notebook saves checkpoints to Drive after **every country bucket** in the blocking step.

If disconnected:
```
Re-run cells 1–6    (Setup — all instant/fast)
Re-run cell 8       (Load preprocessed Parquet from Drive)
Re-run cell 9       (Define blocking functions)
Re-run cell 10      (Resumes from last completed country checkpoint)
Continue normally
```

---

## Estimated Runtimes (Using T4 GPU)

The GPU acceleration reduces blocking time dramatically.

### Free Colab (12 GB RAM + T4 GPU)
| Step | Time |
|------|------|
| Setup + EDA | 5 min |
| Preprocessing | 15-20 min |
| **Train blocking (US + India)** | **~10-20 min** |
| Feature engineering | 2-3 hours |
| LightGBM training | ~5 min |
| **Test blocking (US + India + France)** | **~10 min** |
| Test prediction | 1-2 hours |
| **Total** | **~4-6 hours** (down from 15-20 hours on CPU) |

### Colab Pro (25 GB RAM + T4 GPU)
| Step | Time |
|------|------|
| Setup + EDA | 3 min |
| Preprocessing | 8-10 min |
| **Train blocking** | **~5-10 min** |
| Feature engineering | 1-2 hours |
| LightGBM training | ~3 min |
| **Test blocking** | **~5 min** |
| Test prediction | 45-60 min |
| **Total** | **~2-3 hours**

---

## RAM Usage by Step

| Step | Peak RAM | Free Colab? |
|------|----------|-------------|
| Load all 6 raw TSVs | ~6 GB | ✅ (barely) |
| TF-IDF index (400K S23 chunk) | +~480 MB | ✅ |
| Dense cosine block (100×400K) | +~160 MB | ✅ |
| Feature matrix (83K × 17) | ~12 MB | ✅ |
| LightGBM training | ~500 MB | ✅ |

> ⚠️ **Free Colab warning**: Loading S2 (5M rows) + S3 (5.3M rows) simultaneously
> uses ~4-6 GB RAM. The notebook loads them sequentially and deletes after processing
> where possible. If you get OOM errors, reduce `S23_CHUNK_SIZE` in Cell 4 to `100_000`.

---

## Output Files

After the notebook completes, your Drive will have:

```
My Drive/AmazonML/output/
├── matching_results.tsv   ← SUBMIT THIS to the leaderboard
└── candidate_pairs.tsv    ← include in submission ZIP
```

The notebook also auto-downloads `matching_results.tsv` to your local machine (Cell 18).

---

## Iterating Faster (Dev Runs)

To test pipeline changes quickly without running full 2.2M training:

**Option A**: Run with `--sample 5000` on local machine
```bash
python src/run_pipeline.py train --sample 5000
```
Takes ~4 minutes. Good for testing code changes.

**Option B**: In Colab, the GPU notebook is already configured to sample 50K S1 records in Cell 10 for rapid experimentation:
```python
# Cell 10 in COLAB_PIPELINE_GPU.ipynb
s1_train = s1_train.sample(n=min(50000, len(s1_train)), random_state=42)
```

**Option C**: Run K-sweep on 5K sample first
```python
# Add a new cell after Cell 9:
!python /content/workspace/src/k_sweep_diagnostic.py --sample 5000 --ks 10 30 50 100
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: config` | Re-run Cell 6 (sys.path patch) |
| `FileNotFoundError: train_source1.tsv` | Re-run Cell 5 (data extraction) |
| `cupy` or GPU errors | Ensure you selected **T4 GPU** in Runtime settings |
| OOM during blocking | Reduce `S23_CHUNK_SIZE` in Cell 4 to `100_000` |
| `KeyError: entity_id` | Check TSV header — might have different column names |
| Drive quota exceeded | Delete old `.pkl` checkpoint files from `AmazonML/cache/` |
| Session disconnected during blocking | Re-run cells 1-6, 8-10; checkpoints load automatically |
