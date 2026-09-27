# Distributed Execution Plan

This document outlines the workflow for 4 team members to execute the Entity Resolution pipeline in parallel.

## 1. Dataset Partitioning

The `train_source1.tsv` dataset (approx. 2.2M entities) has been safely partitioned into 4 quarters. 
To guarantee fair resource distribution (memory footprint, candidate hit counts) and balanced training for the LightGBM model, the splits were stratified on three variables:
- **Country**
- **Match Count Bucket** (0 matches, 1, 2, 3-4, 5-8, 9+)
- **Singleton Flag**

Each team member is assigned one quarter (~551K entities). The data is distributed into 4 folders (`person1/`, `person2/`, `person3/`, `person4/`). 
Each folder contains:
- `Q[X]_s1_ids.txt`: The specific S1 entity IDs for this quarter.
- `Q[X]_ground_truth.tsv`: The ground truth corresponding to this S1 subset.
- `train_source2.tsv` & `train_source3.tsv`: The full S2 and S3 files, copied so candidates can be sourced from anywhere.

## 2. Individual Execution (Parallel)

Each person should run the `MASTER_PIPELINE.ipynb` in Colab.

### Pre-computation
1. Upload the contents of your designated folder (e.g. `person1/`) to your Google Drive `dataset/train/` folder.
2. In Cell 2 of the notebook, configure `QUARTER_SPLIT = 'Q1'` (or Q2, Q3, Q4 depending on your assignment).
3. Ensure `FULL_S23 = True` so your S1 entities are tested against all possible combinations.

### Execution
Run all cells. The pipeline will:
1. Only load the S1 entities belonging to your quarter.
2. Generate candidates against the *entire* S2/S3 universe.
3. Compute all 46 pairwise features.
4. Output your candidate lists and features into your local Colab storage or mounted Drive.

*Note: Generating candidates against the full 10.3M S2/S3 universe takes hours. Parallelizing the 551K subsets guarantees each person's instance won't exceed RAM and finishes significantly faster.*

## 3. Merging and Final Model

Once all 4 runs are completed, we will have 4 partial candidate files with fully engineered features.
- We will NOT train 4 different LightGBM models.
- We will NOT fine-tune a model sequentially.

**Final Step:** We will aggregate the 4 generated feature matrices back together into a single global dataset, and train **ONE final LightGBM model** on all of it. This prevents the final chunk from heavily biasing the tree splits, maintaining overall model generalization.
