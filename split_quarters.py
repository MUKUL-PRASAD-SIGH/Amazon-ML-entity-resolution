import pandas as pd
import numpy as np
from pathlib import Path

# Paths (adjust if running locally instead of Colab)
# If on Colab, this would be:
# TRAIN_DIR = Path("/content/drive/MyDrive/AmazonML/dataset/train")
# SPLIT_DIR = Path("/content/drive/MyDrive/AmazonML/splits")

# Using local paths for the workspace
TRAIN_DIR = Path("dataset/train")
SPLIT_DIR = Path("splits")

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
s1 = pd.read_csv(
    TRAIN_DIR / "train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

gt = pd.read_csv(
    TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"S1: {len(s1)}")
print(f"GT: {len(gt)}")

print("Calculating strata...")
gt["num_matches"] = gt["matched_entity_ids"].apply(
    lambda x: 0 if not x else len(x.split(","))
)
gt["is_singleton"] = (gt["num_matches"] == 0).astype(int)

split_df = s1.merge(
    gt[["source1_entity_id", "num_matches", "is_singleton"]],
    left_on="entity_id",
    right_on="source1_entity_id",
    how="left"
)

split_df["num_matches"] = split_df["num_matches"].fillna(0).astype(int)
split_df["is_singleton"] = split_df["is_singleton"].fillna(1).astype(int)

def match_bucket(n):
    if n == 0: return "0"
    elif n == 1: return "1"
    elif n == 2: return "2"
    elif n <= 4: return "3_4"
    elif n <= 8: return "5_8"
    else: return "9_plus"

split_df["match_bucket"] = split_df["num_matches"].apply(match_bucket)
split_df["stratum"] = split_df["country"].astype(str) + "__" + split_df["match_bucket"]

print("Assigning quarters...")
rng = np.random.default_rng(42)
split_df["quarter"] = -1

for _, idx in split_df.groupby("stratum", sort=False).groups.items():
    idx = np.array(list(idx))
    rng.shuffle(idx)
    for i, row_idx in enumerate(idx):
        split_df.loc[row_idx, "quarter"] = i % 4

print("Saving splits...")
for q in range(4):
    qdf = split_df[split_df["quarter"] == q]
    
    # Save S1 IDs
    qdf[["entity_id"]].to_csv(
        SPLIT_DIR / f"Q{q+1}_s1_ids.txt",
        index=False,
        header=False
    )
    
    # Save Ground Truth
    gt_q = gt[gt["source1_entity_id"].isin(set(qdf["entity_id"]))]
    gt_q.to_csv(
        SPLIT_DIR / f"Q{q+1}_ground_truth.tsv",
        sep="\t",
        index=False
    )
    
    print(f"Q{q+1}: {len(qdf):,} S1 entities")

print("Done! You can upload the 'splits' folder to your Google Drive.")
