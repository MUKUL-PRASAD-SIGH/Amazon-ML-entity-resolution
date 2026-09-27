import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json

TRAIN_DIR = Path("dataset/train")
PERSON_DIRS = [Path(f"person{i}") for i in range(1, 5)]

for d in PERSON_DIRS:
    d.mkdir(parents=True, exist_ok=True)

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

print("Validating balance...")
stats = []
for q in range(4):
    qdf = split_df[split_df["quarter"] == q]
    singletons = qdf["is_singleton"].sum()
    total_matches = qdf["num_matches"].sum()
    stats.append({
        "person": q + 1,
        "total_s1": len(qdf),
        "singletons": int(singletons),
        "singleton_pct": round(singletons / len(qdf) * 100, 2),
        "total_matches": int(total_matches)
    })
    
print(json.dumps(stats, indent=2))

print("Distributing files...")
for q in range(4):
    person_dir = PERSON_DIRS[q]
    qdf = split_df[split_df["quarter"] == q]
    
    # Save S1 IDs
    qdf[["entity_id"]].to_csv(
        person_dir / f"Q{q+1}_s1_ids.txt",
        index=False,
        header=False
    )
    
    # Save Ground Truth
    gt_q = gt[gt["source1_entity_id"].isin(set(qdf["entity_id"]))]
    gt_q.to_csv(
        person_dir / f"Q{q+1}_ground_truth.tsv",
        sep="\t",
        index=False
    )
    
    # Copy S2 and S3
    print(f"Copying S2 and S3 for person {q+1}...")
    shutil.copy2(TRAIN_DIR / "train_source2.tsv", person_dir / "train_source2.tsv")
    shutil.copy2(TRAIN_DIR / "train_source3.tsv", person_dir / "train_source3.tsv")

print("Distribution complete!")
