import json
from pathlib import Path

nb_path = Path("MASTER_PIPELINE.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        
        # Patch Cell 2 (Config)
        if any("SAMPLE_S1" in line for line in source) and any("K_CANDIDATES" in line for line in source):
            for i, line in enumerate(source):
                if line.startswith("SAMPLE_S1     = 50_000"):
                    # Insert QUARTER_SPLIT right after
                    source.insert(i+1, "QUARTER_SPLIT = None     # 'Q1', 'Q2', 'Q3', 'Q4' or None. If set, loads only that quarter's S1.\\n")
                    break
            
            for i, line in enumerate(source):
                if line.startswith("print(f'SAMPLE_S1={SAMPLE_S1}"):
                    source[i] = "print(f'SAMPLE_S1={SAMPLE_S1}  QUARTER_SPLIT={QUARTER_SPLIT}  FULL_S23={FULL_S23}  K={K_CANDIDATES}')\\n"
                    break
        
        # Patch Cell 4 (Data Loader)
        if any("def load_data(" in line for line in source):
            for i, line in enumerate(source):
                if line.startswith("def load_data(sample_s1=None, full_s23=False, split='train'):"):
                    source[i] = "def load_data(sample_s1=None, full_s23=False, quarter_split=None, split='train'):\\n"
            
            for i, line in enumerate(source):
                if "s3 = _read(f'{prefix}_source3.tsv', 'source3_entity_id')" in line:
                    insert_idx = i + 1
                    patch = [
                        "\\n",
                        "    if quarter_split and split == 'train':\\n",
                        "        split_dir = DRIVE_ROOT / 'splits'\\n",
                        "        q_file = split_dir / f'{quarter_split}_s1_ids.txt'\\n",
                        "        if q_file.exists():\\n",
                        "            print(f'  Filtering S1 to {quarter_split}...', end=' ')\\n",
                        "            q_ids = set(pd.read_csv(q_file, header=None, dtype=str)[0])\\n",
                        "            s1 = s1[s1['entity_id'].isin(q_ids)].reset_index(drop=True)\\n",
                        "            print(f'kept {len(s1):,} rows.')\\n",
                        "        else:\\n",
                        "            print(f'  Warning: {q_file} not found. Proceeding with full S1.')\\n",
                        "\\n"
                    ]
                    for j, p in enumerate(patch):
                        source.insert(insert_idx + j, p)
                    break
            
            for i, line in enumerate(source):
                if line.startswith("s1, s2, s3, gt_dict = load_data(sample_s1=SAMPLE_S1, full_s23=FULL_S23, split='train')"):
                    source[i] = "s1, s2, s3, gt_dict = load_data(sample_s1=SAMPLE_S1, full_s23=FULL_S23, quarter_split=QUARTER_SPLIT, split='train')\\n"
                    break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully patched MASTER_PIPELINE.ipynb")
