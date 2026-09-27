import json
from pathlib import Path

nb_path = Path("MASTER_PIPELINE.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        
        for i, line in enumerate(source):
            if line.startswith("from google.colab import drive"):
                source[i] = "# " + line
            elif line.startswith("drive.mount"):
                source[i] = "# " + line
            elif line.startswith("subprocess.check_call([sys.executable, '-m', 'pip', 'install'"):
                source[i] = "# " + line
            elif line.startswith("DRIVE_ROOT    = Path('/content/drive/MyDrive/Amazon-ML-entity-resolution')"):
                source[i] = "DRIVE_ROOT    = Path('.')\\n"
            elif line.startswith("LOCAL_DATASET = Path('/content/dataset')"):
                source[i] = "LOCAL_DATASET = Path('dataset')\\n"
            elif line.startswith("ROOT          = Path('/content')"):
                source[i] = "ROOT          = Path('.')\\n"
            elif line.startswith("QUARTER_SPLIT = None"):
                source[i] = "QUARTER_SPLIT = 'Q1'     # Set to Q1 for Person 1\\n"
            elif line.startswith("        split_dir = DRIVE_ROOT / 'splits'"):
                source[i] = "        split_dir = Path('splits')\\n"
        
        cell["source"] = source

with open("MASTER_PIPELINE_LOCAL.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Created MASTER_PIPELINE_LOCAL.ipynb for Q1")
