import json

nb_path = "MASTER_PIPELINE_LOCAL.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if r"\n" in line:
                source[i] = line.replace(r"\n", "\n")
        cell["source"] = source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Fixed syntax error")
