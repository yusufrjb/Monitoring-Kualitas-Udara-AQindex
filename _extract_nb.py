import json, sys, os

with open("archive/ml_model/notebooks/compare_classification.ipynb", "r") as f:
    nb = json.load(f)

lines = ["# -*- coding: utf-8 -*-"]
for c in nb["cells"]:
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        if src.strip():
            lines.append(src)
            lines.append("")

with open("archive/ml_model/notebooks/_run_notebook.py", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(
    f"Extracted {len([c for c in nb['cells'] if c['cell_type'] == 'code'])} code cells"
)
