import json

with open("archive/ml_model/notebooks/compare_classification.ipynb", "r") as f:
    nb = json.load(f)
print("Cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    print(f"\n=== Cell [{i}] ({c['cell_type']}) ===")
    print(src[:500])
    if len(src) > 500:
        print(f"... ({len(src)} chars total)")
