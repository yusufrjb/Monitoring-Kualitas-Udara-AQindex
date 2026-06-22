import json

with open("archive/ml_model/notebooks/compare_classification.ipynb", "r") as f:
    nb = json.load(f)

# Fix Cell [6]: remove stratify (Sangat Tidak Sehat only has 1 sample)
old_cell6 = "".join(nb["cells"][6]["source"])
old_split = "X_train_real, X_test, y_train_real, y_test = train_test_split(\n    X_real, y_real, test_size=0.25, random_state=42, stratify=y_real\n)"
new_split = "X_train_real, X_test, y_train_real, y_test = train_test_split(\n    X_real, y_real, test_size=0.25, random_state=42\n)"
new_cell6 = old_cell6.replace(old_split, new_split)

nb["cells"][6]["source"] = new_cell6.splitlines(True)

with open("archive/ml_model/notebooks/compare_classification.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Removed stratify from train_test_split (Sangat Tidak Sehat hanya 1 sample)")
