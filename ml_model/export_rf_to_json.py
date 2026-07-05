import json, sys, os
import numpy as np
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "random_forest_air_quality.pkl")
OUTPUT_PATH = os.path.join(BASE, "..", "src", "lib", "rf_model.json")

m = joblib.load(MODEL_PATH)

trees = []
for est in m.estimators_:
    tree = est.tree_
    nodes = []
    for i in range(tree.node_count):
        node = {
            "id": int(i),
            "feature": int(tree.feature[i]),
            "threshold": float(tree.threshold[i]),
            "children_left": int(tree.children_left[i]),
            "children_right": int(tree.children_right[i]),
            "value": tree.value[i].tolist(),
            "n_node_samples": int(tree.n_node_samples[i]),
            "impurity": float(tree.impurity[i]),
        }
        nodes.append(node)
    trees.append(nodes)

export = {
    "classes": list(m.classes_),
    "n_features": int(m.n_features_in_),
    "feature_names": list(m.feature_names_in_),
    "n_classes": int(m.n_classes_),
    "n_estimators": len(m.estimators_),
    "trees": trees,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(export, f)

print(
    f"Exported {len(trees)} trees ({sum(len(t) for t in trees)} nodes) to {OUTPUT_PATH}"
)
