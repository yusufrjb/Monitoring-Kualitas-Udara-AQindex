"""
Eksperimen Kritis: Apakah ML hanya belajar max() rule?
====================================================
Dataset: KLHK (ISPU index per polutan + label resmi)
- Fitur: PM10_ISPU, SO2_ISPU, CO_ISPU, O3_ISPU, NO2_ISPU
- Label: Kategori resmi pemerintah
- Baseline: max(pm10, so2, co, o3, no2) -> kategori resmi (OFFICIAL RULE)

Jika ML == official rule -> ML tidak berguna
Jika ML != official rule -> cari tahu perbedaannya
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

np.random.seed(42)

# ── Load KLHK ──
df = pd.read_csv("D:/DashboardAQ_v3/data_train/klhk_combined.csv")
df = df.dropna(subset=["pm10", "so2", "co", "o3", "no2"]).copy()
df = df[df["categori"] != "BERBAHAYA"].copy()

FEATURES = ["pm10", "so2", "co", "o3", "no2"]
LABELS = ["BAIK", "SEDANG", "TIDAK SEHAT", "SANGAT TIDAK SEHAT"]

X = df[FEATURES].values.astype(np.float64)
y_raw = df["categori"].values


# ── Official max-rule ──
# Kategori resmi ditentukan dari max ISPU -> threshold
def ispu_to_cat(ispi):
    if ispi <= 50:
        return "BAIK"
    if ispi <= 100:
        return "SEDANG"
    if ispi <= 200:
        return "TIDAK SEHAT"
    if ispi <= 300:
        return "SANGAT TIDAK SEHAT"
    return "BERBAHAYA"


max_ispu = X.max(axis=1)
y_official = np.array([ispu_to_cat(m) for m in max_ispu])

# ── Compare official vs actual label ──
print("=" * 80)
print("KOMPARASI: Official max-rule vs Actual KLHK label")
print("=" * 80)
same = (y_official == y_raw).sum()
total = len(y_raw)
print(
    "Official max-rule == Actual label: %d/%d (%.2f%%)"
    % (same, total, same / total * 100)
)
print()

# Confusion matrix: official vs actual
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(LABELS)
y_actual_enc = le.transform(y_raw)
y_official_enc = le.transform(y_official)

print("Confusion: Official (rows) vs Actual (cols)")
cm = confusion_matrix(y_actual_enc, y_official_enc)
print("%15s" % "", end="")
for l in le.classes_:
    print("%15s" % l, end="")
print()
for i, l in enumerate(le.classes_):
    print("%15s" % l, end="")
    for j in range(len(le.classes_)):
        print("%15d" % cm[i, j], end="")
    print()

print()
print(
    "Official accuracy vs Actual: %.2f%%"
    % (accuracy_score(y_actual_enc, y_official_enc) * 100)
)
print()

# ── Train ML ──
print("=" * 80)
print("TRAINING RF pada ISPU index (5 fitur)")
print("=" * 80)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_actual_enc, test_size=0.25, random_state=42, stratify=y_actual_enc
)

# NO noise on KLHK — we want to see if ML can learn the official rule
rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_tr, y_tr)

y_pred = rf.predict(X_te)

print()
print("Classification Report (ML pada test set):")
print(classification_report(y_te, y_pred, target_names=le.classes_, digits=4))
print()

# ── Compare ML vs Official rule on test set ──
y_official_te = le.transform(np.array([ispu_to_cat(m) for m in X_te.max(axis=1)]))

ml_vs_official_same = (y_pred == y_official_te).sum()
print(
    "ML == Official rule pada test set: %d/%d (%.2f%%)"
    % (ml_vs_official_same, len(y_te), ml_vs_official_same / len(y_te) * 100)
)

# Breakdown: where does ML differ from official?
diff_mask = y_pred != y_official_te
print()
print("Kasus dimana ML != Official rule: %d sampel" % diff_mask.sum())
if diff_mask.sum() > 0:
    X_diff = X_te[diff_mask]
    y_pred_diff = y_pred[diff_mask]
    y_off_diff = y_official_te[diff_mask]
    y_act_diff = y_te[diff_mask]

    print()
    print("Sample perbedaan (max 20 baris):")
    print(
        "%5s %5s %5s %5s %5s | %10s | %10s | %10s | %10s"
        % ("PM10", "SO2", "CO", "O3", "NO2", "max", "Official", "ML_Pred", "Actual")
    )
    print("-" * 85)
    for i in range(min(20, len(X_diff))):
        mx = X_diff[i].max()
        print(
            "%5.0f %5.0f %5.0f %5.0f %5.0f | %10.0f | %10s | %10s | %10s"
            % (
                X_diff[i][0],
                X_diff[i][1],
                X_diff[i][2],
                X_diff[i][3],
                X_diff[i][4],
                mx,
                le.classes_[y_off_diff[i]],
                le.classes_[y_pred_diff[i]],
                le.classes_[y_act_diff[i]],
            )
        )

# ── Feature importance ──
print()
print("Feature Importance:")
imp = pd.DataFrame({"feature": FEATURES, "importance": rf.feature_importances_})
print(imp.sort_values("importance", ascending=False).to_string(index=False))

# ── Compare edge cases: near boundary ──
print()
print("=" * 80)
print("EDGE CASE ANALYSIS: Sampel dengan max ISPU dekat boundary")
print("=" * 80)
# Find samples where max ISPU is near 50, 100, 200, 300
boundaries = [50, 100, 200, 300]
for b in boundaries:
    near = np.abs(max_ispu - b) <= 5
    if near.sum() == 0:
        continue
    print()
    print("Near boundary %d (+/-5): %d samples" % (b, near.sum()))
    X_near = X[near]
    y_near_actual = y_raw[near]
    y_near_off = np.array([ispu_to_cat(m) for m in max_ispu[near]])

    # ML prediction on these
    ml_near = rf.predict(X_near)

    # ML vs official on boundary
    same_near = (ml_near == le.transform(y_near_off)).sum()
    print(
        "  ML == Official: %d/%d (%.1f%%)"
        % (same_near, len(ml_near), same_near / len(ml_near) * 100)
    )

    # ML vs actual on boundary
    same_actual = (ml_near == le.transform(y_near_actual)).sum()
    print(
        "  ML == Actual:   %d/%d (%.1f%%)"
        % (same_actual, len(ml_near), same_actual / len(ml_near) * 100)
    )

print()
print("KESIMPULAN:")
print("Jika ML == Official rule >99% -> ML hanya belajar max()")
print("Jika ML mirip Actual -> ML belajar dari data, bukan dari max()")
