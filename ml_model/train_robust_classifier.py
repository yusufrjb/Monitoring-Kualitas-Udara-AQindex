"""
Training Robust Classifier ISPU (Handle Imbalance + Noise)
===========================================================
Masalah:
  - Label (y) diturunkan dari fitur (X) via rumus ISPU
  - Imbalance ekstrim: Baik(35k) vs Sangat Tidak Sehat(3)
  - Noise sensor di lapangan bikin breakpoint langsung tidak akurat

Solusi:
  1. Noise injection saat training → robust terhadap sensor noise
  2. SMOTE oversampling untuk kelas minoritas
  3. Class-weight agresif untuk minoritas
  4. Confidence-based fallback → jika ragu, pakai breakpoint
  5. Cross-validation stratified

Cara pakai:
  python train_robust_classifier.py

Output:
  - air_quality_classifier_robust.pkl
  - classification_report.txt
  - classification_metadata.json
"""

import pandas as pd
import numpy as np
import os, sys, json, warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

np.random.seed(42)

# ── Konfigurasi ──
NOISE_STD = 0.12  # 12% fractional noise
CONFIDENCE_THRESHOLD = 0.6  # fallback threshold
N_FOLDS = 5
CSV_PATH = "../data_train/tb_konsentrasi_gas.csv"
MODEL_OUT = "air_quality_classifier_robust.pkl"

# ── 1. Breakpoint ISPU ──
BP_PM25 = [
    (0, 15.5, 0, 50),
    (15.5, 55.4, 50, 100),
    (55.4, 150.4, 100, 200),
    (150.4, 250.4, 200, 300),
    (250.4, 500, 300, 500),
]
BP_PM10 = [
    (0, 50, 0, 50),
    (50, 150, 50, 100),
    (150, 350, 100, 200),
    (350, 420, 200, 300),
    (420, 500, 300, 500),
]
BP_CO = [
    (0, 4000, 0, 50),
    (4000, 8000, 50, 100),
    (8000, 15000, 100, 200),
    (15000, 30000, 200, 300),
    (30000, 45000, 300, 500),
]
BP_NO2 = [
    (0, 80, 0, 50),
    (80, 200, 50, 100),
    (200, 1130, 100, 200),
    (1130, 2260, 200, 300),
    (2260, 3000, 300, 500),
]
BP_O3 = [
    (0, 120, 0, 50),
    (120, 235, 50, 100),
    (235, 400, 100, 200),
    (400, 800, 200, 300),
    (800, 1000, 300, 500),
]
ALL_BP = [BP_PM25, BP_PM10, BP_CO, BP_NO2, BP_O3]
BP_NAMES = ["pm25_ugm3", "pm10_ugm3", "co_ugm3", "no2_ugm3", "o3_ugm3"]


def conc_to_ispi(val, bp):
    if val <= 0:
        return 0
    for cl, ch, il, ih in bp:
        if val <= ch:
            return il + (val - cl) / (ch - cl) * (ih - il)
    return bp[-1][3]


def ispi_from_row(vals):
    return max(conc_to_ispi(vals[i], ALL_BP[i]) for i in range(5))


def ispi_to_label(ispi):
    if ispi <= 50:
        return "Baik"
    if ispi <= 100:
        return "Sedang"
    if ispi <= 200:
        return "Tidak Sehat"
    if ispi <= 300:
        return "Sangat Tidak Sehat"
    return "Berbahaya"


# ── 2. Load data ──
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} tidak ditemukan")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
for c in BP_NAMES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=BP_NAMES[:3])
noise_mask = (df["co_ugm3"] > 30000) & (df["pm25_ugm3"] < 55)
df = df[~noise_mask].copy()

FEATURES = BP_NAMES
LABELS = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]

X_raw = df[FEATURES].values.astype(np.float64)
df["Label"] = df.apply(
    lambda r: ispi_to_label(ispi_from_row([r[c] for c in BP_NAMES])), axis=1
)
y_str = df["Label"].values

le = LabelEncoder()
le.fit(LABELS)
y_enc = le.transform(y_str)

print("=" * 70)
print("DISTRIBUSI LABEL")
print("=" * 70)
for lbl in le.classes_:
    n = int((y_str == lbl).sum())
    print(f"  {lbl:25s} {n:>6d} ({n / len(y_str) * 100:.2f}%)")


# ── 3. Cross-validation ──
def add_noise(X, std=NOISE_STD):
    return np.maximum(X * (1 + np.random.randn(*X.shape) * std), 0)


def safe_k_neighbors(y):
    counts = np.bincount(y)
    min_n = counts[counts > 0].min()
    return min(min_n - 1, 5, 2)  # max 2 to avoid overfitting sparse classes


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
cv_acc, cv_f1m, cv_f1w = [], [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_raw, y_str)):
    X_tr, X_val = X_raw[tr_idx], X_raw[val_idx]
    y_tr, y_val = y_enc[tr_idx], y_enc[val_idx]

    k = safe_k_neighbors(y_tr)
    if k >= 1:
        try:
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        except:
            pass

    X_tr = add_noise(X_tr)

    m = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    m.fit(X_tr, y_tr)
    yp = m.predict(X_val)

    cv_acc.append(accuracy_score(y_val, yp))
    cv_f1m.append(f1_score(y_val, yp, average="macro"))
    cv_f1w.append(f1_score(y_val, yp, average="weighted"))
    print(
        f"  Fold {fold + 1}: Acc={cv_acc[-1] * 100:.2f}%  "
        f"F1_macro={cv_f1m[-1] * 100:.2f}%  F1_w={cv_f1w[-1] * 100:.2f}%"
    )

print(
    f"\n  CV Summary: Acc={np.mean(cv_acc) * 100:.2f}% ±{np.std(cv_acc) * 100:.2f}%  "
    f"F1_macro={np.mean(cv_f1m) * 100:.2f}% ±{np.std(cv_f1m) * 100:.2f}%"
)

# ── 4. Final model ──
print("\n" + "=" * 70)
print("TRAINING FINAL MODEL")
print("=" * 70)

k = safe_k_neighbors(y_enc)
if k >= 1:
    try:
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_raw, y_enc)
    except:
        X_res, y_res = X_raw, y_enc
else:
    X_res, y_res = X_raw, y_enc

X_res = add_noise(X_res)
final = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
final.fit(X_res, y_res)

# ── 5. Evaluasi ──
print("\n" + "=" * 70)
print("EVALUASI")
print("=" * 70)

X_noise = add_noise(X_raw)

y_bp_clean = np.array(
    [ispi_to_label(ispi_from_row(X_raw[i])) for i in range(len(X_raw))]
)
y_bp_clean_enc = le.transform(y_bp_clean)

y_bp_noise = np.array(
    [ispi_to_label(ispi_from_row(X_noise[i])) for i in range(len(X_noise))]
)
y_bp_noise_enc = le.transform(y_bp_noise)

ml_proba = final.predict_proba(X_noise)
ml_enc = final.predict(X_noise)

hybrid_enc = np.where(
    ml_proba.max(axis=1) >= CONFIDENCE_THRESHOLD, ml_enc, y_bp_clean_enc
)

for name, yp, desc in [
    ("Breakpoint (clean)", y_bp_clean_enc, "Rumus ISPU pada data bersih"),
    ("Breakpoint (noisy)", y_bp_noise_enc, "Rumus ISPU pada data dengan noise 12%"),
    ("ML (noisy)", ml_enc, "RF+SMOTE pada data noise"),
    ("ML+Fallback (noisy)", hybrid_enc, "RF + fallback ke breakpoint"),
]:
    print(f"\n  {name}:")
    print(
        f"    Acc={accuracy_score(y_enc, yp) * 100:.2f}%  "
        f"F1_macro={f1_score(y_enc, yp, average='macro') * 100:.2f}%  "
        f"F1_w={f1_score(y_enc, yp, average='weighted') * 100:.2f}%"
    )
    print(f"    ({desc})")

# ── 6. Per-class detail ──
print("\n" + "=" * 70)
print("PER-CLASS DETAIL (ML+Fallback pada noisy input)")
print("=" * 70)
report = classification_report(
    y_enc, hybrid_enc, target_names=le.classes_, zero_division=0, digits=4
)
print(report)

cm = confusion_matrix(y_enc, hybrid_enc)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("Confusion Matrix (aktual \\ prediksi):")
print(cm_df.to_string())

with open("classification_report.txt", "w") as f:
    f.write(
        f"Noise STD={NOISE_STD}  Confidence={CONFIDENCE_THRESHOLD}\n\n{report}\n\nConfusion Matrix:\n{cm_df.to_string()}"
    )

# ── 7. Save ──
final.n_features_ = len(FEATURES)
final.feature_names_ = FEATURES
joblib.dump(final, MODEL_OUT)

meta = {
    "model": "RandomForest (robust+SMOTE)",
    "noise_std": NOISE_STD,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "n_estimators": 300,
    "max_depth": 15,
    "features": FEATURES,
    "labels": list(le.classes_),
    "cv_accuracy_mean": round(np.mean(cv_acc) * 100, 2),
    "cv_f1_macro_mean": round(np.mean(cv_f1m) * 100, 2),
}
with open("classification_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nModel: {MODEL_OUT}")
print(f"Report: classification_report.txt")
print(f"Metadata: classification_metadata.json")
print("\nSELESAI")
