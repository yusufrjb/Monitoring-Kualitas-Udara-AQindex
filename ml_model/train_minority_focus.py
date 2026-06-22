"""
Training Klasifikasi — Fokus Minoritas (tanpa fallback)
=======================================================
Pendekatan: RandomForest standar + noise injection + synthetic minority oversampling
(tidak pakai SMOTE/BalancedRF karena hasilnya kurang optimal untuk ekstrim minority)

Cara pakai: python train_minority_focus.py
"""

import pandas as pd
import numpy as np
import os, json, warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

np.random.seed(42)

NOISE_STD = 0.12
CSV_PATH = "../data_train/tb_konsentrasi_gas.csv"
MODEL_OUT = "air_quality_classifier_minority_focus.pkl"

# ── Breakpoint ISPU ──
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


# ── Load sensor data ──
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)
for c in BP_NAMES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=BP_NAMES[:3])
df = df[~((df["co_ugm3"] > 30000) & (df["pm25_ugm3"] < 55))].copy()

FEATURES = BP_NAMES
LABELS = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]

df["Label"] = df.apply(
    lambda r: ispi_to_label(ispi_from_row([r[c] for c in BP_NAMES])), axis=1
)

le = LabelEncoder()
le.fit(LABELS)

# ── Generate synthetic minority data ──
# Gunakan range ISPU untuk generate data sintetis yang REALISTIS untuk minoritas
SYNTH_RANGES = {
    "Tidak Sehat": [(36, 54), (151, 348), (10100, 16800), (81, 178), (236, 399)],
    "Sangat Tidak Sehat": [
        (56, 149),
        (351, 419),
        (17100, 33800),
        (181, 279),
        (401, 799),
    ],
    "Berbahaya": [(151, 350), (421, 550), (34100, 50000), (281, 500), (801, 1000)],
}


def nearest_posdef(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-6)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# Compute per-class covariance from real data
df_real = df.copy()
per_class_corr = {}
for lbl in ["Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]:
    sub = df_real[df_real["Label"] == lbl]
    if len(sub) >= 5:
        c = sub[FEATURES].corr().fillna(0).values
        per_class_corr[lbl] = nearest_posdef(c)
    else:
        per_class_corr[lbl] = np.eye(5)

N_SYNTH = {"Tidak Sehat": 1500, "Sangat Tidak Sehat": 1500, "Berbahaya": 1500}
synth_rows = []

for label, ranges in SYNTH_RANGES.items():
    n = N_SYNTH[label]
    means = [(lo + hi) / 2 for lo, hi in ranges]
    stds = [(hi - lo) / 4 for lo, hi in ranges]
    D = np.diag(stds)
    cov = D @ per_class_corr[label] @ D
    cov = nearest_posdef(cov)

    try:
        samples = np.random.multivariate_normal(means, cov, n)
    except np.linalg.LinAlgError:
        samples = np.random.multivariate_normal(means, cov + np.eye(5) * 1e-6, n)

    # Clip ke range
    for i in range(5):
        lo, hi = ranges[i]
        samples[:, i] = np.clip(samples[:, i], lo, hi)

    for i in range(n):
        row = {feat: float(samples[i, j]) for j, feat in enumerate(FEATURES)}
        row["Label"] = label
        synth_rows.append(row)

df_synth = pd.DataFrame(synth_rows)

# ── Gabung real + synthetic ──
df_all = pd.concat([df, df_synth], ignore_index=True)
X = df_all[FEATURES].values.astype(np.float64)
y_str = df_all["Label"].values
y_enc = le.transform(y_str)

print("=" * 70)
print("DISTRIBUSI (Real + Synthetic Minoritas)")
print("=" * 70)
for i, lbl in enumerate(le.classes_):
    n = int((y_str == lbl).sum())
    print(f"  [{i}] {lbl:25s} {n:>6d}")

# ── Train/test split ──
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)


# ── Noise injection ──
def add_noise(X, std=NOISE_STD):
    return np.maximum(X * (1 + np.random.randn(*X.shape) * std), 0)


X_tr_noisy = add_noise(X_tr)

# ── Train ──
print("\nTraining RandomForest (500 trees, balanced class_weight)...")
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=14,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_tr_noisy, y_tr)
print("Selesai")

# ── Evaluasi ──
X_te_noise = add_noise(X_te)
y_pred = model.predict(X_te_noise)

acc = accuracy_score(y_te, y_pred)
f1_m = f1_score(y_te, y_pred, average="macro")
f1_w = f1_score(y_te, y_pred, average="weighted")

print("\n" + "=" * 70)
print(f"ACCURACY:  {acc * 100:.2f}%")
print(f"F1 MACRO:  {f1_m * 100:.2f}%")
print(f"F1 WEIGHT: {f1_w * 100:.2f}%")
print("=" * 70)

report = classification_report(y_te, y_pred, target_names=le.classes_, digits=4)
print(report)

cm = confusion_matrix(y_te, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("\nConfusion Matrix:")
print(cm_df.to_string())

# ── Banding: Breakpoint pada noisy data ──
y_bp_noise = np.array(
    [ispi_to_label(ispi_from_row(X_te_noise[i])) for i in range(len(X_te_noise))]
)
y_bp_enc = le.transform(y_bp_noise)

print("\n" + "=" * 70)
print("BANDING: ML vs Breakpoint (pada data noise 12%)")
print("=" * 70)
bp_f1_m = f1_score(y_te, y_bp_enc, average="macro")
bp_acc = accuracy_score(y_te, y_bp_enc)
print(f"  Breakpoint: Acc={bp_acc * 100:.2f}%  F1_macro={bp_f1_m * 100:.2f}%")
print(f"  ML:         Acc={acc * 100:.2f}%  F1_macro={f1_m * 100:.2f}%")

# ── Fokus minoritas ──
print("\n" + "=" * 70)
print("FOKUS MINORITAS (test set)")
print("=" * 70)
for lbl in ["Berbahaya", "Sangat Tidak Sehat", "Tidak Sehat"]:
    idx = list(le.classes_).index(lbl)
    n_te = int((y_te == idx).sum())
    if n_te == 0:
        continue
    ml_ok = int((y_pred[y_te == idx] == idx).sum())
    bp_ok = int((y_bp_enc[y_te == idx] == idx).sum())
    print(
        f"  {lbl:25s} n_test={n_te:>4d}:  ML={ml_ok:>4d}/{n_te}  BP={bp_ok:>4d}/{n_te}"
    )

# ── Save ──
model.n_features_ = len(FEATURES)
model.feature_names_ = FEATURES
joblib.dump(model, MODEL_OUT)

print(f"\nModel: {MODEL_OUT}")
print("SELESAI")
