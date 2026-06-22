import pandas as pd
import numpy as np
import os, warnings, time, json

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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

np.random.seed(42)

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
FEATURES = BP_NAMES
LABELS = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]


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
CSV_PATH = "../data_train/tb_konsentrasi_gas.csv"
df = pd.read_csv(CSV_PATH)
for c in BP_NAMES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=BP_NAMES[:3])
df = df[~((df["co_ugm3"] > 30000) & (df["pm25_ugm3"] < 55))].copy()

df["Label"] = df.apply(
    lambda r: ispi_to_label(ispi_from_row([r[c] for c in BP_NAMES])), axis=1
)

# ── CORRECT PIPELINE: split BEFORE synthetic ──
le = LabelEncoder()
le.fit(LABELS)

X = df[FEATURES].values.astype(np.float64)
y_str = df["Label"].values
y_enc = le.transform(y_str)

X_tr, X_te, y_tr, y_te, y_str_tr, y_str_te = train_test_split(
    X, y_enc, y_str, test_size=0.25, random_state=42, stratify=y_enc
)

print(f"REAL DATA — Train: {len(X_tr)}, Test: {len(X_te)}")
print("\nDistribusi TRAINING SET (real only):")
for i, lbl in enumerate(le.classes_):
    n = int((y_str_tr == lbl).sum())
    print(f"  [{i}] {lbl:25s} {n:>6d}")

print("\nDistribusi TEST SET (real only — murni!):")
for i, lbl in enumerate(le.classes_):
    n = int((y_str_te == lbl).sum())
    print(f"  [{i}] {lbl:25s} {n:>6d}")

# ── Synthetic minority (hanya dari training set stats) ──
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


# Gunakan training set untuk kovarian
df_tr = pd.DataFrame(X_tr, columns=FEATURES)
df_tr["Label"] = y_str_tr

per_class_corr = {}
for lbl in ["Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]:
    sub = df_tr[df_tr["Label"] == lbl]
    if len(sub) >= 5:
        c = sub[FEATURES].corr().fillna(0).values
        per_class_corr[lbl] = nearest_posdef(c)
    else:
        per_class_corr[lbl] = np.eye(5)

MAJORITY_N = max((y_str_tr == lbl).sum() for lbl in ["Baik", "Sedang"])
N_SYNTH = {}
for lbl in SYNTH_RANGES:
    n_real = int((y_str_tr == lbl).sum())
    N_SYNTH[lbl] = max(MAJORITY_N - n_real, 0)

print(f"\nSynthetic per kelas (target = {MAJORITY_N}):")
for lbl, n in N_SYNTH.items():
    print(f"  {lbl:25s}: {n:>6d}")

synth_rows = []
for label, ranges in SYNTH_RANGES.items():
    n = N_SYNTH[label]
    if n <= 0:
        continue
    means = [(lo + hi) / 2 for lo, hi in ranges]
    stds = [(hi - lo) / 4 for lo, hi in ranges]
    D = np.diag(stds)
    cov = D @ per_class_corr[label] @ D
    cov = nearest_posdef(cov)
    try:
        samples = np.random.multivariate_normal(means, cov, n)
    except:
        samples = np.random.multivariate_normal(means, cov + np.eye(5) * 1e-6, n)
    for i in range(5):
        lo, hi = ranges[i]
        samples[:, i] = np.clip(samples[:, i], lo, hi)
    for i in range(n):
        row = {feat: float(samples[i, j]) for j, feat in enumerate(FEATURES)}
        row["Label"] = label
        synth_rows.append(row)

df_synth = pd.DataFrame(synth_rows)
X_synth = df_synth[FEATURES].values.astype(np.float64)
y_synth_str = df_synth["Label"].values
y_synth_enc = le.transform(y_synth_str)

# ── Gabung hanya di training set ──
X_tr_aug = np.vstack([X_tr, X_synth])
y_tr_aug = np.concatenate([y_tr, y_synth_enc])
y_str_tr_aug = np.concatenate([y_str_tr, y_synth_str])

print(
    f"\nTraining set final: {len(X_tr_aug)} baris (real {len(X_tr)} + synthetic {len(X_synth)})"
)
print("Distribusi:")
for i, lbl in enumerate(le.classes_):
    n = int((y_str_tr_aug == lbl).sum())
    print(f"  [{i}] {lbl:25s} {n:>6d}")

# ── Test set: 100% REAL ──
print(
    f"\nTest set: {len(X_te)} baris (100% REAL, {int((y_str_te == 'Baik').sum())} Baik, {int((y_str_te == 'Sedang').sum())} Sedang, {int((y_str_te == 'Tidak Sehat').sum())} Tdk Sehat, {int((y_str_te == 'Sangat Tidak Sehat').sum())} Sgt Tdk Sehat, {int((y_str_te == 'Berbahaya').sum())} Berbahaya)"
)

# ── Model Definitions ──
NOISE_STD = 0.15


def add_noise(X, std=NOISE_STD):
    return np.maximum(X * (1 + np.random.randn(*X.shape) * std), 0)


X_tr_noisy = add_noise(X_tr_aug)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        max_depth=14,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=1,
        random_state=42,
        verbosity=0,
        eval_metric="mlogloss",
    ),
}

# ── Training & Evaluation on 100% REAL test ──
results = []
trained = {}

X_te_noise = add_noise(X_te)
y_bp_noise = np.array(
    [ispi_to_label(ispi_from_row(X_te_noise[i])) for i in range(len(X_te_noise))]
)
y_bp_enc = le.transform(y_bp_noise)


def eval_model(y_true, y_pred, name, train_time):
    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro")
    f1_w = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(
        y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    results.append(
        {
            "Model": name,
            "Accuracy": round(acc * 100, 2),
            "F1_Macro": round(f1_m * 100, 2),
            "F1_Weighted": round(f1_w * 100, 2),
            "Train_time": round(train_time, 3),
            "Report": report,
            "CM": cm,
        }
    )
    print(
        f"  {name:20s} Acc={acc * 100:.2f}%  F1_macro={f1_m * 100:.2f}%  F1_w={f1_w * 100:.2f}%  ({train_time:.2f}s)"
    )
    return report


print("\n========== EVALUASI PADA TEST SET 100% REAL ==========")
print("\nBreakpoint (noisy test, real data):")
eval_model(y_te, y_bp_enc, "Breakpoint (noisy)", 0)

for name, model in models.items():
    t0 = time.time()
    model.fit(X_tr_noisy, y_tr_aug)
    t1 = time.time()
    y_pred = model.predict(X_te_noise)
    print(f"\n{name}:")
    eval_model(y_te, y_pred, name, t1 - t0)
    trained[name] = model

# ── Summary ──
print("\n" + "=" * 80)
print("RINGKASAN — TEST 100% REAL")
print("=" * 80)
summary = pd.DataFrame(results).drop(columns=["Report", "CM"])
print(summary.to_string(index=False))

# ── Per-class report ──
for r in results:
    print(f"\n--- {r['Model']} ---")
    if r["Model"] == "Breakpoint (noisy)":
        y_pred_for_report = y_bp_enc
    else:
        model = trained[r["Model"]]
        y_pred_for_report = model.predict(X_te_noise)
    print(
        classification_report(
            y_te, y_pred_for_report, target_names=le.classes_, digits=4
        )
    )

# ── Fokus minoritas ──
minority_labels = ["Berbahaya", "Sangat Tidak Sehat", "Tidak Sehat"]
print("\n" + "=" * 80)
print("FOKUS MINORITAS (recall per class) — TEST 100% REAL")
print("=" * 80)
for r in results:
    report = r["Report"]
    print(f"{r['Model']:20s}", end="")
    for lbl in minority_labels:
        rec = report[lbl]["recall"] * 100
        n_support = int(report[lbl]["support"])
        print(f"  {lbl:20s} recall={rec:.1f}% (n={n_support})", end="")
    print()

# ── Drift test ──
print("\n" + "=" * 80)
print("DRIFT TEST — TEST 100% REAL")
print("=" * 80)
for drift_std in [0.05, 0.10, 0.15, 0.20, 0.25]:
    X_drift = add_noise(X_te, drift_std)
    y_bp_drift = le.transform(
        [ispi_to_label(ispi_from_row(X_drift[i])) for i in range(len(X_drift))]
    )
    bp_acc = accuracy_score(y_te, y_bp_drift)
    print(f"\n  Noise {drift_std * 100:.0f}%:")
    print(f"    {'Breakpoint':20s} Acc={bp_acc * 100:.1f}%")
    for name, model in trained.items():
        y_pred = model.predict(X_drift)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro")
        winner = "<<<" if acc > bp_acc else ""
        print(
            f"    {name:20s} Acc={acc * 100:.1f}%  F1_macro={f1 * 100:.1f}%  {winner}"
        )

# ── Sanity check ──
print("\n" + "=" * 80)
print("SANITY CHECK — TEST 100% REAL")
print("=" * 80)
sanity_cases = [
    ("Udara gunung (sgt bersih)", [5, 10, 500, 10, 30]),
    ("Kantor AC (bersih)", [8, 15, 1000, 20, 40]),
    ("Kota pagi (bersih)", [12, 25, 2000, 30, 60]),
    ("Jalan macet (sedang)", [25, 60, 6000, 60, 100]),
    ("Dekat pabrik (tdk sehat)", [45, 180, 14000, 120, 250]),
    ("Kabut tebal (sgt tdk sehat)", [100, 300, 20000, 200, 500]),
    ("Pabrik bocor (berbahaya)", [200, 450, 35000, 300, 800]),
]
print(f"{'Kasus':30s} {'BP':18s}", end="")
for name in trained:
    print(f"{name:20s}", end="")
print()
for name, vals in sanity_cases:
    X_s = np.array([vals], dtype=np.float64)
    bp_label = ispi_to_label(ispi_from_row(vals))
    print(f"{name:30s} {bp_label:18s}", end="")
    for mname, model in trained.items():
        pred = le.classes_[model.predict(X_s)[0]]
        proba = model.predict_proba(X_s)[0].max()
        ok = (
            "OK"
            if pred == bp_label
            else (
                "minor"
                if abs(
                    list(le.classes_).index(pred) - list(le.classes_).index(bp_label)
                )
                < 2
                else "KRITIS"
            )
        )
        print(f"{pred:15s}({proba * 100:.0f}%)", end="")
    print()

print("\nSELESAI")
