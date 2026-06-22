"""
Klasifikasi Kualitas Udara — ML sebagai Robustness Layer
===========================================================
ISPU tetap backbone. ML hanya mengurangi fluktuasi akibat noise sensor.

Pipeline:
  Training:  ug/m3 -> noise 10% -> RF 500 trees (label dari ISPU max())
  Inference: ug/m3 -> RF predict -> class + confidence + risk_score
"""

import pandas as pd, numpy as np, os, json, warnings, time

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

np.random.seed(42)
NOISE_STD = 0.10
CSV_PATH = "../data_train/tb_konsentrasi_gas_30d.csv"
MODEL_OUT = "air_quality_classifier_final.pkl"
LABELS = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]
SEVERITY = {l: i for i, l in enumerate(LABELS)}
RISK_W = [0.0, 0.3, 0.6, 0.8, 1.0]

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


def add_noise(X, std=NOISE_STD):
    return np.maximum(X * (1 + np.random.randn(*X.shape) * std), 0)


# ── 1. LOAD ──
print("1. LOAD DATA 30 HARI")
df = pd.read_csv(CSV_PATH)
for c in BP_NAMES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=BP_NAMES[:3])
df = df[~((df["co_ugm3"] > 30000) & (df["pm25_ugm3"] < 55))].copy()
df["Label"] = df.apply(
    lambda r: ispi_to_label(ispi_from_row([r[c] for c in BP_NAMES])), axis=1
)
print(
    f"  {len(df)} baris, {df['created_at'].min()[:10]} to {df['created_at'].max()[:10]}"
)
for l in LABELS:
    print(f"    {l}: {(df['Label'] == l).sum()}")

# ── 2. SPLIT ──
le = LabelEncoder()
le.fit(LABELS)
X, y_str = df[FEATURES].values.astype(np.float64), df["Label"].values
y_enc = le.transform(y_str)
X_tr, X_te, y_tr, y_te, y_str_tr, y_str_te = train_test_split(
    X, y_enc, y_str, test_size=0.25, random_state=42, stratify=y_enc
)
print(f"\n2. SPLIT: Train={len(X_tr)}, Test={len(X_te)} (100% REAL)")

# ── 3. SYNTHETIC ──
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


df_tr = pd.DataFrame(X_tr, columns=FEATURES)
df_tr["Label"] = y_str_tr
per_class_corr = {}
for lbl in ["Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]:
    sub = df_tr[df_tr["Label"] == lbl]
    per_class_corr[lbl] = (
        nearest_posdef(sub[FEATURES].corr().fillna(0).values)
        if len(sub) >= 5
        else np.eye(5)
    )

MAJORITY_N = max((y_str_tr == lbl).sum() for lbl in ["Baik", "Sedang"])
synth_rows = []
for label, ranges in SYNTH_RANGES.items():
    n_real = int((y_str_tr == label).sum())
    n_synth = max(MAJORITY_N - n_real, 0)
    if n_synth <= 0:
        continue
    means = [(lo + hi) / 2 for lo, hi in ranges]
    stds = [(hi - lo) / 4 for lo, hi in ranges]
    cov = np.diag(stds) @ per_class_corr[label] @ np.diag(stds)
    cov = nearest_posdef(cov)
    try:
        samples = np.random.multivariate_normal(means, cov, n_synth)
    except:
        samples = np.random.multivariate_normal(means, cov + np.eye(5) * 1e-6, n_synth)
    for i in range(5):
        lo, hi = ranges[i]
        samples[:, i] = np.clip(samples[:, i], lo, hi)
    for i in range(n_synth):
        synth_rows.append(
            {feat: float(samples[i, j]) for j, feat in enumerate(FEATURES)}
            | {"Label": label}
        )
    print(f"  {label}: synth {n_synth}")
print(f"3. SYNTHETIC: {len(synth_rows)} total")

df_synth = pd.DataFrame(synth_rows)
X_synth = df_synth[FEATURES].values.astype(np.float64)
y_synth_str = df_synth["Label"].values
X_tr_aug = np.vstack([X_tr, X_synth])
y_str_tr_aug = np.concatenate([y_str_tr, y_synth_str])
y_tr_aug = le.transform(y_str_tr_aug)

# ── 4. TRAIN (MULTI-MODEL COMPARISON) ──
print("4. TRAIN — RF vs XGBoost vs LightGBM")
X_tr_n = add_noise(X_tr_aug)

model_defs = {
    "RF": RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
        eval_metric="mlogloss",
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
}

trained = {}
for nm, m in model_defs.items():
    t0 = time.time()
    m.fit(X_tr_n, y_tr_aug)
    print(f"   {nm:10s} ({time.time() - t0:.1f}s)")
    trained[nm] = m

model = trained["RF"]

# ── 5. EVAL (MULTI-MODEL) ──
print("\n5. HASIL (test noise 10%):")
X_te_n = add_noise(X_te)
for nm, m in trained.items():
    yp = m.predict(X_te_n)
    a, f = accuracy_score(y_te, yp), f1_score(y_te, yp, average="macro")
    print(f"   {nm:10s} Acc={a * 100:.1f}%  F1_macro={f * 100:.1f}%")

y_bp = le.transform(
    [ispi_to_label(ispi_from_row(X_te_n[i])) for i in range(len(X_te_n))]
)
bp_a, bp_f = accuracy_score(y_te, y_bp), f1_score(y_te, y_bp, average="macro")
print(f"   {'BP':10s} Acc={bp_a * 100:.1f}%  F1_macro={bp_f * 100:.1f}%")

# ── 6. DRIFT COMPARISON ──
print("\n6. DRIFT TEST (Multi-Model):")
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]
drift_results = {}
for nm in list(trained.keys()) + ["BP"]:
    drift_results[nm] = {}
    for s in NOISE_LEVELS:
        X_d = add_noise(X_te, s)
        if nm == "BP":
            yp = le.transform(
                [ispi_to_label(ispi_from_row(X_d[i])) for i in range(len(X_d))]
            )
        else:
            yp = trained[nm].predict(X_d)
        drift_results[nm][f"{s * 100:.0f}%"] = round(accuracy_score(y_te, yp) * 100, 1)

header = f"{'Model':>10s}" + "".join(f"{s * 100:>8.0f}%  " for s in NOISE_LEVELS)
print(f"   {header}")
for nm in drift_results:
    vals = "  ".join(f"{v:>6.1f}%" for v in drift_results[nm].values())
    avg_drop = np.mean([drift_results[nm][f"{s * 100:.0f}%"] for s in NOISE_LEVELS])
    var = np.std([drift_results[nm][f"{s * 100:.0f}%"] for s in NOISE_LEVELS])
    print(f"   {nm:>10s}  {vals}  | avg={avg_drop:.1f}%  std={var:.2f}")
print(f"\n   {'BP':>10s}  baseline — model dengan std terendah = paling stabil")

drift_rf = dict(drift_results["RF"])
drift_all = {m: dict(drift_results[m]) for m in drift_results}

# ── 7. BOUNDARY ──
bc = []
for pm25 in [14.0, 15.4, 15.6, 17.0, 50.0, 55.4, 56.0, 60.0]:
    for pm10 in [49, 51, 149, 151]:
        for lbl in ["Baik", "Sedang"]:
            if ispi_to_label(ispi_from_row([pm25, pm10, 5000, 50, 50])) == lbl:
                bc.append([pm25, pm10, 5000, 50, 50, lbl])
Xb = np.array([r[:5] for r in bc], dtype=np.float64)
yb_e = le.transform(np.array([r[5] for r in bc]))
Xb_n = add_noise(Xb)
ml_ok = int((model.predict(Xb_n) == yb_e).sum())
bp_ok = int(
    (
        le.transform(
            np.array([ispi_to_label(ispi_from_row(Xb_n[i])) for i in range(len(Xb_n))])
        )
        == yb_e
    ).sum()
)
print(
    f"\n7. BOUNDARY: ML={ml_ok}/{len(Xb)} ({ml_ok / len(Xb) * 100:.0f}%) BP={bp_ok}/{len(Xb)} ({bp_ok / len(Xb) * 100:.0f}%)"
)

# ── 8. SANITY ──
print("\n8. SANITY CHECK:")
cases = [
    ("Udara gunung", [5, 10, 500, 10, 30], "Baik"),
    ("Kantor AC", [8, 15, 1000, 20, 40], "Baik"),
    ("Kota pagi", [12, 25, 2000, 30, 60], "Baik"),
    ("Tepi jalan", [15, 30, 3000, 40, 80], "Baik"),
    ("Jalan macet", [25, 60, 6000, 60, 100], "Sedang"),
    ("Dekat pabrik", [45, 180, 14000, 120, 250], "Tidak Sehat"),
    ("Kabut tebal", [100, 300, 20000, 200, 500], "Sangat Tidak Sehat"),
    ("Pabrik bocor", [200, 450, 35000, 300, 800], "Berbahaya"),
]
for name, vals, exp in cases:
    X_s = np.array([vals], dtype=np.float64)
    pred = le.inverse_transform(model.predict(X_s))[0]
    proba = model.predict_proba(X_s)[0]
    conf = proba.max()
    risk = float((proba * RISK_W).sum())
    delta = abs(SEVERITY[pred] - SEVERITY[exp])
    ok = (
        "OK"
        if delta == 0
        else ("minor" if delta == 1 else ("WASPADA" if delta == 2 else "KRITIS"))
    )
    print(
        f"   {name:15s} exp={exp:20s} pred={pred:20s} conf={conf * 100:.0f}% risk={risk * 100:.0f}% {ok}"
    )

# ── 9. KLHK EXTERNAL VALIDATION ──
print("\n9. KLHK EXTERNAL VALIDATION")
KLHK_PATH = "../data_train/klhk_combined.csv"


def ispi_to_conc(ispi, bp):
    if ispi <= 0:
        return 0
    for cl, ch, il, ih in bp:
        if ispi <= ih:
            return cl + (ispi - il) / (ih - il) * (ch - cl)
    return bp[-1][1]


df_k = pd.read_csv(KLHK_PATH)
df_k = df_k.dropna(subset=["pm10", "co", "o3", "no2", "categori", "pm25"]).copy()
print(f"  KLHK rows with PM2.5: {len(df_k)}")

cat_map = {
    "BAIK": "Baik",
    "SEDANG": "Sedang",
    "TIDAK SEHAT": "Tidak Sehat",
    "SANGAT TIDAK SEHAT": "Sangat Tidak Sehat",
    "BERBAHAYA": "Berbahaya",
}
y_k_actual = np.array([cat_map.get(c.strip().upper(), c) for c in df_k["categori"]])

# Reverse ISPU index → µg/m³
X_k = np.zeros((len(df_k), 5))
X_k[:, 0] = df_k["pm25"].apply(lambda v: ispi_to_conc(v, BP_PM25))
X_k[:, 1] = df_k["pm10"].apply(lambda v: ispi_to_conc(v, BP_PM10))
X_k[:, 2] = df_k["co"].apply(lambda v: ispi_to_conc(v, BP_CO))
X_k[:, 3] = df_k["no2"].apply(lambda v: ispi_to_conc(v, BP_NO2))
X_k[:, 4] = df_k["o3"].apply(lambda v: ispi_to_conc(v, BP_O3))

y_k_pred = le.inverse_transform(model.predict(X_k))
y_k_bp = np.array([ispi_to_label(ispi_from_row(X_k[i])) for i in range(len(X_k))])

ml_acc = accuracy_score(y_k_actual, y_k_pred)
bp_acc = accuracy_score(y_k_actual, y_k_bp)
same = (y_k_pred == y_k_bp).sum()
print(f"  ML  vs KLHK: {ml_acc * 100:.1f}%")
print(f"  BP  vs KLHK: {bp_acc * 100:.1f}%")
print(f"  ML == BP:    {same}/{len(X_k)} ({same / len(X_k) * 100:.1f}%)")

diff_ml = y_k_pred != y_k_actual
diff_bp = y_k_bp != y_k_actual
print(f"  ML wrong: {diff_ml.sum()}  BP wrong: {diff_bp.sum()}")
if diff_ml.sum() > 0:
    both = (diff_ml & diff_bp).sum()
    ml_only = (diff_ml & ~diff_bp).sum()
    bp_only = (~diff_ml & diff_bp).sum()
    print(f"  Both wrong: {both}  ML-only: {ml_only}  BP-only: {bp_only}")

# Breakdown by category
print("  Confusion ML vs KLHK actual:")
from sklearn.metrics import confusion_matrix

cm_k = confusion_matrix(y_k_actual, y_k_pred, labels=LABELS)
print(f"  {'':>20s}" + "".join(f"{l:>14s}" for l in LABELS))
for i, l in enumerate(LABELS):
    print(f"  {l:>20s}" + "".join(f"{cm_k[i, j]:>14d}" for j in range(len(LABELS))))

# ── 10. SAVE ──
print("\n10. SAVE")
model.n_features_ = len(FEATURES)
model.feature_names_ = FEATURES
out = {
    "model": model,
    "label_encoder": le,
    "all_bp": ALL_BP,
    "risk_weights": RISK_W,
    "features": FEATURES,
    "labels": LABELS,
}
joblib.dump(out, MODEL_OUT)

rf_acc = accuracy_score(y_te, trained["RF"].predict(X_te_n))
rf_f1 = f1_score(y_te, trained["RF"].predict(X_te_n), average="macro")
meta = {
    "model": "RF + Noise10%",
    "arsitektur": "ug/m3 langsung (tanpa feature engineering)",
    "noise_std": NOISE_STD,
    "n_estimators": 500,
    "dataset": "30d (12 May - 11 Jun 2026)",
    "test_accuracy": round(rf_acc * 100, 2),
    "test_f1_macro": round(rf_f1 * 100, 2),
    "breakpoint_accuracy": round(bp_a * 100, 2),
    "breakpoint_f1_macro": round(bp_f * 100, 2),
    "drift_test": drift_rf,
    "model_comparison_drift": drift_all,
    "klhk_ml_accuracy": round(ml_acc * 100, 2),
    "klhk_bp_accuracy": round(bp_acc * 100, 2),
    "klhk_n": len(df_k),
}
for nm in ["classification_metadata.json", "classification_final_metadata.json"]:
    with open(nm, "w") as f:
        json.dump(meta, f, indent=2)
print(f"   Model: {MODEL_OUT}")
print("\nSELESAI")
