import pandas as pd, numpy as np, warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

np.random.seed(42)
NOISE_STD = 0.10
LABELS = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]

BP_PM25 = [
    [0, 15.5, 0, 50],
    [15.5, 55.4, 50, 100],
    [55.4, 150.4, 100, 200],
    [150.4, 250.4, 200, 300],
    [250.4, 500, 300, 500],
]
BP_PM10 = [
    [0, 50, 0, 50],
    [50, 150, 50, 100],
    [150, 350, 100, 200],
    [350, 420, 200, 300],
    [420, 500, 300, 500],
]
BP_CO = [
    [0, 4000, 0, 50],
    [4000, 8000, 50, 100],
    [8000, 15000, 100, 200],
    [15000, 30000, 200, 300],
    [30000, 45000, 300, 500],
]
BP_NO2 = [
    [0, 80, 0, 50],
    [80, 200, 50, 100],
    [200, 1130, 100, 200],
    [1130, 2260, 200, 300],
    [2260, 3000, 300, 500],
]
BP_O3 = [
    [0, 120, 0, 50],
    [120, 235, 50, 100],
    [235, 400, 100, 200],
    [400, 800, 200, 300],
    [800, 1000, 300, 500],
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


df = pd.read_csv("data_train/tb_konsentrasi_gas_30d.csv")
for c in BP_NAMES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=BP_NAMES[:3])
df = df[~((df["co_ugm3"] > 30000) & (df["pm25_ugm3"] < 55))].copy()
df["Label"] = df.apply(
    lambda r: ispi_to_label(ispi_from_row([r[c] for c in BP_NAMES])), axis=1
)

le = LabelEncoder()
le.fit(LABELS)
X, y_str = df[FEATURES].values.astype(np.float64), df["Label"].values
y_enc = le.transform(y_str)
X_tr, X_te, y_tr, y_te, y_str_tr, y_str_te = train_test_split(
    X, y_enc, y_str, test_size=0.25, random_state=42, stratify=y_enc
)

saved = joblib.load("ml_model/air_quality_classifier_final.pkl")
model = saved["model"]

X_te_n = add_noise(X_te)
y_pred = model.predict(X_te_n)
y_pred_str = le.inverse_transform(y_pred)

# Count test samples per class
print()
print(
    "Tabel 4.15 Laporan Klasifikasi Random Forest (Skenario Noise 10% pada Data Uji Riil)"
)
print("=" * 95)
print(
    f"{'Kategori ISPU':<25s} {'Precision':>12s} {'Recall':>12s} {'F1-Score':>12s} {'Support':>10s}"
)
print("-" * 95)

for lbl in LABELS:
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_str_te, y_pred_str, labels=[lbl], average=None, zero_division=0
    )
    print(
        f"{lbl:<25s} {prec[0] * 100:>10.2f}% {rec[0] * 100:>10.2f}% {f1[0] * 100:>10.2f}% {int(supp[0]):>10d}"
    )

acc = accuracy_score(y_str_te, y_pred_str)
print("-" * 95)
print(f"{'Accuracy (Overall)':<25s} {acc * 100:>10.2f}%")
print(f"{'Macro Avg':<25s}", end="")
p, r, f, _ = precision_recall_fscore_support(
    y_str_te, y_pred_str, average="macro", zero_division=0
)
print(f" {p * 100:>10.2f}% {r * 100:>10.2f}% {f * 100:>10.2f}%")

print(f"{'Weighted Avg':<25s}", end="")
p2, r2, f2, _ = precision_recall_fscore_support(
    y_str_te, y_pred_str, average="weighted", zero_division=0
)
print(f" {p2 * 100:>10.2f}% {r2 * 100:>10.2f}% {f2 * 100:>10.2f}%")
print("=" * 95)

# Also print raw counts for transparency
print()
print("Distribusi Data Uji:")
for lbl in LABELS:
    n_true = int((y_str_te == lbl).sum())
    n_pred = int((y_pred_str == lbl).sum())
    print(f"  {lbl:<25s} actual={n_true:>4d}  predicted={n_pred:>4d}")
