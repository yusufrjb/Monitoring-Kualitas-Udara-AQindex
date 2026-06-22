"""
Verifikasi 4.52% mismatch: Official max-rule vs Actual KLHK label
===============================================================
1. Distribusi mismatch
2. Apakah error boundary (inklusif/eksklusif)?
3. Apakah ML benar di mismatch cases?
4. Apakah mismatch adalah pola nyata atau artefak?
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# ── Load KLHK ──
df = pd.read_csv("D:/DashboardAQ_v3/data_train/klhk_combined.csv")
df = df.dropna(subset=["pm10", "so2", "co", "o3", "no2"]).copy()
df = df[df["categori"] != "BERBAHAYA"].copy()

LABELS = ["BAIK", "SEDANG", "TIDAK SEHAT", "SANGAT TIDAK SEHAT"]
FEATURES = ["pm10", "so2", "co", "o3", "no2"]

X = df[FEATURES].values.astype(np.float64)
y_actual = df["categori"].values
critical = df["critical"].values
max_vals = df["max"].values


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

# ── 1. Confusion matrix: Official vs Actual ──
print("=" * 70)
print("CONFUSION: Official max-rule (rows) vs Actual KLHK (cols)")
print("=" * 70)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_actual, y_official, labels=LABELS)
print("%15s" % "", end="")
for l in LABELS:
    print("%18s" % l, end="")
print()
for i, l in enumerate(LABELS):
    print("%15s" % l, end="")
    for j in range(len(LABELS)):
        print("%18d" % cm[i, j], end="")
    print()

print()
total = len(y_actual)
same = (y_official == y_actual).sum()
print("Match: %d/%d (%.2f%%)" % (same, total, same / total * 100))
print("Mismatch: %d/%d (%.2f%%)" % (total - same, total, (total - same) / total * 100))

# ── 2. Breakdown mismatch by type ──
print()
print("=" * 70)
print("DISTRIBUSI MISMATCH (Official -> Actual)")
print("=" * 70)
mismatch_mask = y_official != y_actual
mismatch_df = pd.DataFrame(
    {
        "official": y_official[mismatch_mask],
        "actual": y_actual[mismatch_mask],
        "max_ispu": max_ispu[mismatch_mask],
        "critical": critical[mismatch_mask],
        "pm10": X[mismatch_mask, 0],
        "so2": X[mismatch_mask, 1],
        "co": X[mismatch_mask, 2],
        "o3": X[mismatch_mask, 3],
        "no2": X[mismatch_mask, 4],
    }
)

print()
print("a. Jumlah per tipe mismatch:")
mm_types = (
    mismatch_df.groupby(["official", "actual"]).size().sort_values(ascending=False)
)
for (off, act), cnt in mm_types.items():
    print("  %20s -> %-20s : %d kasus" % (off, act, cnt))

print()
print(
    "b. Total mismatch: %d kasus dari %d (%.2f%%)"
    % (len(mismatch_df), total, len(mismatch_df) / total * 100)
)

# ── 3. Boundary inclusivity check ──
print()
print("=" * 70)
print("BOUNDARY INKLUSIVITAS: Cek kasus max=50")
print("=" * 70)
# ISPU: 0-50 = Baik, 51-100 = Sedang
# max=50 should be BAIK by official rule
max_50 = df[max_ispu == 50]
print("Kasus dengan max ISPU = 50: %d baris" % len(max_50))
if len(max_50) > 0:
    off_50 = np.array([ispu_to_cat(50)] * len(max_50))
    act_50 = max_50["categori"].values
    same_50 = (off_50 == act_50).sum()
    print(
        "  Official = Baik, Actual = Baik: %d (%.1f%%)"
        % (same_50, same_50 / len(max_50) * 100)
    )
    print("  Official = Baik, Actual = Sedang: %d" % (len(max_50) - same_50))
    if (len(max_50) - same_50) > 0:
        print()
        print("  Sampel dimana max=50 tapi actual=Semang:")
        for _, r in max_50[max_50["categori"] == "SEDANG"].head(10).iterrows():
            print(
                "    PM10=%.0f SO2=%.0f CO=%.0f O3=%.0f NO2=%.0f crit=%s"
                % (r["pm10"], r["so2"], r["co"], r["o3"], r["no2"], r["critical"])
            )

# ── 4. Train ML and check mismatch prediction ──
print()
print("=" * 70)
print("ML PERFORMANCE PADA MISMATCH CASES")
print("=" * 70)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(LABELS)
y_enc = le.transform(y_actual)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)

# Train RF
rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_tr, y_tr)

# Predict ALL data
y_pred_all = rf.predict(X)

print(
    "a. Overall ML accuracy on ALL data: %.2f%%"
    % ((y_pred_all == y_enc).sum() / len(y_enc) * 100)
)

# On mismatch cases only
y_pred_mismatch = y_pred_all[mismatch_mask]
y_actual_mismatch_enc = le.transform(y_actual[mismatch_mask])
ml_correct_mismatch = (y_pred_mismatch == y_actual_mismatch_enc).sum()
print(
    "b. ML correct on MISMATCH cases: %d/%d (%.2f%%)"
    % (
        ml_correct_mismatch,
        len(y_pred_mismatch),
        ml_correct_mismatch / len(y_pred_mismatch) * 100,
    )
)

# On match cases
match_mask = ~mismatch_mask
y_pred_match = y_pred_all[match_mask]
y_actual_match_enc = le.transform(y_actual[match_mask])
ml_correct_match = (y_pred_match == y_actual_match_enc).sum()
print(
    "c. ML correct on MATCH cases: %d/%d (%.2f%%)"
    % (ml_correct_match, len(y_pred_match), ml_correct_match / len(y_pred_match) * 100)
)

print()
print("d. Detail mismatch types (ML vs Official vs Actual):")
for (off, act), cnt in mm_types.items():
    # Find these indices in mismatch_df
    type_mask = (mismatch_df["official"].values == off) & (
        mismatch_df["actual"].values == act
    )
    actual_type_indices = np.where(mismatch_mask)[0][type_mask]

    ml_pred_on_type = le.inverse_transform(y_pred_all[actual_type_indices])
    ml_correct_type = (ml_pred_on_type == act).sum()

    # What does official say? (we know it's 'off')
    # What does ML say?
    print(
        "  %20s -> %-20s : %d cases | ML=%s correct: %d/%d (%.0f%%)"
        % (
            off,
            act,
            cnt,
            "YES" if ml_correct_type == cnt else "PARTIAL",
            ml_correct_type,
            cnt,
            ml_correct_type / cnt * 100,
        )
    )

# ── 5. ML vs Official on official rule usage ──
print()
print("=" * 70)
print("APAKAH ML MENGGUNAKAN max() RULE?")
print("=" * 70)
# Check: when ML != Official, is ML correct (matching actual)?
ml_diff_mask = y_pred_all != le.transform(y_official)
ml_diff_total = ml_diff_mask.sum()
ml_diff_correct = (y_pred_all[ml_diff_mask] == y_enc[ml_diff_mask]).sum()
print("ML != Official rule: %d cases" % ml_diff_total)
print(
    "  ML correct (matches actual): %d (%.1f%%)"
    % (ml_diff_correct, ml_diff_correct / ml_diff_total * 100)
)
print("  ML wrong (does not match actual): %d" % (ml_diff_total - ml_diff_correct))

# ── 6. Sample analysis of mismatches ──
print()
print("=" * 70)
print("SAMPLE MISMATCH CASES (max 30)")
print("=" * 70)
print(
    "%5s %5s %5s %5s %5s | %5s | %-12s | %-12s | %-12s | %-12s"
    % (
        "PM10",
        "SO2",
        "CO",
        "O3",
        "NO2",
        "MAX",
        "Official",
        "Actual",
        "ML_Pred",
        "Critical",
    )
)
print("-" * 90)
samples = 0
for idx in np.where(mismatch_mask)[0]:
    if samples >= 30:
        break
    r = df.iloc[idx]
    print(
        "%5.0f %5.0f %5.0f %5.0f %5.0f | %5.0f | %-12s | %-12s | %-12s | %-12s"
        % (
            r["pm10"],
            r["so2"],
            r["co"],
            r["o3"],
            r["no2"],
            max_ispu[idx],
            y_official[idx],
            y_actual[idx],
            le.inverse_transform([y_pred_all[idx]])[0],
            r["critical"],
        )
    )
    samples += 1
