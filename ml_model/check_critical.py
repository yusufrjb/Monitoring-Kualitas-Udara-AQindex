import pandas as pd, numpy as np

df = pd.read_csv("D:/DashboardAQ_v3/data_train/klhk_combined.csv")
df = df.dropna(subset=["pm10", "so2", "co", "o3", "no2"]).copy()
df = df[df["categori"] != "BERBAHAYA"].copy()

FEATURES = ["pm10", "so2", "co", "o3", "no2"]
X = df[FEATURES].values.astype(np.float64)


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
y_actual = df["categori"].values

mismatch_mask = y_official != y_actual

print("=== Critical pollutant di mismatch cases ===")
print()
print("Distribusi critical di %d mismatch:" % mismatch_mask.sum())
critical_counts = df[mismatch_mask]["critical"].value_counts()
for val, cnt in critical_counts.items():
    print("  %s: %d (%.1f%%)" % (val, cnt, cnt / mismatch_mask.sum() * 100))

pm25_refs = (
    df[mismatch_mask]["critical"].str.contains("PM2", na=False, case=False).sum()
)
print()
print(
    "PM2.5-related critical: %d/%d (%.1f%%)"
    % (pm25_refs, mismatch_mask.sum(), pm25_refs / mismatch_mask.sum() * 100)
)

non_pm25 = df[mismatch_mask][
    ~df[mismatch_mask]["critical"].str.contains("PM2", na=False, case=False)
]
print()
print("Non-PM2.5 mismatch (%d cases):" % len(non_pm25))
for _, r in non_pm25.iterrows():
    mx = max([r[c] for c in FEATURES])
    off = ispu_to_cat(mx)
    print(
        "  critical=%-6s pm10=%.0f so2=%.0f co=%.0f o3=%.0f no2=%.0f max=%.0f official=%-12s actual=%s"
        % (
            r["critical"],
            r["pm10"],
            r["so2"],
            r["co"],
            r["o3"],
            r["no2"],
            mx,
            off,
            r["categori"],
        )
    )
