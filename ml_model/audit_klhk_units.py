import pandas as pd, numpy as np

df = pd.read_csv("D:/DashboardAQ_v3/data_train/klhk_combined.csv")

# Sample PM10-critical row
pm10_crit = df[df["critical"].str.upper() == "PM10"].iloc[0]
print("=== PM10-critical row ===")
print("categori:", pm10_crit["categori"], " max:", pm10_crit["max"])
for pol in ["pm10", "so2", "co", "o3", "no2"]:
    val = pm10_crit[pol]
    is_crit = "<<<" if pol.upper() == "PM10" else ""
    print("  %5s = %8.2f  %s" % (pol, val, is_crit))

print()

# Sample CO-critical row
co_crit = df[df["critical"].str.upper() == "CO"].iloc[0]
print("=== CO-critical row ===")
print("categori:", co_crit["categori"], " max:", co_crit["max"])
for pol in ["pm10", "so2", "co", "o3", "no2"]:
    val = co_crit[pol]
    is_crit = "<<<" if pol.upper() == "CO" else ""
    print("  %5s = %8.2f  %s" % (pol, val, is_crit))

print()

# Sample O3-critical row
o3_crit = df[df["critical"].str.upper() == "O3"].iloc[0]
print("=== O3-critical row ===")
print("categori:", o3_crit["categori"], " max:", o3_crit["max"])
for pol in ["pm10", "so2", "co", "o3", "no2"]:
    val = o3_crit[pol]
    is_crit = "<<<" if pol.upper() == "O3" else ""
    print("  %5s = %8.2f  %s" % (pol, val, is_crit))

print()
print("=== Max values per column (ISPU range = 0-500) ===")
for pol in ["pm10", "so2", "co", "o3", "no2"]:
    mx = df[pol].max()
    print("  %5s max = %8.2f" % (pol, mx))

print()
print("=== Compare: For PM10-critical rows, what do other columns contain? ===")
pm10_crit = df[df["critical"].str.upper() == "PM10"].head(5)
print("pm10 | so2 | co  | o3  | no2 | max | categori")
for _, r in pm10_crit.iterrows():
    print(
        "%5.0f | %5.0f | %5.0f | %5.0f | %5.0f | %5.0f | %s"
        % (r["pm10"], r["so2"], r["co"], r["o3"], r["no2"], r["max"], r["categori"])
    )

print()
print("=== Check if NON-critical columns hold ISPU index or concentration ===")
# For PM10-critical rows, if O3 is also ISPU index,
# then o3 == o3's ispu (which we don't have directly)
# But we can check: does the o3 value predict the right category?
# O3 BP: <=120 ug/m3 = BAIK (ISPU 0-50), 120-235 = SEDANG (50-100)
# If o3 is concentration: o3=50 ug/m3 -> ISPU 21 -> BAIK
# If o3 is ISPU index: o3=50 -> SEDANG
# For these rows, most have categori = SEDANG or BAIK

# Check CO in PM10-critical rows
print("CO values in PM10-critical rows:")
print(df[df["critical"].str.upper() == "PM10"]["co"].describe())
print()

# Check O3 in PM10-critical rows
print("O3 values in PM10-critical rows:")
print(df[df["critical"].str.upper() == "PM10"]["o3"].describe())
