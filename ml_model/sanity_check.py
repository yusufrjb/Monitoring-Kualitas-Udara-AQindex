import joblib, numpy as np

model = joblib.load("air_quality_classifier_final.pkl")

# LabelEncoder order from training: 0=Baik,1=Berbahaya,2=SgtTdkSehat,3=Sedang,4=TdkSehat
CLASS_NAMES = {
    0: "Baik",
    1: "Berbahaya",
    2: "Sangat Tidak Sehat",
    3: "Sedang",
    4: "Tidak Sehat",
}

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


def get_bp_label(vals):
    ispis = [conc_to_ispi(vals[i], ALL_BP[i]) for i in range(5)]
    ispi = max(ispis)
    if ispi <= 50:
        return "Baik", ispi
    if ispi <= 100:
        return "Sedang", ispi
    if ispi <= 200:
        return "Tidak Sehat", ispi
    if ispi <= 300:
        return "Sangat Tidak Sehat", ispi
    return "Berbahaya", ispi


SCORE = {
    "Baik": 0,
    "Sedang": 1,
    "Tidak Sehat": 2,
    "Sangat Tidak Sehat": 3,
    "Berbahaya": 4,
}

test_cases = [
    ("Udara gunung (sgt bersih)", [5, 10, 500, 10, 30]),
    ("Kantor AC (bersih)", [8, 15, 1000, 20, 40]),
    ("Kota pagi (bersih)", [12, 25, 2000, 30, 60]),
    ("Tepi jalan (baik)", [15, 30, 3000, 40, 80]),
    ("Jalan macet (sedang)", [25, 60, 6000, 60, 100]),
    ("Industri (sedang)", [30, 80, 9000, 70, 150]),
    ("Kabut tipis (sedang)", [40, 100, 5000, 50, 200]),
    ("Asap rokok (tdk sehat)", [50, 200, 12000, 100, 300]),
    ("Dekat pabrik (tdk sehat)", [45, 180, 14000, 120, 250]),
    ("Kebakaran ringan (tdk sehat)", [80, 250, 15000, 150, 350]),
    ("Kabut tebal (sgt tdk sehat)", [100, 300, 20000, 200, 500]),
    ("Kebakaran besar (sgt tdk sehat)", [120, 350, 25000, 220, 600]),
    ("Pabrik bocor (berbahaya)", [200, 450, 35000, 300, 800]),
    ("Polusi ekstrim (berbahaya)", [250, 500, 40000, 350, 900]),
]

print("SANITY CHECK - Sanity check model classification")
print()
hdr = "%-30s %6s %6s %6s %6s %6s  %-20s %-20s %-8s %s" % (
    "Kondisi",
    "PM2.5",
    "PM10",
    "CO",
    "NO2",
    "O3",
    "ML Prediksi",
    "BP Label",
    "Delta",
    "Status",
)
print(hdr)
print("-" * len(hdr))

err_katastropik = 0
err_minor = 0
ok = 0

for name, vals in test_cases:
    X = np.array([vals], dtype=np.float64)
    pred_class = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    ml_label = CLASS_NAMES[pred_class]
    bp_label, bp_ispi = get_bp_label(vals)

    ml_score = SCORE[ml_label]
    bp_score = SCORE[bp_label]
    delta = abs(ml_score - bp_score)

    conf = proba.max()
    pm25, pm10, co, no2, o3 = vals

    if delta >= 3:
        status = "KRITIS"
        err_katastropik += 1
    elif delta >= 2:
        status = "WASPADA"
        err_minor += 1
    elif delta >= 1:
        status = "minor"
        err_minor += 1
    else:
        status = "OK"
        ok += 1

    print(
        "%-30s %6.0f %6.0f %6.0f %6.0f %6.0f  %-20s %-20s delta=%-2d  %s (conf=%.0f%%)"
        % (
            name,
            pm25,
            pm10,
            co,
            no2,
            o3,
            ml_label + (" (conf=%.0f%%)" % (conf * 100) if status != "OK" else ""),
            bp_label,
            delta,
            status,
            conf * 100,
        )
    )

print()
print(
    "Rekap: OK=%d  Minor=%d  WASPADA=%d  KRITIS=%d"
    % (ok, err_minor - len([s for s in ["WASPADA"] if s]), err_minor, err_katastropik)
)
print()
print("Catatan:")
print("- Delta 0 = ML sama persis dengan breakpoint")
print("- Delta 1 = beda 1 level (contoh: Baik vs Sedang) - tolerabel")
print("- Delta 2 = beda 2 level - waspada")
print("- Delta 3+ = KRITIS - Baik jadi Berbahaya, dll")
print()
if err_katastropik > 0:
    print("KESIMPULAN: BERBAHAYA - Ada catastrophic error!")
elif err_minor > 3:
    print("KESIMPULAN: Ada beberapa error minor, perlu dicek")
else:
    print("KESIMPULAN: AMAN - Model waras")
