"""
Klasifikasi Udara — Robustness Layer
========================================
ISPU breakpoint untuk kategori utama.
Random Forest untuk confidence + risk score.

Usage:
  python classify.py <pm25> <pm10> <co> <no2> <o3>
  python classify.py 15.5 30 1200 40 80
"""

import sys, json, warnings, os
import numpy as np
import joblib

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "air_quality_classifier_final.pkl")
LABELS = ["Baik", "Berbahaya", "Sangat Tidak Sehat", "Sedang", "Tidak Sehat"]
RISK_W = [0.0, 1.0, 0.8, 0.3, 0.6]
FEATURES = ["pm25_ugm3", "pm10_ugm3", "co_ugm3", "no2_ugm3", "o3_ugm3"]

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


def conc_to_ispi(val, bp):
    if val <= 0:
        return 0
    for cl, ch, il, ih in bp:
        if val <= ch:
            return il + (val - cl) / (ch - cl) * (ih - il)
    return bp[-1][3]


def ispi_to_label(ispi):
    if ispi <= 50:
        return LABELS[0]
    if ispi <= 100:
        return LABELS[1]
    if ispi <= 200:
        return LABELS[2]
    if ispi <= 300:
        return LABELS[3]
    return LABELS[4]


def classify(pm25, pm10, co, no2, o3):
    # ── ISPU breakpoint (kategori resmi) ──
    ispis = [
        conc_to_ispi(pm25, BP_PM25),
        conc_to_ispi(pm10, BP_PM10),
        conc_to_ispi(co, BP_CO),
        conc_to_ispi(no2, BP_NO2),
        conc_to_ispi(o3, BP_O3),
    ]
    max_ispi = max(ispis)
    bp_label = ispi_to_label(max_ispi)
    dominant = FEATURES[np.argmax(ispis)]

    # ── RF robustness layer (confidence + risk score) ──
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    le = saved["label_encoder"]
    X = np.array([[pm25, pm10, co, no2, o3]], dtype=np.float64)
    proba = model.predict_proba(X)[0]
    ml_label = le.inverse_transform(model.predict(X))[0]
    ml_confidence = float(round(max(proba), 4))

    # proba index matches model.classes_ which = le.classes_ indices
    # le.classes_ (sorted): ["Baik","Berbahaya","Sangat Tidak Sehat","Sedang","Tidak Sehat"]
    # Map each proba[i] to its RISK_W via LABELS
    le_classes = list(le.classes_)
    risk_score = 0.0
    probabilities = {}
    for i in range(len(proba)):
        label_name = le_classes[i]
        risk_score += proba[i] * RISK_W[LABELS.index(label_name)]
        probabilities[label_name] = round(float(proba[i]), 4)
    risk_score = round(risk_score, 4)

    return {
        "bp_category": str(bp_label),
        "bp_ispu": float(round(max_ispi, 2)),
        "bp_dominant": str(dominant),
        "ml_category": str(ml_label),
        "ml_confidence": ml_confidence,
        "risk_score": risk_score,
        "probabilities": probabilities,
        "features": {f: float(v) for f, v in zip(FEATURES, [pm25, pm10, co, no2, o3])},
    }


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(json.dumps({"error": "Usage: classify.py <pm25> <pm10> <co> <no2> <o3>"}))
        sys.exit(1)
    try:
        vals = [float(v) for v in sys.argv[1:6]]
        result = classify(*vals)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
