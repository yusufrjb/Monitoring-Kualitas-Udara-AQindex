"""
predict_time_series.py
======================
Prediksi PM2.5, PM10, CO menggunakan model XGBoost timeseries.
Dengan proper recursive prediction dan feature update.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from supabase import create_client, Client
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_DIR = Path(__file__).parent
PREDICTION_STEPS = 60  # 60 menit ke depan
DATA_WINDOW_MIN = 1440  # 24 jam terakhir

TABLE_DATA = "tb_konsentrasi_gas"
TABLE_HOURLY = "tb_prediksi_hourly"

# Breakpoint untuk ISPU
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


def conc_to_ispi(val, bp):
    if val <= 0:
        return 0
    for cl, ch, il, ih in bp:
        if val <= ch:
            return il + (val - cl) / (ch - cl) * (ih - il)
    return bp[-1][3]


# ============================================================
# LOAD ENV
# ============================================================
def load_env():
    env_paths = [
        MODEL_DIR / ".env",
        MODEL_DIR.parent / ".env.local",
        MODEL_DIR.parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


# ============================================================
# FETCH DATA
# ============================================================
def fetch_recent_data(supabase: Client, minutes: int = DATA_WINDOW_MIN) -> pd.DataFrame:
    since = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    all_rows, offset = [], 0

    while True:
        resp = (
            supabase.table(TABLE_DATA)
            .select(
                "pm25_ugm3,pm10_corrected_ugm3,co_ugm3,temperature,humidity,created_at"
            )
            .gte("created_at", since)
            .order("created_at", desc=False)
            .range(offset, offset + 999)
            .execute()
        )
        batch = resp.data
        if not batch:
            break
        all_rows.extend(batch)
        offset += len(batch)
        if len(batch) < 1000:
            break

    if not all_rows:
        raise ValueError(f"Tidak ada data dalam {minutes} menit terakhir.")

    df = pd.DataFrame(all_rows)
    for col in [
        "pm25_ugm3",
        "pm10_corrected_ugm3",
        "co_ugm3",
        "temperature",
        "humidity",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
    df = df.dropna(subset=["pm25_ugm3", "pm10_corrected_ugm3", "co_ugm3"])
    df = df.set_index("created_at").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.rename(
        columns={"pm25_ugm3": "pm25", "pm10_corrected_ugm3": "pm10", "co_ugm3": "co"}
    )

    print(f"Data diambil: {len(df)} baris")
    return df


# ============================================================
# BUILD FEATURES - Enhanced with ACF-based lags
# ============================================================
# Default ACF lags - will be updated from model
DEFAULT_ACF_LAGS = [1, 3, 5, 10, 15, 30]


def build_features(
    series: pd.Series, df: pd.DataFrame = None, acf_lags: list = None
) -> pd.DataFrame:
    col = series.name
    feat = pd.DataFrame(index=series.index)
    feat[col] = series

    # Use ACF-based lags or default
    lags_to_use = acf_lags if acf_lags else DEFAULT_ACF_LAGS

    # Lag features based on ACF analysis
    for lag in lags_to_use:
        feat[f"lag_{lag}"] = feat[col].shift(lag)

    # Rolling features with multiple windows
    for window in [3, 5, 10, 15, 30]:
        feat[f"rolling_mean_{window}"] = feat[col].rolling(window).mean()
        feat[f"rolling_std_{window}"] = feat[col].rolling(window).std()
        if window <= 15:
            feat[f"rolling_min_{window}"] = feat[col].rolling(window).min()
            feat[f"rolling_max_{window}"] = feat[col].rolling(window).max()

    # Time features
    feat["hour"] = feat.index.hour
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
    feat["day_of_week"] = feat.index.dayofweek
    feat["is_weekend"] = (feat.index.dayofweek >= 5).astype(int)

    # Diff features
    feat["diff_1"] = feat[col].diff(1)
    feat["diff_5"] = feat[col].diff(5)
    feat["diff_15"] = feat[col].diff(15)

    # External features (temperature, humidity) - current values
    if df is not None and "temperature" in df.columns:
        feat["temperature"] = df["temperature"]
    if df is not None and "humidity" in df.columns:
        feat["humidity"] = df["humidity"]

    feat = feat.dropna()
    return feat


# ============================================================
# HYBRID PREDICTION
# ============================================================
def hybrid_predict(
    df: pd.DataFrame,
    model_data: dict,
    param: str,
    steps: int = PREDICTION_STEPS,
    base_time: datetime = None,
):
    series = df[param].dropna()
    if len(series) < 30:
        print(f"Data terlalu sedikit untuk prediksi {param}")
        return []

    # Get ACF lags from model or use default
    acf_lags = model_data.get("acf_lags", DEFAULT_ACF_LAGS)

    # Build features with ACF-based lags
    feat = build_features(series, df, acf_lags)
    if feat.empty:
        return []

    col = param
    feature_cols = model_data.get("features", [])
    model = model_data["model"]

    # Validasi fitur
    available_cols = [c for c in feature_cols if c in feat.columns]
    if not available_cols:
        available_cols = [c for c in feat.columns if c != col]

    if not available_cols:
        print("Tidak ada fitur valid")
        return []

    # Calculate statistics
    recent = series.tail(60)
    mean_val = recent.mean()
    std_val = recent.std()

    # Get temperature and humidity if available
    temp_val = df["temperature"].tail(10).mean() if "temperature" in df.columns else 25
    humidity_val = df["humidity"].tail(10).mean() if "humidity" in df.columns else 60

    # Enhanced trend detection
    if len(recent) >= 10:
        short_trend = (recent.iloc[-1] - recent.iloc[-10]) / 10
    else:
        short_trend = 0

    # Build hourly pattern with variance
    series_with_hour = series.copy()
    series_with_hour = pd.DataFrame({col: series_with_hour})
    series_with_hour["hour"] = series_with_hour.index.hour
    hourly_pattern = series_with_hour.groupby("hour")[col].mean().to_dict()
    hourly_std = series_with_hour.groupby("hour")[col].std().fillna(std_val).to_dict()

    # Build minute-of-hour pattern
    series_with_minute = series.copy()
    series_with_minute = pd.DataFrame({col: series_with_minute})
    series_with_minute["minute"] = series_with_minute.index.minute
    minute_pattern = series_with_minute.groupby("minute")[col].mean().to_dict()

    predictions = []
    now = base_time or datetime.now(timezone.utc)

    # Initialize lag values based on ACF lags
    max_lag = max(acf_lags) if acf_lags else 30
    lag_values = list(series.tail(max_lag).values)

    # Initialize rolling values
    rolling_values = {w: list(series.tail(w).values) for w in [3, 5, 10, 15, 30]}
    diff_values = {k: list(series.diff(k).tail(5).values) for k in [1, 5, 15]}

    for step in range(1, steps + 1):
        feature_vec = {}

        # Lag features based on ACF lags
        for lag in acf_lags:
            if lag <= len(lag_values):
                feature_vec[f"lag_{lag}"] = lag_values[-lag]
            else:
                feature_vec[f"lag_{lag}"] = mean_val

        # Rolling features
        for window in [3, 5, 10, 15, 30]:
            vals = rolling_values.get(window, [])
            feature_vec[f"rolling_mean_{window}"] = (
                np.mean(vals) if len(vals) > 0 else mean_val
            )
            feature_vec[f"rolling_std_{window}"] = (
                np.std(vals) if len(vals) > 0 else std_val
            )
            if window <= 15:
                feature_vec[f"rolling_min_{window}"] = (
                    np.min(vals) if len(vals) > 0 else mean_val
                )
                feature_vec[f"rolling_max_{window}"] = (
                    np.max(vals) if len(vals) > 0 else mean_val
                )

        # Time features
        target_time = now + timedelta(minutes=step)
        target_hour = target_time.hour
        target_minute = target_time.minute
        feature_vec["hour"] = target_hour
        feature_vec["hour_sin"] = np.sin(2 * np.pi * target_hour / 24)
        feature_vec["hour_cos"] = np.cos(2 * np.pi * target_hour / 24)
        feature_vec["day_of_week"] = target_time.weekday()
        feature_vec["is_weekend"] = 1 if target_time.weekday() >= 5 else 0

        # Diff features
        feature_vec["diff_1"] = diff_values[1][-1] if len(diff_values[1]) > 0 else 0
        feature_vec["diff_5"] = diff_values[5][-1] if len(diff_values[5]) > 0 else 0
        feature_vec["diff_15"] = diff_values[15][-1] if len(diff_values[15]) > 0 else 0

        # External features
        feature_vec["temperature"] = temp_val
        feature_vec["humidity"] = humidity_val

        # XGBoost prediction
        try:
            X_pred = pd.DataFrame([feature_vec])[available_cols]
            xgb_pred = float(model.predict(X_pred)[0])
        except:
            xgb_pred = mean_val

        # Enhanced pattern injection
        hourly_avg = hourly_pattern.get(target_hour, mean_val)
        hourly_s = hourly_std.get(target_hour, std_val)
        minute_avg = minute_pattern.get(target_minute, mean_val)

        # Use model output directly (no blending). If model prediction fails earlier,
        # xgb_pred already set to mean_val fallback.
        pred = xgb_pred
        # Ensure non-negative
        pred = max(0, pred)

        # Update rolling values
        lag_values.append(pred)
        if len(lag_values) > max_lag:
            lag_values = lag_values[-max_lag:]

        for window in [3, 5, 10, 15, 30]:
            if window in rolling_values:
                rolling_values[window].append(pred)
                if len(rolling_values[window]) > window:
                    rolling_values[window] = rolling_values[window][-window:]

        diff_values[1] = (
            np.append(diff_values[1][1:], pred - lag_values[-2])
            if len(lag_values) > 1
            else np.array([0])
        )
        diff_values[5] = (
            np.append(diff_values[5][1:], pred - lag_values[-6])
            if len(lag_values) >= 6
            else diff_values[5]
        )
        diff_values[15] = (
            np.append(diff_values[15][1:], pred - lag_values[-16])
            if len(lag_values) >= 16
            else diff_values[15]
        )

        predictions.append({"target_at": target_time, param: round(pred, 2)})

    print(
        f"  {param.upper()}: {len(predictions)} prediksi, range: {min(p[param] for p in predictions):.1f} - {max(p[param] for p in predictions):.1f}"
    )
    return predictions


# ============================================================
# ISPU CLASSIFICATION
# ============================================================
def classify_predictions(forecast_rows: list):
    result = []
    for row in forecast_rows:
        pm25 = row.get("pm25_pred", 0)
        pm10 = row.get("pm10_pred", 0)
        co = row.get("co_pred", 0)

        ispu_pm25 = conc_to_ispi(pm25, BP_PM25)
        ispu_pm10 = conc_to_ispi(pm10, BP_PM10)
        ispu_co = conc_to_ispi(co, BP_CO)

        max_ispu = max(ispu_pm25, ispu_pm10, ispu_co)

        if max_ispu <= 50:
            category = "Baik"
            color = "#10b981"
        elif max_ispu <= 100:
            category = "Sedang"
            color = "#3b82f6"
        elif max_ispu <= 200:
            category = "Tidak Sehat"
            color = "#f59e0b"
        elif max_ispu <= 300:
            category = "Sangat Tidak Sehat"
            color = "#ef4444"
        else:
            category = "Berbahaya"
            color = "#7c3aed"

        result.append(
            {
                "ispu": round(max_ispu, 1),
                "category": category,
                "color": color,
            }
        )
    return result


# ============================================================
# MAIN
# ============================================================
def get_results():
    load_env()

    sb_url = os.environ.get("SUPABASE_URL", "")
    sb_key = os.environ.get("SUPABASE_ANON_KEY", "")

    if not sb_url or not sb_key:
        print("ERROR: SUPABASE_URL atau SUPABASE_KEY tidak ditemukan")
        return {}

    sb_url = sb_url.replace("http://", "https://")
    supabase: Client = create_client(sb_url, sb_key)

    # Simpan base time SEBELUM fetch/prediksi apapun
    prediction_base = datetime.now(timezone.utc)

    # Fetch data
    df = fetch_recent_data(supabase)

    # Load models
    models = {}
    for param in ["pm25", "pm10", "co"]:
        model_path = MODEL_DIR / f"xgb_{param}_timeseries.pkl"
        if model_path.exists():
            models[param] = joblib.load(model_path)
            print(f"Model loaded: {model_path.name}")
        else:
            print(f"Model tidak ditemukan: {model_path.name}")

    # Generate predictions dengan pattern blending
    all_forecasts = {}
    for param in ["pm25", "pm10", "co"]:
        if param in models:
            # Get ACF lags from model
            acf_lags = models[param].get("acf_lags", DEFAULT_ACF_LAGS)

            # Build features with ACF lags
            feat = build_features(df[param].dropna(), df, acf_lags)
            models[param]["features"] = [c for c in feat.columns if c != param]

            preds = hybrid_predict(
                df, models[param], param, PREDICTION_STEPS, prediction_base
            )
            if preds:
                all_forecasts[param] = preds

    if not all_forecasts:
        print("ERROR: Tidak ada prediksi yang berhasil")
        return {}

    # Combine forecasts
    forecast_rows = []
    now = prediction_base
    for i in range(PREDICTION_STEPS):
        target_at = now + timedelta(minutes=i + 1)
        row = {"target_at": target_at, "forecast_at": now}
        for param, preds in all_forecasts.items():
            if i < len(preds):
                row[f"{param}_pred"] = preds[i].get(param, 0)
        forecast_rows.append(row)

    # ISPU Classification
    classifications = classify_predictions(forecast_rows)

    # Simpan ke Supabase (tanpa ispu - nullable)
    try:
        rows_to_insert = []
        for row, cls in zip(forecast_rows, classifications):
            rows_to_insert.append(
                {
                    "forecast_at": row["forecast_at"].isoformat(),
                    "target_at": row["target_at"].isoformat(),
                    "pm25_pred": round(row.get("pm25_pred", 0), 2),
                    "pm10_pred": round(row.get("pm10_pred", 0), 2),
                    "co_pred": round(row.get("co_pred", 0), 2),
                    "ispu": cls["ispu"],
                    "category": cls["category"],
                    "is_historical": False,
                }
            )

        if rows_to_insert:
            supabase.table(TABLE_HOURLY).insert(rows_to_insert).execute()
            print(f"Disimpan ke {TABLE_HOURLY}: {len(rows_to_insert)} baris")
    except Exception as e:
        print(f"Gagal simpan ke Supabase: {e}")

    # Return JSON
    return {
        "forecast_at": now.isoformat(),
        "method": "XGBoost dengan Pola Harian",
        "parameters": list(all_forecasts.keys()),
        "forecast": [
            {
                "target_at": row["target_at"].isoformat(),
                "pm25": float(row.get("pm25_pred", 0)),
                "pm10": float(row.get("pm10_pred", 0)),
                "co": float(row.get("co_pred", 0)),
            }
            for row in forecast_rows
        ],
    }


if __name__ == "__main__":
    import sys

    # Check for --json flag - suppress all debug output
    json_only = len(sys.argv) > 1 and sys.argv[1] == "--json"

    if json_only:
        # Suppress stdout temporarily for debug output
        import io

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    print("=" * 60)
    print("XGBoost dengan Pola Harian")
    print("=" * 60)

    result = get_results()

    if json_only:
        # Restore stdout and print only JSON
        sys.stdout = old_stdout
        print(json.dumps(result))
    else:
        if result:
            print("\n" + "=" * 60)
            print("HASIL PREDIKSI:")
            print("=" * 60)
            print(f"Method: {result.get('method')}")
            print(f"Parameters: {result.get('parameters')}")
            print(f"Total predictions: {len(result.get('forecast', []))}")
            print("\n---JSON_OUTPUT_START---")
            print(json.dumps(result))
            print("---JSON_OUTPUT_END---")
        else:
            print("Prediksi gagal!")
            sys.exit(1)
