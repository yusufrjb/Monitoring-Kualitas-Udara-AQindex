"""
predict_hourly_multi.py
-----------------------
Multi-parameter forecasting: PM2.5, PM10, CO — 60 menit ke depan.
- Load model .pkl (PM2.5, PM10, CO)
- Enhanced forecast: XGBoost ML (1-5 min) + pola harian (6-60 min) + adaptive noise
- ISPU classification untuk setiap step
- Simpan ke tb_prediksi_hourly di Supabase
- Return dict forecast + classification

Jalankan via live_forecast_watcher.py setiap 5 menit.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from supabase import create_client, Client

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent / "forecast_hourly_log.txt",
            encoding="utf-8",
            mode="a",
        ),
    ],
)
log = logging.getLogger(__name__)

TABLE_DATA = "tb_konsentrasi_gas"
TABLE_HOURLY = "tb_prediksi_hourly"
FORECAST_MINUTES = 60
DATA_WINDOW_MIN = 1440

MODEL_DIR = Path(__file__).parent
MODEL_FILES = {
    # Use "best" models for PM2.5 and PM10 (root-level files), keep CO unchanged
    "pm25": MODEL_DIR.parent / "best_pm25_forecast_1hour.pkl",
    "pm10": MODEL_DIR.parent / "best_pm10_forecast_1hour.pkl",
    "co": MODEL_DIR / "xgb_co.pkl",
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

RAW_COLS_DB = [
    "pm25_ugm3",
    "pm10_corrected_ugm3",
    "co_ugm3",
    "no2_ugm3",
    "o3_ugm3",
    "temperature",
    "humidity",
]

CAT_COLORS = {
    "Baik": "#10b981",
    "Sedang": "#3b82f6",
    "Tidak Sehat": "#f59e0b",
    "Sangat Tidak Sehat": "#ef4444",
    "Berbahaya": "#7c3aed",
}


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
                    k = k.strip()
                    if not os.environ.get(k):
                        os.environ[k] = v.strip().strip('"')


def conc_to_ispi(val, bp):
    if val <= 0:
        return 0
    for cl, ch, il, ih in bp:
        if val <= ch:
            return il + (val - cl) / (ch - cl) * (ih - il)
    return bp[-1][3]


def ispi_to_label_cat(ispu):
    if ispu <= 50:
        return "Baik"
    if ispu <= 100:
        return "Sedang"
    if ispu <= 200:
        return "Tidak Sehat"
    if ispu <= 300:
        return "Sangat Tidak Sehat"
    return "Berbahaya"


def get_ispi(pm25, pm10, co, no2, o3_ugm3):
    ispis = {
        "PM2.5": conc_to_ispi(pm25, BP_PM25),
        "PM10": conc_to_ispi(pm10, BP_PM10),
        "CO": conc_to_ispi(co, BP_CO),
        "NO2": conc_to_ispi(no2, BP_NO2),
        "O3": conc_to_ispi(o3_ugm3, BP_O3),
    }
    max_ispu = max(ispis.values())
    dominant = max(ispis, key=ispis.get)
    return round(max_ispu, 1), ispi_to_label_cat(max_ispu), dominant


def fetch_recent_data(supabase: Client, minutes: int = DATA_WINDOW_MIN) -> pd.DataFrame:
    since = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    all_rows, offset = [], 0
    while True:
        resp = (
            supabase.table(TABLE_DATA)
            .select(",".join(RAW_COLS_DB) + ",created_at")
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
    for col in RAW_COLS_DB:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
    df = df.dropna(subset=["pm25_ugm3", "pm10_corrected_ugm3", "co_ugm3"])
    df = df.set_index("created_at").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.rename(
        columns={
            "pm25_ugm3": "pm25",
            "pm10_corrected_ugm3": "pm10",
            "co_ugm3": "co",
            "no2_ugm3": "no2",
            "o3_ugm3": "o3",
            "temperature": "temp",
            "humidity": "humid",
        }
    )
    log.info(f"Data diambil: {len(df)} baris")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    for col in ["pm25", "pm10", "co"]:
        for lag in [1, 3, 5, 10, 15, 30]:
            feat[f"{col}_lag_{lag}"] = feat[col].shift(lag)
        feat[f"{col}_rm_5"] = feat[col].rolling(5).mean()
        feat[f"{col}_rm_15"] = feat[col].rolling(15).mean()
        feat[f"{col}_rm_30"] = feat[col].rolling(30).mean()
    feat["temp_roll_mean_15"] = feat["temp"].rolling(15).mean()
    feat["humid_roll_mean_15"] = feat["humid"].rolling(15).mean()
    feat["hour"] = feat.index.hour
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
    feat["pm25_diff1"] = feat["pm25"].diff(1)
    feat["pm10_diff1"] = feat["pm10"].diff(1)
    feat["co_diff1"] = feat["co"].diff(1)
    feat = feat.dropna()
    return feat


def build_hourly_pattern(df: pd.DataFrame, col: str) -> dict:
    df2 = df.copy()
    df2["hour"] = df2.index.hour
    return df2.groupby("hour")[col].mean().to_dict()


def fallback_holt_winters(
    df: pd.DataFrame, col: str, max_val: float, n_steps: int = FORECAST_MINUTES
) -> list[dict]:
    series = df[col].iloc[-60:].values
    if len(series) < 2:
        series = np.full(60, series[0]) if len(series) > 0 else np.zeros(60)
    alpha, beta = 0.3, 0.1
    level, trend = series[0], series[1] - series[0] if len(series) > 1 else 0
    for v in series[1:]:
        new_level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
    fc = []
    now = datetime.now(timezone.utc)
    params = ["pm25", "pm10", "co"]
    param_names = {"pm25": "pm25", "pm10": "pm10", "co": "co"}
    for i in range(1, n_steps + 1):
        noise = (np.random.random() - 0.5) * level * 0.05
        fc.append(
            {
                "target_at": now + timedelta(minutes=i),
                param_names[col]: round(
                    max(0, min(max_val, level + i * trend + noise)), 2
                ),
            }
        )
    return fc


def forecast_one_param(
    df: pd.DataFrame,
    model_data: dict,
    raw_col: str,
    max_val: float,
    n_steps: int = FORECAST_MINUTES,
) -> list[dict]:
    series = df[raw_col].dropna()
    if len(series) < 30:
        return []

    model = model_data["model"] if isinstance(model_data, dict) else model_data
    recent = series.tail(60)
    mean_val = recent.mean()
    now = datetime.now(timezone.utc)

    feat = build_features(df)
    if feat.empty:
        return []
    feature_cols = [c for c in feat.columns if c != raw_col]

    lag_steps = [1, 3, 5, 10, 15, 30]
    raw_lag_values = {
        col: list(df[col].dropna().tail(30).values) for col in ["pm25", "pm10", "co"]
    }
    raw_series = {
        col: list(df[col].dropna().tail(60).values) for col in ["pm25", "pm10", "co"]
    }
    temp_series = (
        list(df["temp"].dropna().tail(30).values) if "temp" in df.columns else [25] * 30
    )
    humid_series = (
        list(df["humid"].dropna().tail(30).values)
        if "humid" in df.columns
        else [60] * 30
    )

    rolling_windows = {}
    for col in ["pm25", "pm10", "co"]:
        vals = raw_series[col]
        rolling_windows[col] = {
            w: vals[-w:] if len(vals) >= w else vals[:] for w in [5, 15, 30]
        }

    # Short-term trend (last 15 min)
    recent_vals = series.tail(15).values
    x = np.arange(len(recent_vals))
    short_trend, _ = np.polyfit(x, recent_vals, 1)

    # Hourly & minute pattern
    hourly_pattern = build_hourly_pattern(df, raw_col)
    overall_mean = series.mean()
    series_2 = pd.DataFrame({raw_col: series})
    series_2["minute"] = series_2.index.minute
    minute_pattern = series_2.groupby("minute")[raw_col].mean().to_dict()

    # Patterns for all params (cross-param feature update)
    all_hourly_patterns = {
        col: build_hourly_pattern(df, col) for col in ["pm25", "pm10", "co"]
    }
    all_minute_patterns = {}
    for col in ["pm25", "pm10", "co"]:
        s2 = pd.DataFrame({col: df[col]})
        s2["minute"] = s2.index.minute
        all_minute_patterns[col] = s2.groupby("minute")[col].mean().to_dict()
    all_means = {col: df[col].mean() for col in ["pm25", "pm10", "co"]}

    # Recent std for noise scaling
    recent_std = series.tail(30).std() if len(series) >= 30 else series.std()

    results = []
    prev_pred = float(raw_series[raw_col][-1]) if raw_series[raw_col] else 0

    for step in range(1, n_steps + 1):
        feature_vec = {}
        target_time = now + timedelta(minutes=step)
        # --- Lag features ---
        for col in ["pm25", "pm10", "co"]:
            vals = raw_lag_values[col]
            for lag in lag_steps:
                feature_vec[f"{col}_lag_{lag}"] = (
                    vals[-lag] if lag <= len(vals) else mean_val
                )
        # --- Rolling means ---
        for col in ["pm25", "pm10", "co"]:
            wins = rolling_windows[col]
            for w in [5, 15, 30]:
                arr = wins[w]
                feature_vec[f"{col}_rm_{w}"] = (
                    float(np.mean(arr)) if len(arr) > 0 else mean_val
                )
        # --- Diff1 ---
        for col in ["pm25", "pm10", "co"]:
            vals = raw_series[col]
            feature_vec[f"{col}_diff1"] = (
                float(vals[-1] - vals[-2]) if len(vals) >= 2 else 0
            )
        # --- Temp / humid ---
        feature_vec["temp_roll_mean_15"] = (
            float(np.mean(temp_series[-15:]))
            if len(temp_series) >= 15
            else (temp_series[-1] if temp_series else 25)
        )
        feature_vec["humid_roll_mean_15"] = (
            float(np.mean(humid_series[-15:]))
            if len(humid_series) >= 15
            else (humid_series[-1] if humid_series else 60)
        )
        # --- Time features ---
        feature_vec["hour"] = target_time.hour
        feature_vec["hour_sin"] = np.sin(2 * np.pi * target_time.hour / 24)
        feature_vec["hour_cos"] = np.cos(2 * np.pi * target_time.hour / 24)

        # Build X
        available_cols = [c for c in feature_cols if c in feature_vec]
        if not available_cols:
            available_cols = list(feature_vec.keys())
        X_pred = pd.DataFrame(
            [{k: feature_vec.get(k, mean_val) for k in available_cols}]
        )

        try:
            xgb_pred = float(model.predict(X_pred)[0])
        except Exception:
            xgb_pred = prev_pred

        # Weighted blending: XGBoost + hourly/minute pattern + persistence
        hour = target_time.hour
        minute = target_time.minute
        hour_dev = hourly_pattern.get(hour, overall_mean) - overall_mean
        minute_dev = minute_pattern.get(minute, overall_mean) - overall_mean

        horizon_weight = min(step / 30, 1.0)
        xgb_weight = 0.6 - (horizon_weight * 0.3)
        pattern_weight = 0.2 + (horizon_weight * 0.3)
        persist_weight = max(0, 1 - xgb_weight - pattern_weight)

        xgb_with_trend = xgb_pred + short_trend * step * 0.3
        pattern_adj = hour_dev * 0.7 + minute_dev * 0.3

        blended = (
            xgb_with_trend * xgb_weight
            + (overall_mean + pattern_adj) * pattern_weight
            + prev_pred * persist_weight
        )

        if step <= 30:
            noise_scale = recent_std * 0.6
        else:
            noise_scale = recent_std * 0.3 * (1 - horizon_weight * 0.2)
        pred = blended + np.random.normal(0, noise_scale)
        pred = max(0.0, min(max_val, pred))

        results.append({"target_at": target_time, raw_col: round(pred, 2)})

        # Update state with prediction
        for col in ["pm25", "pm10", "co"]:
            vals = raw_series[col]
            if col == raw_col:
                vals.append(pred)
            else:
                other_hour_dev = (
                    all_hourly_patterns[col].get(hour, all_means[col]) - all_means[col]
                )
                other_min_dev = (
                    all_minute_patterns[col].get(minute, all_means[col])
                    - all_means[col]
                )
                base_val = df[col].dropna().iloc[-1] if len(df[col].dropna()) > 0 else 0
                other_adj = other_hour_dev * 0.7 + other_min_dev * 0.3
                vals.append(max(0.0, base_val + other_adj * pattern_weight))
            if len(vals) > 60:
                vals.pop(0)
            lvals = raw_lag_values[col]
            lvals.append(vals[-1])
            if len(lvals) > 30:
                lvals.pop(0)
            for w in [5, 15, 30]:
                arr = rolling_windows[col][w]
                arr.append(vals[-1])
                if len(arr) > w:
                    arr.pop(0)

        if temp_series:
            temp_series.append(temp_series[-1])
        if len(temp_series) > 30:
            temp_series.pop(0)
        if humid_series:
            humid_series.append(humid_series[-1])
        if len(humid_series) > 30:
            humid_series.pop(0)

        prev_pred = xgb_pred

    log.info(
        f"  {raw_col.upper()}: {len(results)} prediksi, range: {min(r[raw_col] for r in results):.1f} - {max(r[raw_col] for r in results):.1f}"
    )
    return results


def classify_forecast(
    forecast_rows: list, no2_latest: float, o3_latest: float
) -> list[dict]:
    result = []
    for row in forecast_rows:
        pm25 = row.get("pm25", 0)
        pm10 = row.get("pm10", 0)
        co = row.get("co", 0)
        ispu, label, dominant = get_ispi(pm25, pm10, co, no2_latest, o3_latest)
        result.append(
            {
                "target_at": row["target_at"],
                "ispu": ispu,
                "category": label,
                "dominant": dominant,
                "color": CAT_COLORS.get(label, "#94a3b8"),
            }
        )
    return result


def save_hourly_predictions(
    supabase: Client, forecast_rows: list, forecast_at: datetime, classifications: list
):
    rows = []
    for f_row, cls in zip(forecast_rows, classifications):
        rows.append(
            {
                "forecast_at": forecast_at.isoformat(),
                "target_at": f_row["target_at"].isoformat(),
                "pm25_pred": round(f_row.get("pm25", 0), 2),
                "pm10_pred": round(f_row.get("pm10", 0), 2),
                "co_pred": round(f_row.get("co", 0), 2),
                "ispu": cls["ispu"],
                "category": cls["category"],
                "is_historical": False,
            }
        )
    if rows:
        try:
            supabase.table(TABLE_HOURLY).upsert(rows, on_conflict="target_at").execute()
            log.info(f"  {len(rows)} baris diupsert ke {TABLE_HOURLY}")
        except Exception as e:
            log.warning(f"  Gagal batch upsert: {e}")
    return rows


def get_results() -> dict:
    LOCK_FILE = MODEL_DIR / ".predict_hourly.lock"
    pid = os.getpid()

    # PID lock to prevent concurrent runs
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            if old_pid != pid:
                # Check if old process is still alive (cross-platform)
                alive = False
                try:
                    if os.name == "nt":
                        import subprocess

                        r = subprocess.run(
                            ["tasklist", "/FI", f"PID eq {old_pid}", "/NH"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        alive = str(old_pid) in r.stdout
                    else:
                        os.kill(old_pid, 0)
                        alive = True
                except Exception:
                    alive = False

                if alive:
                    log.warning(f"Instance lain berjalan (PID {old_pid}), skip")
                    return {}

                log.info(f"Lock file stale (PID {old_pid}), melanjutkan...")
        except Exception:
            pass
    LOCK_FILE.write_text(str(pid))

    try:
        return _do_get_results()
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def _do_get_results() -> dict:
    load_env()

    sb_url = os.environ.get("SUPABASE_URL", "")
    sb_key = os.environ.get("SUPABASE_ANON_KEY", "")
    if not sb_url or not sb_key:
        log.error("SUPABASE_URL/SUPABASE_KEY tidak ditemukan")
        return {}

    sb_url = sb_url.replace("http://", "https://")
    supabase: Client = create_client(sb_url, sb_key)

    df = fetch_recent_data(supabase)
    forecast_at = datetime.now(timezone.utc)

    no2_latest = (
        float(df["no2"].iloc[-1]) if "no2" in df.columns and len(df) > 0 else 50
    )
    o3_latest = float(df["o3"].iloc[-1]) if "o3" in df.columns and len(df) > 0 else 40

    all_forecasts = {}

    models_loaded = {}
    for param, pkl_path in MODEL_FILES.items():
        if pkl_path.exists():
            try:
                m = joblib.load(pkl_path)
                if isinstance(m, dict):
                    models_loaded[param] = m
                else:
                    models_loaded[param] = {"model": m, "features": []}
                log.info(f"Loaded {pkl_path.name}")
            except Exception as e:
                log.warning(f"Gagal load {pkl_path.name}: {e}")
        else:
            log.warning(f"PKL tidak ditemukan: {pkl_path.name}")

    for param, col, max_val in [
        ("pm25", "pm25", 300),
        ("pm10", "pm10", 600),
        ("co", "co", 50000),
    ]:
        model_data = models_loaded.get(param)
        if model_data:
            log.info("--- %s (XGBoost) ---" % param.upper())
            fc = forecast_one_param(df, model_data, col, max_val)
            if fc:
                all_forecasts[param] = fc
                log.info(f"  {param.upper()}: {len(fc)} prediksi")
            else:
                log.warning(f"Gagal forecast {param}, gunakan fallback")
                all_forecasts[param] = fallback_holt_winters(df, col, max_val)
        else:
            log.info("--- %s (Holt-Winters fallback) ---" % param.upper())
            all_forecasts[param] = fallback_holt_winters(df, col, max_val)
            log.info(f"  {param.upper()}: {len(all_forecasts[param])} prediksi")

    if not all_forecasts:
        log.error("Tidak ada forecast yang berhasil dibuat.")
        return {}

    n_steps = FORECAST_MINUTES
    forecast_rows = []
    for i in range(n_steps):
        target_at = forecast_at + timedelta(minutes=i + 1)
        row = {"target_at": target_at}
        for param, forecasts in all_forecasts.items():
            if i < len(forecasts):
                row[param] = forecasts[i].get(param, 0)
        forecast_rows.append(row)

    classifications = classify_forecast(forecast_rows, no2_latest, o3_latest)

    try:
        save_hourly_predictions(supabase, forecast_rows, forecast_at, classifications)
    except Exception as e:
        log.warning(f"  Gagal simpan ke Supabase: {e}")

    return {
        "method": "XGBoost Forecasting",
        "forecast_at": forecast_at.isoformat(),
        "forecast": [
            {
                "target_at": row["target_at"].isoformat(),
                "pm25": round(row.get("pm25", 0), 2),
                "pm10": round(row.get("pm10", 0), 2),
                "co": round(row.get("co", 0), 2),
            }
            for row in forecast_rows
        ],
        "classification": [
            {
                "target_at": row["target_at"].isoformat(),
                "ispu": cls["ispu"],
                "category": cls["category"],
                "dominant": cls["dominant"],
                "color": cls["color"],
            }
            for row, cls in zip(forecast_rows, classifications)
        ],
    }


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Hourly Multi-Parameter Forecast — PM2.5, PM10, CO (60 min)")
    log.info("=" * 60)
    result = get_results()
    if result:
        print(json.dumps(result, indent=2, default=str))
        log.info("Selesai!")
    else:
        log.error("Forecast gagal.")
        sys.exit(1)
