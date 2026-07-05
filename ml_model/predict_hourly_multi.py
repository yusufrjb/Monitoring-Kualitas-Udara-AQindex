"""
predict_hourly_multi.py
-----------------------
Multi-parameter forecasting: PM2.5, PM10, CO — 60 menit ke depan.
- Load 3 XGBoost v2 models (PM2.5, PM10, CO)
- Multivariate step-by-step recursive forecast: tiap step predict PM25 → PM10 → CO
  dengan shared state agar cross-param dynamics terjaga
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
TABLE_FITTED = "tb_fitted_values"
FORECAST_MINUTES = 60
DATA_WINDOW_MIN = 4320

MODEL_DIR = Path(__file__).parent
MODEL_FILES = {
    "pm25": MODEL_DIR / "xgb_pm25_v2.pkl",
    "pm10": MODEL_DIR / "xgb_pm10_v2.pkl",
    "co": MODEL_DIR / "xgb_co_v2.pkl",
}
RF_MODEL_PATH = MODEL_DIR / "random_forest_air_quality.pkl"

RAW_COLS_DB = [
    "pm25_ugm3",
    "pm10_ugm3",
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


ISPU_MIDPOINT = {
    "Baik": 25,
    "Sedang": 75,
    "Tidak Sehat": 150,
    "Sangat Tidak Sehat": 250,
    "Berbahaya": 400,
}
RF_FEATURES = ["PM2.5", "PM10", "CO", "NO2", "O3"]


def classify_rf(rf_model, pm25, pm10, co, no2, o3_ugm3):
    X = pd.DataFrame(
        [
            {
                "PM2.5": pm25,
                "PM10": pm10,
                "CO": co,
                "NO2": no2,
                "O3": o3_ugm3,
            }
        ],
        columns=RF_FEATURES,
    )
    proba = rf_model.predict_proba(X)[0]
    label = rf_model.predict(X)[0]
    ml_confidence = float(round(max(proba), 4))
    ispu = ISPU_MIDPOINT.get(label, 0)
    return ispu, label, ml_confidence, proba


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
    df = df.dropna(subset=["pm25_ugm3", "pm10_ugm3", "co_ugm3"])
    df = df.set_index("created_at").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.rename(
        columns={
            "pm25_ugm3": "pm25",
            "pm10_ugm3": "pm10",
            "co_ugm3": "co",
            "no2_ugm3": "no2",
            "o3_ugm3": "o3",
        }
    )
    log.info(f"Data diambil: {len(df)} baris")
    return df


RAW_COLS_FEAT = ["pm25", "pm10", "co", "no2", "temperature", "humidity"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    for col in RAW_COLS_FEAT:
        if col not in feat.columns:
            continue
        feat[f"{col}_lag_1min"] = feat[col].shift(1)
        feat[f"{col}_lag_5min"] = feat[col].shift(5)
        feat[f"{col}_lag_15min"] = feat[col].shift(15)
        feat[f"{col}_lag_60min"] = feat[col].shift(60)
        feat[f"{col}_rolling_mean_5min"] = feat[col].rolling(window=5).mean()
        feat[f"{col}_rolling_std_5min"] = feat[col].rolling(window=5).std()
        feat[f"{col}_rolling_mean_15min"] = feat[col].rolling(window=15).mean()
    feat["minute"] = feat.index.minute
    feat["hour"] = feat.index.hour
    feat["dayofweek"] = feat.index.dayofweek
    feat = feat.dropna()
    return feat


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
    z_score = 1.96
    residual_std = float(np.std(series)) if len(series) > 1 else level * 0.1
    fc = []
    now = datetime.now(timezone.utc)
    params = ["pm25", "pm10", "co"]
    param_names = {"pm25": "pm25", "pm10": "pm10", "co": "co"}
    for i in range(1, n_steps + 1):
        noise = (np.random.random() - 0.5) * level * 0.05
        val = max(0, min(max_val, level + i * trend + noise))
        noise_scale = residual_std * (0.6 if i <= 30 else 0.3)
        upper = min(max_val, val + z_score * noise_scale)
        lower = max(0.0, val - z_score * noise_scale)
        fc.append(
            {
                "target_at": now + timedelta(minutes=i),
                param_names[col]: round(val, 2),
                f"{param_names[col]}_upper": round(upper, 2),
                f"{param_names[col]}_lower": round(lower, 2),
            }
        )
    return fc


FORECAST_PARAMS = ["pm25", "pm10", "co"]


def forecast_all_params(
    df: pd.DataFrame,
    models: dict,
    max_vals: dict,
    n_steps: int = FORECAST_MINUTES,
) -> dict:
    for p in FORECAST_PARAMS:
        if df[p].dropna().empty or len(df[p].dropna()) < 60:
            return {}

    now = datetime.now(timezone.utc)

    # Unwrap dict-wrapped models
    models_clean = {
        k: (m.get("model", m) if isinstance(m, dict) else m) for k, m in models.items()
    }

    feat = build_features(df)
    if feat.empty:
        return {}
    model_features = [
        c for c in feat.columns if c not in ["pm25", "pm10", "co", "no2", "o3"]
    ]

    # Shared rolling state
    raw_series = {col: list(df[col].dropna().tail(60).values) for col in RAW_COLS_FEAT}

    # --- Blending prep per param ---
    hourly_patterns = {}
    minute_patterns = {}
    overall_means = {}
    short_trends = {}
    recent_stds = {}

    for p in FORECAST_PARAMS:
        series = df[p].dropna()
        overall_means[p] = float(series.mean())

        sh = pd.DataFrame({p: series})
        sh["hour"] = sh.index.hour
        hourly_patterns[p] = sh.groupby("hour")[p].mean().to_dict()

        sm = pd.DataFrame({p: series})
        sm["minute"] = sm.index.minute
        minute_patterns[p] = sm.groupby("minute")[p].mean().to_dict()

        recent15 = series.tail(15).values
        if len(recent15) >= 2:
            short_trends[p], _ = np.polyfit(np.arange(len(recent15)), recent15, 1)
        else:
            short_trends[p] = 0.0

        recent_stds[p] = (
            float(series.tail(30).std()) if len(series) >= 30 else float(series.std())
        )

    results = {p: [] for p in FORECAST_PARAMS}
    prev_preds = {
        p: float(raw_series[p][-1]) if raw_series[p] else 0 for p in FORECAST_PARAMS
    }

    def _build_fv(ts):
        fv = {}
        for col in RAW_COLS_FEAT:
            vals = raw_series.get(col, [])
            fv[f"{col}_lag_1min"] = vals[-1] if len(vals) >= 1 else 0
            fv[f"{col}_lag_5min"] = (
                vals[-5] if len(vals) >= 5 else (vals[-1] if vals else 0)
            )
            fv[f"{col}_lag_15min"] = (
                vals[-15] if len(vals) >= 15 else (vals[-1] if vals else 0)
            )
            fv[f"{col}_lag_60min"] = (
                vals[-60] if len(vals) >= 60 else (vals[-1] if vals else 0)
            )
            win5 = vals[-5:] if len(vals) >= 5 else vals
            win15 = vals[-15:] if len(vals) >= 15 else vals
            fv[f"{col}_rolling_mean_5min"] = float(np.mean(win5)) if win5 else 0
            fv[f"{col}_rolling_std_5min"] = float(np.std(win5)) if len(win5) > 1 else 0
            fv[f"{col}_rolling_mean_15min"] = float(np.mean(win15)) if win15 else 0
        fv["temperature"] = (
            raw_series["temperature"][-1] if raw_series["temperature"] else 0
        )
        fv["humidity"] = raw_series["humidity"][-1] if raw_series["humidity"] else 0
        fv["minute"] = ts.minute
        fv["hour"] = ts.hour
        fv["dayofweek"] = ts.weekday()
        return fv

    def _blend(p, xgb_raw, step, hour, minute):
        hw = min(step / 30, 1.0)
        xgb_w = 0.6 - hw * 0.3
        pat_w = 0.2 + hw * 0.3
        per_w = max(0.0, 1.0 - xgb_w - pat_w)

        hour_dev = hourly_patterns[p].get(hour, overall_means[p]) - overall_means[p]
        minute_dev = minute_patterns[p].get(minute, overall_means[p]) - overall_means[p]

        trend_adj = short_trends[p] * step * 0.3
        pattern_adj = hour_dev * 0.7 + minute_dev * 0.3

        blended = (
            (xgb_raw + trend_adj) * xgb_w
            + (overall_means[p] + pattern_adj) * pat_w
            + prev_preds[p] * per_w
        )

        noise_scale = recent_stds[p] * (0.6 if step <= 30 else 0.3 * (1 - hw * 0.2))
        return blended + np.random.normal(0, noise_scale), noise_scale

    Z_SCORE = 1.96

    for step in range(1, n_steps + 1):
        target_time = now + timedelta(minutes=step)
        hour = target_time.hour
        minute = target_time.minute

        for p in FORECAST_PARAMS:
            fv = _build_fv(target_time)
            X_pred = pd.DataFrame([{k: fv.get(k, 0) for k in model_features}])
            try:
                xgb_pred = float(models_clean[p].predict(X_pred)[0])
            except Exception:
                xgb_pred = prev_preds[p]
            xgb_pred = max(0.0, min(max_vals[p], xgb_pred))

            final_pred, noise_scale = _blend(p, xgb_pred, step, hour, minute)
            final_pred = max(0.0, min(max_vals[p], final_pred))

            upper = min(max_vals[p], final_pred + Z_SCORE * noise_scale)
            lower = max(0.0, final_pred - Z_SCORE * noise_scale)

            results[p].append(
                {
                    "target_at": target_time,
                    p: round(final_pred, 2),
                    f"{p}_upper": round(upper, 2),
                    f"{p}_lower": round(lower, 2),
                }
            )
            raw_series[p].append(final_pred)
            prev_preds[p] = final_pred
            if len(raw_series[p]) > 60:
                raw_series[p].pop(0)

        # Freeze non-target cols
        for col in ["no2", "temperature", "humidity"]:
            raw_series[col].append(raw_series[col][-1] if raw_series[col] else 0)
            if len(raw_series[col]) > 60:
                raw_series[col].pop(0)

    for p in FORECAST_PARAMS:
        vals = [r[p] for r in results[p]]
        log.info(
            f"  {p.upper()}: {len(results[p])} prediksi, range: {min(vals):.1f} - {max(vals):.1f}"
        )
    return results


def classify_forecast(
    forecast_rows: list, no2_latest: float, o3_latest: float, rf_model
) -> list[dict]:
    result = []
    for row in forecast_rows:
        pm25 = row.get("pm25", 0)
        pm10 = row.get("pm10", 0)
        co = row.get("co", 0)
        ispu, label, ml_confidence, proba = classify_rf(
            rf_model, pm25, pm10, co, no2_latest, o3_latest
        )
        result.append(
            {
                "target_at": row["target_at"],
                "ispu": ispu,
                "category": label,
                "dominant": "",
                "ml_confidence": ml_confidence,
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
                "pm25_upper": round(f_row.get("pm25_upper", 0), 2),
                "pm25_lower": round(f_row.get("pm25_lower", 0), 2),
                "pm10_pred": round(f_row.get("pm10", 0), 2),
                "pm10_upper": round(f_row.get("pm10_upper", 0), 2),
                "pm10_lower": round(f_row.get("pm10_lower", 0), 2),
                "co_pred": round(f_row.get("co", 0), 2),
                "co_upper": round(f_row.get("co_upper", 0), 2),
                "co_lower": round(f_row.get("co_lower", 0), 2),
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


def save_fitted_values(
    supabase: Client, df: pd.DataFrame, models: dict, max_vals: dict
):
    log.info("--- Fitted Values (In-Sample Predictions) ---")
    feat = build_features(df)
    if feat.empty:
        log.warning("  Feature matrix kosong, skip fitted")
        return

    model_features = [
        c for c in feat.columns if c not in ["pm25", "pm10", "co", "no2", "o3"]
    ]

    # Compute hourly/minute patterns
    hourly_pat, minute_pat, overall_means = {}, {}, {}
    for p in FORECAST_PARAMS:
        series = df[p].dropna()
        overall_means[p] = float(series.mean())
        sh = pd.DataFrame({p: series})
        sh["hour"] = sh.index.hour
        hourly_pat[p] = sh.groupby("hour")[p].mean().to_dict()
        sm = pd.DataFrame({p: series})
        sm["minute"] = sm.index.minute
        minute_pat[p] = sm.groupby("minute")[p].mean().to_dict()

    rows = []
    prev_actuals = {p: None for p in FORECAST_PARAMS}

    for idx, row in feat.iterrows():
        actual_pm25 = row.get("pm25")
        actual_pm10 = row.get("pm10")
        actual_co = row.get("co")
        if pd.isna(actual_pm25) or pd.isna(actual_pm10) or pd.isna(actual_co):
            continue

        X = pd.DataFrame([{k: row.get(k, 0) for k in model_features}])

        try:
            preds = {
                "pm25": float(models["pm25"].predict(X)[0]),
                "pm10": float(models["pm10"].predict(X)[0]),
                "co": float(models["co"].predict(X)[0]),
            }
        except Exception:
            continue

        actuals = {
            "pm25": float(actual_pm25),
            "pm10": float(actual_pm10),
            "co": float(actual_co),
        }

        hour, minute = idx.hour, idx.minute
        blended = {}
        for p in FORECAST_PARAMS:
            xgb_raw = max(0.0, min(max_vals[p], preds[p]))
            prev = prev_actuals[p] if prev_actuals[p] is not None else xgb_raw
            hd = hourly_pat[p].get(hour, overall_means[p]) - overall_means[p]
            md = minute_pat[p].get(minute, overall_means[p]) - overall_means[p]
            pattern = overall_means[p] + (hd * 0.7 + md * 0.3)
            blended[p] = round(xgb_raw * 0.5 + prev * 0.3 + pattern * 0.2, 2)

        rows.append(
            {
                "timestamp": idx.isoformat(),
                "pm25_actual": round(float(actual_pm25), 2),
                "pm25_fitted": blended["pm25"],
                "pm10_actual": round(float(actual_pm10), 2),
                "pm10_fitted": blended["pm10"],
                "co_actual": round(float(actual_co), 2),
                "co_fitted": blended["co"],
            }
        )

        for p in FORECAST_PARAMS:
            prev_actuals[p] = actuals[p]

    if rows:
        BATCH = 500
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            try:
                supabase.table(TABLE_FITTED).upsert(
                    batch, on_conflict="timestamp"
                ).execute()
                log.info(
                    f"  {len(batch)} fitted values diupsert ke {TABLE_FITTED} ({i + 1}-{i + len(batch)})"
                )
            except Exception as e:
                log.warning(f"  Gagal upsert fitted batch: {e}")
    else:
        log.info("  Tidak ada fitted values yang dihasilkan")


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
                models_loaded[param] = (
                    m if not isinstance(m, dict) else m.get("model", m)
                )
                log.info(f"Loaded {pkl_path.name}")
            except Exception as e:
                log.warning(f"Gagal load {pkl_path.name}: {e}")
        else:
            log.warning(f"PKL tidak ditemukan: {pkl_path.name}")

    rf_model = None
    if RF_MODEL_PATH.exists():
        try:
            rf_model = joblib.load(RF_MODEL_PATH)
            log.info(f"Loaded RF classifier: {RF_MODEL_PATH.name}")
        except Exception as e:
            log.warning(f"Gagal load RF model: {e}")

    max_vals = {"pm25": 300, "pm10": 600, "co": 50000}
    if all(p in models_loaded for p in FORECAST_PARAMS):
        log.info("--- Multivariate XGBoost v2 (60-step) ---")
        all_forecasts = forecast_all_params(df, models_loaded, max_vals)

    if not all_forecasts:
        log.warning("Multivariate forecast gagal, fallback Holt-Winters per-param")
        for param, col, max_val in [
            ("pm25", "pm25", 300),
            ("pm10", "pm10", 600),
            ("co", "co", 50000),
        ]:
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
                for k, v in forecasts[i].items():
                    if k == "target_at":
                        continue
                    row[k] = v
        forecast_rows.append(row)

    classifications = classify_forecast(forecast_rows, no2_latest, o3_latest, rf_model)

    try:
        save_hourly_predictions(supabase, forecast_rows, forecast_at, classifications)
    except Exception as e:
        log.warning(f"  Gagal simpan ke Supabase: {e}")

    # Fitted values: hanya jika XGBoost models digunakan (bukan fallback)
    if len(models_loaded) >= 3:
        try:
            save_fitted_values(supabase, df, models_loaded, max_vals)
        except Exception as e:
            log.warning(f"  Gagal compute fitted values: {e}")

    return {
        "method": "XGBoost Forecasting",
        "forecast_at": forecast_at.isoformat(),
        "forecast": [
            {
                "target_at": row["target_at"].isoformat(),
                "pm25": round(row.get("pm25", 0), 2),
                "pm25_upper": round(row.get("pm25_upper", 0), 2),
                "pm25_lower": round(row.get("pm25_lower", 0), 2),
                "pm10": round(row.get("pm10", 0), 2),
                "pm10_upper": round(row.get("pm10_upper", 0), 2),
                "pm10_lower": round(row.get("pm10_lower", 0), 2),
                "co": round(row.get("co", 0), 2),
                "co_upper": round(row.get("co_upper", 0), 2),
                "co_lower": round(row.get("co_lower", 0), 2),
            }
            for row in forecast_rows
        ],
        "classification": [
            {
                "target_at": row["target_at"].isoformat(),
                "ispu": cls["ispu"],
                "category": cls["category"],
                "dominant": cls["dominant"],
                "ml_confidence": cls["ml_confidence"],
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
