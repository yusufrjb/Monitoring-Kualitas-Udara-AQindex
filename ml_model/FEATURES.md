# Fitur Forecasting — DashboardAQ

## Arsitektur

```
live_forecast_watcher.py  (scheduler tiap 1 menit)
        │
        ▼
predict_hourly_multi.py   (engine forecasting)
        │
        ├── fetch_recent_data()    ← Supabase: 3 hari terakhir
        ├── build_features()       ← 47 fitur dari 6 kolom raw
        ├── forecast_all_params()  ← XGBoost v2 recursive (60 step)
        ├── fallback_holt_winters()← fallback jika model gagal
        ├── classify_forecast()    ← Random Forest → ISPU
        └── save_hourly_predictions() → Supabase: tb_prediksi_hourly
```

## Model

| Parameter | Model | File |
|-----------|-------|------|
| PM2.5 | XGBRegressor | `xgb_pm25_v2.pkl` |
| PM10 | XGBRegressor | `xgb_pm10_v2.pkl` |
| CO | XGBRegressor | `xgb_co_v2.pkl` |
| ISPU | RandomForestClassifier | `random_forest_air_quality.pkl` |

- Model di-training via `train_pm25_pm10_co.ipynb` dari data historis Supabase
- Setiap model adalah pure XGBRegressor tanpa scaler/transformer

## Feature Engineering

6 raw columns → 47 fitur:

- **Raw temperature & humidity** — dimasukkan langsung ke feature vector
- **Lag features** — lag_1min, lag_5min, lag_15min, lag_60min (untuk tiap kolom)
- **Rolling statistics** — rolling_mean_5min, rolling_std_5min, rolling_mean_15min
- **Time features** — minute, hour, dayofweek

## Data Flow

1. **Fetch** — ambil data dari `tb_konsentrasi_gas` (3 hari = 4320 menit)
2. **Rename** — kolom DB (`pm25_ugm3`, `pm10_ugm3`, `co_ugm3`, dll) → nama pendek
3. **Build features** — transform ke 47 fitur
4. **Predict recursive** — 60 step, tiap step predict PM25 → PM10 → CO dengan shared state
5. **Classify** — tiap prediksi diklasifikasi ke kategori ISPU via Random Forest
6. **Save** — upsert ke `tb_prediksi_hourly` di Supabase

## Multivariate Recursive Prediction

- Tiap step menggunakan **shared state**: hasil prediksi step sebelumnya digunakan sebagai input step berikutnya
- Prediksi 3 polutan dalam urutan: PM2.5 → PM10 → CO (setiap step)
- Non-target kolom (NO2, temperature, humidity) di-freeze dengan nilai terakhir untuk menjaga stabilitas
- Output di-clip ke batas aman: PM2.5 (300), PM10 (600), CO (50000)

## Fallback System

Jika model XGBoost gagal atau data tidak mencukupi:
- **Holt-Winters** exponential smoothing per parameter
- Alpha=0.3, Beta=0.1
- Noise random ±5% untuk variasi

## ISPU Classification

- **Input**: PM2.5, PM10, CO, NO2, O3 (NO2/O3 dari data terakhir sensor)
- **Model**: Random Forest (5 kelas)
- **Kategori**: Baik → Sedang → Tidak Sehat → Sangat Tidak Sehat → Berbahaya
- Output: `ispu` (nilai numerik midpoint), `category`, `color`

## Scheduler

- `live_forecast_watcher.py` berjalan setiap **1 menit**
- Cek ketersediaan data sensor (threshold: 30 menit)
- PID lock file mencegah instance ganda
- Logging ke stdout + `watcher_log.txt`
- Dijalankan via `npm run dev` (concurrently dengan Next.js)

## Persistence (Supabase)

**Table**: `tb_prediksi_hourly`
**Columns**: `forecast_at`, `target_at`, `pm25_pred`, `pm10_pred`, `co_pred`, `ispu`, `category`, `is_historical`
**Strategy**: upsert on conflict `target_at`

## Safety

- **PID lock** — cegah concurrent run di `predict_hourly_multi.py` dan `live_forecast_watcher.py`
- **NaN handling** — dropna di fetch, fallback value di feature builder
- **Clip bounds** — semua output di-clip ke rentang aman

## Data Window

- `DATA_WINDOW_MIN = 4320` (3 hari) — karena sensor data intermittent dengan gap panjang
- Minimal 60 baris kontigu diperlukan untuk shift/lag features
