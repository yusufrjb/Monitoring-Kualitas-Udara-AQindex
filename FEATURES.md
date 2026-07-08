# Fitur DashboardAQ — Web Dashboard

## Navigasi
- **Overview** (`/`) — Dashboard utama monitoring
- **Statistik** (`/statistik`) — Analisis & prediksi mendalam
- **Pengaturan** (`/pengaturan`) — Tampilan, notifikasi, data
- **Tentang** (`/tentang`) — Info aplikasi & tim

---

## Halaman UTama (`/`)

### 1. Header
- Judul "Monitoring Kualitas Udara"
- Jam digital real-time (WIB) dengan detik + pulsing green dot

### 2. Klasifikasi ISPU — Circular Gauge
- Ring berwarna (SVG arc) skala 0-500
- Label kategori (Baik/Sedang/Tidak Sehat/dll) + polutan dominan
- Panel deskripsi status dari standar KLHK
- Legenda 5 warna dengan rentang ISPU
- Rekomendasi kesehatan (collapse/expand): Kelompok Sensitif & Setiap Orang
- Label "Random Forest" di pojok

### 3. Kartu Polutan (5 kartu)
- PM2.5, PM10, NO2, CO, O3
- Masing-masing: nilai ISPU, warna gradien sesuai status, dot status, progress bar, label Aman/Berisiko
- Klik untuk toggle highlight

### 4. Ringkasan PM2.5
- Rata-rata PM2.5 (7 hari terakhir) dengan warna gradien
- Bar gradien dengan marker posisi rata-rata
- Statistik: Terendah, Tertinggi, Hampir Tertinggi (P95)
- Indikator tren (naik/turun/stabil) + tooltip standar deviasi

### 5. Kondisi Lingkungan
- Suhu (°C)
- Kelembapan (%)

### 6. Data BMKG
- Ikonic cuaca + deskripsi
- 4 mini card: Suhu, Kelembaban, Angin (km/h), Arah angin
- Tombol refresh + error banner retry

### 7. Pola Harian PM2.5 — Line Chart
- Rata-rata per jam: garis biru (hari kerja) + oranye putus-putus (akhir pekan)
- Sumbu X: 00:00-23:00, Sumbu Y: ISPU
- Tooltip hover

### 8. Distribusi Jam Sibuk — Bar Chart
- Rentang 07:00-09:00 WIB (Sen-Jum, 7 hari terakhir)
- 5 bar warna sesuai kategori ISPU
- Tooltip: kategori, rentang ISPU, jumlah + persentase

### 9. Trend Data — Line Chart (PM2.5 & PM10)
- Time series PM2.5 (biru) + PM10 (hijau)
- Pilihan periode: 1H / 7H / 14H / 30H / 90H
- Tooltip waktu + nilai ISPU

### 10. Kalender Kontribusi Bulan — Heatmap
- Grid 7 kolom (Sen-Min) dengan warna per hari sesuai rata-rata ISPU PM2.5
- Navigasi bulan (prev/next)
- Tooltip: tanggal, nilai ISPU + kategori, atau "Tidak ada data sensor"

### 11. Sebaran CO — Area Chart
- Distribusi kepadatan ISPU CO (120 data terakhir)
- Sumbu X: ISPU CO, Sumbu Y: frekuensi
- Gradient fill oranye

---

## Halaman Statistik (`/statistik`)

### 1. Prediksi 1 Jam (ISPU)
- Kartu status saat ini: badge kategori + ISPU + 3 nilai mini (PM2.5, PM10, CO)
- Tab: **Grafik** (area chart 3 series: PM2.5 ungu, PM10 hijau, CO oranye) / **Tabel** (Waktu, ISPU per polutan, ISPU Total, Kategori)

### 2. Forecasting & Analitik (XGBoost)
3 seksi identik untuk PM2.5, PM10, CO:
- **Area chart**: garis solid (aktual) + garis putus-putus (prediksi) dengan gradient fill
- **Side panel**: nilai "Saat Ini" (µg/m³) + "Prediksi 1 Jam" (µg/m³)
- Indikator arah tren

---

## Halaman Pengaturan (`/pengaturan`)

- **Mode Gelap** — toggle on/off
- **Notifikasi** — toggle + threshold checkbox (ISPU >100/200/300)
- **Auto Refresh** — toggle refresh tiap 1 menit
- **Sumber Data** — 3 bullet color-coded (real-time, prediksi, agregasi)
- **Refresh Data Sekarang** — tombol manual
- **Keamanan** — status: API Key, Koneksi DB, Terakhir Diperbarui

---

## Halaman Tentang (`/tentang`)

- Header card: logo + deskripsi + tech stack
- Fitur Utama (2x2 grid): Monitoring Real-time, Data Agregasi, ML Forecasting, Indeks ISPU
- Parameter Polutan (5 kartu dengan deskripsi)
- Standar ISPU KLHK (5 warna + rentang)
- Teknologi: Next.js 15, Radix UI, Tailwind CSS, Recharts, Supabase
- Kontak: GitHub, Email, Website
