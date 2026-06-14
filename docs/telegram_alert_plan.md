# Telegram Alert System — Rencana Implementasi

Dokumen ini merangkum rencana implementasi sistem notifikasi Telegram untuk DashboardAQ v3.0. Simpan dokumen ini sebagai referensi sebelum implementasi.

## Fase 1: Setup Bot Telegram
1. Buat bot via @BotFather, dapatkan bot token
2. Dapatkan chat ID (mis. via @userinfobot atau dengan memanggil getUpdates setelah mengirim pesan ke bot)
3. Tambahkan variabel ke file `.env`:

```env
TELEGRAM_BOT_TOKEN="7372812345:AAH..."
TELEGRAM_CHAT_ID="123456789"
```

## Fase 2: Supabase Table — `tb_alert_config`
Rancangan tabel untuk menyimpan preferensi notifikasi dan status cooldown.

Kolom yang disarankan:

| Column | Type | Keterangan |
|--------|------|------------|
| id | int8 PK | Auto-increment |
| chat_id | text | Telegram chat ID tujuan |
| notifications_enabled | bool | Master toggle (default: true) |
| threshold_100 | bool | ISPU > 100 (default: true) |
| threshold_200 | bool | ISPU > 200 (default: true) |
| threshold_300 | bool | ISPU > 300 (default: true) |
| cooldown_minutes | int4 | Cooldown antar notifikasi (default: 30) |
| last_notified_100 | timestamptz | Waktu terakhir notifikasi untuk threshold 100 |
| last_notified_200 | timestamptz | Waktu terakhir notifikasi untuk threshold 200 |
| last_notified_300 | timestamptz | Waktu terakhir notifikasi untuk threshold 300 |

Catatan: Alternatif lebih sederhana bisa menyimpan hanya `last_notified` per konfigurasi jika hanya satu chat/konfigurasi.

## Fase 3: Python — `ml_model/telegram_alert.py`
Modul baru yang akan:

1. Membaca `TELEGRAM_BOT_TOKEN` dari `.env`
2. Query ke Supabase, tabel `tb_prediksi_kualitas_udara` atau `tb_konsentrasi_gas` untuk mendapatkan ISPU terkini
3. Menentukan polutan dominan (parameter dengan ISPU tertinggi)
4. Menentukan kategori ISPU (contoh: >100, >200, >300)
5. Mengecek apakah threshold yang sesuai aktif pada `tb_alert_config`
6. Mengecek cooldown (mis. 30 menit) berdasarkan kolom `last_notified_*`
7. Jika lolos, mengirim pesan ke Telegram (HTTP API) dan memperbarui `last_notified_*` di Supabase

Contoh alur fungsi utama (pseudo-code):

```python
def check_and_alert():
    cfg = read_alert_config_from_supabase()
    if not cfg.notifications_enabled:
        return

    latest = read_latest_ispu_from_supabase()
    dominant, ispu_value = get_dominant_pollutant(latest)
    level = categorize_ispu(ispu_value)

    if should_notify(cfg, level):
        msg = build_message(latest, dominant, level)
        send_telegram_message(cfg.chat_id, msg)
        update_last_notified(cfg, level)

```

Catatan implementasi:
- Gunakan package `python-dotenv` untuk membaca `.env` pada runtime watcher jika diperlukan.
- Gunakan `requests` untuk memanggil Telegram `https://api.telegram.org/bot{TOKEN}/sendMessage`.
- Hindari menyimpan token bot pada tabel publik. Simpan token di environment; chat_id dan preferences di Supabase.

## Fase 4: Integrasi ke `live_forecast_watcher.py`
Tambahkan pemanggilan `check_and_alert()` di akhir `run_forecast()` sehingga setiap siklus prediksi juga mengecek apakah perlu mengirim peringatan.

Contoh (setelah perintah prediksi selesai):

```python
from telegram_alert import check_and_alert
check_and_alert()
```

## Fase 5: Frontend — `/pengaturan` Jadi Fungsional
- Ganti kontrol statis di `src/app/pengaturan/page.tsx` menjadi binding ke `tb_alert_config` via API
- Sediakan form untuk mengatur `chat_id`, toggle per-threshold, dan `cooldown_minutes`
- Simpan perubahan ke Supabase melalui endpoint API (server-side) atau langsung via Supabase client pada frontend, tergantung pertimbangan keamanan

## Format Pesan Telegram (Template)

Pesan harus informatif dan singkat. Contoh format:

```
Peringatan Kualitas Udara
Waktu: 14 Jun 2026 13:45 WIB
Lokasi: Stasiun Pemantau Surabaya

Polutan Dominan: PM2.5
Nilai ISPU: 185
Kategori: Tidak Sehat

Parameter Lain:
- PM10: 120 ISPU
- CO: 45 ISPU
- NO2: 30 ISPU

Rekomendasi: Kelompok sensitif wajib mengurangi aktivitas di luar ruangan dan mengenakan masker.

Lihat dashboard: <URL_DASHBOARD>
```

Gunakan Markdown parsing pada `sendMessage` (`parse_mode=Markdown`) untuk menebalkan bagian penting.

## Opsional / Pertimbangan
- Jika ada lebih dari satu penerima, simpan banyak `chat_id` atau tabel relasi `alert_subscribers`.
- Untuk skalabilitas, bisa dipisah menjadi microservice notifikasi (mis. function/container) dan dipanggil oleh watcher atau cron.
- Pertimbangkan enkripsi/secret management untuk token jika menyebar ke lingkungan produksi.

## Langkah Selanjutnya
1. Jika setuju: saya akan membuat file `ml_model/telegram_alert.py` dasar dan menambahkan pemanggilan di `live_forecast_watcher.py`.
2. Buat migration/tabel `tb_alert_config` di Supabase (SQL atau dashboard).
3. Update frontend `/pengaturan` untuk mengelola konfigurasi.

---

File ini adalah dokumentasi rencana implementasi. Untuk eksekusi, ikuti urutan "Langkah Selanjutnya" di atas.
