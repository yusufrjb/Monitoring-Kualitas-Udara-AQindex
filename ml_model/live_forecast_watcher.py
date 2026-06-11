import time
import subprocess
import logging
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent / "watcher_log.txt", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger("Watcher")

SUPABASE_URL_KEY = "SUPABASE_URL"
SUPABASE_ANON_KEY = "SUPABASE_ANON_KEY"
TABLE_DATA = "tb_konsentrasi_gas"
STALE_THRESHOLD_MIN = 30


def sensor_has_recent_data() -> bool:
    """Cek apakah ada data sensor dalam STALE_THRESHOLD_MIN menit terakhir."""
    env_paths = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env.local",
        Path(__file__).parent.parent / ".env",
    ]
    url = anon = None
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip('"')
                    if k == SUPABASE_URL_KEY and not url:
                        url = v
                    elif k == SUPABASE_ANON_KEY and not anon:
                        anon = v
    if not url or not anon:
        log.error("Supabase credentials tidak ditemukan di .env")
        return True  # fallback: jalankan saja

    try:
        from supabase import create_client

        sb = create_client(url, anon)
        since = (
            datetime.now(timezone.utc) - timedelta(minutes=STALE_THRESHOLD_MIN)
        ).isoformat()
        resp = (
            sb.table(TABLE_DATA)
            .select("created_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            log.warning("Tidak ada data sensor sama sekali.")
            return False
        latest = resp.data[0]["created_at"]
        is_fresh = latest >= since
        if not is_fresh:
            log.warning(
                f"Data sensor terakhir: {latest} — lebih dari {STALE_THRESHOLD_MIN} menit. Sensor offline."
            )
        return is_fresh
    except Exception as e:
        log.error(f"Gagal cek data sensor: {e}")
        return True  # fallback: jalankan saja


def run_forecast():
    log.info("Memulai pipeline peramalan...")
    try:
        # Jalankan Peramalan Multi-Parameter (60 min, untuk ISPU 1 jam)
        log.info("  -> Menjalankan predict_hourly_multi.py")
        res3 = subprocess.run(
            [sys.executable, "predict_hourly_multi.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        if res3.stdout:
            for line in res3.stdout.splitlines():
                if "Selesai" in line or "simpan" in line:
                    log.info(f"     [Hourly] {line}")

        if res3.returncode == 0:
            log.info("Pipeline peramalan berhasil.")
        else:
            log.warning(f"Pipeline gagal")

    except Exception as e:
        log.error(f"Error saat menjalankan pipeline: {e}")


def main():
    LOCK_FILE = Path(__file__).parent / ".watcher.lock"
    pid = os.getpid()

    # PID lock to prevent duplicate watcher instances
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            if old_pid != pid:
                alive = False
                try:
                    if os.name == "nt":
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
                    log.warning(f"Watcher lain sudah berjalan (PID {old_pid}), exit")
                    return
                log.info(f"Watcher lock stale (PID {old_pid}), melanjutkan...")
        except Exception:
            pass
    LOCK_FILE.write_text(str(pid))

    try:
        interval = 60  # 1 menit dalam detik

        while True:
            try:
                log.info("=" * 50)
                log.info("LIVE FORECAST WATCHER (Interval: 1 Menit)")
                log.info("=" * 50)

                while True:
                    start_time = time.time()

                    if sensor_has_recent_data():
                        run_forecast()
                    else:
                        log.info("Sensor offline — lewati siklus prediksi.")

                    elapsed = time.time() - start_time
                    sleep_time = max(0, interval - elapsed)

                    log.info(
                        f"Selesai dalam {elapsed:.2f} detik. Tidur selama {sleep_time / 60:.2f} menit..."
                    )
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                log.info("Watcher dimatikan oleh pengguna.")
                break
            except Exception as e:
                log.error(f"Watcher crash: {e}. Restart dalam 5 detik...")
                time.sleep(5)
    finally:
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
