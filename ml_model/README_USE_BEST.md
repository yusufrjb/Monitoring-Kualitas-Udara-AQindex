Usage notes for switching to best models

- Feature flag: set environment variable `USE_BEST_MODEL=true` (default true) to prefer
  artifacts named `best_{param}_forecast_1hour.pkl`.
- If best artifacts missing, the code falls back to legacy `xgb_*` filenames when available.
- The change removed blending/ensemble logic: model outputs are used directly.

Testing
- Run `python predict_time_series.py --json` to produce JSON-only output for quick checks.
- Run `python predict_hourly_multi.py` to execute the hourly multi-parameter forecast.

Notes
- Keep backup of previous blending logic in version control. If you want to preserve
  the blending behavior for analysis, run both versions on a historical dataset and
  compare distributions before full rollout.
