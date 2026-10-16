# timeseries-peakbench, stage 3 of 12: Baseline forecasters

Day-ahead forecasting without any foundation model. Four forecasters with one signature, each limited to data from before the midnight that starts the day it predicts, and the runner that issues one forecast per day.

This folder is a cumulative snapshot: everything from the earlier stages plus this one, and it runs on its own. The finished project is `v12` (identical to the repo root). See `STAGE.md` for what this stage adds and how to review it.

## What's here

| File | What it is |
|---|---|
| `src/__init__.py` | makes src a package |
| `src/config.py` | paths, the data-source switch, warm-up and context lengths |
| `tests/test_config.py` | paths and the .env reader |
| `pytest.ini` | keeps test collection inside tests/ |
| `LICENSE` | MIT |
| `.gitignore` | what git skips: environments, caches, the raw UCI download, versions/ |
| `.gitattributes` | line endings (LF for the Makefile and Python files) |
| `src/data.py` | synthetic building, real CSV loader, holiday calendars, data fingerprint |
| `tests/conftest.py` | shared fixtures |
| `tests/test_data.py` | synthetic data shape, CSV loader, holidays |
| `src/forecast.py` | day-ahead forecasters: oracle, seasonal naive, profile, ridge, and the runner |
| `tests/test_forecast.py` | no forecaster can see its target day |

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q          # 19 tests
```

## Try it

Ridge, day-ahead, over the first four weeks:

```bash
python -c "from src.data import synthetic_load, evaluation_days; from src.forecast import day_ahead, ridge; kw = synthetic_load()['kw']; fc = day_ahead(ridge, kw, evaluation_days(kw)[:28]); a = kw.reindex(fc.index); print('ridge MAPE:', round(100 * ((fc - a).abs() / a).mean(), 1), '%')"
```

## Not in this stage yet

Later stages add these. They are simply absent here, nothing is stubbed out:

- `v4`: Foundation models: TimesFM and TimeGPT
- `v5`: Tariff and the monthly bill
- `v6`: Battery and dispatch policies
- `v7`: Evaluation pipeline
- `v8`: Charts
- `v9`: Static report
- `v10`: Dashboard
- `v11`: Real meter data (UCI)
- `v12`: Real-meter evaluation (complete project)
