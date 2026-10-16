# timeseries-peakbench, stage 2 of 12: Building load data

The data layer. A synthetic Boston-area office building at 15-minute resolution (occupancy, weather-driven cooling, random equipment spikes), a loader that reads real interval-data CSVs into the same shape, and the holiday calendars that decide what counts as a business day.

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

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q          # 8 tests
```

## Try it

Monthly peaks of the synthetic building:

```bash
python -c "from src.data import synthetic_load; kw = synthetic_load()['kw'].loc['2025']; print(kw.resample('MS').max().round().to_string())"
```

## Not in this stage yet

Later stages add these. They are simply absent here, nothing is stubbed out:

- `v3`: Baseline forecasters
- `v4`: Foundation models: TimesFM and TimeGPT
- `v5`: Tariff and the monthly bill
- `v6`: Battery and dispatch policies
- `v7`: Evaluation pipeline
- `v8`: Charts
- `v9`: Static report
- `v10`: Dashboard
- `v11`: Real meter data (UCI)
- `v12`: Real-meter evaluation (complete project)
