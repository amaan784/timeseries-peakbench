# timeseries-peakbench, stage 9 of 13: Static report

A standalone HTML report that reads only the evaluation output, so every number and sentence on the page comes from a run. Also writes docs/index.html for GitHub Pages.

This folder is a cumulative snapshot: everything from the earlier stages plus this one, and it runs on its own. The finished project is `v13` (identical to the repo root). See `STAGE.md` for what this stage adds and how to review it.

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
| `src/foundation.py` | TimesFM and TimeGPT, zero-shot, with the forecast cache |
| `tests/test_foundation.py` | foundation-model batching, cache keys, the no-key skip |
| `requirements-fm.txt` | optional: TimesFM and the Nixtla client |
| `src/tariff.py` | illustrative MA tariff and the monthly bill |
| `tests/test_tariff.py` | the bill, demand charge = max interval x rate |
| `src/dispatch.py` | battery model, daily target bisection, the three dispatch policies |
| `tests/test_dispatch.py` | SOC bounds, energy balance, charging cap, power limits |
| `src/evaluate.py` | forecaster x policy x timing grid, writes out/results.json and CSVs |
| `src/charts.py` | plotly figures shared by the report and the dashboard |
| `tests/test_charts.py` | money formatting, policy colours, the timing chart |
| `src/report.py` | standalone HTML report and the generated README results |

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q          # 215 tests
```

## Try it

Build the report (after the evaluation):

```bash
python -m src.report
```

## Not in this stage yet

Later stages add these. They are simply absent here, nothing is stubbed out:

- `v10`: Dashboard
- `v11`: Real meter data (UCI)
- `v12`: Real-meter evaluation
- `v13`: Mistral's Timestral (complete project)
