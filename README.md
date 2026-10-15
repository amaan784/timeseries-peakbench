# timeseries-peakbench, stage 1 of 12: Project scaffold and configuration

The skeleton everything else hangs off: where outputs and the forecast cache live, the one-line switch between the synthetic building and a real meter CSV, the warm-up and context lengths, and a tiny .env reader so an API key can live outside git. Plus the pytest setup and the licence.

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

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q          # 3 tests
```

## Try it

Where outputs and the forecast cache go:

```bash
python -c "from src import config; print(config.OUT); print(config.CACHE); print(config.DATA_SOURCE)"
```

## Not in this stage yet

Later stages add these. They are simply absent here, nothing is stubbed out:

- `v2`: Building load data
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
