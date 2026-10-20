# Stage 12 of 13: Real-meter evaluation

The check against real data. The same forecasters, policies and tariff on the six real meters, with Portuguese holidays and each battery sized from the building's 2013 peak. Also brings in the full README, Makefile and requirements.

## What to review in this layer

- `src/real_meters.py`: the experiment per building, written to out/real/
- The cached TimesFM forecasts for the six meters
- The real-meter results appear in the report and the dashboard
- The full README with generated results, the complete Makefile and requirements

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/real_meters.py` | 147 lines | the same experiment on the six real meters |
| `tests/test_real_meters.py` | 16 lines | battery sizing, the six picked meters are present |
| `README.md` | 214 lines | the full project README |
| `requirements.txt` | 8 lines | the full dependency list |
| `Makefile` | 33 lines | every target: install, run, real, uci, test, report, app |
| `out/cache/timesfm-3.0/227a678ffa.parquet` | 0.7 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timesfm-3.0/2d3f7f743b.parquet` | 0.6 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timesfm-3.0/4f5021e674.parquet` | 0.7 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timesfm-3.0/672404e66e.parquet` | 0.7 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timesfm-3.0/fe1a8764dd.parquet` | 0.6 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timesfm-3.0/febcf0a347.parquet` | 0.8 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |

Every file is copied byte for byte from the repo root, except files a later stage changes: until that stage, they are the earlier version kept in `versions/before/`.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (6.3s) 221 passed in 2.67s
- `python -m src.evaluate`: ok (16.3s) INFO peakshave.evaluate: wrote C:\Users\dell\Desktop\time series project\versions\v12\out\results.json (+ CSVs) in 13.9s
- `python -m src.real_meters`: ok (35.4s) INFO peakshave.real: MT_164 done (32s so far)
- `python -m src.report`: ok (3.4s) refreshed the results section of README.md
- `dashboard (streamlit AppTest, headless)`: ok (7.2s) 2026-10-17 01:39:27.968 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
- style (no comprehensions, ASCII only): clean

## Commit

**Run the experiment on the real meters, and add the full README**

```
src/real_meters.py runs the same forecasters, policies and tariff on the six UCI
buildings with Portuguese holidays, sizing each battery at 30% of its 2013 peak. Adds
their cached TimesFM forecasts; their results now appear in the report and dashboard.
The README with generated results, the full Makefile and requirements.
```
