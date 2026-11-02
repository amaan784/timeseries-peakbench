# Stage 13 of 13: Mistral's Timestral (complete project)

A third foundation model: Mistral's Timestral 209M, zero-shot on CPU through the same batching and cache as TimesFM. It runs on the synthetic building and the six real meters, and the report, README and dashboard compare it with ridge the same way. This stage is identical to the repo root.

## What to review in this layer

- `TimestralZeroShot` in src/foundation.py: 28 days in, 96 steps out, median plus the 10th and 90th percentiles
- Registered as `timestral-209m` next to TimesFM, and run on the real meters as well
- The report's foundation-model findings and real-meter table now cover every foundation model, not only TimesFM 3.0
- The cached Timestral forecasts, one Parquet file per dataset (two synthetic, six real)
- Tests with a fake model: reshaping the output, NaN padding, skipping cleanly without the package

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `tests/test_timestral.py` | 93 lines | the Timestral wrapper, with a fake model |
| `out/cache/timestral-209m/227a678ffa.parquet` | 0.7 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timestral-209m/2d3f7f743b.parquet` | 0.6 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timestral-209m/4f5021e674.parquet` | 0.7 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timestral-209m/672404e66e.parquet` | 0.7 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timestral-209m/7886d719ee.parquet` | 0.7 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timestral-209m/8fa4b9177a.parquet` | 0.7 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timestral-209m/fe1a8764dd.parquet` | 0.6 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timestral-209m/febcf0a347.parquet` | 0.8 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |

## Files changed in this stage

| File | Lines | What changed |
|---|---:|---|
| `src/foundation.py` | +71 / -3 | adds `TimestralZeroShot` and `timestral_zeroshot()` |
| `src/forecast.py` | +6 / -3 | `timestral-209m` in LABELS, FM_MODELS and build_forecasters |
| `src/real_meters.py` | +6 / -6 | Timestral runs on the real meters too |
| `src/report.py` | +99 / -53 | foundation-model text and the real-meter table cover every model |
| `src/evaluate.py` | +3 / -3 | the docstring and `--no-fm` help mention Timestral |
| `app.py` | +3 / -3 | two sentences mention Timestral |
| `requirements-fm.txt` | +5 / -2 | the timestral package, pinned to a commit |
| `Makefile` | +1 / -1 | the install-fm comment |
| `README.md` | +37 / -32 | results with Timestral, how it is run, its licence |

Every file is copied byte for byte from the repo root, except files a later stage changes: until that stage, they are the earlier version kept in `versions/before/`.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (11.1s) 226 passed in 7.04s
- `python -m src.evaluate`: ok (16.5s) INFO peakshave.evaluate: wrote C:\Users\dell\Desktop\time series project\versions\v13\out\results.json (+ CSVs) in 14.1s
- `python -m src.real_meters`: ok (36.7s) INFO peakshave.real: MT_164 done (33s so far)
- `python -m src.report`: ok (3.7s) refreshed the results section of README.md
- `dashboard (streamlit AppTest, headless)`: ok (7.2s) 2026-10-17 01:40:43.694 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
- `try-it snippet`: ok (2.7s) Timestral 209M: 35040 cached intervals
- style (no comprehensions, ASCII only): clean

## Commit

**Add Mistral's Timestral as a third foundation model**

```
Timestral 209M runs zero-shot on CPU through the same batching and cache as TimesFM
(28 days in, 96 steps out, median plus 10th/90th percentiles), on the synthetic
building and the six real meters. The report, README and dashboard now compare every
foundation model with ridge, not only TimesFM 3.0. Includes the cached forecasts.
```
