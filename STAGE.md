# Stage 4 of 13: Foundation models: TimesFM and TimeGPT

Two zero-shot foundation models behind the same signature. TimesFM runs locally on CPU, TimeGPT calls Nixtla's API. Both batch a year of day-ahead forecasts and cache them in one Parquet file per model and dataset; the cached TimesFM forecasts for the synthetic building are included (four small files), so nothing here needs torch or a key.

## What to review in this layer

- TimesFM 3.0 and 2.5 zero-shot on CPU (28 days of context in, 96 steps out)
- TimeGPT via Nixtla's client, skipped with a logged message when there is no key
- The forecast cache: one Parquet file per model and dataset, out/cache/<model>/<data fingerprint>.parquet
- Tests with a fake model: the batched path can't see the future, the cache file, the no-key skip

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/foundation.py` | 313 lines | TimesFM and TimeGPT, zero-shot, with the forecast cache |
| `tests/test_foundation.py` | 87 lines | foundation-model batching, cache keys, the no-key skip |
| `requirements-fm.txt` | 6 lines | optional: TimesFM and the Nixtla client |
| `out/cache/timesfm-2.5/7886d719ee.parquet` | 0.8 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timesfm-2.5/8fa4b9177a.parquet` | 0.8 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timesfm-3.0/7886d719ee.parquet` | 0.8 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |
| `out/cache/timesfm-3.0/8fa4b9177a.parquet` | 0.8 MB | a year of cached day-ahead forecasts for one dataset, a row per 15 minutes |

Every file is copied byte for byte from the repo root, except files a later stage changes: until that stage, they are the earlier version kept in `versions/before/`.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (5.2s) 23 passed in 1.79s
- `try-it snippet`: ok (2.6s) TimesFM 3.0: 35040 cached intervals
- style (no comprehensions, ASCII only): clean

## Commit

**Add TimesFM and TimeGPT forecasters with a forecast cache**

```
TimesFM 3.0 and 2.5 run zero-shot on CPU; TimeGPT calls Nixtla's API and skips cleanly
without NIXTLA_API_KEY. Both batch a year of day-ahead forecasts and cache them in one
Parquet file per model and dataset under out/cache/. Includes the cached TimesFM
forecasts for the synthetic building, so nothing needs torch or a key.
```
