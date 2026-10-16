# Stage 2 of 12: Building load data

The data layer. A synthetic Boston-area office building at 15-minute resolution (occupancy, weather-driven cooling, random equipment spikes), a loader that reads real interval-data CSVs into the same shape, and the holiday calendars that decide what counts as a business day.

## What to review in this layer

- `synthetic_load()`: one year of 15-minute kW plus a warm-up, deterministic from a seed
- `load_interval_csv()`: utility or ISO-NE exports into the same shape, units and DST handled
- US and Portuguese holiday calendars behind one `is_business_day()`
- `fingerprint()`: a hash of the data, used later to key the forecast cache

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/data.py` | 313 lines | synthetic building, real CSV loader, holiday calendars, data fingerprint |
| `tests/conftest.py` | 52 lines | shared fixtures |
| `tests/test_data.py` | 66 lines | synthetic data shape, CSV loader, holidays |

Every file is copied byte for byte from the repo root: stages differ only in which files are present.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (8.3s) 8 passed in 1.51s
- `try-it snippet`: ok (1.6s) Freq: MS
- style (no comprehensions, ASCII only): clean

## Commit

**Add the synthetic building and an interval-data loader**

```
A generated Boston-area office at 15-minute resolution: occupancy, weather-driven cooling,
light winter heating and random weekday equipment spikes that set most monthly peaks.
load_interval_csv() reads real meter exports into the same shape, handling units, DST
and coarser-than-15-minute data. US and Portuguese holiday calendars.
```
