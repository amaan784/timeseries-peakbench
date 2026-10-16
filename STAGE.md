# Stage 3 of 12: Baseline forecasters

Day-ahead forecasting without any foundation model. Four forecasters with one signature, each limited to data from before the midnight that starts the day it predicts, and the runner that issues one forecast per day.

## What to review in this layer

- Oracle (perfect foresight, the ceiling), seasonal naive, 28-day profile
- Ridge: Fourier time-of-day terms plus 1, 2 and 7-day lags, refit every day
- `check_history()` and a test that poisons the future to prove nothing leaks
- `day_ahead()`: one forecast per day, each from that day's history only

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/forecast.py` | 178 lines | day-ahead forecasters: oracle, seasonal naive, profile, ridge, and the runner |
| `tests/test_forecast.py` | 64 lines | no forecaster can see its target day |

Every file is copied byte for byte from the repo root: stages differ only in which files are present.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (17.4s) 19 passed in 7.76s
- `try-it snippet`: ok (7.3s) ridge MAPE: 3.3 %
- style (no comprehensions, ASCII only): clean

## Commit

**Add day-ahead baseline forecasters**

```
Oracle (perfect foresight, the ceiling), seasonal naive, 28-day profile and ridge
(Fourier time of day plus 1/2/7-day lags, refit daily). All share one signature and
only see data from before the target day; a test overwrites the future with garbage
and checks that no forecast changes.
```
