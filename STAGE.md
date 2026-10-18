# Stage 6 of 13: Battery and dispatch policies

The physics. A battery model and a simulator with three dispatch policies that all start from the same forecast-based daily kW target, so any difference between them comes from how each handles a wrong forecast.

## What to review in this layer

- `Battery` and `daily_targets()`: the lowest target the battery could hold if the forecast were right
- `simulate()`: window (follows the forecast's schedule), threshold (reacts to the meter), ratchet
- SOC limits, no export, off-peak charging that is paced and never goes above the day's target
- Tests for SOC bounds, energy balance, the charging cap and power limits

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/dispatch.py` | 260 lines | battery model, daily target bisection, the three dispatch policies |
| `tests/test_dispatch.py` | 114 lines | SOC bounds, energy balance, charging cap, power limits |

Every file is copied byte for byte from the repo root, except files a later stage changes: until that stage, they are the earlier version kept in `versions/before/`.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (6.1s) 212 passed in 2.39s
- `try-it snippet`: ok (1.0s) saved with perfect foresight: $ 52197
- style (no comprehensions, ASCII only): clean

## Commit

**Add the battery model and three dispatch policies**

```
All three policies share one forecast-based daily target: window follows the
forecast's schedule, threshold reacts to the meter, ratchet also never shaves below a
peak already billed this month. With a perfect forecast window and threshold are
identical. Charging is off-peak only, paced, and never above the day's target.
```
