# Stage 5 of 13: Tariff and the monthly bill

The money. An illustrative Massachusetts C&I time-of-use rate with a monthly demand charge on the single highest 15-minute interval, and the monthly bill that every later stage scores against.

## What to review in this layer

- `Tariff`: demand charge, on/off-peak energy rates, the 8am-9pm business-day window
- `bill()`: peak kW, kWh, energy charge, demand charge and total per month
- Tests: the demand charge is exactly the month's highest interval times the rate

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/tariff.py` | 69 lines | illustrative MA tariff and the monthly bill |
| `tests/test_tariff.py` | 44 lines | the bill, demand charge = max interval x rate |

Every file is copied byte for byte from the repo root, except files a later stage changes: until that stage, they are the earlier version kept in `versions/before/`.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (5.7s) 27 passed in 1.85s
- `try-it snippet`: ok (1.2s) year: 579413
- style (no comprehensions, ASCII only): clean

## Commit

**Add the illustrative MA tariff and the monthly bill**

```
A demand charge on the month's single highest 15-minute interval plus time-of-use
energy (on-peak 8am-9pm on business days). bill() returns peak kW, kWh, energy and
demand charges per month. Illustrative, not a filed rate.
```
