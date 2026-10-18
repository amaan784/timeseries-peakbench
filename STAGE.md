# Stage 8 of 13: Charts

The chart library, used by both the report and the dashboard. Five plotly figures in light and dark, one accent colour, units on every axis, and state of charge in its own panel instead of a second axis.

## What to review in this layer

- Savings vs timing shift, with the gap at +/-30 minutes bracketed
- Savings by forecaster, the peak day, monthly demand charges, error vs dollars
- Money formatting helpers and a fixed colour per policy

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/charts.py` | 384 lines | plotly figures shared by the report and the dashboard |
| `tests/test_charts.py` | 35 lines | money formatting, policy colours, the timing chart |

Every file is copied byte for byte from the repo root, except files a later stage changes: until that stage, they are the earlier version kept in `versions/before/`.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (6.8s) 215 passed in 3.29s
- `python -m src.evaluate`: ok (15.2s) INFO peakshave.evaluate: wrote C:\Users\dell\Desktop\time series project\versions\v8\out\results.json (+ CSVs) in 12.8s
- `try-it snippet`: ok (2.0s) wrote out/timing_chart.html
- style (no comprehensions, ASCII only): clean

## Commit

**Add the chart library**

```
Plotly figures shared by the report and the dashboard: savings vs timing shift with the
+/-30 minute gap bracketed, savings by forecaster, the peak day with state of charge in
its own panel, monthly demand charges, and error vs dollars. Light and dark themes.
```
