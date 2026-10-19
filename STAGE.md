# Stage 10 of 13: Dashboard

The interactive version. A Streamlit app that re-runs the simulation live for any battery size, demand charge, forecaster, policy, timing shift and month range, laid out like a utility bill.

## What to review in this layer

- Sidebar controls; every number recomputed with the same functions as the evaluation
- The bill as a ledger, then the five charts with generated captions
- Cached simulations, so moving a slider back is instant

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `app.py` | 479 lines | Streamlit dashboard |
| `.streamlit/config.toml` | 23 lines | dashboard theme |

Every file is copied byte for byte from the repo root, except files a later stage changes: until that stage, they are the earlier version kept in `versions/before/`.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (6.5s) 215 passed in 2.86s
- `python -m src.evaluate`: ok (15.7s) INFO peakshave.evaluate: wrote C:\Users\dell\Desktop\time series project\versions\v10\out\results.json (+ CSVs) in 13.3s
- `python -m src.report`: ok (3.3s) wrote C:\Users\dell\Desktop\time series project\versions\v10\out\report.html and C:\Users\dell\Desktop\time series project\versions\v10\docs\index.html (0.4 MB)
- `dashboard (streamlit AppTest, headless)`: ok (6.3s) 2026-10-17 01:37:44.988 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
- style (no comprehensions, ASCII only): clean

## Commit

**Add the Streamlit dashboard**

```
Re-runs the simulation live for any battery, demand charge, forecaster, policy, timing
shift and month range. Shows the bill as a ledger and the same charts as the report.
```
