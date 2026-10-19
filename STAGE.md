# Stage 11 of 13: Real meter data (UCI)

Real data. Six buildings from UCI's ElectricityLoadDiagrams20112014 (Portugal, 15-minute kW, CC BY 4.0), picked by a rule fixed before any results were run, cleaned and stored as small CSVs with the attribution.

## What to review in this layer

- `src/uci.py`: download, DST fixes, end-of-interval to start-of-interval timestamps
- The selection rule: complete data, a 2013 peak of 100-2,000 kW, an office-like weekly pattern
- data/uci/: the six cleaned meters, selection.json, and the CC BY 4.0 attribution

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/uci.py` | 182 lines | downloads the UCI meters, cleans them, picks six by a fixed rule |
| `tests/test_uci.py` | 40 lines | UCI DST fixes and timestamps |
| `data/uci` | 8 files | six cleaned real meters (CC BY 4.0) and how they were picked |

Every file is copied byte for byte from the repo root, except files a later stage changes: until that stage, they are the earlier version kept in `versions/before/`.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (6.2s) 219 passed in 2.66s
- `python -m src.evaluate`: ok (16.1s) INFO peakshave.evaluate: wrote C:\Users\dell\Desktop\time series project\versions\v11\out\results.json (+ CSVs) in 13.5s
- `python -m src.report`: ok (4.4s) wrote C:\Users\dell\Desktop\time series project\versions\v11\out\report.html and C:\Users\dell\Desktop\time series project\versions\v11\docs\index.html (0.4 MB)
- `dashboard (streamlit AppTest, headless)`: ok (6.8s) 2026-10-17 01:38:18.516 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
- `try-it snippet`: ok (1.1s) MT_128: 457 days, peak 772 kW
- style (no comprehensions, ASCII only): clean

## Commit

**Add six real meters from the UCI dataset**

```
src/uci.py downloads ElectricityLoadDiagrams20112014 (Portugal, 15-minute kW, CC BY
4.0), fixes its DST quirks and end-of-interval timestamps, and picks six buildings
with a rule fixed before any results: complete data, a 2013 peak of 100-2,000 kW and
an office-like weekly pattern. The cleaned meters are in data/uci/ with attribution.
```
