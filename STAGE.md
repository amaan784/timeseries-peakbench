# Stage 7 of 13: Evaluation pipeline

The experiment itself. Every forecaster under every policy, plus a timing sweep that shifts the forecast up to an hour early or late, scored as dollars saved on the bill. Runs the synthetic building and its spike-free twin and writes everything to out/.

## What to review in this layer

- Forecaster x policy x timing-shift grid, scored on the full bill
- Per-day forecast error joined to per-day dollars, and the dollars-per-kW-of-error slope
- Battery-size sensitivity and the no-spike counterfactual building
- out/results.json plus tidy CSVs; a sanity check that window and threshold agree with a perfect forecast

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/evaluate.py` | 532 lines | forecaster x policy x timing grid, writes out/results.json and CSVs |

Every file is copied byte for byte from the repo root, except files a later stage changes: until that stage, they are the earlier version kept in `versions/before/`.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (5.8s) 212 passed in 2.34s
- `python -m src.evaluate`: ok (15.9s) INFO peakshave.evaluate: wrote C:\Users\dell\Desktop\time series project\versions\v7\out\results.json (+ CSVs) in 13.4s
- style (no comprehensions, ASCII only): clean

## Commit

**Add the evaluation pipeline**

```
Runs every forecaster under every policy plus a +/-60 minute timing sweep and scores
each on the full bill. Also per-day error vs dollars, battery-size sensitivity and the
spike-free twin of the building. Writes out/results.json and tidy CSVs.
```
