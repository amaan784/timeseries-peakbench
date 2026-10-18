# Stage 9 of 13: Static report

A standalone HTML report that reads only the evaluation output, so every number and sentence on the page comes from a run. Also writes docs/index.html for GitHub Pages.

## What to review in this layer

- Findings, captions and headings generated from the results, including when they go against the thesis
- The bill as a ledger, five chart sections with toggles, accuracy and robustness tables
- Light/dark themes and the chart toggles in plain JS

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/report.py` | 1208 lines | standalone HTML report and the generated README results |

Every file is copied byte for byte from the repo root, except files a later stage changes: until that stage, they are the earlier version kept in `versions/before/`.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (6.8s) 215 passed in 3.26s
- `python -m src.evaluate`: ok (15.1s) INFO peakshave.evaluate: wrote C:\Users\dell\Desktop\time series project\versions\v9\out\results.json (+ CSVs) in 12.4s
- `python -m src.report`: ok (3.4s) wrote C:\Users\dell\Desktop\time series project\versions\v9\out\report.html and C:\Users\dell\Desktop\time series project\versions\v9\docs\index.html (0.4 MB)
- style (no comprehensions, ASCII only): clean

## Commit

**Add the standalone HTML report**

```
Reads only out/results.json and the CSVs, so all of its text and numbers are generated
from a run. Writes out/report.html and docs/index.html for GitHub Pages.
```
