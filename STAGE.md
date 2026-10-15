# Stage 1 of 12: Project scaffold and configuration

The skeleton everything else hangs off: where outputs and the forecast cache live, the one-line switch between the synthetic building and a real meter CSV, the warm-up and context lengths, and a tiny .env reader so an API key can live outside git. Plus the pytest setup and the licence.

## What to review in this layer

- `src/config.py`: paths, `DATA_SOURCE`, `WARMUP_DAYS`, `FM_CONTEXT_DAYS`
- `load_dotenv()`: reads `.env` without overriding real environment variables
- `pytest.ini` so test collection stays inside `tests/`

## Files added in this stage

| File | Size | What it is |
|---|---:|---|
| `src/__init__.py` | 0 lines | makes src a package |
| `src/config.py` | 35 lines | paths, the data-source switch, warm-up and context lengths |
| `tests/test_config.py` | 23 lines | paths and the .env reader |
| `pytest.ini` | 4 lines | keeps test collection inside tests/ |
| `LICENSE` | 21 lines | MIT |
| `.gitignore` | 25 lines | what git skips: environments, caches, the raw UCI download, versions/ |
| `.gitattributes` | 4 lines | line endings (LF for the Makefile and Python files) |

Every file is copied byte for byte from the repo root: stages differ only in which files are present.

## Checks (`--verify`)

- isolated (imports this folder's own `src`): yes
- `python -m pytest -q`: ok (2.8s) 3 passed in 0.04s
- `try-it snippet`: ok (0.2s) synthetic
- style (no comprehensions, ASCII only): clean

## Commit

**Set up the project: configuration and test scaffolding**

```
Paths, the one-line switch between the synthetic building and a real meter CSV,
warm-up and context lengths, and a small .env reader so a Nixtla key can stay out of
git. pytest config and the MIT licence.
```
