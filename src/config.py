"""Paths and settings."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
CACHE = OUT / "cache"
DOCS = ROOT / "docs"

# "synthetic" = the generated demo building. To use a real meter, point this at
# a CSV (path relative to the repo root), e.g. "data/my_building_15min.csv".
DATA_SOURCE = "synthetic"

# history needed before the first evaluated day (28-day profile, 7-day lag,
# and the foundation models' context all fit in this)
WARMUP_DAYS = 35

# how many days of history the foundation models get
FM_CONTEXT_DAYS = 28


def load_dotenv(path: Path = ROOT / ".env") -> None:
    # tiny .env reader so NIXTLA_API_KEY can live in a gitignored file
    # without adding python-dotenv as a dependency
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
