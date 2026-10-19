"""Real meter data from UCI's ElectricityLoadDiagrams20112014.

370 anonymous clients in Portugal, 15-minute kW readings, 2011-2014.
License CC BY 4.0. Source: Artur Trindade, UCI Machine Learning Repository.

The raw file is 700 MB, so it isn't in the repo. This script downloads it,
picks a handful of buildings with a fixed rule, and writes small cleaned CSVs
for just those buildings to data/uci/ (those are committed):

    python -m src.uci

Quirks of the raw file handled here:
  - semicolon separated, comma as the decimal point
  - timestamps mark the END of each 15-min interval (first row is 00:15)
  - spring DST: the four readings labelled 01:00-01:45 are all zero
  - autumn DST: the same four readings hold two hours of consumption
"""

from __future__ import annotations

import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from . import config
from .data import FREQ, is_business_day, set_holidays

URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
RAW_DIR = config.ROOT / "data" / "raw" / "uci"  # gitignored
RAW_FILE = RAW_DIR / "LD2011_2014.txt"
OUT_DIR = config.ROOT / "data" / "uci"
SELECTION_FILE = OUT_DIR / "selection.json"
SOURCE = "UCI ElectricityLoadDiagrams20112014 (Trindade, 2015), CC BY 4.0"

# The selection rule. Fixed before looking at any battery results, so the
# buildings can't be picked to suit the conclusion.
RULE = {
    "years": [2013, 2014],
    "max_zero_share": 0.005,  # basically complete data over both years
    "peak_min_kw": 100,  # 2013 peak in the C&I range where demand charges matter
    "peak_max_kw": 2000,
    "min_weekday_ratio": 1.3,  # business-day daytime load >= 1.3x weekends: office-ish
    "n_pick": 6,
    "seed": 2014,
}
EXPORT_START = "2013-10-01"  # same warm-up as the synthetic building
EXPORT_END = "2014-12-31 23:45"


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "electricityloaddiagrams20112014.zip"
    if not zip_path.exists():
        print(f"downloading {URL} (about 250 MB)...")
        urllib.request.urlretrieve(URL, zip_path)
    if not RAW_FILE.exists():
        zipfile.ZipFile(zip_path).extract("LD2011_2014.txt", RAW_DIR)


def read_raw(start: str = "2013-01-01", end: str = "2015-01-01 00:00") -> pd.DataFrame:
    """All 370 clients between start and end, cleaned, interval-start timestamps."""
    cached = RAW_DIR / f"clean_{start[:4]}_{end[:4]}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    df = pd.read_csv(RAW_FILE, sep=";", decimal=",", index_col=0)
    df = clean(df, start, end)
    df.to_parquet(cached)
    return df


def clean(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Fix the DST rows and move to interval-start timestamps."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    # end-of-interval labels, keep a little extra so the shift below has whole days
    df = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    # DST fixes, done on the original labels
    for year in sorted(set(df.index.year)):
        spring = _last_sunday(year, 3)
        autumn = _last_sunday(year, 10)
        spring_rows = pd.date_range(spring + pd.Timedelta("01:00:00"), periods=4, freq=FREQ)
        autumn_rows = pd.date_range(autumn + pd.Timedelta("01:00:00"), periods=4, freq=FREQ)
        spring_rows = spring_rows[spring_rows.isin(df.index)]
        autumn_rows = autumn_rows[autumn_rows.isin(df.index)]
        df.loc[spring_rows] = np.nan  # the zeros aren't real readings
        df.loc[autumn_rows] = df.loc[autumn_rows] / 2  # two hours squashed into one
    df = df.interpolate(limit=8, limit_area="inside")

    # move to interval-start labels like the rest of the project
    df.index = df.index - pd.Timedelta(FREQ)
    df = df.loc[pd.Timestamp(start) : pd.Timestamp(end) - pd.Timedelta(FREQ)]
    df.index.name = "timestamp"
    return df


def _last_sunday(year: int, month: int) -> pd.Timestamp:
    last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return last_day - pd.Timedelta(days=(last_day.dayofweek + 1) % 7)


def client_stats(df: pd.DataFrame) -> pd.DataFrame:
    # one row per client with the numbers the selection rule looks at
    set_holidays("PT")
    year_one = df.loc[str(RULE["years"][0])]
    hours = year_one.index.hour
    daytime = (hours >= 8) & (hours < 19)
    biz = is_business_day(year_one.index)
    weekend = year_one.index.dayofweek >= 5

    rows = []
    for client in df.columns:
        s = df[client]
        y1 = year_one[client]
        weekend_mean = y1[daytime & weekend].mean()
        ratio = y1[daytime & biz].mean() / weekend_mean if weekend_mean > 0 else np.nan
        rows.append({
            "client": client,
            "zero_share": float((s <= 0).mean()),
            "peak_2013_kw": float(y1.max()),
            "mean_2013_kw": float(y1.mean()),
            "weekday_ratio": float(ratio),
        })
    stats = pd.DataFrame(rows).set_index("client")
    stats["eligible"] = (
        (stats.zero_share <= RULE["max_zero_share"])
        & (stats.peak_2013_kw >= RULE["peak_min_kw"])
        & (stats.peak_2013_kw <= RULE["peak_max_kw"])
        & (stats.weekday_ratio >= RULE["min_weekday_ratio"])
    )
    return stats


def select(stats: pd.DataFrame) -> list[str]:
    eligible = sorted(stats.index[stats.eligible])
    rng = np.random.default_rng(RULE["seed"])
    picked = rng.choice(eligible, size=min(RULE["n_pick"], len(eligible)), replace=False)
    return sorted(picked.tolist())


def export(df: pd.DataFrame, clients: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for client in clients:
        s = df[client].loc[EXPORT_START:EXPORT_END].rename("kw")
        s.to_csv(OUT_DIR / f"{client}.csv", float_format="%.3f")


def main() -> None:
    download()
    print("reading the raw file (takes a minute)...")
    df = read_raw()
    stats = client_stats(df)
    picked = select(stats)
    export(df, picked)

    picked_stats = {}
    for client in picked:
        row = stats.loc[client]
        picked_stats[client] = {
            "peak_2013_kw": round(row.peak_2013_kw, 1),
            "mean_2013_kw": round(row.mean_2013_kw, 1),
            "weekday_ratio": round(row.weekday_ratio, 2),
        }
    selection = {
        "source": SOURCE,
        "url": URL,
        "rule": RULE,
        "n_clients": int(len(stats)),
        "n_eligible": int(stats.eligible.sum()),
        "picked": picked_stats,
        "export": [EXPORT_START, EXPORT_END],
    }
    SELECTION_FILE.write_text(json.dumps(selection, indent=1))
    print(f"{int(stats.eligible.sum())} of {len(stats)} clients pass the rule; picked {', '.join(picked)}")


if __name__ == "__main__":
    main()
