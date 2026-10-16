"""Load data for the experiment.

Either a synthetic office building or a real interval-data CSV. Both come back
in the same shape: 15-minute DatetimeIndex (naive local time, 96 rows per day)
and a single `kw` column.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Easter,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMemorialDay,
    USThanksgivingDay,
)
from pandas.tseries.offsets import Day

from . import config

FREQ = "15min"
INTERVALS_PER_DAY = 96
DT_HOURS = 0.25

SYNTHETIC_LABEL = "Synthetic data: a generated C&I building, not a real meter"


class _BuildingHolidays(AbstractHolidayCalendar):
    # days the building runs its weekend schedule
    rules = [
        Holiday("New Year's Day", month=1, day=1),
        USMemorialDay,
        Holiday("Independence Day", month=7, day=4),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25),
    ]


class _PortugalHolidays(AbstractHolidayCalendar):
    # Portuguese national holidays, for the UCI meters. Four of them were
    # suspended from 2013 to 2015, which covers the years we use.
    rules = [
        Holiday("Ano Novo", month=1, day=1),
        GoodFriday,
        Holiday("Liberdade", month=4, day=25),
        Holiday("Trabalhador", month=5, day=1),
        Holiday("Portugal", month=6, day=10),
        Holiday("Assuncao", month=8, day=15),
        Holiday("Imaculada Conceicao", month=12, day=8),
        Holiday("Natal", month=12, day=25),
        Holiday("Corpo de Deus", month=1, day=1, offset=[Easter(), Day(60)], end_date="2012-12-31"),
        Holiday("Corpo de Deus ", month=1, day=1, offset=[Easter(), Day(60)], start_date="2016-01-01"),
        Holiday("Republica", month=10, day=5, end_date="2012-12-31"),
        Holiday("Republica ", month=10, day=5, start_date="2016-01-01"),
        Holiday("Todos os Santos", month=11, day=1, end_date="2012-12-31"),
        Holiday("Todos os Santos ", month=11, day=1, start_date="2016-01-01"),
        Holiday("Restauracao", month=12, day=1, end_date="2012-12-31"),
        Holiday("Restauracao ", month=12, day=1, start_date="2016-01-01"),
    ]


_CALENDARS = {"US": _BuildingHolidays, "PT": _PortugalHolidays}
_active_calendar = "US"


def set_holidays(name: str) -> None:
    # The synthetic building is in Massachusetts, the UCI meters are in
    # Portugal. Everything that asks "is this a business day?" goes through
    # is_business_day(), so switching here switches it everywhere.
    global _active_calendar
    if name not in _CALENDARS:
        raise ValueError(f"unknown holiday calendar {name!r}")
    _active_calendar = name


def holiday_calendar() -> str:
    return _active_calendar


@lru_cache(maxsize=None)
def _holidays_for_years(first: int, last: int, calendar: str = "US") -> pd.DatetimeIndex:
    # building the calendar is slow, and we ask for it a lot
    return _CALENDARS[calendar]().holidays(start=f"{first}-01-01", end=f"{last}-12-31")


def holidays(start, end) -> pd.DatetimeIndex:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hol = _holidays_for_years(start.year, end.year, _active_calendar)
    return hol[(hol >= start.normalize()) & (hol <= end)]


def is_business_day(index: pd.DatetimeIndex) -> np.ndarray:
    days = index.normalize()
    hol = _holidays_for_years(days.min().year, days.max().year, _active_calendar)
    return np.asarray((index.dayofweek < 5) & ~days.isin(hol))


# Synthetic building


def synthetic_load(
    start: str = "2024-10-01",
    end: str = "2025-12-31",
    seed: int = 7,
    spikes: bool = True,
) -> pd.DataFrame:
    """Mid-size Boston-area office, roughly 800 kW at peak.

    Base load, a weekday occupancy curve, cooling driven by a made-up but
    Boston-ish temperature series, a bit of electric heat in winter, and short
    equipment spikes on weekdays. The spikes are what set most monthly peaks,
    and no day-ahead forecast can know when they'll happen.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta("23:45:00"), freq=FREQ)
    n_days = len(idx) // INTERVALS_PER_DAY
    day_starts = idx[::INTERVALS_PER_DAY]
    hour = (idx.hour + idx.minute / 60).to_numpy()
    business = is_business_day(idx)
    business_day = is_business_day(day_starts)

    def ramp(h, center, width):
        return 1 / (1 + np.exp(-(h - center) / width))

    # occupancy 0..1: in around 7:30, out around 6pm; weekends/holidays at a quarter
    occ_weekday = ramp(hour, 7.4, 0.55) * (1 - ramp(hour, 18.3, 0.7))
    occ_weekend = 0.25 * ramp(hour, 9.0, 0.6) * (1 - ramp(hour, 15.5, 0.6))
    occ = np.where(business, occ_weekday, occ_weekend)

    # temperature in F: seasonal curve + AR(1) weather anomaly per day + daily swing
    doy = day_starts.dayofyear.to_numpy()
    seasonal = 51.5 - 23.0 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    anomaly = np.zeros(n_days)
    for d in range(1, n_days):
        anomaly[d] = 0.72 * anomaly[d - 1] + rng.normal(0, 4.6)
    daily_mean_t = np.repeat(seasonal + anomaly, INTERVALS_PER_DAY)
    temp = daily_mean_t + 8.5 * np.sin(2 * np.pi * (hour - 9.5) / 24)
    # the building lags the weather a bit, so use the temperature from 90 min ago
    temp_lagged = np.concatenate([np.full(6, temp[0]), temp[:-6]])

    cooling = 8.8 * np.clip(temp_lagged - 63.0, 0, None) * (0.30 + 0.70 * occ)
    heating = 1.9 * np.clip(48.0 - temp_lagged, 0, None) * (0.45 + 0.55 * occ)

    # some days are just busier than others
    activity = np.repeat(rng.normal(1.0, 0.045, n_days), INTERVALS_PER_DAY)
    base = 165.0 + 10.0 * np.sin(2 * np.pi * (hour - 3) / 24)
    kw = base + 265.0 * occ * activity + cooling + heating

    # small autocorrelated noise so it doesn't look too clean
    eps = np.zeros(len(idx))
    shocks = rng.normal(0, 0.012, len(idx))
    for i in range(1, len(idx)):
        eps[i] = 0.6 * eps[i - 1] + shocks[i]
    kw *= 1 + eps

    if spikes:
        kw += _equipment_spikes(rng, len(idx), business_day)

    out = pd.DataFrame({"kw": np.round(kw, 1)}, index=idx)
    out.index.name = "timestamp"
    return out


def _equipment_spikes(rng, n, business_day) -> np.ndarray:
    # chiller restarts, compressors etc: 15-45 min, 60-170 kW, business hours only
    spikes = np.zeros(n)
    for d, is_biz in enumerate(business_day):
        if not is_biz:
            continue
        for _ in range(rng.poisson(0.6)):
            start = d * INTERVALS_PER_DAY + rng.integers(8 * 4, 18 * 4)
            length = rng.integers(1, 4)
            # two events at once don't add up, take the bigger one
            spikes[start : start + length] = np.maximum(spikes[start : start + length], rng.uniform(60, 170))
    return spikes


# Real interval data

_TS_NAMES = ("timestamp", "datetime", "date_time", "interval_start", "interval start", "start", "time", "date", "local time")
_KW_NAMES = ("kw", "demand", "load", "value", "usage", "kwh", "mw", "mwh")


def _find_timestamp_col(columns) -> str:
    lowered = {}
    for c in columns:
        lowered[c.lower().strip()] = c
    for name in _TS_NAMES:
        if name in lowered:
            return lowered[name]
    return columns[0]


def _find_value_col(columns) -> str | None:
    # match "kw", "kW demand", "Usage (kWh)" and so on
    for c in columns:
        name = c.lower().strip()
        for k in _KW_NAMES:
            if name == k or name.startswith(k + " ") or name.endswith(" " + k):
                return c
    return None


def load_interval_csv(
    path: str | Path,
    timestamp_col: str | None = None,
    value_col: str | None = None,
    units: str | None = None,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Read a utility interval export (or an ISO-NE load file) into our shape.

    Column names are guessed if not given. Units can be kw, kwh, mw or mwh
    (guessed from the column name). Anything coarser than 15 min, like hourly
    data, is held flat across its 15-min slots; anything finer is averaged.
    DST: the repeated fall-back hour is averaged and the missing spring hour
    is interpolated, so every day ends up with 96 rows.
    """
    df = pd.read_csv(path)
    ts_col = timestamp_col or _find_timestamp_col(list(df.columns))
    v_col = value_col or _find_value_col(list(df.columns))
    if v_col is None:
        numeric = df.drop(columns=[ts_col]).select_dtypes("number").columns
        if len(numeric) == 0:
            raise ValueError(f"No numeric load column found in {path}")
        v_col = numeric[0]
    units = (units or _infer_units(v_col)).lower()

    # timestamps like 2025-03-09T01:00-05:00 switch offset at DST, which pandas
    # won't parse as one column unless we go through UTC first
    has_offset = df[ts_col].astype(str).str.contains(r"(?:[+-]\d\d:?\d\d|Z)$", regex=True).any()
    ts = pd.to_datetime(df[ts_col], utc=bool(has_offset))
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(tz).dt.tz_localize(None)
    s = pd.Series(pd.to_numeric(df[v_col], errors="coerce").to_numpy(), index=ts).sort_index()

    step = s.index.to_series().diff().median()
    hours = step / pd.Timedelta("1h")
    if units in ("kwh", "mwh"):
        s = s / hours
    if units in ("mw", "mwh"):
        s = s * 1000.0

    s = s.groupby(level=0).mean()  # fall-back hour shows up twice
    if step <= pd.Timedelta(FREQ):
        s = s.resample(FREQ).mean()
    else:
        s = s.resample(FREQ).ffill(limit=int(step / pd.Timedelta(FREQ)) - 1)
    s = s.interpolate(limit=8, limit_area="inside")

    # keep whole days only (a day is whole if it reaches 23:45), and drop any
    # day that still has holes
    first = s.index.min().ceil("D")
    last = (s.index.max() + pd.Timedelta(FREQ)).floor("D") - pd.Timedelta(FREQ)
    s = s.loc[first:last]
    complete = s.groupby(s.index.normalize()).transform(lambda x: x.notna().all())
    s = s[complete.astype(bool)]

    out = pd.DataFrame({"kw": s.astype(float)})
    out.index.name = "timestamp"
    return out


def _infer_units(col: str) -> str:
    c = col.lower()
    for u in ("mwh", "kwh", "mw", "kw"):
        if u in c:
            return u
    return "kw"


def load_building(source: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Load whatever config.DATA_SOURCE points at. Returns (load, meta)."""
    source = source or config.DATA_SOURCE
    if source == "synthetic":
        load = synthetic_load()
        meta = {"source": "synthetic", "label": SYNTHETIC_LABEL, "synthetic": True}
    else:
        path = Path(source)
        if not path.is_absolute():
            path = config.ROOT / path
        load = load_interval_csv(path)
        meta = {"source": str(source), "label": f"Metered interval data: {path.name}", "synthetic": False}
    meta["fingerprint"] = fingerprint(load["kw"])
    return load, meta


def evaluation_days(load: pd.DataFrame | pd.Series, n_days: int = 365) -> pd.DatetimeIndex:
    # last n_days full days, skipping the warm-up the forecasters need
    days = load.index.normalize().unique()
    usable = days[config.WARMUP_DAYS :]
    if len(usable) == 0:
        raise ValueError(f"Need more than {config.WARMUP_DAYS} days of data")
    return usable[-n_days:]


def fingerprint(kw: pd.Series) -> str:
    # hash of the data, used as the forecast cache key. Rounded first so tiny
    # float differences between machines don't change it.
    h = hashlib.sha1()
    h.update(np.round(kw.to_numpy(dtype=float), 1).tobytes())
    h.update(str(kw.index[0]).encode())
    h.update(str(len(kw)).encode())
    return h.hexdigest()[:10]
