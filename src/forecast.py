"""Day-ahead load forecasters.

All of them share one signature:

    forecast(history, target_index) -> pd.Series

where history has to stop before midnight of the target day (we pretend the
forecast is made the evening before). check_history() enforces that and
tests/test_forecast.py checks nothing leaks.

The foundation models (TimesFM, TimeGPT) live in foundation.py and plug into
the same day-ahead runner.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .data import FREQ, INTERVALS_PER_DAY, is_business_day

Forecaster = Callable[[pd.Series, pd.DatetimeIndex], pd.Series]
ONE_DAY = pd.Timedelta("1D")


class LookaheadError(ValueError):
    pass


def check_history(history: pd.Series, target_index: pd.DatetimeIndex) -> None:
    if len(history) and history.index[-1] >= target_index[0].normalize():
        raise LookaheadError(
            f"history ends {history.index[-1]}, but must end before {target_index[0].normalize()}"
        )


def target_index_for(day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(day, periods=INTERVALS_PER_DAY, freq=FREQ)


def history_before(load: pd.Series, day: pd.Timestamp) -> pd.Series:
    # everything up to (not including) midnight at the start of `day`
    return load.iloc[: load.index.searchsorted(day)]


def _day_matrix(history: pd.Series, first_day: pd.Timestamp, n_days: int) -> np.ndarray:
    # n_days x 96 grid, NaN where history has no value
    grid = pd.date_range(first_day, periods=n_days * INTERVALS_PER_DAY, freq=FREQ)
    return history.reindex(grid).to_numpy(dtype=float).reshape(n_days, INTERVALS_PER_DAY)


def _one_day(target_index: pd.DatetimeIndex, values: np.ndarray) -> pd.Series:
    day = target_index_for(target_index[0].normalize())
    return pd.Series(values, index=day, name="kw").reindex(target_index)


# Baselines


def oracle(actual: pd.Series) -> Forecaster:
    # Perfect foresight. Not a real forecaster, it's the ceiling we compare
    # against, and the only thing in here that's allowed to look at the future.
    def forecast(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        check_history(history, target_index)
        return actual.reindex(target_index).rename("kw")

    forecast.is_oracle = True
    return forecast


def seasonal_naive(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    # same day last week
    check_history(history, target_index)
    values = history.reindex(target_index - 7 * ONE_DAY).to_numpy(dtype=float)
    return pd.Series(values, index=target_index, name="kw")


def profile(history: pd.Series, target_index: pd.DatetimeIndex, days: int = 28) -> pd.Series:
    # average of the last 28 days for each time of day, using only days of the
    # same type (business vs weekend/holiday) as the target
    check_history(history, target_index)
    day0 = target_index[0].normalize()
    M = _day_matrix(history, day0 - days * ONE_DAY, days)
    biz = is_business_day(pd.date_range(day0 - days * ONE_DAY, periods=days, freq="D"))
    target_is_biz = is_business_day(pd.DatetimeIndex([day0]))[0]
    same = M[biz == target_is_biz]
    if len(same) == 0:
        same = M  # e.g. a holiday with no other holiday in the window
    return _one_day(target_index, np.nanmean(same, axis=0))


# Ridge: calendar features + lags of 1, 2 and 7 days

_HARMONICS = 8
_SLOT = np.arange(INTERVALS_PER_DAY)
_ANGLE = 2 * np.pi * _SLOT[:, None] * np.arange(1, _HARMONICS + 1)[None, :] / INTERVALS_PER_DAY
_FOURIER = np.hstack([np.sin(_ANGLE), np.cos(_ANGLE)])  # 96 x 16, smooth time-of-day shape


def _ridge_features(M: np.ndarray, biz: np.ndarray, rows: np.ndarray) -> np.ndarray:
    # One row per interval for the days in `rows`. I tried a full 192-column
    # one-hot for time of day first; same accuracy, 4x slower, so Fourier terms.
    k = len(rows)
    b = np.repeat(biz[rows], INTERVALS_PER_DAY).astype(float)
    tod = np.tile(_FOURIER, (k, 1))
    lag1 = M[rows - 1].ravel()
    lag2 = M[rows - 2].ravel()
    lag7 = M[rows - 7].ravel()
    # yesterday is a worse guide on Mondays and after holidays
    same_type = np.repeat(biz[rows] == biz[rows - 1], INTERVALS_PER_DAY)
    prev_mean = np.repeat(np.nanmean(M[rows - 1], axis=1), INTERVALS_PER_DAY)
    return np.column_stack([tod, tod * b[:, None], lag1, lag2, lag7, lag1 * same_type, prev_mean, b])


def ridge(history: pd.Series, target_index: pd.DatetimeIndex, train_days: int = 56, alpha: float = 1.0) -> pd.Series:
    # refit every day on the last 8 weeks
    check_history(history, target_index)
    day0 = target_index[0].normalize()
    n = train_days + 7  # +7 so the oldest training day still has its weekly lag
    M = np.vstack([_day_matrix(history, day0 - n * ONE_DAY, n), np.full((1, INTERVALS_PER_DAY), np.nan)])
    biz = is_business_day(pd.date_range(day0 - n * ONE_DAY, periods=n + 1, freq="D"))
    train_rows = np.arange(7, n)
    X = _ridge_features(M, biz, train_rows)
    y = M[train_rows].ravel()
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    model = Ridge(alpha=alpha).fit(X[ok], y[ok])
    X_target = _ridge_features(M, biz, np.array([n]))
    X_target = np.nan_to_num(X_target, nan=float(np.nanmean(y)))
    return _one_day(target_index, model.predict(X_target))


def day_ahead(forecaster: Forecaster, load: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    # one forecast per day, each only seeing data from before that day
    if hasattr(forecaster, "forecast_days"):
        return forecaster.forecast_days(load, days)
    parts = []
    for d in days:
        parts.append(forecaster(history_before(load, d), target_index_for(d)))
    return pd.concat(parts).rename("kw")


# display order and names used everywhere else
LABELS = {
    "oracle": "Oracle (perfect foresight)",
    "seasonal_naive": "Seasonal naive",
    "profile": "28-day profile",
    "ridge": "Ridge",
    "timesfm-3.0": "TimesFM 3.0 (zero-shot)",
    "timesfm-2.5": "TimesFM 2.5 (zero-shot)",
    "timegpt": "TimeGPT (zero-shot)",
}


FM_MODELS = ("timesfm-3.0", "timesfm-2.5", "timegpt")


def build_forecasters(load: pd.Series, include_fm: bool = True, fm_models=FM_MODELS) -> dict[str, Forecaster]:
    fcs = {
        "oracle": oracle(load),
        "seasonal_naive": seasonal_naive,
        "profile": profile,
        "ridge": ridge,
    }
    if not include_fm:
        return fcs
    # imported here rather than at the top because foundation.py imports this module
    from .foundation import TimeGPTZeroShot, TimesFMZeroShot

    if "timesfm-3.0" in fm_models:
        fcs["timesfm-3.0"] = TimesFMZeroShot("3.0")
    if "timesfm-2.5" in fm_models:
        fcs["timesfm-2.5"] = TimesFMZeroShot("2.5")
    if "timegpt" in fm_models:
        fcs["timegpt"] = TimeGPTZeroShot()
    return fcs
