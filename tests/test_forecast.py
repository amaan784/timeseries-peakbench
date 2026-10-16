"""Make sure no forecaster can peek at the day it's forecasting."""

import numpy as np
import pandas as pd
import pytest

from src import forecast as F

REAL = {"seasonal_naive": F.seasonal_naive, "profile": F.profile, "ridge": F.ridge}


def poison_from(series: pd.Series, day: pd.Timestamp) -> pd.Series:
    # overwrite everything from `day` onwards with garbage
    out = series.copy()
    out[out.index >= day] = 1e7
    return out


@pytest.mark.parametrize("name", REAL)
def test_history_that_reaches_the_target_day_is_rejected(name, load, days):
    day = days[5]
    leaky = load.loc[: day + pd.Timedelta("15min")]  # includes the first target interval
    with pytest.raises(F.LookaheadError):
        REAL[name](leaky, F.target_index_for(day))


@pytest.mark.parametrize("name", REAL)
def test_poisoned_future_does_not_change_the_forecast(name, load, days):
    # If a forecaster could see the future, garbage there would leak into its output.
    targets = days[[3, 17, 40]]
    clean = F.day_ahead(REAL[name], load, targets)
    for day in targets:
        poisoned = F.day_ahead(REAL[name], poison_from(load, day), pd.DatetimeIndex([day]))
        np.testing.assert_allclose(poisoned.to_numpy(), clean.loc[str(day.date())].to_numpy())
        assert poisoned.max() < 1e6


@pytest.mark.parametrize("name", REAL)
def test_day_ahead_hands_over_history_ending_before_midnight(name, load, days):
    seen = []

    def spy(history, target_index):
        seen.append((history.index[-1], target_index[0]))
        return REAL[name](history, target_index)

    F.day_ahead(spy, load, days[:10])
    for last, first_target in seen:
        assert last < first_target.normalize()
        assert last == first_target - pd.Timedelta("15min")


def test_oracle_is_the_only_forecaster_that_reads_the_future(load, days):
    day = days[4]
    oracle = F.oracle(load)
    assert getattr(oracle, "is_oracle", False)
    out = oracle(F.history_before(load, day), F.target_index_for(day))
    np.testing.assert_allclose(out.to_numpy(), load.loc[F.target_index_for(day)].to_numpy())
    for f in REAL.values():
        assert not getattr(f, "is_oracle", False)


def test_baselines_without_foundation_models(load):
    fcs = F.build_forecasters(load, include_fm=False)
    assert list(fcs) == ["oracle", "seasonal_naive", "profile", "ridge"]
