import numpy as np
import pandas as pd
import pytest

from src.data import evaluation_days, synthetic_load


@pytest.fixture(scope="session")
def load() -> pd.Series:
    # May-Aug: short enough to be quick, crosses month boundaries and has summer peaks
    return synthetic_load(start="2025-05-01", end="2025-08-31")["kw"]


@pytest.fixture(scope="session")
def days(load) -> pd.DatetimeIndex:
    return evaluation_days(load, n_days=60)


@pytest.fixture(scope="session")
def actual(load, days) -> pd.Series:
    return load.loc[days[0] : days[-1] + pd.Timedelta(hours=23, minutes=45)]


# battery and tariff are imported inside their fixtures so the data-only tests
# still run in the early build stages (versions/v1), before those modules exist


@pytest.fixture(scope="session")
def battery():
    from src.dispatch import Battery

    return Battery()


@pytest.fixture(scope="session")
def tariff():
    from src.tariff import Tariff

    return Tariff()


@pytest.fixture(scope="session")
def bad_forecasts(actual) -> dict[str, pd.Series]:
    # forecasts that are wrong in the ways that hurt a battery
    rng = np.random.default_rng(0)
    return {
        "perfect": actual,
        "too_low": actual * 0.85,
        "too_high": actual * 1.15,
        "noisy": actual * (1 + rng.normal(0, 0.10, len(actual))),
        "flat": pd.Series(actual.mean(), index=actual.index),
    }
