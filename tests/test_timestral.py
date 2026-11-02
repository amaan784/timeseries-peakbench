"""The Timestral wrapper, with a fake model so it runs without torch or the checkpoint."""

import sys

import numpy as np
import pandas as pd
import pytest

from src import forecast as F
from src import foundation as FM


class FakeForecast:
    # same shape as timestral's Forecast: (days, variates, steps)
    def __init__(self, base):
        self.base = base

    @property
    def median(self):
        return self.base

    def quantile(self, level):
        return self.base + (level - 0.5) * 10


class FakeTimestral:
    # forecasts each day as the mean of its context, and keeps what it was given
    def __init__(self):
        self.batches = []

    def forecast(self, context, prediction_length):
        self.batches.append(context)
        means = np.nanmean(context, axis=1)
        base = np.repeat(means[:, None, None], prediction_length, axis=2)
        return FakeForecast(base)


def fake_model(tmp_path):
    model = FM.TimestralZeroShot("209M", cache_root=tmp_path)
    model._model = FakeTimestral()
    return model


def test_outputs_are_reshaped_into_one_frame_per_day(load, days, tmp_path):
    model = fake_model(tmp_path)
    targets = days[:3]
    fc = model.forecast_days(load, targets)
    assert len(fc) == 3 * F.INTERVALS_PER_DAY
    assert fc.index.equals(pd.DatetimeIndex(np.concatenate([
        F.target_index_for(targets[0]).to_numpy(),
        F.target_index_for(targets[1]).to_numpy(),
        F.target_index_for(targets[2]).to_numpy(),
    ])))
    # all three days went in one batch of 28-day contexts
    assert len(model._model.batches) == 1
    assert model._model.batches[0].shape == (3, 28 * F.INTERVALS_PER_DAY)

    # the median is the forecast; the 10th and 90th percentiles are kept in the cache
    cached = model._cached()
    np.testing.assert_allclose(cached["q10"], cached["kw"] - 4, atol=0.011)
    np.testing.assert_allclose(cached["q90"], cached["kw"] + 4, atol=0.011)


def test_shorter_contexts_are_padded_with_nan_on_the_left(load, days, tmp_path):
    model = fake_model(tmp_path)
    long_ctx = load.loc[: days[5] - pd.Timedelta("15min")].iloc[-200:]
    short_ctx = load.loc[: days[6] - pd.Timedelta("15min")].iloc[-120:]
    model._predict([long_ctx, short_ctx], [days[5], days[6]])
    batch = model._model.batches[0]
    assert batch.shape == (2, 200)
    assert np.isnan(batch[1, :80]).all()
    np.testing.assert_allclose(batch[1, 80:], short_ctx.to_numpy(dtype=np.float32))


def test_missing_package_skips_cleanly(load, days, tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "timestral", None)  # makes the import fail
    model = FM.TimestralZeroShot("209M", cache_root=tmp_path)
    with caplog.at_level("WARNING", logger="peakshave.foundation"):
        assert model.forecast_days(load, days[:2]) is None
    assert "timestral: package not installed" in caplog.text


def test_sizes_and_names():
    assert FM.TimestralZeroShot("28M").name == "timestral-28m"
    with pytest.raises(ValueError):
        FM.TimestralZeroShot("1B")


def test_build_forecasters_includes_timestral(load):
    fcs = F.build_forecasters(load, include_fm=True, fm_models=("timestral-209m",))
    assert list(fcs) == ["oracle", "seasonal_naive", "profile", "ridge", "timestral-209m"]
    assert isinstance(fcs["timestral-209m"], FM.TimestralZeroShot)
    assert fcs["timestral-209m"].name in F.LABELS
