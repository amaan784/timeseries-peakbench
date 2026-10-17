"""The foundation-model plumbing: batching, the cache, and skipping without a key.

These use a fake model so they run without TimesFM installed or a Nixtla key.
"""

import numpy as np
import pandas as pd

from src import forecast as F
from src import foundation as FM


def poison_from(series: pd.Series, day: pd.Timestamp) -> pd.Series:
    out = series.copy()
    out[out.index >= day] = 1e7
    return out


class EchoModel(FM.ZeroShotForecaster):
    # fake foundation model: repeats the last day of context and remembers what it was given
    name = "echo-test"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.context_ends = []

    def _predict(self, contexts, days):
        out = []
        for ctx, d in zip(contexts, days):
            self.context_ends.append((ctx.index[-1], d))
            last_day = ctx.to_numpy()[-F.INTERVALS_PER_DAY :]
            out.append(pd.DataFrame({"kw": last_day}, index=F.target_index_for(d)))
        return out


def test_batching_cannot_see_the_future(load, days, tmp_path):
    targets = days[[2, 9, 30]]
    model = EchoModel(cache_root=tmp_path)
    clean = model.forecast_days(load, targets)
    assert len(model.context_ends) == len(targets)
    for end, day in model.context_ends:
        assert end < day

    # same poisoning trick as test_forecast.py, through the batched path
    for day in targets:
        fresh = EchoModel(cache_root=tmp_path / f"p{day:%m%d}")
        poisoned = fresh.forecast_days(poison_from(load, day), pd.DatetimeIndex([day]))
        np.testing.assert_allclose(poisoned.to_numpy(), clean.loc[str(day.date())].to_numpy())


def test_cache_is_one_file_per_model_and_data(load, days, tmp_path):
    model = EchoModel(cache_root=tmp_path)
    model.forecast_days(load, days[:2])

    files = []
    for p in tmp_path.rglob("*"):
        if p.is_file():
            files.append(p.relative_to(tmp_path).as_posix())
    assert files == [f"echo-test/{model.fingerprint}.parquet"]
    cached = pd.read_parquet(tmp_path / files[0])
    assert len(cached) == 2 * F.INTERVALS_PER_DAY
    assert sorted(cached.index.normalize().unique()) == list(days[:2])

    # a second run comes straight from the cache...
    again = EchoModel(cache_root=tmp_path)
    again.forecast_days(load, days[:2])
    assert again.context_ends == []

    # ...and asking for one more day only runs that day
    more = EchoModel(cache_root=tmp_path)
    more.forecast_days(load, days[:3])
    assert len(more.context_ends) == 1


def test_timegpt_without_key_skips_cleanly(load, days, tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("NIXTLA_API_KEY", raising=False)
    monkeypatch.setattr(FM.config, "load_dotenv", lambda *a, **k: None)
    model = FM.TimeGPTZeroShot(cache_root=tmp_path)
    with caplog.at_level("WARNING", logger="peakshave.foundation"):
        assert model.forecast_days(load, days[:3]) is None
    assert "NIXTLA_API_KEY not set" in caplog.text


def test_build_forecasters_picks_the_requested_models(load):
    fcs = F.build_forecasters(load, include_fm=True, fm_models=("timesfm-3.0",))
    assert list(fcs) == ["oracle", "seasonal_naive", "profile", "ridge", "timesfm-3.0"]
    assert isinstance(fcs["timesfm-3.0"], FM.TimesFMZeroShot)
