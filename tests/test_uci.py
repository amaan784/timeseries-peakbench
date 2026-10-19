import numpy as np
import pandas as pd

from src.uci import _last_sunday, clean


def raw_like(start, end, value=10.0):
    # a tiny file in the UCI layout: end-of-interval labels, one column per client
    idx = pd.date_range(start, end, freq="15min")
    return pd.DataFrame({"MT_001": value, "MT_002": 2 * value}, index=idx.strftime("%Y-%m-%d %H:%M:%S"))


def test_last_sunday():
    assert _last_sunday(2014, 3) == pd.Timestamp("2014-03-30")
    assert _last_sunday(2014, 10) == pd.Timestamp("2014-10-26")


def test_spring_dst_zeros_are_filled_in():
    raw = raw_like("2014-03-29 00:15", "2014-04-01 00:00")
    zeros = pd.date_range("2014-03-30 01:00", periods=4, freq="15min").strftime("%Y-%m-%d %H:%M:%S")
    raw.loc[zeros] = 0.0
    out = clean(raw, "2014-03-29", "2014-04-01 00:00")
    assert (out > 0).all().all()
    np.testing.assert_allclose(out["MT_001"].to_numpy(), 10.0)


def test_autumn_dst_hour_is_halved():
    raw = raw_like("2014-10-25 00:15", "2014-10-28 00:00")
    doubled = pd.date_range("2014-10-26 01:00", periods=4, freq="15min").strftime("%Y-%m-%d %H:%M:%S")
    raw.loc[doubled] = raw.loc[doubled] * 2
    out = clean(raw, "2014-10-25", "2014-10-28 00:00")
    np.testing.assert_allclose(out["MT_002"].to_numpy(), 20.0)


def test_timestamps_move_to_interval_start():
    raw = raw_like("2014-05-01 00:15", "2014-05-03 00:00")
    out = clean(raw, "2014-05-01", "2014-05-03 00:00")
    assert out.index[0] == pd.Timestamp("2014-05-01 00:00")
    assert out.index[-1] == pd.Timestamp("2014-05-02 23:45")
    assert len(out) == 2 * 96
