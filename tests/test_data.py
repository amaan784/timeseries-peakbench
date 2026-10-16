import numpy as np
import pandas as pd

from src.data import INTERVALS_PER_DAY, is_business_day, load_interval_csv, synthetic_load


def test_synthetic_shape_and_story():
    df = synthetic_load(start="2025-01-01", end="2025-12-31")
    assert list(df.columns) == ["kw"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 365 * INTERVALS_PER_DAY
    assert (df["kw"] > 0).all()
    # Summer cooling drives the annual peak; weekends run lighter than weekdays.
    assert df["kw"].idxmax().month in (6, 7, 8, 9)
    biz = is_business_day(df.index)
    assert df.loc[biz, "kw"].mean() > 1.3 * df.loc[~biz, "kw"].mean()


def test_spikes_only_add_load_on_business_days():
    a = synthetic_load(start="2025-03-01", end="2025-05-31")["kw"]
    b = synthetic_load(start="2025-03-01", end="2025-05-31", spikes=False)["kw"]
    diff = a - b
    assert (diff >= -1e-9).all()
    assert (diff[~is_business_day(diff.index)] == 0).all()
    assert (diff > 50).sum() > 0


def test_load_interval_csv_matches_synthetic_shape(tmp_path):
    # Hourly kWh export with a tz-aware timestamp column, a DST transition and a gap.
    idx = pd.date_range("2025-03-01", "2025-03-20 23:00", freq="h", tz="America/New_York")
    values = np.linspace(100, 200, len(idx))
    df = pd.DataFrame({"Interval Start": idx.astype(str), "Usage (kWh)": values})
    df = df.drop(index=50)  # a missing hour, interpolated
    path = tmp_path / "meter.csv"
    df.to_csv(path, index=False)

    out = load_interval_csv(path, tz="America/New_York")
    assert list(out.columns) == ["kw"]
    assert pd.infer_freq(out.index[:500]) == "15min"
    assert out.index.tz is None
    assert len(out) % INTERVALS_PER_DAY == 0
    assert out["kw"].notna().all()
    # Hourly kWh becomes kW held across the hour's four 15-minute slots.
    assert out["kw"].iloc[0] == out["kw"].iloc[3] == values[0]


def test_load_interval_csv_keeps_a_complete_last_day(tmp_path):
    # data running right up to 23:45 on the last day shouldn't lose that day
    idx = pd.date_range("2025-01-01", "2025-01-10 23:45", freq="15min")
    pd.DataFrame({"timestamp": idx, "kw": 100.0}).to_csv(tmp_path / "meter.csv", index=False)
    out = load_interval_csv(tmp_path / "meter.csv")
    assert len(out) == 10 * INTERVALS_PER_DAY
    assert out.index[-1] == pd.Timestamp("2025-01-10 23:45")


def test_portugal_holidays_switch():
    from src.data import set_holidays

    set_holidays("PT")
    try:
        # 25 April and Good Friday 2014 are holidays in Portugal, not in the US
        days = pd.DatetimeIndex(["2014-04-25", "2014-04-18", "2014-04-22"])
        assert list(is_business_day(days)) == [False, False, True]
    finally:
        set_holidays("US")
    assert list(is_business_day(pd.DatetimeIndex(["2014-04-25"]))) == [True]
