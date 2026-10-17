import numpy as np
import pandas as pd
import pytest

from src.data import DT_HOURS
from src.tariff import Tariff


def test_demand_charge_is_max_interval_times_rate(actual, tariff):
    bill = tariff.bill(actual)
    expected = actual.groupby(actual.index.to_period("M")).max() * tariff.demand_charge
    np.testing.assert_allclose(bill["demand_cost"].to_numpy(), expected.to_numpy())
    np.testing.assert_allclose(bill["peak_kw"].to_numpy(), actual.groupby(actual.index.to_period("M")).max().to_numpy())


def test_single_spike_sets_the_demand_charge():
    idx = pd.date_range("2025-03-01", "2025-03-31 23:45", freq="15min")
    kw = pd.Series(100.0, index=idx)
    kw.iloc[1234] = 437.5
    t = Tariff(demand_charge=20.0)
    bill = t.bill(kw)
    assert len(bill) == 1
    assert bill["peak_kw"].iloc[0] == 437.5
    assert bill["demand_cost"].iloc[0] == pytest.approx(437.5 * 20.0)


def test_energy_cost_uses_tou_rates():
    # A Saturday and a Monday: only Monday 08:00-21:00 is on-peak.
    idx = pd.date_range("2025-03-01", "2025-03-03 23:45", freq="15min")
    kw = pd.Series(100.0, index=idx)
    t = Tariff(on_peak_rate=0.30, off_peak_rate=0.10)
    on_peak_hours = 13
    kwh_total = 100.0 * DT_HOURS * len(idx)
    kwh_on = 100.0 * on_peak_hours
    expected = kwh_on * 0.30 + (kwh_total - kwh_on) * 0.10
    bill = t.bill(kw)
    assert bill["energy_cost"].sum() == pytest.approx(expected)
    assert bill["kwh"].sum() == pytest.approx(kwh_total)
    assert (bill["total"] == bill["energy_cost"] + bill["demand_cost"]).all()


def test_bill_splits_months(actual, tariff):
    bill = tariff.bill(actual)
    assert list(bill.index.astype(str)) == sorted(set(actual.index.strftime("%Y-%m")))
