import itertools

import numpy as np
import pytest

from src.data import DT_HOURS, INTERVALS_PER_DAY
from src.dispatch import POLICIES, Battery, daily_target, simulate

SHIFTS = (-4, 0, 3)
# every policy against every kind of bad forecast, at a few timing shifts
CASES = list(itertools.product(POLICIES, ["perfect", "too_low", "too_high", "noisy", "flat"], SHIFTS))


def case_id(case):
    policy, forecast, shift = case
    return f"{policy}-{forecast}-{shift}"


@pytest.fixture(scope="module")
def sims(actual, bad_forecasts, battery, tariff):
    out = {}
    for policy, fc, shift in CASES:
        out[(policy, fc, shift)] = simulate(actual, bad_forecasts[fc], battery, tariff, policy, shift)
    return out


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_soc_never_leaves_bounds(sims, battery, case):
    soc = sims[case]["soc_kwh"]
    assert soc.min() >= battery.e_min - 1e-6
    assert soc.max() <= battery.e_max + 1e-6


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_energy_balance(sims, battery, case):
    sim = sims[case]
    b = sim["battery_kw"]
    discharged = b.clip(lower=0).sum() * DT_HOURS  # kWh into the building
    charged = (-b).clip(lower=0).sum() * DT_HOURS  # kWh pulled from the grid
    eta = battery.one_way_efficiency
    delta_soc = sim["soc_kwh"].iloc[-1] - battery.e_max  # starts the year full

    # exact: stored energy changes by charge * eta - discharge / eta
    assert delta_soc == pytest.approx(charged * eta - discharged / eta, abs=1e-6)
    # and the usual rule of thumb, discharged ~ charged * RTE, give or take
    # whatever is left in the battery at the end
    assert discharged == pytest.approx(charged * battery.round_trip_efficiency, abs=battery.capacity_kwh)
    if charged > 0:
        assert abs(discharged - charged * battery.round_trip_efficiency) <= eta * abs(delta_soc) + 1e-6


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_charging_never_raises_net_load_above_days_target(sims, tariff, case):
    sim = sims[case]
    charging = sim["battery_kw"] < 0
    if case[1] in ("perfect", "noisy", "too_low"):
        assert charging.any()  # these should all empty and refill the battery at some point
    assert (sim.loc[charging, "net_kw"] <= sim.loc[charging, "day_target_kw"] + 1e-6).all()

    # day target is one number per day, and nothing defends a level below it
    assert (sim["day_target_kw"].groupby(sim.index.normalize()).nunique() == 1).all()
    assert (sim["target_kw"] >= sim["day_target_kw"] - 1e-6).all()

    # and charging is off-peak only
    on_peak = tariff.is_on_peak(sim.index)
    assert not (charging & on_peak).any()


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_power_limit_and_no_export(sims, battery, case):
    sim = sims[case]
    assert sim["battery_kw"].abs().max() <= battery.power_kw + 1e-6
    assert sim["net_kw"].min() >= -1e-6


def test_window_and_threshold_coincide_with_perfect_forecast(sims):
    # the whole experiment leans on this: with a perfect forecast the two
    # policies should do exactly the same thing
    a = sims[("threshold", "perfect", 0)]["net_kw"]
    b = sims[("window", "perfect", 0)]["net_kw"]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-6)


def test_threshold_ignores_timing_error_with_perfect_shape(sims, tariff):
    # shifting the forecast barely changes the daily target, so threshold shouldn't care
    base = sims[("threshold", "perfect", 0)]
    shifted = sims[("threshold", "perfect", 3)]
    base_bill = tariff.bill(base["net_kw"])["total"].sum()
    shifted_bill = tariff.bill(shifted["net_kw"])["total"].sum()
    assert shifted_bill == pytest.approx(base_bill, rel=0.01)


def test_daily_target_bisection_is_tight():
    battery = Battery(capacity_kwh=100, power_kw=1000, round_trip_efficiency=1.0, soc_min=0.0, soc_max=1.0)
    day = np.full(INTERVALS_PER_DAY, 100.0)
    day[40:48] = 300.0  # two hours at 300 kW
    # 100 kWh over 2 hours should take 50 kW off the top
    assert daily_target(day, battery) == pytest.approx(250.0, abs=0.1)

    # lots of energy but only 80 kW of inverter: power is the limit
    small_inverter = Battery(capacity_kwh=10_000, power_kw=80, round_trip_efficiency=1.0, soc_min=0, soc_max=1)
    assert daily_target(day, small_inverter) == pytest.approx(220.0, abs=1e-6)


def test_ratchet_never_defends_below_month_to_date_peak(actual, bad_forecasts, battery, tariff):
    sim = simulate(actual, bad_forecasts["noisy"], battery, tariff, "ratchet")
    month = sim.index.to_period("M")
    peak_so_far = sim["net_kw"].groupby(month).transform(lambda s: s.cummax().shift(fill_value=0.0))
    assert (sim["target_kw"] >= peak_so_far - 1e-6).all()


def test_simulate_rejects_partial_days(actual, battery, tariff):
    with pytest.raises(ValueError):
        simulate(actual.iloc[:-1], actual.iloc[:-1], battery, tariff)
