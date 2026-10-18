"""Battery dispatch for demand-charge shaving.

Every policy gets the same forecast and computes the same daily kW target from
it (the lowest target the battery could hold if the forecast were right).
What differs is what they do with it:

  window     follows a schedule made from the forecast: discharge
             (forecast - target) in the intervals where the forecast is above
             the target, no matter what the meter actually says
  threshold  discharges whenever the real meter reading is above the target
  ratchet    like threshold, but never bothers shaving below a peak that has
             already been set this month (the bill is the monthly max, so
             there's no point)

If the forecast is perfect, window and threshold do exactly the same thing.
So any difference between them is down to how they cope with forecast error.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import DT_HOURS, INTERVALS_PER_DAY, holiday_calendar
from .tariff import Tariff

POLICIES = ("threshold", "ratchet", "window")
POLICY_LABELS = {
    "threshold": "Threshold",
    "ratchet": "Threshold + ratchet",
    "window": "Window",
}

# Charge at 1.5x the rate that would just finish by the next on-peak window,
# so there's some slack if headroom under the target is tight for a while.
CHARGE_PACE = 1.5


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float = 500.0
    power_kw: float = 250.0
    round_trip_efficiency: float = 0.88
    soc_min: float = 0.10  # as a fraction of capacity
    soc_max: float = 0.95

    @property
    def one_way_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def e_min(self) -> float:
        return self.capacity_kwh * self.soc_min

    @property
    def e_max(self) -> float:
        return self.capacity_kwh * self.soc_max

    @property
    def usable_kwh(self) -> float:
        return self.e_max - self.e_min

    @property
    def deliverable_kwh(self) -> float:
        # what actually reaches the building from a full battery, after losses
        return self.usable_kwh * self.one_way_efficiency

    def to_dict(self) -> dict:
        return asdict(self)


def daily_targets(forecast_days: np.ndarray, battery: Battery, iters: int = 16) -> np.ndarray:
    """Lowest target each day the battery could hold if the forecast were exact.

    forecast_days is n_days x 96. Bisection on the target, all days at once.
    A target works if the forecast energy above it fits in the battery and the
    forecast peak is within power_kw of it. 16 iterations gets well under 0.01 kW.
    """
    f = np.atleast_2d(np.asarray(forecast_days, dtype=float))
    peak = np.nanmax(f, axis=1)
    budget = battery.deliverable_kwh

    def energy_above(t):
        return np.clip(f - t[:, None], 0, None).sum(axis=1) * DT_HOURS

    lo = peak - battery.power_kw  # can't shave deeper than the inverter allows
    hi = peak.copy()
    done = energy_above(lo) <= budget
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = energy_above(mid) <= budget
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return np.where(done, peak - battery.power_kw, hi)


def daily_target(day_forecast: np.ndarray, battery: Battery) -> float:
    return float(daily_targets(np.asarray(day_forecast)[None, :], battery)[0])


_CALENDAR: dict = {}


def _calendar(idx: pd.DatetimeIndex, tariff: Tariff):
    # Off-peak mask, month number, and how many off-peak intervals are left
    # before on-peak starts again. The last one is for pacing the charging:
    # without it the battery recharges flat out at 9pm and draws a big
    # rebound spike. Same for every simulation on this index, so cache it.
    key = (holiday_calendar(), idx[0], idx[-1], len(idx), tariff.on_peak_start_hour, tariff.on_peak_end_hour)
    if key not in _CALENDAR:
        off_peak = ~tariff.is_on_peak(idx)
        on_idx = np.flatnonzero(~off_peak)
        i = np.arange(len(idx))
        if len(on_idx):
            pos = np.searchsorted(on_idx, i)
            next_on = np.where(pos < len(on_idx), on_idx[np.minimum(pos, len(on_idx) - 1)], len(idx))
        else:
            next_on = np.full(len(idx), len(idx))
        _CALENDAR[key] = (off_peak, idx.month.to_numpy(), np.maximum(next_on - i, 1))
    return _CALENDAR[key]


def shift_forecast(forecast: pd.Series, intervals: int) -> pd.Series:
    # timing error knob: +k means the forecast runs k intervals late
    if intervals == 0:
        return forecast
    return forecast.shift(intervals).bfill().ffill()


def simulate(
    actual: pd.Series,
    forecast: pd.Series,
    battery: Battery,
    tariff: Tariff,
    policy: str = "threshold",
    window_shift_intervals: int = 0,
) -> pd.DataFrame:
    """Simulate one policy over whole days of 15-min data.

    Rules: SOC stays within limits, battery kW within power_kw, sqrt(RTE)
    loss each way, no export to the grid, and charging only off-peak, never
    above the day's target (so charging can't create a new peak).

    Returns a frame with actual_kw, forecast_kw (after any shift),
    day_target_kw, target_kw (what's actually being defended, which is higher
    than the day target only under the ratchet), battery_kw (+ discharge,
    - charge), net_kw and soc_kwh.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if len(actual) % INTERVALS_PER_DAY:
        raise ValueError("simulate() needs whole days of 15-minute data")

    idx = actual.index
    fc = forecast if forecast.index.equals(idx) else forecast.reindex(idx)
    fc = shift_forecast(fc, window_shift_intervals)
    a = actual.to_numpy(dtype=float)
    f = fc.to_numpy(dtype=float)
    day_target = np.repeat(daily_targets(f.reshape(-1, INTERVALS_PER_DAY), battery), INTERVALS_PER_DAY)
    plan = np.clip(f - day_target, 0, battery.power_kw)  # what the window policy will discharge
    off_peak, month, left = _calendar(idx, tariff)

    closed_loop = policy != "window"
    ratchet = policy == "ratchet"
    p_max = battery.power_kw
    e_min, e_max = battery.e_min, battery.e_max
    k_out = battery.one_way_efficiency / DT_HOURS  # stored kWh -> kW delivered over one interval
    k_in = 1.0 / (battery.one_way_efficiency * DT_HOURS)  # free kWh -> kW of charging that fills it

    n = len(a)
    out_b = [0.0] * n
    out_soc = [0.0] * n
    out_t = [0.0] * n
    soc = e_max  # battery starts the year full
    mtd = 0.0  # highest net load so far this month
    cur_month = month[0]

    # plain python floats here, numpy scalar access in a loop this long is slow
    a_l = a.tolist()
    t_l = day_target.tolist()
    p_l = plan.tolist()
    off_l = off_peak.tolist()
    m_l = month.tolist()
    left_l = left.tolist()

    for i in range(n):
        ai = a_l[i]
        td = t_l[i]
        ti = td
        if ratchet:
            if m_l[i] != cur_month:
                cur_month = m_l[i]
                mtd = 0.0
            if mtd > ti:
                ti = mtd

        if closed_loop:
            want = ai - ti
        else:
            want = p_l[i]

        b = 0.0
        if want > 0:
            # discharge, limited by power, by the load itself (no export) and by what's left
            b = want
            if b > p_max:
                b = p_max
            if b > ai:
                b = ai
            avail = (soc - e_min) * k_out
            if b > avail:
                b = avail
            if b > 0:
                soc -= b / k_out
            else:
                b = 0.0
        elif off_l[i]:
            # charge, but never push net load above the day's target
            c = td - ai
            if c > p_max:
                c = p_max
            room = (e_max - soc) * k_in
            if c > room:
                c = room
            paced = room * CHARGE_PACE / left_l[i]
            if c > paced:
                c = paced
            if c > 0:
                soc += c / k_in
                b = -c

        # guard against float drift
        if soc < e_min:
            soc = e_min
        elif soc > e_max:
            soc = e_max

        net = ai - b
        if net > mtd:
            mtd = net
        out_b[i] = b
        out_soc[i] = soc
        out_t[i] = ti

    b_arr = np.asarray(out_b)
    return pd.DataFrame(
        {
            "actual_kw": a,
            "forecast_kw": f,
            "day_target_kw": day_target,
            "target_kw": np.asarray(out_t),
            "battery_kw": b_arr,
            "net_kw": a - b_arr,
            "soc_kwh": np.asarray(out_soc),
        },
        index=idx,
    )
