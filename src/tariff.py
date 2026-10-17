"""A made-up but realistic Massachusetts C&I tariff.

Demand charge in the low $20s per kW-month and an 8am-9pm weekday on-peak
window, roughly what you'd see on a general-service TOU rate in MA. This is
NOT a real filed tariff: no customer charge, no ratchet, no supply split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import DT_HOURS, holiday_calendar, is_business_day

# bill() runs thousands of times on the same index, so the calendar bits are cached
_CACHE: dict = {}


@dataclass(frozen=True)
class Tariff:
    demand_charge: float = 22.0  # $/kW-month, applied to the month's highest 15-min interval
    on_peak_rate: float = 0.19  # $/kWh
    off_peak_rate: float = 0.13  # $/kWh
    on_peak_start_hour: int = 8  # on-peak is [start, end) on business days
    on_peak_end_hour: int = 21
    name: str = "Illustrative MA C&I time-of-use rate"

    def is_on_peak(self, index: pd.DatetimeIndex) -> np.ndarray:
        key = (holiday_calendar(), self.on_peak_start_hour, self.on_peak_end_hour, index[0], index[-1], len(index))
        if key not in _CACHE:
            hour = index.hour
            mask = is_business_day(index) & (hour >= self.on_peak_start_hour) & (hour < self.on_peak_end_hour)
            _CACHE[key] = np.asarray(mask)
        return _CACHE[key]

    def energy_rate(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.where(self.is_on_peak(index), self.on_peak_rate, self.off_peak_rate)

    def _months(self, idx: pd.DatetimeIndex):
        key = ("months", holiday_calendar(), self.on_peak_rate, self.off_peak_rate, self.on_peak_start_hour,
               self.on_peak_end_hour, idx[0], idx[-1], len(idx))
        if key not in _CACHE:
            code = idx.year.to_numpy() * 12 + idx.month.to_numpy() - 1
            starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])  # assumes the index is sorted
            months = pd.PeriodIndex(idx[starts], freq="M", name="month")
            _CACHE[key] = (starts, months, self.energy_rate(idx))
        return _CACHE[key]

    def bill(self, kw: pd.Series) -> pd.DataFrame:
        """Monthly bill with columns peak_kw, kwh, energy_cost, demand_cost, total."""
        v = kw.to_numpy(dtype=float)
        starts, months, rate = self._months(kw.index)
        kwh = v * DT_HOURS
        monthly = pd.DataFrame(
            {
                "peak_kw": np.maximum.reduceat(v, starts),
                "kwh": np.add.reduceat(kwh, starts),
                "energy_cost": np.add.reduceat(kwh * rate, starts),
            },
            index=months,
        )
        monthly["demand_cost"] = monthly["peak_kw"] * self.demand_charge
        monthly["total"] = monthly["energy_cost"] + monthly["demand_cost"]
        return monthly

    def to_dict(self) -> dict:
        return asdict(self)
