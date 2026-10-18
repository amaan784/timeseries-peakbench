"""Run every forecaster x policy combination and write the results.

    python -m src.evaluate            # foundation models come from the cache
    python -m src.evaluate --no-fm    # skip TimesFM / TimeGPT

Writes out/results.json plus tidy CSVs. The dashboard, report and README all
read from these, so no number is typed in by hand anywhere downstream.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import config
from .data import INTERVALS_PER_DAY, evaluation_days, fingerprint, is_business_day, load_building, synthetic_load
from .dispatch import POLICIES, POLICY_LABELS, Battery, simulate
from .forecast import FM_MODELS, LABELS, build_forecasters, day_ahead
from .tariff import Tariff

log = logging.getLogger("peakshave.evaluate")

SHIFTS = tuple(range(-4, 5))  # in 15-min intervals, so -60 to +60 minutes
MINUTES_PER_INTERVAL = 15
DEFAULT_FORECASTER = "ridge"

BUILDINGS = {
    "with_spikes": "With equipment spikes",
    "no_spikes": "Smooth (no equipment spikes)",
    "metered": "Metered building",
}


@dataclass
class Experiment:
    actual: pd.Series  # just the evaluated days
    forecasts: dict[str, pd.Series]  # name -> day-ahead forecast on the same index
    meta: dict = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        # forecasters we actually have, in display order
        names = []
        for n in LABELS:
            if n in self.forecasts:
                names.append(n)
        return names


def _days(s: pd.Series) -> np.ndarray:
    # reshape a series to n_days x 96
    return s.to_numpy(dtype=float).reshape(-1, INTERVALS_PER_DAY)


def _nanmean_rows(x: np.ndarray) -> np.ndarray:
    # per-day mean that quietly gives NaN for a day with no usable readings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=1)


def _ape(f: np.ndarray, a: np.ndarray) -> np.ndarray:
    # absolute percentage error, NaN where the meter read zero (real meters
    # have the odd outage, and dividing by zero there means nothing)
    safe = np.where(a > 0, a, np.nan)
    return np.abs(f - a) / safe


def prepare(include_fm: bool = True, load: pd.DataFrame | None = None, meta: dict | None = None,
            fm_models=FM_MODELS) -> Experiment:
    if load is None:
        load, meta = load_building()
    kw = load["kw"]
    days = evaluation_days(kw)
    actual = kw.loc[days[0] : days[-1] + pd.Timedelta(hours=23, minutes=45)]

    forecasts, timings = {}, {}
    for name, fn in build_forecasters(kw, include_fm=include_fm, fm_models=fm_models).items():
        t0 = time.time()
        fc = day_ahead(fn, kw, days)
        if fc is None:  # e.g. TimeGPT with no key and nothing cached
            continue
        forecasts[name] = fc.reindex(actual.index)
        timings[name] = round(time.time() - t0, 2)

    meta = dict(meta or {})
    meta["eval_start"] = str(days[0].date())
    meta["eval_end"] = str(days[-1].date())
    meta["n_days"] = len(days)
    meta["forecast_seconds"] = timings
    return Experiment(actual=actual, forecasts=forecasts, meta=meta)


def experiments(include_fm: bool = True) -> dict[str, Experiment]:
    # The configured building, plus (for synthetic data) the same building
    # with the equipment spikes switched off. Same random draws otherwise,
    # so the difference between the two is purely the spikes.
    load, meta = load_building()
    exps = {}
    if meta.get("synthetic"):
        exps["with_spikes"] = prepare(include_fm, load, meta)
        smooth = synthetic_load(spikes=False)
        meta_ns = dict(meta)
        meta_ns["label"] = meta["label"] + " (equipment spikes removed)"
        meta_ns["fingerprint"] = fingerprint(smooth["kw"])
        exps["no_spikes"] = prepare(include_fm, smooth, meta_ns)
    else:
        exps["metered"] = prepare(include_fm, load, meta)
    return exps


def load_saved(building: str) -> Experiment:
    # What the dashboard uses: actual load + all forecasts from the last run.
    # Means the app starts instantly and doesn't need TimesFM or a Nixtla key.
    results = config.OUT / "results.json"
    main = None
    if results.exists():
        main = json.loads(results.read_text())["meta"]["main_building"]
    real = config.OUT / "real" / f"forecasts_{building}.parquet"  # the UCI meters (src/real_meters.py)
    if building == main:
        path = config.OUT / "forecasts.parquet"
    elif real.exists():
        path = real
    else:
        path = config.OUT / f"forecasts_{building}.parquet"
    if not path.exists():
        return experiments()[building]
    df = pd.read_parquet(path)
    actual = df.pop("actual")
    forecasts = {}
    for c in df.columns:
        forecasts[c] = df[c]
    return Experiment(actual=actual, forecasts=forecasts)


def score(actual: pd.Series, net: pd.Series, tariff: Tariff, before: pd.DataFrame | None = None) -> dict:
    if before is None:
        before = tariff.bill(actual)
    after = tariff.bill(net)
    return {
        "bill_without": before["total"].sum(),
        "bill_with": after["total"].sum(),
        "savings_usd": before["total"].sum() - after["total"].sum(),
        "savings_pct": 100 * (1 - after["total"].sum() / before["total"].sum()),
        "demand_savings_usd": before["demand_cost"].sum() - after["demand_cost"].sum(),
        "energy_savings_usd": before["energy_cost"].sum() - after["energy_cost"].sum(),
        "annual_peak_cut_kw": actual.max() - net.max(),
        "avg_monthly_peak_cut_kw": (before["peak_kw"] - after["peak_kw"]).mean(),
    }


def run_grid(exp: Experiment, battery: Battery, tariff: Tariff, names=None, policies=POLICIES, shifts=SHIFTS):
    """Simulate every forecaster x policy x shift. Returns (grid, sims at shift 0)."""
    names = names or exp.names
    before = tariff.bill(exp.actual)
    biz = is_business_day(exp.actual.index[::INTERVALS_PER_DAY])
    rows, sims = [], {}
    for name in names:
        for policy in policies:
            for shift in shifts:
                sim = simulate(exp.actual, exp.forecasts[name], battery, tariff, policy, shift)
                row = {"forecaster": name, "policy": policy, "shift_intervals": shift}
                row.update(score(exp.actual, sim["net_kw"], tariff, before))

                # Days a meter-driven policy ran out of energy while the meter was
                # still above target. Skipped for window: it plans to empty the
                # battery every day anyway, so "empty" doesn't mean anything there.
                if policy == "window":
                    row["ran_out_days_pct"] = np.nan
                else:
                    empty = _days(sim["soc_kwh"]) <= battery.e_min + 0.5
                    over = _days(sim["net_kw"] - sim["target_kw"]) > 0.5
                    ran_out = (empty & over).any(axis=1)
                    row["ran_out_days_pct"] = 100 * float(ran_out[biz].mean())

                rows.append(row)
                if shift == 0:
                    sims[(name, policy)] = sim

    grid = pd.DataFrame(rows)
    grid["shift_minutes"] = grid["shift_intervals"] * MINUTES_PER_INTERVAL
    oracle = grid[(grid.forecaster == "oracle") & (grid.shift_intervals == 0)].set_index("policy")["savings_usd"]
    grid["pct_of_oracle"] = 100 * grid["savings_usd"] / grid["policy"].map(oracle)
    # Each policy's perfect-foresight result is its own ceiling, but those
    # ceilings can differ a lot (on some real buildings the aggressive daily
    # targets can't be refilled overnight, even with a perfect forecast).
    # So also measure everything against the best ceiling of the three.
    grid["pct_of_best_oracle"] = 100 * grid["savings_usd"] / oracle.max()
    return grid, sims


def daily_table(exp: Experiment, sims: dict, tariff: Tariff) -> pd.DataFrame:
    """Per-day forecast error next to per-day peak shaving, in dollars.

    The demand charge is monthly, so a single day's "value" here is what its
    peak reduction would have been worth if that day had set the month's peak.
    vs_oracle_usd compares that with what perfect foresight managed on the
    same day, which is the cost of that day's forecast error.
    """
    a = _days(exp.actual)
    dates = exp.actual.index[::INTERVALS_PER_DAY]
    biz = is_business_day(dates)
    a_peak = a.max(axis=1)
    a_slot = a.argmax(axis=1)

    oracle_shave = {}
    for p in POLICIES:
        if ("oracle", p) in sims:
            oracle_shave[p] = a_peak - _days(sims[("oracle", p)]["net_kw"]).max(axis=1)

    frames = []
    for (name, policy), sim in sims.items():
        f = _days(exp.forecasts[name])
        f_peak = f.max(axis=1)
        net_peak = _days(sim["net_kw"]).max(axis=1)
        shave = a_peak - net_peak
        frames.append(
            pd.DataFrame(
                {
                    "date": dates.date,
                    "forecaster": name,
                    "policy": policy,
                    "business_day": biz,
                    "actual_peak_kw": a_peak,
                    "forecast_peak_kw": f_peak,
                    "peak_error_kw": f_peak - a_peak,
                    "abs_peak_error_kw": np.abs(f_peak - a_peak),
                    "peak_time_error_min": (f.argmax(axis=1) - a_slot) * MINUTES_PER_INTERVAL,
                    "mape_pct": 100 * _nanmean_rows(_ape(f, a)),
                    "target_kw": _days(sim["day_target_kw"])[:, 0],
                    "net_peak_kw": net_peak,
                    "shave_kw": shave,
                    "shave_value_usd": shave * tariff.demand_charge,
                    "oracle_shave_value_usd": oracle_shave[policy] * tariff.demand_charge,
                    "vs_oracle_usd": (shave - oracle_shave[policy]) * tariff.demand_charge,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def error_slopes(daily: pd.DataFrame) -> pd.DataFrame:
    # Linear fit of dollars lost per day vs |peak error| on business days.
    # The slope is $ lost per kW of forecast error; flatter = more forgiving.
    d = daily[daily.business_day & (daily.forecaster != "oracle")].copy()
    d["lost"] = -d["vs_oracle_usd"]

    groups = []
    for (name, policy), g in d.groupby(["forecaster", "policy"]):
        groups.append((name, policy, g))
    for policy, g in d.groupby("policy"):
        groups.append(("all", policy, g))  # all forecasters pooled

    rows = []
    for name, policy, g in groups:
        slope, intercept = np.polyfit(g["abs_peak_error_kw"], g["lost"], 1)
        r = np.corrcoef(g["abs_peak_error_kw"], g["lost"])[0, 1]
        rows.append({
            "forecaster": name, "policy": policy, "usd_lost_per_kw_error": slope,
            "intercept_usd": intercept, "r": r, "n_days": len(g),
        })
    return pd.DataFrame(rows)


def monthly_table(exp: Experiment, sims: dict, tariff: Tariff) -> pd.DataFrame:
    before = tariff.bill(exp.actual)
    frames = []
    for (name, policy), sim in sims.items():
        after = tariff.bill(sim["net_kw"])
        frames.append(
            pd.DataFrame(
                {
                    "forecaster": name,
                    "policy": policy,
                    "peak_kw_before": before["peak_kw"],
                    "peak_kw_after": after["peak_kw"],
                    "demand_cost_before": before["demand_cost"],
                    "demand_cost_after": after["demand_cost"],
                    "energy_cost_before": before["energy_cost"],
                    "energy_cost_after": after["energy_cost"],
                    "total_before": before["total"],
                    "total_after": after["total"],
                }
            )
        )
    out = pd.concat(frames)
    out.index = out.index.astype(str)
    return out.reset_index()


def accuracy_table(exp: Experiment) -> pd.DataFrame:
    a = _days(exp.actual)
    biz = is_business_day(exp.actual.index[::INTERVALS_PER_DAY])
    rows = []
    for name in exp.names:
        f = _days(exp.forecasts[name])
        ape = _ape(f, a)
        pk_err = f.max(axis=1) - a.max(axis=1)
        slot_err = np.abs(f.argmax(axis=1) - a.argmax(axis=1)) * MINUTES_PER_INTERVAL
        rows.append(
            {
                "forecaster": name,
                "label": LABELS[name],
                "mape_pct": 100 * np.nanmean(ape),
                "business_day_mape_pct": 100 * np.nanmean(ape[biz]),
                "rmse_kw": float(np.sqrt(((f - a) ** 2).mean())),
                "peak_mae_kw": np.abs(pk_err[biz]).mean(),
                "peak_bias_kw": pk_err[biz].mean(),
                "peak_time_mae_min": slot_err[biz].mean(),
            }
        )
    return pd.DataFrame(rows)


def battery_sensitivity(exp: Experiment, battery: Battery, tariff: Tariff, base_grid: pd.DataFrame) -> pd.DataFrame:
    # does the ranking hold up with a battery half or twice the size?
    half = replace(battery, capacity_kwh=battery.capacity_kwh / 2, power_kw=battery.power_kw / 2)
    double = replace(battery, capacity_kwh=battery.capacity_kwh * 2, power_kw=battery.power_kw * 2)
    cases = [("Half-size battery", half), ("Base battery", battery), ("Double-size battery", double)]

    rows = []
    for label, bat in cases:
        if bat == battery:
            grid = base_grid[base_grid.shift_intervals == 0]  # already simulated
        else:
            grid, _ = run_grid(exp, bat, tariff, shifts=(0,))
        for r in grid.itertuples():
            rows.append({
                "case": label, "capacity_kwh": bat.capacity_kwh, "power_kw": bat.power_kw,
                "forecaster": r.forecaster, "policy": r.policy,
                "savings_usd": r.savings_usd, "pct_of_oracle": r.pct_of_oracle,
            })
    return pd.DataFrame(rows)


def evaluate_building(exp: Experiment, battery: Battery, tariff: Tariff, full: bool = True):
    # full=False is for the spike-free twin: no timing sweep or battery sizes,
    # nothing downstream looks at those for it
    shifts = SHIFTS if full else (0,)
    grid, sims = run_grid(exp, battery, tariff, shifts=shifts)

    # With a perfect forecast, threshold and window should come out identical.
    # If they don't, something is broken in the simulator.
    o = grid[(grid.forecaster == "oracle") & (grid.shift_intervals == 0)].set_index("policy")["savings_usd"]
    if not np.isclose(o["threshold"], o["window"], rtol=1e-6):
        raise AssertionError(f"oracle threshold/window disagree: {o.to_dict()}")

    summary = grid[grid.shift_intervals == 0].drop(columns=["shift_intervals", "shift_minutes"]).copy()
    summary.insert(1, "label", summary["forecaster"].map(LABELS))
    summary.insert(3, "policy_label", summary["policy"].map(POLICY_LABELS))
    daily = daily_table(exp, sims, tariff)
    tables = {
        "summary": summary,
        "timing_sweep": grid[["forecaster", "policy", "shift_intervals", "shift_minutes", "savings_usd", "pct_of_oracle"]],
        "daily": daily,
        "error_slopes": error_slopes(daily),
        "monthly": monthly_table(exp, sims, tariff),
        "forecast_accuracy": accuracy_table(exp),
    }
    if full:
        tables["battery_sensitivity"] = battery_sensitivity(exp, battery, tariff, grid)
    return tables, sims


def headline(summary: pd.DataFrame, sweep: pd.DataFrame, slopes: pd.DataFrame, accuracy: pd.DataFrame) -> dict:
    # the few numbers the README and report open with
    real = summary[summary.forecaster != "oracle"]
    pct = real.pivot(index="forecaster", columns="policy", values="pct_of_oracle")
    sav = real.pivot(index="forecaster", columns="policy", values="savings_usd")
    oracle = summary[summary.forecaster == "oracle"].set_index("policy")["savings_usd"]
    acc = accuracy.set_index("forecaster")

    timing = {}
    for f in sweep.forecaster.unique():
        timing[f] = {}
        for p in POLICIES:
            timing[f][p] = {}
            for m in sorted(sweep.shift_minutes.unique()):
                row = sweep[(sweep.forecaster == f) & (sweep.policy == p) & (sweep.shift_minutes == m)]
                timing[f][p][f"{m:+d}"] = float(row["savings_usd"].iloc[0])

    pct_of_oracle, savings, fc_spread, fc_spread_pct = {}, {}, {}, {}
    for p in POLICIES:
        pct_of_oracle[p] = pct[p].round(2).to_dict()
        savings[p] = sav[p].round(0).to_dict()
        # what picking the best forecaster is worth if the policy is fixed
        fc_spread[p] = float(sav[p].max() - sav[p].min())
        fc_spread_pct[p] = [float(pct[p].min()), float(pct[p].max())]

    pooled = slopes[slopes.forecaster == "all"].set_index("policy")["usd_lost_per_kw_error"]
    return {
        "oracle_savings_usd": oracle.round(0).to_dict(),
        "pct_of_oracle": pct_of_oracle,
        "savings_usd": savings,
        "best_policy": pct.mean().idxmax(),
        "forecaster_spread_usd": fc_spread,
        "forecaster_spread_pct": fc_spread_pct,
        # and what picking the best policy is worth if the forecaster is fixed
        "policy_spread_usd": (sav.max(axis=1) - sav.min(axis=1)).round(0).to_dict(),
        "best_forecaster_by_peak_mae": str(acc.drop(index="oracle")["peak_mae_kw"].idxmin()),
        "timing": timing,
        "usd_lost_per_kw_error_pooled": pooled.round(2).to_dict(),
    }


def _records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = df.copy()
    for c in out.select_dtypes("number").columns:
        out[c] = out[c].round(4)
    return json.loads(out.to_json(orient="records", date_format="iso"))


def run(include_fm: bool = True, battery: Battery | None = None, tariff: Tariff | None = None) -> dict:
    t0 = time.time()
    battery = battery or Battery()
    tariff = tariff or Tariff()
    config.OUT.mkdir(parents=True, exist_ok=True)

    exps = experiments(include_fm)
    main_key = next(iter(exps))
    tables = {}
    per_building = {}

    for key, exp in exps.items():
        res, sims = evaluate_building(exp, battery, tariff, full=(key == main_key))
        for name, df in res.items():
            tables.setdefault(name, []).append(df.assign(building=key))

        # the dashboard loads actual + every forecast from this file
        suffix = "" if key == main_key else f"_{key}"
        pd.DataFrame({"actual": exp.actual, **exp.forecasts}).to_parquet(config.OUT / f"forecasts{suffix}.parquet")

        per_building[key] = {
            "label": BUILDINGS[key],
            "data_label": exp.meta.get("label"),
            "fingerprint": exp.meta.get("fingerprint"),
            "forecast_seconds": exp.meta.get("forecast_seconds"),
            "annual_peak_timestamp": str(exp.actual.idxmax()),
            "annual_peak_kw": float(exp.actual.max()),
            "baseline_bill_usd": float(tariff.bill(exp.actual)["total"].sum()),
            "headline": headline(res["summary"], res["timing_sweep"], res["error_slopes"], res["forecast_accuracy"]),
        }

        if key == main_key:
            # the year's peak day under each policy, for the report's day chart
            peak_day = exp.actual.idxmax().normalize()
            day = slice(peak_day, peak_day + pd.Timedelta(hours=23, minutes=45))
            parts = []
            for p in POLICIES:
                parts.append(sims[(DEFAULT_FORECASTER, p)].loc[day].assign(policy=p))
            pd.concat(parts).to_csv(config.OUT / "peak_day.csv", index_label="timestamp")

    merged = {}
    for name, dfs in tables.items():
        df = pd.concat(dfs, ignore_index=True)
        cols = ["building"]
        for c in df.columns:
            if c != "building":
                cols.append(c)
        merged[name] = df[cols]
        merged[name].round(4).to_csv(config.OUT / f"{name}.csv", index=False)

    main = exps[main_key]
    forecaster_names = {}
    for n in main.names:
        forecaster_names[n] = LABELS[n]
    shift_minutes = []
    for s in SHIFTS:
        shift_minutes.append(s * MINUTES_PER_INTERVAL)

    results = {
        "meta": {
            "label": main.meta.get("label"),
            "synthetic": bool(main.meta.get("synthetic")),
            "source": main.meta.get("source"),
            "eval_start": main.meta["eval_start"],
            "eval_end": main.meta["eval_end"],
            "n_days": main.meta["n_days"],
            "main_building": main_key,
            "battery": battery.to_dict(),
            "tariff": tariff.to_dict(),
            "forecasters": forecaster_names,
            "policies": POLICY_LABELS,
            "default_forecaster": DEFAULT_FORECASTER,
            "fm_context_days": config.FM_CONTEXT_DAYS,
            "shift_minutes": shift_minutes,
            "runtime_seconds": None,
        },
        "buildings": per_building,
    }
    # per-day rows are big, they only go in out/daily.csv
    for name, df in merged.items():
        if name != "daily":
            results[name] = _records(df)

    results["meta"]["runtime_seconds"] = round(time.time() - t0, 1)
    with open(config.OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    log.info("wrote %s (+ CSVs) in %.1fs", config.OUT / "results.json", time.time() - t0)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-fm", action="store_true", help="skip TimesFM and TimeGPT")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    res = run(include_fm=not args.no_fm)

    print(f"\n{res['meta']['label']}")
    print(f"Evaluated {res['meta']['eval_start']} .. {res['meta']['eval_end']}  ({res['meta']['runtime_seconds']}s)")
    for b in res["buildings"].values():
        h = b["headline"]
        print(f"\n[{b['label']}]")
        for pol in POLICIES:
            parts = []
            for k, v in h["pct_of_oracle"][pol].items():
                parts.append(f"{k} {v:.0f}%")
            print(f"  {POLICY_LABELS[pol]:20s} perfect foresight ${h['oracle_savings_usd'][pol]:>8,.0f}/yr | captured: " + ", ".join(parts))


if __name__ == "__main__":
    main()
