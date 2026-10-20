"""The same experiment, on real meters: the six UCI buildings in data/uci/.

    python -m src.real_meters

Same forecasters (TimesFM 3.0 is the only foundation model here; 2.5 would
double the runtime and the synthetic run already compares the two), same
three policies, same illustrative tariff, Portuguese holidays. Each battery
is sized to its building: 30% of the 2013 peak in kW, two hours of storage,
about the same ratio as the synthetic case. Sizing uses 2013 so nothing from
the evaluated year (2014) leaks into it.

Writes out/real/results.json plus tidy CSVs, one row group per building.
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd

from . import config
from .data import fingerprint, load_interval_csv, set_holidays
from .dispatch import POLICIES, POLICY_LABELS, Battery
from .evaluate import SHIFTS, _records, evaluate_building, headline, prepare, run_grid
from .forecast import LABELS
from .tariff import Tariff
from .uci import OUT_DIR as UCI_DIR
from .uci import SELECTION_FILE, SOURCE

log = logging.getLogger("peakshave.real")

OUT = config.OUT / "real"
POWER_SHARE = 0.30  # battery kW as a share of the building's 2013 peak
HOURS = 2.0  # kWh of storage per kW
FM_MODELS = ("timesfm-3.0",)


def battery_for(peak_kw: float) -> Battery:
    # round to 5 kW like a real quote would
    power = 5 * round(POWER_SHARE * peak_kw / 5)
    return Battery(capacity_kwh=power * HOURS, power_kw=power)


def building_ids() -> list[str]:
    if not SELECTION_FILE.exists():
        return []
    return list(json.loads(SELECTION_FILE.read_text())["picked"])


def run() -> dict:
    t0 = time.time()
    set_holidays("PT")
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION_FILE.read_text())
    tariff = Tariff()

    tables = {}
    buildings = {}
    for client, info in selection["picked"].items():
        load = load_interval_csv(UCI_DIR / f"{client}.csv")
        meta = {
            "source": SOURCE,
            "label": f"UCI meter {client} (Portugal)",
            "synthetic": False,
            "fingerprint": fingerprint(load["kw"]),
        }
        exp = prepare(True, load, meta, fm_models=FM_MODELS)
        battery = battery_for(info["peak_2013_kw"])

        res, _ = evaluate_building(exp, battery, tariff, full=False)
        # timing sweep with the perfect-shape forecast only (that's the clean timing test)
        sweep, _ = run_grid(exp, battery, tariff, names=["oracle"], shifts=SHIFTS)
        res["timing_sweep"] = sweep[["forecaster", "policy", "shift_intervals", "shift_minutes", "savings_usd", "pct_of_oracle"]]

        for name, df in res.items():
            tables.setdefault(name, []).append(df.assign(building=client))
        pd.DataFrame({"actual": exp.actual, **exp.forecasts}).to_parquet(OUT / f"forecasts_{client}.parquet")

        buildings[client] = {
            "label": f"Real meter {client}",
            "data_label": meta["label"],
            "fingerprint": meta["fingerprint"],
            "peak_2013_kw": info["peak_2013_kw"],
            "annual_peak_kw": float(exp.actual.max()),
            "annual_peak_timestamp": str(exp.actual.idxmax()),
            "battery": battery.to_dict(),
            "baseline_bill_usd": float(tariff.bill(exp.actual)["total"].sum()),
            "eval_start": exp.meta["eval_start"],
            "eval_end": exp.meta["eval_end"],
            "headline": headline(res["summary"], res["timing_sweep"], res["error_slopes"], res["forecast_accuracy"]),
        }
        log.info("%s done (%.0fs so far)", client, time.time() - t0)

    merged = {}
    for name, dfs in tables.items():
        df = pd.concat(dfs, ignore_index=True)
        cols = ["building"]
        for c in df.columns:
            if c != "building":
                cols.append(c)
        merged[name] = df[cols]
        merged[name].round(4).to_csv(OUT / f"{name}.csv", index=False)

    forecasters = {}
    for n in LABELS:
        if n in set(merged["summary"].forecaster):
            forecasters[n] = LABELS[n]
    results = {
        "meta": {
            "source": SOURCE,
            "selection": selection,
            "holidays": "PT",
            "tariff": tariff.to_dict(),
            "battery_rule": {"power_share_of_2013_peak": POWER_SHARE, "hours": HOURS},
            "forecasters": forecasters,
            "policies": POLICY_LABELS,
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "buildings": buildings,
    }
    for name, df in merged.items():
        if name != "daily":
            results[name] = _records(df)
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    res = run()
    summary = pd.DataFrame(res["summary"])
    print(f"\n{len(res['buildings'])} real buildings, {res['meta']['runtime_seconds']}s")
    for client, b in res["buildings"].items():
        s = summary[(summary.building == client) & (summary.forecaster != "oracle")]
        piv = s.pivot(index="forecaster", columns="policy", values="pct_of_oracle")
        parts = []
        for p in POLICIES:
            parts.append(f"{p} {piv[p].min():.0f}-{piv[p].max():.0f}%")
        bat = b["battery"]
        print(f"  {client} peak {b['annual_peak_kw']:.0f} kW, battery {bat['power_kw']:.0f} kW/{bat['capacity_kwh']:.0f} kWh: " + ", ".join(parts))


if __name__ == "__main__":
    main()
