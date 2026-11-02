"""Build the standalone HTML report (and refresh the generated results).

    python -m src.report

Writes out/report.html, a copy at docs/index.html for GitHub Pages, the short
results summary in README.md and the full tables in docs/results.md. Only
reads what src/evaluate.py wrote, so every number here comes from the last run.
"""

from __future__ import annotations

import html as H
import json
import re
from datetime import date

import pandas as pd
import plotly.io as pio
from plotly.offline import get_plotlyjs_version

from . import charts, config
from .charts import money, usd
from .forecast import FM_MODELS

POLICY_ORDER = ("ratchet", "threshold", "window")
BATTERY_CASES = ("Half-size battery", "Base battery", "Double-size battery")
# generated text goes between these two comments in README.md and docs/results.md
RESULTS_START = "<!-- results:start -->"
RESULTS_END = "<!-- results:end -->"


def _fig(builder, *args, **kw) -> dict:
    # light and dark version of the same figure, as plotly JSON
    out = {}
    for mode in ("light", "dark"):
        res = builder(*args, mode=mode, **kw)
        fig = res[0] if isinstance(res, tuple) else res
        out[mode] = json.loads(pio.to_json(fig, validate=False))
    return out


def _count(n: int) -> str:
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    if n < len(words):
        return words[n]
    return str(n)


def _and(items: list) -> str:
    # "a", "a and b", "a, b and c"
    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _range(values) -> str:
    lo, hi = min(values), max(values)
    if round(lo) == round(hi):
        return f"{lo:.0f}%"
    return f"{lo:.0f}-{hi:.0f}%"


def _short(label: str) -> str:
    # "TimesFM 3.0 (zero-shot)" -> "TimesFM 3.0"
    return label.split(" (")[0]


def _spreads(g: pd.DataFrame):
    # For one case: the smallest gap between policies (over all forecasters),
    # the smallest gap between the two best policies, and the largest gap
    # between forecasters (over all policies). Deliberately conservative.
    piv = g[g.forecaster != "oracle"].pivot(index="forecaster", columns="policy", values="savings_usd")
    policy_gap = float((piv.max(axis=1) - piv.min(axis=1)).min())
    top_two = []
    for _, row in piv.iterrows():
        best = row.nlargest(2)
        top_two.append(best.iloc[0] - best.iloc[1])
    forecaster_gap = float((piv.max() - piv.min()).max())
    return policy_gap, float(min(top_two)), forecaster_gap


def _load_inputs() -> dict:
    res = json.loads((config.OUT / "results.json").read_text())
    meta = res["meta"]
    main = meta["main_building"]
    c = {
        "res": res,
        "meta": meta,
        "main": main,
        "pl": meta["policies"],
        "labels": meta["forecasters"],
        "fc": meta["default_forecaster"],
        "bat": meta["battery"],
        "tar": meta["tariff"],
        "hb": res["buildings"][main],
        "summary": pd.DataFrame(res["summary"]),
        "acc": pd.DataFrame(res["forecast_accuracy"]),
        "monthly": pd.DataFrame(res["monthly"]),
        "sens": pd.DataFrame(res["battery_sensitivity"]),
        "daily": pd.read_csv(config.OUT / "daily.csv"),
        "peak_day": pd.read_csv(config.OUT / "peak_day.csv", index_col="timestamp", parse_dates=True),
    }
    sweep = pd.DataFrame(res["timing_sweep"])
    c["sw"] = sweep[sweep.building == main]

    S = c["summary"][c["summary"].building == main]
    real = S[S.forecaster != "oracle"]
    c["S"] = S
    c["pct"] = real.pivot(index="forecaster", columns="policy", values="pct_of_oracle")
    c["sav"] = real.pivot(index="forecaster", columns="policy", values="savings_usd")
    c["empty"] = real.pivot(index="forecaster", columns="policy", values="ran_out_days_pct")
    c["A"] = c["acc"][c["acc"].building == main].set_index("forecaster")

    c["real_names"] = []
    for f in c["labels"]:
        if f != "oracle" and f in c["pct"].index:
            c["real_names"].append(f)
    c["fms"] = []
    for f in c["real_names"]:
        if f in FM_MODELS:
            c["fms"].append(f)

    # the spike-free twin, if there is one
    c["other"] = None
    c["pct_o"] = c["sav_o"] = None
    for b in res["buildings"]:
        if b != main:
            c["other"] = b
            So = c["summary"][(c["summary"].building == b) & (c["summary"].forecaster != "oracle")]
            c["pct_o"] = So.pivot(index="forecaster", columns="policy", values="pct_of_oracle")
            c["sav_o"] = So.pivot(index="forecaster", columns="policy", values="savings_usd")
            break

    head = c["hb"]["headline"]
    c["policy_spread"] = head["policy_spread_usd"][c["fc"]]
    c["forecaster_spread"] = head["forecaster_spread_usd"]["ratchet"]
    c["best"] = S[(S.forecaster == c["fc"]) & (S.policy == "ratchet")].iloc[0]
    return c


def _timing_loss(sw: pd.DataFrame, forecaster: str, policy: str) -> float:
    # savings lost when the forecast is 30 min early/late, averaged over both
    d = sw[(sw.forecaster == forecaster) & (sw.policy == policy)].set_index("shift_minutes").savings_usd
    return float(d.loc[0] - d.loc[[-30, 30]].mean())


def _fm_text(c: dict):
    """Sentences comparing each foundation model with ridge, plus the TimeGPT note."""
    A, sav, labels, pl = c["A"], c["sav"], c["labels"], c["pl"]
    ridge_mae = A.loc["ridge", "peak_mae_kw"]
    ridge_mape = A.loc["ridge", "business_day_mape_pct"]

    lines = []
    for f in c["fms"]:
        beat = A.loc[f, "peak_mae_kw"] < ridge_mae
        verdict = "beat" if beat else "did not beat"
        bias = ""
        if A.loc[f, "peak_bias_kw"] < -20:
            bias = f"; its peaks run {abs(A.loc[f, 'peak_bias_kw']):.0f} kW low on average"
        mape_better = A.loc[f, "business_day_mape_pct"] < ridge_mape
        mape_word = "better" if mape_better else "worse"
        # "while" only when the two metrics disagree
        if beat == mape_better:
            link = "and"
            mape_word += " too"
        else:
            link = "while"
        dollars = []
        for p in POLICY_ORDER:
            dollars.append(f"{money(sav.loc[f, p] - sav.loc['ridge', p], signed=True)} under {pl[p].lower()}")
        lines.append(
            f"<b>{H.escape(labels[f])}</b> {verdict} ridge on day-ahead peak error "
            f"({A.loc[f, 'peak_mae_kw']:.0f} vs {ridge_mae:.0f} kW mean absolute error{bias}), "
            f"{link} its business-day MAPE was {mape_word} ({A.loc[f, 'business_day_mape_pct']:.1f}% vs {ridge_mape:.1f}%). "
            f"Dollars vs ridge: {', '.join(dollars)}."
        )

    note = ""
    if "timegpt" not in labels:
        note = ("TimeGPT is implemented (<code>timegpt_zeroshot()</code>) but did not run for this report: no "
                "<code>NIXTLA_API_KEY</code> was available, so it is left out of every comparison rather than estimated.")
    return lines, note


def _fm_moves(c: dict) -> float:
    # biggest dollar difference any foundation model made vs ridge, any policy
    biggest = 0.0
    for f in c["fms"]:
        for p in POLICY_ORDER:
            biggest = max(biggest, abs(c["sav"].loc[f, p] - c["sav"].loc["ridge", p]))
    return biggest


def _findings(c: dict, timing_window: float, timing_closed: float, fm_lines: list, timegpt_note: str) -> list[str]:
    pl, labels, fc = c["pl"], c["labels"], c["fc"]
    pct, empty = c["pct"], c["empty"]
    findings = []

    findings.append(
        f"<b>Policy beats forecaster.</b> Holding the forecaster fixed ({labels[fc]}), the three dispatch policies are "
        f"{money(c['policy_spread'])} a year apart. Holding the policy fixed ({pl['ratchet'].lower()}), the "
        f"{_count(len(c['real_names']))} forecasters are {money(c['forecaster_spread'])} apart."
    )
    findings.append(
        f"<b>The obvious threshold policy is the worst one.</b> Setting each day's target as low as the forecast says the "
        f"battery can hold captures only {_range(pct['threshold'])} of perfect-foresight savings. It spends the whole battery on "
        f"the forecast's peak, so when the real peak is higher the battery is empty before it arrives: it ran out with the meter "
        f"still above target on {_range(empty['threshold'])} of business days (the ratchet: {_range(empty['ratchet'])})."
    )
    findings.append(
        f"<b>Reading the meter only pays if the policy respects the bill.</b> Adding one rule (never shave below a peak "
        f"already set this month) lifts the same closed-loop policy to {_range(pct['ratchet'])}, against "
        f"{_range(pct['window'])} for a forecast-scheduled window."
    )
    if timing_closed < 50:
        closed_text = "don't move."
    else:
        closed_text = f"move by {money(timing_closed)}."
    findings.append(
        f"<b>Timing error is what kills schedules.</b> A perfect forecast running 30 minutes late costs the window policy "
        f"{money(timing_window)} a year; the meter-driven policies {closed_text}"
    )

    if c["other"]:
        findings.append(
            f"<b>But no policy wins everywhere.</b> Remove the short equipment spikes from the same building and the ranking "
            f"flips: the window schedule captures {_range(c['pct_o']['window'])}, the ratchet {_range(c['pct_o']['ratchet'])}. "
            f"Spikes are what a forecast can't place and a meter can catch; on a smooth load, the risk is "
            f"under-forecasting the level, and a schedule bounded by its plan degrades more gently."
        )

    if c["fms"]:
        A, sav = c["A"], c["sav"]
        ridge_mae = A.loc["ridge", "peak_mae_kw"]
        beat = []
        better_mape = True
        for f in c["fms"]:
            if A.loc[f, "peak_mae_kw"] < ridge_mae:
                beat.append(f)
            if A.loc[f, "business_day_mape_pct"] >= A.loc["ridge", "business_day_mape_pct"]:
                better_mape = False

        two = len(c["fms"]) == 2
        if not beat:
            lead = ("neither foundation model" if two else "no foundation model") + " beat ridge on day-ahead peak error"
        elif len(beat) == len(c["fms"]):
            lead = ("both foundation models" if two else "every foundation model") + " beat ridge on day-ahead peak error"
        else:
            names = []
            for f in beat:
                names.append(_short(labels[f]))
            others = "the other did not" if len(c["fms"]) - len(beat) == 1 else "the others did not"
            lead = _and(names) + " beat ridge on day-ahead peak error; " + others

        maes, deltas = [], []
        for f in c["fms"]:
            maes.append(f"{_short(labels[f])} {A.loc[f, 'peak_mae_kw']:.0f} kW")
            delta = sav.loc[f, "ratchet"] - sav.loc["ridge", "ratchet"]
            deltas.append(f"{money(delta, signed=True)} ({_short(labels[f])})")

        changed = "did not change" if _fm_moves(c) < c["policy_spread"] else "changed"
        despite = ", despite a better average MAPE" if better_mape and not beat else ""
        text = (
            f"<b>Foundation models {changed} the answer.</b> On the synthetic building, {lead} "
            f"(mean absolute error: {', '.join(maes)}, ridge {ridge_mae:.0f} kW){despite}. Under {pl['ratchet'].lower()} they moved annual savings by "
            f"{', '.join(deltas)}, against {money(c['policy_spread'])} between policies."
        )
        if timegpt_note:
            text += " " + timegpt_note
        findings.append(text)
    return findings


def _figures(c: dict, timing_window: float, timing_closed: float):
    """All report figures (light + dark) and the caption that goes with each variant."""
    pl, labels, fc, sw, res = c["pl"], c["labels"], c["fc"], c["sw"], c["res"]
    figs, captions = {}, {}

    # 1. timing
    def sweep_for(f):
        return sw[sw.forecaster == f][["policy", "shift_minutes", "savings_usd"]]

    if timing_closed < 50:
        closed_text = "the meter-driven policies don't move."
    else:
        closed_text = f"the meter-driven policies move by {money(timing_closed)}."
    figs["fig-timing"] = {
        "oracle": _fig(charts.money_chart, sweep_for("oracle"), pl),
        fc: _fig(charts.money_chart, sweep_for(fc), pl),
    }
    captions["fig-timing"] = {
        "oracle": f"Run a perfect forecast 30 minutes late and the window schedule loses {money(timing_window)} a year; {closed_text}",
        fc: f"With the real {labels[fc]} forecast, already blurry in time, a 30-minute shift costs the window "
        f"{money(_timing_loss(sw, fc, 'window'))}; its problem is the spikes it can't place, not the clock.",
    }

    # 2. forecaster vs policy, one variant per building
    figs["fig-forecasters"] = {}
    captions["fig-forecasters"] = {}
    for b in res["buildings"]:
        Sb = c["summary"][c["summary"].building == b]
        present = set(Sb.forecaster)
        bar_labels = {}
        for f in labels:
            if f in present:
                bar_labels[f] = labels[f]
        figs["fig-forecasters"][b] = _fig(charts.forecaster_bars, Sb, bar_labels, pl, selected=fc)

        Pb = Sb[Sb.forecaster != "oracle"].pivot(index="forecaster", columns="policy", values="pct_of_oracle")
        winner = Pb.mean().idxmax()
        name = res["buildings"][b]["label"]
        if (Pb.idxmax(axis=1) == winner).all():
            captions["fig-forecasters"][b] = (
                f"{name}: {pl[winner].lower()} wins with every forecaster ({_range(Pb[winner])} of perfect foresight); "
                f"within a policy, forecasters differ by a few points."
            )
        else:
            captions["fig-forecasters"][b] = f"{name}: the best policy depends on the forecaster."

    # 3. the annual peak day under each policy
    pk = pd.Timestamp(c["hb"]["annual_peak_timestamp"])
    figs["fig-day"], captions["fig-day"] = {}, {}
    for p in POLICY_ORDER:
        d = c["peak_day"][c["peak_day"].policy == p]
        figs["fig-day"][p] = _fig(charts.day_explorer, d, net_label=f"Net load ({pl[p].lower()})")
        captions["fig-day"][p] = (
            f"{pk:%A %B} {pk.day}, the year's peak day. The {labels[fc]} forecast said {d.forecast_kw.max():,.0f} kW, "
            f"the meter hit {d.actual_kw.max():,.0f} kW, and with {pl[p].lower()} the building drew at most "
            f"{d.net_kw.max():,.0f} kW."
        )

    # 4. monthly
    M = c["M"]
    cut = M.demand_cost_before - M.demand_cost_after
    weakest = pd.Timestamp(M.loc[cut.idxmin(), "month"])
    caption = f"Demand charges fall {money(cut.sum())} over the year ({labels[fc]}, {pl['ratchet'].lower()})"
    # only blame a single bad day if that month really did collapse
    if cut.min() < 0.25 * cut.mean():
        caption += f", but only {money(cut.min())} in {weakest:%B}: one day the battery couldn't cover sets the whole month's charge."
    else:
        caption += f"; the smallest monthly cut is {money(cut.min())}, in {weakest:%B}."
    figs["fig-monthly"] = {"ratchet": _fig(charts.monthly_chart, M)}
    captions["fig-monthly"] = {"ratchet": caption}

    # 5. error vs dollars
    daily = c["daily"]
    D = daily[(daily.building == c["main"]) & (daily.forecaster == fc) & daily.business_day]
    _, slopes = charts.error_scatter(D, "ratchet", pl)
    parts = []
    for p in POLICY_ORDER:
        if p in slopes:
            parts.append(f"{money(slopes[p])} a day under {pl[p].lower()}")
    figs["fig-scatter"] = {"ratchet": _fig(charts.error_scatter, D, "ratchet", pl)}
    captions["fig-scatter"] = {
        "ratchet": "Each kW of day-ahead peak error costs " + ", ".join(parts)
        + f" ({labels[fc]}). The ratchet's line is the flattest: it degrades most gracefully."
    }
    return figs, captions


def _robustness(c: dict) -> list:
    # (case name, {policy: [pct of oracle for each real forecaster]})
    sens, pl, main = c["sens"], c["pl"], c["main"]
    rob = []
    for case in BATTERY_CASES:
        rows = sens[(sens.building == main) & (sens.case == case) & (sens.forecaster != "oracle")]
        size = rows.iloc[0]
        name = f"{case.replace(' battery', '')} battery ({size.capacity_kwh:,.0f} kWh / {size.power_kw:,.0f} kW)"
        values = {}
        for p in POLICY_ORDER:
            values[p] = rows[rows.policy == p].pct_of_oracle.tolist()
        rob.append((name, values))
    if c["other"]:
        values = {}
        for p in POLICY_ORDER:
            values[p] = c["pct_o"][p].tolist()
        rob.append((f"{c['res']['buildings'][c['other']]['label']}, base battery", values))
    return rob


def _robustness_heading(c: dict, rob: list) -> str:
    # which policy wins each case, by its average share of perfect foresight
    pl = c["pl"]
    winners = {}
    for name, values in rob:
        best, best_avg = None, None
        for p in POLICY_ORDER:
            avg = sum(values[p]) / len(values[p])
            if best_avg is None or avg > best_avg:
                best, best_avg = p, avg
        winners[name] = best

    battery_winners = set()
    for name, w in winners.items():
        if "battery (" in name:
            battery_winners.add(w)
    if len(battery_winners) == 1 and c["other"] and winners[rob[-1][0]] not in battery_winners:
        return "Battery size shifts the numbers, not the ranking. The building's load shape changes the ranking."
    if len(set(winners.values())) == 1:
        return f"{pl[next(iter(winners.values()))]} comes out ahead in every case."
    parts = []
    for name, w in winners.items():
        parts.append(f"{name.split(' (')[0].lower()}: {pl[w].lower()}")
    return "The best policy depends on the case: " + "; ".join(parts) + "."


def _spread_cases(c: dict) -> list:
    sens, main, res = c["sens"], c["main"], c["res"]
    cases = []
    for case in BATTERY_CASES:
        g = sens[(sens.building == main) & (sens.case == case)]
        cases.append((f"{case}, {res['buildings'][main]['label'].lower()}", _spreads(g)))
    if c["other"]:
        g = c["summary"][c["summary"].building == c["other"]]
        cases.append((f"Base battery, {res['buildings'][c['other']]['label'].lower()}", _spreads(g)))
    return cases


def _spread_note(spread_cases: list) -> str:
    ratios, narrow = [], []
    for name, (pol, top2, fcs) in spread_cases:
        if fcs > 0:
            ratios.append(pol / fcs)
        if top2 < fcs:
            narrow.append(name.lower())

    if min(ratios) > 1:
        note = (f"In every case the spread across policies exceeds the spread across forecasters, "
                f"by {min(ratios):.1f}x to {max(ratios):.1f}x.")
    else:
        note = "In some cases the forecaster spread is the larger one."
    if narrow:
        plural = "s" if len(narrow) != 1 else ""
        note += (f" The two best policies come closer than that, and in {_count(len(narrow))} case{plural} "
                 f"({'; '.join(narrow)}) closer to each other than the best and worst forecasters are.")
    else:
        note += " Even the two best policies are further apart than the best and worst forecasters."
    return note


def _fm_heading(c: dict) -> str:
    A, fms = c["A"], c["fms"]
    if not fms:
        return "No foundation-model forecasts were available for this run."
    ridge_mae = A.loc["ridge", "peak_mae_kw"]
    ridge_mape = A.loc["ridge", "business_day_mape_pct"]

    mixed = 0  # models with better average accuracy but worse peaks
    beat = 0
    for f in fms:
        if A.loc[f, "business_day_mape_pct"] < ridge_mape and A.loc[f, "peak_mae_kw"] > ridge_mae:
            mixed += 1
        if A.loc[f, "peak_mae_kw"] < ridge_mae:
            beat += 1
    all_beat = beat == len(fms)

    if mixed and beat:
        first = "Better average accuracy did not reliably buy better peaks"
    elif mixed:
        first = "Better average accuracy did not buy better peaks"
    elif all_beat:
        first = "Foundation models beat ridge on peak error"
    else:
        first = "Foundation models did not beat ridge on peak error"

    moves = _fm_moves(c)
    if moves < c["policy_spread"]:
        second = f"and the dollars moved by at most {money(moves)}, against {money(c['policy_spread'])} for the choice of policy."
    else:
        second = f"and the dollars moved by up to {money(moves)}."
    return f"{first}, {second}"


def _html_tables(c: dict, rob: list, spread_cases: list) -> dict:
    A, sav, pct, pl, labels, tar = c["A"], c["sav"], c["pct"], c["pl"], c["labels"], c["tar"]

    acc_head = ""
    rob_head = ""
    for p in POLICY_ORDER:
        acc_head += f"<th>{pl[p]}</th>"
        rob_head += f"<th>{pl[p]}</th>"

    acc_rows = ""
    for f in c["real_names"]:
        acc_rows += (
            f"<tr><td>{H.escape(labels[f])}</td><td>{A.loc[f, 'business_day_mape_pct']:.1f}%</td>"
            f"<td>{A.loc[f, 'peak_mae_kw']:.0f} kW</td><td>{A.loc[f, 'peak_bias_kw']:+.0f} kW</td>"
            f"<td>{A.loc[f, 'peak_time_mae_min']:.0f} min</td>"
        )
        for p in POLICY_ORDER:
            acc_rows += f"<td>{usd(sav.loc[f, p])} <span class='m'>{pct.loc[f, p]:.0f}%</span></td>"
        acc_rows += "</tr>"

    rob_rows = ""
    for name, values in rob:
        rob_rows += f"<tr><td>{H.escape(name)}</td>"
        for p in POLICY_ORDER:
            rob_rows += f"<td>{_range(values[p])}</td>"
        rob_rows += "</tr>"

    spread_rows = ""
    for name, (pol, top2, fcs) in spread_cases:
        spread_rows += (f"<tr><td>{H.escape(name)}</td><td>{usd(pol)}</td><td>{usd(top2)}</td>"
                        f"<td>{usd(fcs)}</td><td>{pol / fcs:.1f}x</td></tr>")

    # the bill for the default forecaster under the ratchet
    M = c["M"]
    before = M[["demand_cost_before", "energy_cost_before", "total_before"]].sum()
    after = M[["demand_cost_after", "energy_cost_after", "total_after"]].sum()
    ledger_rows = (
        f"<tr><td>Energy charges</td><td>{usd(before.energy_cost_before)}</td><td>{usd(after.energy_cost_after)}</td>"
        f"<td>{usd(after.energy_cost_after - before.energy_cost_before, True)}</td></tr>"
        f"<tr><td>Demand charges (12 monthly peaks &times; ${tar['demand_charge']:.2f}/kW)</td>"
        f"<td>{usd(before.demand_cost_before)}</td><td>{usd(after.demand_cost_after)}</td>"
        f"<td>{usd(after.demand_cost_after - before.demand_cost_before, True)}</td></tr>"
        f"<tr class='total'><td>Total bill</td><td>{usd(before.total_before)}</td><td>{usd(after.total_after)}</td>"
        f"<td>{usd(after.total_after - before.total_before, True)} ({-c['best'].savings_pct:+.1f}%)</td></tr>"
        f"<tr><td>Average monthly peak</td><td>{M.peak_kw_before.mean():,.0f} kW</td><td>{M.peak_kw_after.mean():,.0f} kW</td>"
        f"<td>{M.peak_kw_after.mean() - M.peak_kw_before.mean():+,.0f} kW</td></tr>"
    )
    return {
        "acc_head": acc_head, "acc_rows": acc_rows, "rob_head": rob_head, "rob_rows": rob_rows,
        "spread_rows": spread_rows, "ledger_rows": ledger_rows,
    }


def _load_real(c: dict) -> dict | None:
    # results from src/real_meters.py, if it has been run
    path = config.OUT / "real" / "results.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    r = {
        "res": res,
        "buildings": res["buildings"],
        "summary": pd.DataFrame(res["summary"]),
        "acc": pd.DataFrame(res["forecast_accuracy"]),
        "sweep": pd.DataFrame(res["timing_sweep"]),
    }
    r["fms"] = []
    ran = set(r["acc"].forecaster)
    for f in FM_MODELS:
        if f in ran:
            r["fms"].append(f)
    fc = c["fc"]
    pl = c["pl"]
    rows = []
    for b, info in r["buildings"].items():
        S = r["summary"][r["summary"].building == b]
        real = S[S.forecaster != "oracle"]
        pct_best = real.pivot(index="forecaster", columns="policy", values="pct_of_best_oracle")
        sav = real.pivot(index="forecaster", columns="policy", values="savings_usd")
        A = r["acc"][r["acc"].building == b].set_index("forecaster")
        sw = r["sweep"][(r["sweep"].building == b) & (r["sweep"].forecaster == "oracle")]

        # winner under the default forecaster, and whether it wins with every forecaster
        winner = pct_best.loc[fc].idxmax()
        wins_everywhere = bool((pct_best.idxmax(axis=1) == winner).all())

        window_at_0 = float(sw[(sw.policy == "window") & (sw.shift_minutes == 0)].savings_usd.iloc[0])
        window_loss = _timing_loss(sw, "oracle", "window")
        meter_loss = max(abs(_timing_loss(sw, "oracle", "threshold")), abs(_timing_loss(sw, "oracle", "ratchet")))

        row = {
            "building": b,
            "peak_kw": info["annual_peak_kw"],
            "battery_kw": info["battery"]["power_kw"],
            "battery_kwh": info["battery"]["capacity_kwh"],
            "mape": A.loc[fc, "business_day_mape_pct"],
            "winner": winner,
            "wins_everywhere": wins_everywhere,
            "policy_spread": float(sav.loc[fc].max() - sav.loc[fc].min()),
            "forecaster_spread": float((sav.max() - sav.min()).max()),
            "window_loss_share": 100 * window_loss / window_at_0 if window_at_0 > 0 else float("nan"),
            "meter_loss": meter_loss,
        }
        for p in POLICY_ORDER:
            row[p] = pct_best.loc[fc, p]
        row["ridge_mae"] = A.loc["ridge", "peak_mae_kw"]
        row["fm"] = {}
        for f in r["fms"]:
            if f in A.index:
                row["fm"][f] = {
                    "mae": A.loc[f, "peak_mae_kw"],
                    "mape": A.loc[f, "business_day_mape_pct"],
                    "usd": float(sav.loc[f, "ratchet"] - sav.loc["ridge", "ratchet"]),
                }
        rows.append(row)
    r["rows"] = rows

    # one line of text summing it up, built from the counts
    n = len(rows)
    win_counts = {}
    for p in POLICY_ORDER:
        win_counts[p] = 0
    policy_beats_fc = 0
    for row in rows:
        win_counts[row["winner"]] += 1
        if row["policy_spread"] > row["forecaster_spread"]:
            policy_beats_fc += 1
    # per foundation model: on how many buildings it beat ridge, and by how many dollars
    r["fm_counts"] = {}
    for f in r["fms"]:
        counts = {"n": 0, "beat": 0, "mape_better": 0, "more": 0, "usd": []}
        for row in rows:
            if f not in row["fm"]:
                continue
            counts["n"] += 1
            counts["usd"].append(row["fm"][f]["usd"])
            if row["fm"][f]["mae"] < row["ridge_mae"]:
                counts["beat"] += 1
            if row["fm"][f]["mape"] < row["mape"]:
                counts["mape_better"] += 1
            if row["fm"][f]["usd"] > 0:
                counts["more"] += 1
        r["fm_counts"][f] = counts
    r["n"] = n
    r["win_counts"] = win_counts
    r["policy_beats_fc"] = policy_beats_fc

    top = max(POLICY_ORDER, key=lambda p: win_counts[p])
    if win_counts[top] == n:
        r["heading"] = (f"On all {_count(n)} real buildings, {pl[top].lower()} is the best policy, and the policy spread beats "
                        f"the forecaster spread on {_count(policy_beats_fc)} of {_count(n)}.")
    else:
        parts = []
        for p in POLICY_ORDER:
            if win_counts[p]:
                parts.append(f"{pl[p].lower()} on {_count(win_counts[p])}")
        r["heading"] = (f"Across {_count(n)} real buildings the best policy varies: " + ", ".join(parts)
                        + f". The policy spread beats the forecaster spread on {_count(policy_beats_fc)} of {_count(n)}.")

    values = {}
    for p in POLICY_ORDER:
        values[p] = []
        for row in rows:
            values[p].append(row[p])
    losses = []
    for row in rows:
        losses.append(row["window_loss_share"])
    mapes = []
    for row in rows:
        mapes.append(row["mape"])
    r["finding"] = (
        f"<b>Real meters {'back this up' if win_counts['ratchet'] == n else 'only partly agree'}.</b> On metered 15-minute "
        f"data from {_count(n)} buildings in Portugal (UCI, 2014, picked by a fixed rule), {pl[top].lower()} was the best policy "
        f"on {_count(win_counts[top])} of {_count(n)} with the {c['labels'][fc]} forecast, capturing {_range(values[top])} of the "
        f"best perfect-foresight savings. A perfect forecast 30 minutes off cost the window schedule "
        f"{_range(losses)} of its savings. Real loads are harder to forecast ({c['labels'][fc]} MAPE {_range(mapes)} on "
        f"business days)"
        + "."
    )
    for f in r["fms"]:
        k = r["fm_counts"][f]
        if not k["n"]:
            continue
        r["finding"] += (
            f" {_short(c['labels'][f])} vs ridge: lower MAPE on {_count(k['mape_better'])} of {_count(k['n'])} buildings, "
            f"lower peak error on {_count(k['beat'])}, more {pl['ratchet'].lower()} savings on {_count(k['more'])} "
            f"({money(min(k['usd']), signed=True)} to {money(max(k['usd']), signed=True)} a year per building)."
        )
    r["top"] = top
    r["window_losses"] = losses
    return r


def _mae_header(c: dict, r: dict) -> str:
    # one column for the peak errors: ridge first, then each foundation model
    names = ["ridge"]
    for f in r["fms"]:
        names.append(_short(c["labels"][f]))
    return "Peak MAE, kW: " + " / ".join(names)


def _mae_cell(r: dict, row: dict) -> str:
    values = [f"{row['ridge_mae']:.0f}"]
    for f in r["fms"]:
        if f in row["fm"]:
            values.append(f"{row['fm'][f]['mae']:.0f}")
        else:
            values.append("n/a")
    return " / ".join(values)


def _real_figure(c: dict, r: dict) -> dict:
    fc = c["fc"]
    rows = r["summary"][(r["summary"].forecaster == fc)]
    labels = {}
    for row in r["rows"]:
        labels[row["building"]] = f"{row['building']} ({row['peak_kw']:,.0f} kW peak)"
    return _fig(charts.building_bars, rows, labels, c["pl"])


def _real_table(c: dict, r: dict) -> str:
    pl = c["pl"]
    head = "<th>Building</th><th>2014 peak</th><th>Battery</th><th>MAPE</th>"
    for p in POLICY_ORDER:
        head += f"<th>{pl[p]}</th>"
    head += "<th>Policy spread</th><th>Forecaster spread</th><th>Peak MAE, kW</th>"
    body = ""
    for row in r["rows"]:
        body += (f"<tr><td>{row['building']}</td><td>{row['peak_kw']:,.0f} kW</td>"
                 f"<td>{row['battery_kw']:,.0f} kW / {row['battery_kwh']:,.0f} kWh</td><td>{row['mape']:.1f}%</td>")
        for p in POLICY_ORDER:
            body += f"<td>{row[p]:.0f}%</td>"
        body += f"<td>{usd(row['policy_spread'])}</td><td>{usd(row['forecaster_spread'])}</td>"
        body += f"<td>{_mae_cell(r, row)}</td></tr>"
    return f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody>"


def _toggle(target: str, options: list) -> str:
    # segmented control above a chart; the JS at the bottom of the page handles clicks
    buttons = ""
    for i, (key, text) in enumerate(options):
        pressed = "true" if i == 0 else "false"
        buttons += f'<button type="button" data-key="{H.escape(key)}" aria-pressed="{pressed}">{H.escape(text)}</button>'
    return f'<div class="seg" data-target="{target}" role="group">{buttons}</div>'


def _section(n: int, kicker: str, fig_id: str, how: str, figs: dict, captions: dict, controls: str = "") -> str:
    first = next(iter(figs[fig_id]))
    return (
        f'<section class="sec"><div class="n">{n:02d}</div><div class="body"><div class="kick">{kicker}</div>'
        f'<h2 id="{fig_id}-cap">{captions[fig_id][first]}</h2><p class="how">{how}</p>{controls}'
        f'<div class="fig" id="{fig_id}" data-key="{H.escape(first)}"></div></div></section>'
    )


def build() -> tuple[str, dict]:
    c = _load_inputs()
    pl, labels, fc, meta, bat, tar = c["pl"], c["labels"], c["fc"], c["meta"], c["bat"], c["tar"]
    monthly = c["monthly"]
    c["M"] = monthly[(monthly.building == c["main"]) & (monthly.forecaster == fc) & (monthly.policy == "ratchet")]

    timing_window = _timing_loss(c["sw"], "oracle", "window")
    timing_closed = max(abs(_timing_loss(c["sw"], "oracle", "threshold")), abs(_timing_loss(c["sw"], "oracle", "ratchet")))

    fm_lines, timegpt_note = _fm_text(c)
    findings = _findings(c, timing_window, timing_closed, fm_lines, timegpt_note)
    figs, captions = _figures(c, timing_window, timing_closed)

    rob = _robustness(c)
    spread_cases = _spread_cases(c)
    spread_note = _spread_note(spread_cases)
    findings[0] += " " + spread_note.split(" The two best")[0]
    tables = _html_tables(c, rob, spread_cases)

    real = _load_real(c)
    real_html = ""
    if real:
        findings.insert(len(findings) - 1, real["finding"])
        figs["fig-real"] = {"ridge": _real_figure(c, real)}
        captions["fig-real"] = {"ridge": real["heading"]}
        real_html = (
            _section(8, "Real meters", "fig-real",
                     f"Metered 15-minute load from {_count(real['n'])} buildings in UCI's ElectricityLoadDiagrams20112014 "
                     f"(Portugal, 2014; CC BY 4.0), picked by a rule fixed before any results: complete data, a 2013 peak "
                     f"between 100 and 2,000 kW, and an office-like weekly pattern. Each battery is sized at 30% of the "
                     f"building's 2013 peak with two hours of storage. Bars use the {labels[fc]} forecast and show savings as a "
                     f"share of the best perfect-foresight result across the three policies.", figs, captions)
            # the table goes after the chart div, inside the section body
            .replace("></div></div></section>", f'></div><div class="tablewrap" style="margin-top:12px"><table>{_real_table(c, real)}</table></div>'
                     f'<p class="how" style="margin-top:10px">Policy columns: share of the best perfect-foresight savings, '
                     f'{labels[fc]} forecast. Spreads are annual dollars. '
                     f'{H.escape(_mae_header(c, real).replace("Peak MAE, kW: ", "Peak MAE: "))}.</p></div></section>')
        )

    building_options = []
    for b in c["res"]["buildings"]:
        building_options.append((b, c["res"]["buildings"][b]["label"]))
    policy_options = []
    for p in POLICY_ORDER:
        policy_options.append((p, pl[p]))

    sections = [
        _section(1, "Timing error", "fig-timing",
                 "Annual savings when the forecast is shifted earlier or later in time. The window policy discharges on the "
                 "forecast's clock; threshold and ratchet discharge when the meter crosses a kW level, so only the forecast's "
                 "magnitude reaches them.", figs, captions,
                 _toggle("fig-timing", [("oracle", "Perfect-shape forecast"), (fc, f"{labels[fc]} forecast")])),
        _section(2, "Forecaster vs policy", "fig-forecasters",
                 "Annual savings for every forecaster under each policy; bar labels give the share of perfect-foresight "
                 "savings captured.", figs, captions, _toggle("fig-forecasters", building_options)),
        _section(3, "The day that set the peak", "fig-day",
                 "Top: actual load, the day-ahead forecast, the kW level the policy defends, and net load after the battery "
                 "(the shaded band is energy supplied or absorbed). Bottom: energy stored.",
                 figs, captions, _toggle("fig-day", policy_options)),
        _section(4, "Month by month", "fig-monthly",
                 "Each month's demand charge, set by its single highest 15-minute interval, without and with the battery.",
                 figs, captions),
        _section(5, "Error vs dollars", "fig-scatter",
                 "Each dot is a business day: how much less peak the battery shaved than perfect foresight, priced at the "
                 "demand charge (what the day was worth had it set the month's peak). Lines are least-squares fits per policy.",
                 figs, captions),
    ]

    policy_spread, forecaster_spread = c["policy_spread"], c["forecaster_spread"]
    if policy_spread >= forecaster_spread:
        title = (f"Choosing the dispatch policy is worth <em>{money(policy_spread)}</em> a year. "
                 f"Choosing the forecaster is worth {money(forecaster_spread)}.")
    else:
        title = (f"Here the forecaster mattered more than the policy: {money(forecaster_spread)} "
                 f"against {money(policy_spread)} a year.")

    if meta["synthetic"]:
        notice = ("The building load is generated (weekday occupancy, Boston-like weather, a summer cooling peak, random "
                  "equipment spikes), not read from a meter. The tariff is illustrative, not a filed rate.")
    else:
        notice = "Tariff is illustrative, not a filed rate."

    findings_html = ""
    for f in findings:
        findings_html += f"<li><div>{f}</div></li>"
    sections_html = ""
    for sec in sections:
        sections_html += sec

    fm_html = " ".join(fm_lines)
    if timegpt_note:
        fm_html += " " + timegpt_note

    oracle_r = c["S"][(c["S"].forecaster == "oracle") & (c["S"].policy == "ratchet")].iloc[0]
    data_json = json.dumps({"figs": figs, "captions": captions}).replace("</", "<\\/")

    page = TEMPLATE.format(
        plotly_v=get_plotlyjs_version(),
        eval_start=meta["eval_start"],
        eval_end=meta["eval_end"],
        source="synthetic building" if meta["synthetic"] else H.escape(meta["label"]),
        title=title,
        dek=(
            f"When a commercial battery uses a day-ahead load forecast to cut demand charges, how much money does forecast "
            f"error cost? A {bat['capacity_kwh']:,.0f} kWh / {bat['power_kw']:,.0f} kW battery, an office building that peaks "
            f"at {c['hb']['annual_peak_kw']:,.0f} kW, a ${tar['demand_charge']:.2f}/kW monthly demand charge, "
            f"{_count(len(c['real_names']))} day-ahead forecasters and three dispatch policies, simulated at 15-minute "
            f"resolution over {meta['n_days']} days."
        ),
        notice=notice,
        findings=findings_html,
        ledger_sub=f"{H.escape(labels[fc])} forecast &middot; {pl['ratchet'].lower()}",
        saved=usd(c["best"].savings_usd),
        saved_pct=f"{c['best'].savings_pct:.1f}",
        left=usd(oracle_r.savings_usd - c["best"].savings_usd),
        captured=f"{c['best'].pct_of_oracle:.0f}",
        sections=sections_html,
        real_section=real_html,
        method_n="09" if real_html else "08",
        fm_text=fm_html,
        fm_heading=_fm_heading(c),
        rob_heading=_robustness_heading(c, rob),
        spread_note=spread_note,
        runtime=meta["runtime_seconds"],
        context_days=meta["fm_context_days"],
        rte=f"{bat['round_trip_efficiency']:.0%}",
        soc=f"{bat['soc_min']:.0%} to {bat['soc_max']:.0%}",
        on_peak=f"{tar['on_peak_start_hour']}:00 to {tar['on_peak_end_hour']}:00",
        on_rate=f"${tar['on_peak_rate']:.2f}",
        off_rate=f"${tar['off_peak_rate']:.2f}",
        generated=date.today().isoformat(),
        data=data_json,
        **tables,
    )

    c["real"] = real
    c.update(title=title, findings=findings, fm_lines=fm_lines, timegpt_note=timegpt_note,
             fm_heading=_fm_heading(c), rob=rob, rob_heading=_robustness_heading(c, rob),
             spread_cases=spread_cases, spread_note=spread_note,
             timing_window=timing_window, timing_closed=timing_closed)
    return page, c


# README


def _md(text: str) -> str:
    # the report's bits of inline HTML, turned into markdown
    text = re.sub(r"</?em>", "", text)
    text = re.sub(r"</?b>", "**", text)
    text = re.sub(r"</?code>", "`", text)
    text = re.sub(r"<[^>]+>", "", text)
    return H.unescape(text)


def _md_row(cells: list) -> str:
    return "| " + " | ".join(cells) + " |"


def _best_policy(pct: pd.DataFrame) -> str:
    # the policy with the highest average share of perfect-foresight savings
    return pct[list(POLICY_ORDER)].mean().idxmax()


def summary_block(c: dict) -> str:
    """The short results summary at the top of the README."""
    labels, pl, pct, A, fc = c["labels"], c["pl"], c["pct"], c["A"], c["fc"]
    policy_spread, forecaster_spread = c["policy_spread"], c["forecaster_spread"]
    out = [f"**{_md(c['title'])}**", ""]

    # share of perfect-foresight savings per policy, with and without the spikes
    cases = [(c["res"]["buildings"][c["main"]]["label"], pct)]
    if c["other"] is not None:
        cases.append((c["res"]["buildings"][c["other"]]["label"], c["pct_o"]))
    header = ["Policy"]
    for name, _ in cases:
        header.append(name)
    out += [_md_row(header), "|---|" + "---:|" * len(cases)]
    for p in POLICY_ORDER:
        cells = [pl[p]]
        for _, table in cases:
            cells.append(_range(table[p]))
        out.append(_md_row(cells))
    where = "synthetic building" if c["meta"]["synthetic"] else H.escape(c["meta"]["label"])
    out += ["", f"Share of perfect-foresight savings each policy captured, across the {_count(len(c['real_names']))} "
            f"forecasters ({where}, one year at 15-minute resolution).", ""]

    if policy_spread >= forecaster_spread:
        lead = "Policy beats forecaster."
    else:
        lead = "Here the forecaster beats the policy."
    out.append(f"- **{lead}** With the forecaster fixed ({labels[fc]}), the three policies are {money(policy_spread)} a "
               f"year apart; with the policy fixed, the forecasters are {money(forecaster_spread)} apart "
               f"({policy_spread / forecaster_spread:.1f}x).")

    worst = pct[list(POLICY_ORDER)].mean().idxmin()
    best = _best_policy(pct)
    lead = "The obvious policy is the worst one." if worst == "threshold" else f"{pl[worst]} is the worst policy."
    out.append(f"- **{lead}** {pl[worst]} captures {_range(pct[worst])} of perfect-foresight savings; "
               f"{pl[best].lower()} captures {_range(pct[best])}.")

    window_at_0 = float(c["sw"][(c["sw"].forecaster == "oracle") & (c["sw"].policy == "window")
                                & (c["sw"].shift_minutes == 0)].savings_usd.iloc[0])
    closed = "don't move" if c["timing_closed"] < 50 else f"move by {money(c['timing_closed'])}"
    out.append(f"- **Timing error breaks schedules.** A perfect forecast 30 minutes late costs the window policy "
               f"{money(c['timing_window'])} a year ({100 * c['timing_window'] / window_at_0:.0f}% of its savings); the "
               f"meter-driven policies {closed}.")

    if c["other"] is not None:
        best_o = _best_policy(c["pct_o"])
        if best_o != best:
            out.append(f"- **No policy wins everywhere.** Without the equipment spikes, {pl[best_o].lower()} comes out on "
                       f"top ({_range(c['pct_o'][best_o])}) and {pl[best].lower()} drops to {_range(c['pct_o'][best])}.")
        else:
            out.append(f"- **The ranking holds without the spikes:** {pl[best].lower()} still captures the most "
                       f"({_range(c['pct_o'][best])}).")

    r = c.get("real")
    if r:
        top = r["top"]
        out.append(f"- **Real meters.** On {_count(r['n'])} metered buildings (UCI, Portugal, 2014), {pl[top].lower()} was "
                   f"the best policy on {_count(r['win_counts'][top])} of {_count(r['n'])}, the policy spread beat the "
                   f"forecaster spread on {_count(r['policy_beats_fc'])} of {_count(r['n'])}, and a perfect forecast 30 "
                   f"minutes off cost the window schedule {_range(r['window_losses'])} of its savings.")

    if c["fms"]:
        names = []
        for f in c["fms"]:
            names.append(_short(labels[f]))
        most_accurate = A.loc[c["real_names"], "business_day_mape_pct"].idxmin()
        delta = c["sav"].loc[most_accurate, "ratchet"] - c["sav"].loc["ridge", "ratchet"]
        out.append(f"- **Better forecasts, little money.** The foundation models ({_and(names)}) moved annual savings by "
                   f"at most {money(_fm_moves(c))} against ridge. The most accurate forecaster by MAPE, "
                   f"{_short(labels[most_accurate])} ({A.loc[most_accurate, 'business_day_mape_pct']:.1f}% vs "
                   f"{A.loc['ridge', 'business_day_mape_pct']:.1f}% for ridge), changed {pl['ratchet'].lower()} savings by "
                   f"{money(delta, signed=True)} a year.")

    out += ["", "Every table, the real-meter breakdown and what the numbers mean: [docs/results.md](docs/results.md).", "",
            "<sub>Generated by `make report` from `out/results.json`. Edit the code, not these numbers.</sub>"]
    return "\n".join(out)


def results_block(c: dict) -> str:
    """The full generated results in docs/results.md."""
    labels, pl, pct, sav, empty, A, S, meta = c["labels"], c["pl"], c["pct"], c["sav"], c["empty"], c["A"], c["S"], c["meta"]
    bat, tar, hb = c["bat"], c["tar"], c["hb"]
    oracle = S[S.forecaster == "oracle"].set_index("policy").savings_usd

    policy_names = []
    for p in POLICY_ORDER:
        policy_names.append(pl[p])

    out = [f"**{_md(c['title'])}**", ""]
    for i, f in enumerate(c["findings"], 1):
        out.append(f"{i}. {_md(f)}")

    # main results table
    out += ["", f"#### Annual savings by forecaster and policy ({hb['label'].lower()}, the main case)", "",
            "Each cell: savings vs the no-battery bill, and the share of perfect-foresight savings captured under that policy.", "",
            _md_row(["Forecaster"] + policy_names + ["Peak error, MAE", "MAPE, business days"]),
            "|---|" + "---:|" * (len(POLICY_ORDER) + 2)]
    cells = [labels["oracle"]]
    for p in POLICY_ORDER:
        cells.append(f"{usd(oracle[p])} (ceiling)")
    out.append(_md_row(cells + ["0 kW", "0%"]))
    for f in c["real_names"]:
        cells = [labels[f]]
        for p in POLICY_ORDER:
            cells.append(f"{usd(sav.loc[f, p])} ({pct.loc[f, p]:.0f}%)")
        cells += [f"{A.loc[f, 'peak_mae_kw']:.0f} kW", f"{A.loc[f, 'business_day_mape_pct']:.1f}%"]
        out.append(_md_row(cells))

    ran_out = []
    for p in ("threshold", "ratchet"):
        ran_out.append(f"{pl[p].lower()} {_range(empty[p])}")
    out += ["", f"Bill without a battery: {usd(hb['baseline_bill_usd'])} a year. Battery {bat['capacity_kwh']:,.0f} kWh / "
            f"{bat['power_kw']:,.0f} kW, {bat['round_trip_efficiency']:.0%} round trip. Demand charge "
            f"${tar['demand_charge']:.2f}/kW-month. Business days a meter-driven policy ran out of energy with the meter "
            f"still above its target: " + "; ".join(ran_out) + "."]

    # the spike-free twin
    if c["other"] is not None:
        po, so = c["pct_o"], c["sav_o"]
        out += ["", f"#### {c['res']['buildings'][c['other']]['label']}: the same building with the equipment spikes removed", "",
                _md_row(["Forecaster"] + policy_names), "|---|" + "---:|" * len(POLICY_ORDER)]
        for f in c["real_names"]:
            if f in po.index:
                cells = [labels[f]]
                for p in POLICY_ORDER:
                    cells.append(f"{usd(so.loc[f, p])} ({po.loc[f, p]:.0f}%)")
                out.append(_md_row(cells))

    # timing table
    sw = c["sw"]
    shifts = sorted(sw.shift_minutes.unique())
    header = ["Policy"]
    for m in shifts:
        header.append(f"{m:+d} min" if m else "0")
    out += ["", "#### Timing error: a perfect-shape forecast shifted in time (annual savings)", "",
            _md_row(header), "|---|" + "---:|" * len(shifts)]
    for p in POLICY_ORDER:
        row = sw[(sw.forecaster == "oracle") & (sw.policy == p)].set_index("shift_minutes").savings_usd
        cells = [pl[p]]
        for m in shifts:
            cells.append(usd(row[m]))
        out.append(_md_row(cells))

    # foundation models
    out += ["", "#### Foundation models vs ridge (synthetic building)", "", f"**{_md(c['fm_heading'])}**", ""]
    for line in c["fm_lines"]:
        out.append(f"- {_md(line)}")
    if c["timegpt_note"]:
        out.append(f"- {_md(c['timegpt_note'])}")

    # robustness
    out += ["", f"#### {_md(c['rob_heading'])}", "", _md_row(["Case"] + policy_names), "|---|" + "---:|" * len(POLICY_ORDER)]
    for name, values in c["rob"]:
        cells = [name]
        for p in POLICY_ORDER:
            cells.append(_range(values[p]))
        out.append(_md_row(cells))

    out += ["", f"**Does the policy still matter more than the forecaster?** {c['spread_note']}", "",
            _md_row(["Case", "Spread across policies", "Best vs second-best policy", "Spread across forecasters", "Ratio"]),
            "|---|---:|---:|---:|---:|"]
    for name, (pol, top2, fcs) in c["spread_cases"]:
        out.append(_md_row([name, usd(pol), usd(top2), usd(fcs), f"{pol / fcs:.1f}x"]))
    out += ["", "Annual savings. Policy columns: the smallest gap over all forecasters. Forecaster column: the largest gap over all policies."]

    r = c.get("real")
    if r:
        out += ["", "#### Real meters: six UCI buildings, Portugal, 2014", "", f"**{_md(r['heading'])}**", "",
                f"From UCI's ElectricityLoadDiagrams20112014 (CC BY 4.0), picked by the fixed rule in "
                f"`src/uci.py`. Battery sized at 30% of each building's 2013 peak, two hours of storage. Policy columns are savings as a "
                f"share of the best perfect-foresight result across the three policies, with the {labels[c['fc']]} forecast.", ""]
        header = ["Building", "2014 peak", "Battery", "MAPE"] + policy_names + ["Policy spread", "Forecaster spread",
                                                                                 _mae_header(c, r)]
        out += [_md_row(header), "|---|" + "---:|" * (len(header) - 1)]
        for row in r["rows"]:
            cells = [row["building"], f"{row['peak_kw']:,.0f} kW", f"{row['battery_kw']:,.0f} kW / {row['battery_kwh']:,.0f} kWh",
                     f"{row['mape']:.1f}%"]
            for p in POLICY_ORDER:
                cells.append(f"{row[p]:.0f}%")
            cells += [usd(row["policy_spread"]), usd(row["forecaster_spread"]), _mae_cell(r, row)]
            out.append(_md_row(cells))
    out += ["", f"<sub>Generated by `make report` from `out/results.json` ({meta['eval_start']} to {meta['eval_end']}, "
            f"evaluation runtime {meta['runtime_seconds']}s). Edit the code, not these numbers.</sub>"]
    return "\n".join(out)


def fill_markers(path, block: str) -> bool:
    # swap in new text between the two marker comments; False if the file or the markers are missing
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if RESULTS_START not in text or RESULTS_END not in text:
        return False
    before, rest = text.split(RESULTS_START, 1)
    _, after = rest.split(RESULTS_END, 1)
    path.write_text(f"{before}{RESULTS_START}\n{block}\n{RESULTS_END}{after}", encoding="utf-8")
    return True


def main() -> None:
    page, c = build()
    config.OUT.mkdir(exist_ok=True)
    (config.OUT / "report.html").write_text(page, encoding="utf-8")
    config.DOCS.mkdir(exist_ok=True)
    (config.DOCS / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {config.OUT / 'report.html'} and {config.DOCS / 'index.html'} ({len(page) / 1e6:.1f} MB)")
    if fill_markers(config.ROOT / "README.md", summary_block(c)):
        print("refreshed the results summary in README.md")
    if fill_markers(config.DOCS / "results.md", results_block(c)):
        print("refreshed the full results in docs/results.md")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Time Series Peakbench</title>
<meta name="description" content="How much does load-forecast error cost a demand-charge battery? A simulation study on synthetic data.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{plotly_v}/plotly.min.js"></script>
<style>
:root {{
  color-scheme: light;
  --paper:#f6f4ee; --ink:#1c1b18; --ink2:#4a4842; --muted:#7d7a70; --rule:#d6d1c4; --hair:#e4e0d5;
  --accent:#d24f1c; --box:#fbfaf6; --btn:#ece9e0;
  --sans:'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --mono:'IBM Plex Mono', ui-monospace, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --paper:#171614; --ink:#f2efe6; --ink2:#c9c5b9; --muted:#8d897e; --rule:#3a3933; --hair:#2a2925;
    --accent:#e4692f; --box:#1d1c1a; --btn:#24231f;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --paper:#171614; --ink:#f2efe6; --ink2:#c9c5b9; --muted:#8d897e; --rule:#3a3933; --hair:#2a2925;
  --accent:#e4692f; --box:#1d1c1a; --btn:#24231f;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin:0; background:var(--paper); color:var(--ink); }}
body {{ font-family:var(--sans); font-size:16px; line-height:1.6; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:24px 24px 80px; }}
.mast {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; font-family:var(--mono);
        font-size:12px; color:var(--muted); border-bottom:1px solid var(--ink); padding-bottom:8px; }}
.mast b {{ color:var(--ink); font-weight:600; }}
.mast button {{ font:inherit; color:var(--ink2); background:none; border:1px solid var(--rule); padding:2px 8px; cursor:pointer; }}
.mast button:hover {{ border-color:var(--ink2); }}
h1 {{ font-size:clamp(1.9rem, 4.2vw, 2.85rem); line-height:1.1; font-weight:600; letter-spacing:-0.022em; margin:40px 0 16px; max-width:860px; }}
h1 em {{ font-style:normal; color:var(--accent); }}
.dek {{ font-size:1.12rem; color:var(--ink2); max-width:760px; margin:0 0 20px; }}
.notice {{ display:flex; gap:14px; align-items:baseline; border:1px solid var(--rule); border-left:3px solid var(--ink); background:var(--box);
          padding:9px 14px; font-size:14px; color:var(--ink2); max-width:760px; margin:0 0 40px; }}
.notice .k {{ font-family:var(--mono); font-size:11.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--ink); white-space:nowrap; }}
.short {{ border-top:2px solid var(--ink); padding-top:10px; margin-bottom:44px; }}
.short .k, .kick, .bill .s {{ font-family:var(--mono); font-size:12px; color:var(--muted); }}
.short ol {{ margin:10px 0 0; padding:0; list-style:none; counter-reset:f; display:grid; gap:12px; }}
.short li {{ counter-increment:f; display:grid; grid-template-columns:2.6rem 1fr; color:var(--ink2); max-width:860px; }}
.short li::before {{ content:counter(f, decimal-leading-zero); font-family:var(--mono); font-size:13px; color:var(--muted); padding-top:2px; }}
.short li b {{ color:var(--ink); font-weight:600; }}
.bill {{ border-top:2px solid var(--ink); margin-bottom:44px; }}
.bill-head {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; padding:10px 0 6px; }}
.bill-head .t {{ font-weight:600; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{ font-weight:500; font-size:12px; color:var(--muted); text-align:right; padding:7px 0 7px 14px; border-bottom:1px solid var(--rule); white-space:nowrap; }}
td {{ padding:8px 0 8px 14px; border-bottom:1px solid var(--hair); text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; padding-left:0; }}
td:first-child {{ font-family:var(--sans); color:var(--ink2); white-space:normal; }}
tr.total td {{ border-top:1px solid var(--ink); border-bottom:1px solid var(--ink); font-weight:600; color:var(--ink); }}
td .m {{ color:var(--muted); font-size:12px; margin-left:4px; }}
.tablewrap {{ overflow-x:auto; }}
.bill-foot {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; padding-top:10px; font-size:14px; color:var(--ink2); }}
.bill-foot b {{ font-family:var(--mono); color:var(--ink); }}
.sec {{ display:grid; grid-template-columns:3.4rem minmax(0,1fr); border-top:1px solid var(--ink); padding-top:14px; margin-top:40px; }}
.sec .n {{ font-family:var(--mono); font-size:13px; color:var(--muted); padding-top:2px; }}
.sec h2 {{ font-size:1.42rem; line-height:1.3; font-weight:600; letter-spacing:-0.012em; margin:4px 0 6px; max-width:820px; }}
.sec .how, .prose p {{ font-size:15px; color:var(--ink2); margin:0 0 12px; max-width:760px; }}
.seg {{ display:inline-flex; flex-wrap:wrap; border:1px solid var(--rule); margin:4px 0 8px; }}
.seg button {{ font:500 13px var(--sans); color:var(--ink2); background:none; border:0; border-right:1px solid var(--rule); padding:6px 12px; cursor:pointer; }}
.seg button:last-child {{ border-right:0; }}
.seg button[aria-pressed="true"] {{ background:var(--btn); color:var(--ink); box-shadow:inset 0 -2px 0 var(--accent); }}
.seg button:focus-visible, .mast button:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.fig {{ width:100%; min-height:380px; }}
.prose {{ max-width:760px; }}
.prose h3 {{ font-size:1.05rem; font-weight:600; margin:24px 0 6px; }}
.prose ul {{ margin:0 0 12px; padding-left:18px; color:var(--ink2); font-size:15px; }}
code {{ font-family:var(--mono); font-size:.88em; color:var(--ink); }}
pre {{ font-family:var(--mono); font-size:13px; background:var(--box); border:1px solid var(--rule); padding:12px 14px; overflow-x:auto; }}
footer {{ margin-top:56px; border-top:1px solid var(--ink); padding-top:10px; font-family:var(--mono); font-size:12px; color:var(--muted);
          display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }}
a {{ color:var(--ink); text-decoration-color:var(--accent); text-underline-offset:3px; }}
@media (max-width:680px) {{
  .wrap {{ padding:16px 16px 64px; }}
  .sec {{ grid-template-columns:1fr; }} .sec .n {{ display:none; }}
  .short li {{ grid-template-columns:2rem 1fr; }}
  td, th {{ padding-left:8px; font-size:12.5px; }}
}}
</style>
</head>
<body>
<div class="wrap">
<header class="mast">
  <span><b>timeseries-peakbench</b> &nbsp;/&nbsp; working note</span>
  <span>{source} &middot; illustrative MA tariff &middot; {eval_start} to {eval_end} &nbsp;<button type="button" id="theme" aria-label="Toggle dark mode">dark</button></span>
</header>

<h1>{title}</h1>
<p class="dek">{dek}</p>
<div class="notice"><span class="k">Synthetic data</span><span>{notice}</span></div>

<section class="short">
  <span class="k">In short</span>
  <ol>{findings}</ol>
</section>

<section class="bill">
  <div class="bill-head"><span class="t">The bill, one year</span><span class="s">{ledger_sub}</span></div>
  <div class="tablewrap"><table>
    <thead><tr><th></th><th>Without battery</th><th>With battery</th><th>Change</th></tr></thead>
    <tbody>{ledger_rows}</tbody>
  </table></div>
  <div class="bill-foot"><span>Saved <b>{saved}</b> ({saved_pct}% of the bill)</span>
  <span>Left on the table vs perfect foresight: <b>{left}</b> ({captured}% of the possible savings captured)</span></div>
</section>

{sections}

<section class="sec"><div class="n">06</div><div class="body">
  <div class="kick">Did the foundation models help? (synthetic building)</div>
  <h2>{fm_heading}</h2>
  <p class="how">{fm_text}</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Forecaster</th><th>MAPE, business days</th><th>Peak error (MAE)</th><th>Peak bias</th><th>Peak timing error</th>{acc_head}</tr></thead>
    <tbody>{acc_rows}</tbody>
  </table></div>
  <p class="how" style="margin-top:10px">Peak columns are over business days. Policy columns: annual savings and share of perfect-foresight savings captured.</p>
</div></section>

<section class="sec"><div class="n">07</div><div class="body">
  <div class="kick">Robustness</div>
  <h2>{rob_heading}</h2>
  <p class="how">Share of perfect-foresight savings captured, range across the real forecasters.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Case</th>{rob_head}</tr></thead>
    <tbody>{rob_rows}</tbody>
  </table></div>
  <p class="how" style="margin-top:22px">Does the policy still matter more than the forecaster? {spread_note}</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Case</th><th>Spread across policies</th><th>Best vs second-best policy</th><th>Spread across forecasters</th><th>Ratio</th></tr></thead>
    <tbody>{spread_rows}</tbody>
  </table></div>
  <p class="how" style="margin-top:10px">Annual savings. Policy columns: the smallest gap over all forecasters. Forecaster column: the largest gap over all policies.</p>
</div></section>

{real_section}

<section class="sec"><div class="n">{method_n}</div><div class="body prose">
  <div class="kick">Method and limits</div>
  <h2>How the numbers were made</h2>
  <p>Each forecaster issues a 96-interval forecast for day D using only data up to midnight before D; a test poisons the future and
  checks that no forecast changes. Foundation models see a {context_days}-day trailing context, zero-shot, and their forecasts are cached per
  (model, data fingerprint, date). The battery has {rte} round-trip efficiency (square root on each leg) and a state of charge kept between {soc}.
  It charges only off-peak, never above the day's target, and paced to refill before the next on-peak window ({on_peak} on business days,
  {on_rate}/kWh; {off_rate}/kWh otherwise). Savings are full-bill: demand plus time-of-use energy.</p>
  <h3>Limitations</h3>
  <ul>
    <li>The main study is one generated building. The six real meters are a check, not a census: anonymous Portuguese
    clients, one year each, with the illustrative MA tariff applied to them.</li>
    <li>Illustrative tariff. No customer charges, ratchets, coincident-peak or capacity charges.</li>
    <li>No battery degradation, so cycling looks free and the threshold policy's extra cycles are not penalised.</li>
    <li>No grid-services or wholesale revenue stacked on top of bill savings.</li>
    <li>A single meter and a single battery; no portfolio effects.</li>
    <li>Day-ahead only: no intraday re-forecasting, which a production controller would do.</li>
  </ul>
  <h3>Reproduce</h3>
<pre>make install
make run      # evaluation, about {runtime}s on a laptop CPU (foundation models served from cache)
make test
make report   # this page
streamlit run app.py</pre>
</div></section>

<footer><span>timeseries-peakbench &middot; MIT licence</span><span>generated {generated} from out/results.json</span></footer>
</div>

<script id="report-data" type="application/json">{data}</script>
<script>
(function () {{
  const DATA = JSON.parse(document.getElementById('report-data').textContent);
  const root = document.documentElement;
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  const cfg = {{ displayModeBar: false, responsive: true }};

  function mode() {{
    return root.dataset.theme || (media.matches ? 'dark' : 'light');
  }}

  function draw(id) {{
    const el = document.getElementById(id);
    const fig = DATA.figs[id][el.dataset.key][mode()];
    Plotly.react(el, fig.data, fig.layout, cfg);
  }}

  function drawAll() {{
    for (const id of Object.keys(DATA.figs)) draw(id);
  }}

  // the button groups above some charts swap which variant is shown
  for (const group of document.querySelectorAll('.seg')) {{
    const id = group.dataset.target;
    const buttons = group.querySelectorAll('button');
    for (const btn of buttons) {{
      btn.addEventListener('click', function () {{
        for (const b of buttons) b.setAttribute('aria-pressed', String(b === btn));
        document.getElementById(id).dataset.key = btn.dataset.key;
        const cap = DATA.captions[id] && DATA.captions[id][btn.dataset.key];
        if (cap) document.getElementById(id + '-cap').innerHTML = cap;
        draw(id);
      }});
    }}
  }}

  const toggle = document.getElementById('theme');
  function label() {{
    toggle.textContent = mode() === 'dark' ? 'light' : 'dark';
  }}
  toggle.addEventListener('click', function () {{
    root.dataset.theme = mode() === 'dark' ? 'light' : 'dark';
    label();
    drawAll();
  }});
  media.addEventListener('change', function () {{
    if (!root.dataset.theme) {{ label(); drawAll(); }}
  }});

  label();
  if (window.Plotly) drawAll();
  else window.addEventListener('load', drawAll);
}})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
