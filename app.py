"""Streamlit dashboard for the peak-shaving experiment.

    streamlit run app.py

Everything on the page is recomputed from the saved forecasts using whatever
battery, tariff and policy are picked in the sidebar.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src import charts, config
from src.charts import money, usd
from src.data import set_holidays
from src.dispatch import POLICIES, POLICY_LABELS, Battery, simulate
from src.evaluate import BUILDINGS, DEFAULT_FORECASTER, MINUTES_PER_INTERVAL, SHIFTS, Experiment, daily_table, load_saved
from src.forecast import LABELS
from src.tariff import Tariff

st.set_page_config(page_title="Time Series Peakbench", layout="wide", initial_sidebar_state="expanded")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
DEFAULTS = Battery()


def html(markup: str) -> None:
    # st.markdown reads two $ signs as a LaTeX formula, so escape them
    st.markdown(markup.replace("$", "&#36;"), unsafe_allow_html=True)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root {--paper:#f6f4ee; --ink:#1c1b18; --ink2:#4a4842; --muted:#7d7a70; --rule:#d6d1c4; --hair:#e4e0d5; --accent:#d24f1c;
       --mono:'IBM Plex Mono', ui-monospace, Consolas, monospace;}
.block-container {max-width: 1180px; padding-top: 1.6rem; padding-bottom: 5rem;}
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stHeaderActionElements"] {display: none;}
[data-testid="stDecoration"] {display: none;}

.mast {display:flex; justify-content:space-between; align-items:baseline; gap:1rem; flex-wrap:wrap;
       font-family:var(--mono); font-size:12px; color:var(--muted); border-bottom:1px solid var(--ink); padding-bottom:.5rem;}
.mast b {color:var(--ink); font-weight:600;}
h1.headline {font-size:2.5rem; line-height:1.1; font-weight:600; letter-spacing:-0.022em; color:var(--ink); margin:1.6rem 0 .9rem; max-width:52rem;}
h1.headline em {font-style:normal; color:var(--accent);}
p.dek {font-size:1.08rem; line-height:1.6; color:var(--ink2); max-width:46rem; margin:0 0 1.2rem;}
.notice {display:flex; gap:.9rem; align-items:baseline; border:1px solid var(--rule); border-left:3px solid var(--ink);
         padding:.55rem .85rem; font-size:13.5px; color:var(--ink2); max-width:46rem; margin-bottom:2rem; background:#fbfaf6;}
.notice .k {font-family:var(--mono); font-size:11.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--ink); white-space:nowrap;}

.bill {border-top:2px solid var(--ink); margin:0 0 2.6rem;}
.bill-head {display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:.5rem; padding:.6rem 0 .4rem;}
.bill-head .t {font-size:1.02rem; font-weight:600; color:var(--ink);}
.bill-head .s {font-family:var(--mono); font-size:12px; color:var(--muted);}
table.ledger {width:100%; border-collapse:collapse; font-size:14px;}
table.ledger th {font-weight:500; color:var(--muted); font-size:12px; text-align:right; padding:.45rem 0 .45rem 1rem; border-bottom:1px solid var(--rule);}
table.ledger th:first-child {text-align:left; padding-left:0;}
table.ledger td {padding:.5rem 0 .5rem 1rem; border-bottom:1px solid var(--hair); text-align:right;
                 font-family:var(--mono); font-variant-numeric:tabular-nums; color:var(--ink);}
table.ledger td:first-child {text-align:left; padding-left:0; font-family:inherit; color:var(--ink2);}
table.ledger tr.total td {border-top:1px solid var(--ink); border-bottom:1px solid var(--ink); font-weight:600; color:var(--ink);}
table.ledger td.save {color:var(--ink); font-weight:600;}
.bill-foot {display:flex; justify-content:space-between; flex-wrap:wrap; gap:.6rem; padding:.7rem 0 0; font-size:14px; color:var(--ink2);}
.bill-foot .v {font-family:var(--mono); font-weight:600; color:var(--ink);}

.sec {display:grid; grid-template-columns:3.2rem 1fr; gap:0 .4rem; border-top:1px solid var(--ink); padding-top:.9rem; margin-top:1.4rem;}
.sec .n {font-family:var(--mono); font-size:13px; color:var(--muted); padding-top:.2rem;}
.sec .kick {font-family:var(--mono); font-size:12px; color:var(--muted); letter-spacing:.02em;}
.sec h2 {font-size:1.38rem; line-height:1.3; font-weight:600; letter-spacing:-0.012em; color:var(--ink); margin:.2rem 0 .35rem; padding:0;}
.sec p {font-size:14.5px; line-height:1.6; color:var(--ink2); margin:0; max-width:48rem;}
.sec p b {color:var(--ink); font-weight:600;}
[data-testid="stExpander"] details {border:none; border-top:1px solid var(--hair); border-radius:0; background:transparent;}
[data-testid="stExpander"] summary {font-family:var(--mono); font-size:12px; color:var(--muted); padding-left:0;}
.foot {font-size:13px; color:var(--muted); line-height:1.65; border-top:1px solid var(--ink); padding-top:.9rem; margin-top:2.6rem; max-width:52rem;}
.foot code {font-family:var(--mono); font-size:12px; background:none; color:var(--ink2); padding:0;}

section[data-testid="stSidebar"] .side-h {font-family:var(--mono); font-size:11.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
       color:var(--ink); border-top:1px solid var(--ink); padding-top:.45rem; margin:1.3rem 0 .3rem;}
section[data-testid="stSidebar"] .side-h:first-of-type {margin-top:.2rem;}
@media (max-width: 760px) {h1.headline {font-size:1.85rem;} .sec {grid-template-columns:1fr;} .sec .n {display:none;}
                           table.ledger {font-size:12.5px;} table.ledger td, table.ledger th {padding-left:.5rem;}}
</style>
""",
    unsafe_allow_html=True,
)


# cached data + simulation


@st.cache_data(show_spinner=False)
def run_meta() -> dict | None:
    path = config.OUT / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def real_meta() -> dict:
    # results for the real UCI meters, if src/real_meters.py has been run
    path = config.OUT / "real" / "results.json"
    if not path.exists():
        return {"buildings": {}}
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def inputs(building: str) -> tuple[pd.Series, dict[str, pd.Series]]:
    exp = load_saved(building)
    return exp.actual, exp.forecasts


@st.cache_data(show_spinner=False, max_entries=2000)
def sim(building: str, forecaster: str, policy: str, shift: int, cap: float, pwr: float, rte: float) -> pd.DataFrame:
    # cached per setting, so moving a slider back and forth is instant
    actual, fcs = inputs(building)
    out = simulate(actual, fcs[forecaster], Battery(cap, pwr, rte), Tariff(), policy, shift)
    return out.drop(columns=["actual_kw"])


def bill(series: pd.Series, tariff: Tariff, months: list[int]) -> pd.DataFrame:
    b = tariff.bill(series)
    return b[b.index.month.isin(months)]


def savings(building, forecaster, policy, shift, bat, tariff, months) -> float:
    actual, _ = inputs(building)
    net = sim(building, forecaster, policy, shift, *bat)["net_kw"]
    return float(bill(actual, tariff, months)["total"].sum() - bill(net, tariff, months)["total"].sum())


def count(n: int) -> str:
    words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    if n < len(words):
        return words[n]
    return str(n)


def short(name: str) -> str:
    return LABELS[name].split(" (")[0]


def section(n: int, kicker: str, takeaway: str, how_to_read: str) -> None:
    html(f'<div class="sec"><div class="n">{n:02d}</div><div><div class="kick">{kicker}</div>'
         f"<h2>{takeaway}</h2><p>{how_to_read}</p></div></div>")


# sidebar

meta = run_meta()
if meta is None:
    st.error("No results yet. Run `make run` (or `python -m src.evaluate`) first.")
    st.stop()

real = real_meta()["buildings"]
building_names = {}
for b in meta["buildings"]:
    building_names[b] = "Synthetic: " + BUILDINGS[b].lower() if meta["meta"]["synthetic"] else BUILDINGS[b]
for b, info in real.items():
    building_names[b] = f"Real meter {b} ({info['annual_peak_kw']:,.0f} kW peak)"

with st.sidebar:
    html('<div class="side-h">Building</div>')
    building = st.selectbox("Building", list(building_names), format_func=building_names.get, label_visibility="collapsed")
    is_real = building in real
    if is_real:
        st.caption("Metered 15-minute load, UCI ElectricityLoadDiagrams, Portugal, 2014.")
    elif building == "no_spikes":
        st.caption("The synthetic building with its equipment spikes removed.")
    else:
        st.caption("Generated office building; short equipment spikes set most monthly peaks.")
    # Portuguese holidays for the UCI meters, US ones for the synthetic building
    set_holidays("PT" if is_real else "US")
    actual, fcs = inputs(building)

    # default battery: the size used in the study for this building
    default_bat = Battery(**real[building]["battery"]) if is_real else DEFAULTS
    html('<div class="side-h">Battery</div>')
    cap = st.slider("Energy (kWh)", 50, 2000, int(default_bat.capacity_kwh), step=10)
    pwr = st.slider("Power (kW)", 10, 1000, int(default_bat.power_kw), step=5)
    rte = st.slider("Round-trip efficiency", 0.70, 0.98, default_bat.round_trip_efficiency, step=0.01, format="%.2f")

    html('<div class="side-h">Tariff</div>')
    dc = st.slider("Demand charge ($/kW-month)", 5.0, 40.0, Tariff().demand_charge, step=0.5, format="$%.2f")

    html('<div class="side-h">Forecast and dispatch</div>')
    real_fcs = []
    for f in LABELS:
        if f in fcs and f != "oracle":
            real_fcs.append(f)
    forecaster = st.selectbox("Forecaster", real_fcs + ["oracle"], index=real_fcs.index(DEFAULT_FORECASTER), format_func=LABELS.get)
    policy = st.radio(
        "Dispatch policy", list(POLICIES), index=list(POLICIES).index("ratchet"), format_func=POLICY_LABELS.get,
        captions=[
            "Discharge when the meter crosses the forecast's daily target",
            "Same, but never shave below this month's peak so far",
            "Follow the forecast's schedule; ignore the meter",
        ],
    )
    shift_options = []
    for s in SHIFTS:
        shift_options.append(s * MINUTES_PER_INTERVAL)
    shift_min = st.select_slider(
        "Forecast timing shift", options=shift_options, value=0,
        format_func=lambda m: f"{m:+d} min" if m else "none",
    )
    shift = shift_min // MINUTES_PER_INTERVAL

    html('<div class="side-h">Months</div>')
    m0, m1 = st.select_slider("Months", options=MONTHS, value=(MONTHS[0], MONTHS[-1]), label_visibility="collapsed")
    months = list(range(MONTHS.index(m0) + 1, MONTHS.index(m1) + 2))

tariff = Tariff(demand_charge=dc)
bat = (float(cap), float(pwr), float(rte))
if len(months) == 12:
    period = "year"
elif m0 == m1:
    period = m0
else:
    period = f"{m0}&ndash;{m1}"
over = "a year" if period == "year" else f"over {period}"

names = []
for f in LABELS:
    if f in fcs:
        names.append(f)


# savings for every forecaster x policy with the current settings

with st.spinner("Simulating the year..."):
    rows = []
    for f in names:
        for p in POLICIES:
            s = 0 if f == "oracle" else shift  # the oracle is never shifted, it's the reference
            rows.append({"forecaster": f, "policy": p, "savings_usd": savings(building, f, p, s, bat, tariff, months)})
    grid = pd.DataFrame(rows)

oracle_sav = grid[grid.forecaster == "oracle"].set_index("policy")["savings_usd"]
grid["pct_of_oracle"] = 100 * grid.savings_usd / grid.policy.map(oracle_sav).replace(0, np.nan)
real_grid = grid[grid.forecaster != "oracle"]

before = bill(actual, tariff, months)
sel_sim = sim(building, forecaster, policy, shift, *bat)
after = bill(sel_sim["net_kw"], tariff, months)
saved = float(before.total.sum() - after.total.sum())
left = float(oracle_sav[policy] - saved)

# headline numbers: how far apart the policies are for this forecaster, and how
# far apart the forecasters are for this policy
ref = forecaster if forecaster != "oracle" else DEFAULT_FORECASTER
by_policy = grid[grid.forecaster == ref].set_index("policy").savings_usd
policy_spread = float(by_policy.max() - by_policy.min())
this_policy = real_grid[real_grid.policy == policy].savings_usd
forecaster_spread = float(this_policy.max() - this_policy.min())


# header

if is_real:
    source = f"UCI meter {building}, Portugal"
    eval_start, eval_end = real[building]["eval_start"], real[building]["eval_end"]
else:
    source = "synthetic building" if meta["meta"]["synthetic"] else meta["meta"]["label"]
    eval_start, eval_end = meta["meta"]["eval_start"], meta["meta"]["eval_end"]
html(f'<div class="mast"><span><b>timeseries-peakbench</b> &nbsp;/&nbsp; dashboard</span>'
     f'<span>{source} &middot; illustrative MA tariff &middot; {eval_start} to {eval_end}</span></div>')

if policy_spread >= forecaster_spread:
    headline = (f"Choosing the dispatch policy is worth <em>{money(policy_spread)}</em> {over}. "
                f"Choosing the forecaster is worth {money(forecaster_spread)}.")
else:
    headline = (f"Here the forecaster matters more: <em>{money(forecaster_spread)}</em> {over}, "
                f"against {money(policy_spread)} for the dispatch policy.")
html(f'<h1 class="headline">{headline}</h1>')
html(
    f'<p class="dek">A {cap:,} kWh / {pwr:,} kW battery on {"a real building (UCI meter " + building + ")" if is_real else "an office building"} '
    f"that peaks at {actual.max():,.0f} kW, billed ${dc:.2f} per kW on each month's single highest 15-minute interval. "
    f"{count(len(real_fcs))} day-ahead forecasters, from "
    f"last week's load to Google's TimesFM and Mistral's Timestral, each drive three dispatch policies. The policy spread is measured for the "
    f"{short(ref)} forecast; the forecaster spread for {POLICY_LABELS[policy].lower()}.</p>"
)
if is_real:
    html('<div class="notice"><span class="k">Real meter</span><span>Metered 15-minute load from UCI&#39;s '
         f"ElectricityLoadDiagrams20112014 (anonymous client {building}, Portugal, 2014; CC BY 4.0). The tariff is still "
         "the illustrative MA rate, with Portuguese public holidays.</span></div>")
elif meta["meta"]["synthetic"]:
    html('<div class="notice"><span class="k">Synthetic data</span><span>The building load is generated (weekday occupancy, '
         "Boston-like weather, a summer cooling peak, random equipment spikes), not read from a meter. The tariff is "
         "illustrative, not a filed rate. The real-meter buildings in the sidebar are metered data.</span></div>")


# 1. the bill, laid out like a utility bill instead of KPI tiles

pct_saved = 100 * saved / float(before.total.sum())
ledger = [
    ("Energy charges", before.energy_cost.sum(), after.energy_cost.sum()),
    (f"Demand charges ({len(before)} monthly peaks &times; ${dc:.2f}/kW)", before.demand_cost.sum(), after.demand_cost.sum()),
]
body = ""
for name, b, a in ledger:
    body += f"<tr><td>{name}</td><td>{usd(b)}</td><td>{usd(a)}</td><td>{usd(a - b, signed=True)}</td></tr>"
body += (f'<tr class="total"><td>Total bill</td><td>{usd(before.total.sum())}</td><td>{usd(after.total.sum())}</td>'
         f'<td class="save">{usd(-saved, signed=True)} ({-pct_saved:+.1f}%)</td></tr>')
body += (f"<tr><td>Average monthly peak</td><td>{before.peak_kw.mean():,.0f} kW</td><td>{after.peak_kw.mean():,.0f} kW</td>"
         f"<td>{after.peak_kw.mean() - before.peak_kw.mean():+,.0f} kW</td></tr>")

captured = 100 * saved / oracle_sav[policy] if oracle_sav[policy] else float("nan")
bill_title = "full year" if period == "year" else period
shift_text = "no timing shift" if shift_min == 0 else f"forecast shifted {shift_min:+d} min"
html(
    f'<div class="bill"><div class="bill-head"><span class="t">The bill, {bill_title}</span>'
    f'<span class="s">{LABELS[forecaster]} &middot; {POLICY_LABELS[policy].lower()} &middot; {shift_text}</span></div>'
    f'<table class="ledger"><thead><tr><th></th><th>Without battery</th><th>With battery</th><th>Change</th></tr></thead>'
    f"<tbody>{body}</tbody></table>"
    f'<div class="bill-foot"><span>Saved <span class="v">{usd(saved)}</span> ({pct_saved:.1f}% of the bill)</span>'
    f'<span>Left on the table vs perfect foresight: <span class="v">{usd(left)}</span> '
    f"({captured:.0f}% of the possible savings captured)</span></div></div>"
)


# 2. savings vs timing error

sweep_options = ["oracle"]
if forecaster != "oracle":
    sweep_options.append(forecaster)
sweep_src = st.session_state.get("sweep_src", "oracle")
if sweep_src not in sweep_options:
    sweep_src = "oracle"

rows = []
for p in POLICIES:
    for s in SHIFTS:
        rows.append({"policy": p, "shift_minutes": s * MINUTES_PER_INTERVAL,
                     "savings_usd": savings(building, sweep_src, p, s, bat, tariff, months)})
sweep = pd.DataFrame(rows)
piv = sweep.pivot(index="shift_minutes", columns="policy", values="savings_usd")
loss_window = float(piv.loc[0, "window"] - piv.loc[[-30, 30], "window"].mean())
meter_policies = ["threshold", "ratchet"]
loss_closed = float((piv.loc[0, meter_policies] - piv.loc[[-30, 30], meter_policies].mean()).abs().max())

src_label = "a perfect forecast" if sweep_src == "oracle" else f"the {short(sweep_src)} forecast"
if loss_closed < 50:
    closed_text = "the meter-driven policies don't move"
else:
    closed_text = f"the meter-driven policies move by {money(loss_closed)}"
section(
    1, "Timing error",
    f"Run {src_label} 30 minutes late and the schedule loses {money(loss_window)}; {closed_text}.",
    "Annual savings when the forecast is shifted earlier or later in time. The window policy discharges on the forecast's "
    "clock; threshold and ratchet discharge when the meter crosses a kW level, so only the forecast's magnitude reaches them. "
    "Brackets mark the gap at &plusmn;30 minutes.",
)
st.segmented_control(
    "Forecast being shifted", sweep_options, key="sweep_src", default="oracle",
    format_func=lambda f: "Perfect-shape forecast" if f == "oracle" else f"{short(f)} forecast", label_visibility="collapsed",
)
st.plotly_chart(charts.money_chart(sweep, POLICY_LABELS), config=PLOT_CONFIG, theme=None, width="stretch")
with st.expander("Table"):
    st.dataframe(piv.rename(columns=POLICY_LABELS).style.format("${:,.0f}"), width="stretch")


# 3. savings by forecaster

sel = real_grid[real_grid.policy == policy].set_index("forecaster")
best_f = sel.savings_usd.idxmax()
section(
    2, "Forecaster vs policy",
    f"Across {count(len(real_fcs)).lower()} forecasters, {POLICY_LABELS[policy].lower()} captures "
    f"{sel.pct_of_oracle.min():.0f}&ndash;{sel.pct_of_oracle.max():.0f}% of what perfect foresight would save. "
    f"Swapping the policy moves savings by up to {money(policy_spread)}.",
    f"Annual savings for every forecaster under each policy; labels give the share of perfect-foresight savings captured. "
    f"Best forecaster under {POLICY_LABELS[policy].lower()}: <b>{short(best_f)}</b>.",
)
bar_labels = {}
for f in names:
    bar_labels[f] = LABELS[f]
st.plotly_chart(charts.forecaster_bars(grid, bar_labels, POLICY_LABELS, selected=forecaster),
                config=PLOT_CONFIG, theme=None, width="stretch")
with st.expander("Table"):
    table = grid.assign(forecaster=grid.forecaster.map(LABELS), policy=grid.policy.map(POLICY_LABELS))
    st.dataframe(table.pivot(index="forecaster", columns="policy", values="savings_usd").style.format("${:,.0f}"), width="stretch")


# 4. day explorer, starts on the day that set the (filtered) peak

in_months = actual[actual.index.month.isin(months)]
peak_day = in_months.idxmax().date()
day_key = f"day-{building}-{m0}-{m1}"  # new key when the month filter changes, so the picker resets
day = st.session_state.get(day_key, peak_day)
d0 = pd.Timestamp(day)
span = slice(d0, d0 + pd.Timedelta(hours=23, minutes=45))
one = sel_sim.loc[span].assign(actual_kw=actual.loc[span])
kicker = "Day explorer"
if day == peak_day:
    kicker += " &middot; the day that set the peak"
section(
    3, kicker,
    f"{d0:%A %B} {d0.day}: the forecast said {one.forecast_kw.max():,.0f} kW, the meter hit {one.actual_kw.max():,.0f} kW, "
    f"and the building drew at most {one.net_kw.max():,.0f} kW with the battery.",
    "Top: actual load, the day-ahead forecast, the kW level the policy defends, and net load after the battery (the shaded band "
    "is energy the battery supplied or absorbed). Bottom: energy stored. Pick any day below.",
)
st.date_input("Day", value=peak_day, min_value=in_months.index.min().date(), max_value=in_months.index.max().date(), key=day_key)
st.plotly_chart(charts.day_explorer(one), config=PLOT_CONFIG, theme=None, width="stretch")


# 5. demand charge per month

monthly = pd.DataFrame(
    {
        "month": before.index.astype(str),
        "demand_cost_before": before.demand_cost.to_numpy(),
        "demand_cost_after": after.demand_cost.to_numpy(),
        "peak_kw_before": before.peak_kw.to_numpy(),
        "peak_kw_after": after.peak_kw.to_numpy(),
    }
)
cut = monthly.demand_cost_before - monthly.demand_cost_after
weakest = pd.Timestamp(monthly.loc[cut.idxmin(), "month"])
# only blame "one bad day" when a month's saving really collapsed
if cut.min() < 0.25 * cut.mean():
    month_text = f", but only {money(cut.min())} in {weakest:%B}: one uncovered day sets the whole month's charge."
else:
    month_text = f"; the smallest monthly cut is {money(cut.min())}, in {weakest:%B}."
section(
    4, "Month by month",
    f"Demand charges fall {money(cut.sum())} {over}{month_text}",
    "Each month's demand charge (its single highest 15-minute interval &times; the rate) without and with the battery. "
    "Hover for the peak kW behind each bar.",
)
st.plotly_chart(charts.monthly_chart(monthly), config=PLOT_CONFIG, theme=None, width="stretch")
with st.expander("Table"):
    st.dataframe(monthly.set_index("month").style.format("{:,.0f}"), width="stretch")


# 6. daily forecast error vs dollars

if forecaster == "oracle":
    section(5, "Error vs dollars", "Perfect foresight has no forecast error.", "Pick a real forecaster in the sidebar to see this chart.")
else:
    exp = Experiment(actual=actual, forecasts={forecaster: fcs[forecaster], "oracle": fcs["oracle"]})
    sims = {}
    for p in POLICIES:
        sims[(forecaster, p)] = sim(building, forecaster, p, shift, *bat)
        sims[("oracle", p)] = sim(building, "oracle", p, 0, *bat)
    daily = daily_table(exp, sims, tariff)
    in_range = pd.to_datetime(daily.date).dt.month.isin(months)
    daily = daily[(daily.forecaster == forecaster) & daily.business_day & in_range]
    fig, slopes = charts.error_scatter(daily, policy, POLICY_LABELS)

    others = []
    for p in POLICIES:
        if p != policy and p in slopes:
            others.append(f"{money(slopes[p])} under {POLICY_LABELS[p].lower()}")
    section(
        5, "Error vs dollars",
        f"Each kW of day-ahead peak error costs {money(slopes[policy])} a day under {POLICY_LABELS[policy].lower()}, "
        f"against {' and '.join(others)}.",
        "Each dot is a business day: how much less peak the battery shaved than perfect foresight would have, priced at the "
        "demand charge (what the day was worth had it set the month's peak). Lines are least-squares fits per policy; "
        "a flatter line degrades more gracefully.",
    )
    st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, width="stretch")

html(
    """<div class="foot">Forecasts are day-ahead: each sees only data up to the midnight before the day it predicts
    (enforced in <code>tests/test_forecast.py</code>). The TimesFM and Timestral forecasts are served from the committed
    cache in <code>out/cache/</code>. Not modelled: battery degradation, grid-services revenue, tariff ratchets, multiple meters.
    Method and limitations are in
    <a href="https://github.com/amaan784/timeseries-peakbench/blob/main/docs/method.md">docs/method.md</a>.</div>"""
)
