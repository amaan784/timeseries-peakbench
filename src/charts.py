"""Plotly figures, shared by the Streamlit app and the static report.

Style: one accent colour (orange) for the ratchet policy and for "with
battery", greys for everything else. No dual axes; state of charge gets its
own small panel under the load chart instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Consolas, monospace"

# Paper-ish background, one accent. I ran these through a colour-blindness
# checker: the accent and both greys stay distinguishable and have at least
# 3:1 contrast on the background in light and dark mode.
THEMES = {
    "light": {
        "surface": "#f6f4ee",
        "ink": "#1c1b18",
        "ink2": "#4a4842",
        "muted": "#7d7a70",
        "grid": "#e4e0d5",
        "axis": "#cfcabd",
        "accent": "#d24f1c",
        "accent_wash": "rgba(210,79,28,0.13)",
        "neutral_dark": "#2f2e2a",
        "neutral_light": "#868276",
        "neutral_bar": "#bdb8aa",
        "neutral_wash": "rgba(47,46,42,0.07)",
    },
    "dark": {
        "surface": "#171614",
        "ink": "#f2efe6",
        "ink2": "#c9c5b9",
        "muted": "#8d897e",
        "grid": "#2a2925",
        "axis": "#3a3933",
        "accent": "#e4692f",
        "accent_wash": "rgba(228,105,47,0.18)",
        "neutral_dark": "#dcd8cc",
        "neutral_light": "#7a766c",
        "neutral_bar": "#4a4842",
        "neutral_wash": "rgba(220,216,204,0.08)",
    },
}

# each policy keeps its colour no matter which one is winning
POLICY_TOKEN = {"ratchet": "accent", "window": "neutral_dark", "threshold": "neutral_light"}
POLICY_ORDER = ("ratchet", "threshold", "window")


def policy_color(policy: str, mode: str = "light") -> str:
    return THEMES[mode][POLICY_TOKEN[policy]]


def money(x: float, signed: bool = False) -> str:
    # short form for headlines: $893, $3.0k, $38k, $1.25M
    sign = ""
    if x < 0:
        sign = "-"
    elif signed and x > 0:
        sign = "+"
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.2f}M"
    if round(x / 1e3, 1) >= 10:
        return f"{sign}${x / 1e3:.0f}k"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.1f}k"
    return f"{sign}${x:,.0f}"


def usd(x: float, signed: bool = False) -> str:
    # exact dollars, like on a bill: $402,118
    sign = ""
    if x < 0:
        sign = "-"
    elif signed and x > 0:
        sign = "+"
    return f"{sign}${abs(x):,.0f}"


def _day_label(dates) -> list[str]:
    # "Tue Jul 22". Built by hand because %-d doesn't work on Windows.
    d = pd.to_datetime(pd.Series(dates))
    return (d.dt.strftime("%a %b ") + d.dt.day.astype(str)).tolist()


def _layout(fig: go.Figure, mode: str, height: int, transparent: bool = True) -> go.Figure:
    t = THEMES[mode]
    bg = "rgba(0,0,0,0)" if transparent else t["surface"]
    axis = dict(
        gridcolor=t["grid"], gridwidth=1, zeroline=False, showline=False, ticks="",
        tickfont=dict(color=t["muted"], size=11.5, family=MONO), title_font=dict(color=t["ink2"], size=12),
        linecolor=t["axis"], automargin=True,
    )
    fig.update_layout(
        height=height,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family=FONT, size=13, color=t["ink2"]),
        margin=dict(l=8, r=24, t=36, b=8),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["axis"], font=dict(family=MONO, size=12, color=t["ink"])),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            font=dict(color=t["ink2"], size=12), bgcolor="rgba(0,0,0,0)", itemclick="toggleothers",
        ),
        bargap=0.35,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _end_labels(fig, mode, x, items, min_gap):
    # Label each line at its right end. If two labels would overlap, skip the
    # lower one and let the legend do the job.
    t = THEMES[mode]
    placed = []
    for text, y in sorted(items, key=lambda item: -item[1]):
        too_close = False
        for other in placed:
            if abs(y - other) < min_gap:
                too_close = True
                break
        if too_close:
            continue
        placed.append(y)
        fig.add_annotation(
            x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=8,
            font=dict(size=12, color=t["ink2"]),
        )


def money_chart(sweep: pd.DataFrame, policy_labels: dict, mode: str = "light", height: int = 400) -> go.Figure:
    """Savings vs forecast timing shift, one line per policy.

    sweep needs columns policy, shift_minutes, savings_usd (for one forecaster).
    """
    t = THEMES[mode]
    fig = go.Figure()
    ends = []
    for policy in POLICY_ORDER:
        d = sweep[sweep.policy == policy].sort_values("shift_minutes")
        if d.empty:
            continue
        color = policy_color(policy, mode)
        fig.add_trace(
            go.Scatter(
                x=d.shift_minutes, y=d.savings_usd, name=policy_labels[policy], mode="lines+markers",
                line=dict(color=color, width=2.5 if policy == "ratchet" else 2, shape="linear"),
                marker=dict(size=8, color=color, line=dict(color=t["surface"], width=2)),
                hovertemplate="%{y:$,.0f}<extra>" + policy_labels[policy] + "</extra>",
            )
        )
        ends.append((policy_labels[policy], float(d.savings_usd.iloc[-1])))

    # bracket the gap at -30 and +30 min between the better meter-driven policy and window
    piv = sweep.pivot(index="shift_minutes", columns="policy", values="savings_usd")
    closed = []
    for p in ("threshold", "ratchet"):
        if p in piv:
            closed.append(p)
    if "window" in piv and closed:
        for m in (-30, 30):
            if m not in piv.index:
                continue
            top = float(piv.loc[m, closed].max())
            low = float(piv.loc[m, "window"])
            fig.add_shape(type="line", x0=m, x1=m, y0=low, y1=top, line=dict(color=t["ink2"], width=1))
            fig.add_shape(type="line", x0=m - 1.6, x1=m + 1.6, y0=low, y1=low, line=dict(color=t["ink2"], width=1))
            fig.add_shape(type="line", x0=m - 1.6, x1=m + 1.6, y0=top, y1=top, line=dict(color=t["ink2"], width=1))
            fig.add_annotation(
                x=m, y=(top + low) / 2, text=f"<b>{money(top - low)}</b><br>gap", showarrow=False,
                xanchor="right" if m < 0 else "left", xshift=-8 if m < 0 else 8,
                font=dict(size=12, color=t["ink"]), align="right" if m < 0 else "left",
            )

    ymax = float(sweep.savings_usd.max())
    _end_labels(fig, mode, float(sweep.shift_minutes.max()), ends, min_gap=0.07 * max(ymax, 1))
    _layout(fig, mode, height)
    fig.update_layout(hovermode="x unified", margin=dict(r=150))

    shifts = sorted(sweep.shift_minutes.unique())
    tick_text = []
    for m in shifts:
        tick_text.append(f"{m:+d}" if m else "0")
    fig.update_xaxes(
        title="Forecast timing error (minutes; positive = forecast runs late)",
        tickvals=shifts,
        ticktext=tick_text,
        range=[sweep.shift_minutes.min() - 4, sweep.shift_minutes.max() + 4],
    )
    fig.update_yaxes(title="Annual savings", tickprefix="$", tickformat="~s", rangemode="tozero")
    return fig


def forecaster_bars(
    summary: pd.DataFrame, labels: dict, policy_labels: dict, selected: str | None = None,
    mode: str = "light", height: int | None = None,
) -> go.Figure:
    """Horizontal bars: savings for each forecaster, one bar per policy.

    summary needs forecaster, policy, savings_usd, pct_of_oracle.
    """
    t = THEMES[mode]
    present = set(summary.forecaster)
    order = []
    for f in labels:
        if f in present:
            order.append(f)
    tick = {}
    for f in order:
        tick[f] = f"<b>{labels[f]}</b>" if f == selected else labels[f]  # bold the selected one

    fig = go.Figure()
    # y axis is reversed below, so the first trace ends up on top within each group
    for policy in POLICY_ORDER:
        d = summary[summary.policy == policy].set_index("forecaster").reindex(order).dropna(subset=["savings_usd"])
        ys, text = [], []
        for f, p in zip(d.index, d.pct_of_oracle):
            ys.append(tick[f])
            text.append("ceiling" if f == "oracle" else f"{p:.0f}%")
        fig.add_trace(
            go.Bar(
                y=ys, x=d.savings_usd, orientation="h", name=policy_labels[policy],
                marker=dict(color=policy_color(policy, mode), cornerradius=4),
                text=text, textposition="outside", cliponaxis=False,
                textfont=dict(color=t["ink2"], size=11),
                customdata=np.c_[d.pct_of_oracle],
                hovertemplate="%{x:$,.0f} a year, %{customdata[0]:.0f}% of perfect foresight<extra>"
                + policy_labels[policy] + "</extra>",
            )
        )
    _layout(fig, mode, height or 70 + 62 * len(order))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.12, margin=dict(r=48))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=t["ink2"], size=12))
    fig.update_xaxes(title="Annual savings (label: % of perfect-foresight savings captured)",
                     tickprefix="$", tickformat="~s", rangemode="tozero")
    return fig


def building_bars(rows: pd.DataFrame, labels: dict, policy_labels: dict, mode: str = "light", height: int | None = None) -> go.Figure:
    """Real meters: one row per building, one bar per policy.

    The buildings are very different sizes, so the x axis is the share of the
    best perfect-foresight savings rather than dollars. rows needs building,
    policy, pct_of_best_oracle and savings_usd.
    """
    t = THEMES[mode]
    order = list(labels)
    fig = go.Figure()
    for policy in POLICY_ORDER:
        d = rows[rows.policy == policy].set_index("building").reindex(order).dropna(subset=["pct_of_best_oracle"])
        ys, text = [], []
        for b, p in zip(d.index, d.pct_of_best_oracle):
            ys.append(labels[b])
            text.append(f"{p:.0f}%")
        fig.add_trace(
            go.Bar(
                y=ys, x=d.pct_of_best_oracle, orientation="h", name=policy_labels[policy],
                marker=dict(color=policy_color(policy, mode), cornerradius=4),
                text=text, textposition="outside", cliponaxis=False, textfont=dict(color=t["ink2"], size=11),
                customdata=np.c_[d.savings_usd],
                hovertemplate="%{x:.0f}% of the best perfect-foresight savings, %{customdata[0]:$,.0f} a year<extra>"
                + policy_labels[policy] + "</extra>",
            )
        )
    _layout(fig, mode, height or 70 + 62 * len(order))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.12, margin=dict(r=48))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=t["ink2"], size=12))
    fig.update_xaxes(title="Savings as a share of the best perfect-foresight result (%)", ticksuffix="%", rangemode="tozero")
    return fig


def day_explorer(day: pd.DataFrame, mode: str = "light", height: int = 520, net_label: str = "Net load (with battery)") -> go.Figure:
    """One day of simulate() output: load on top, stored energy underneath."""
    t = THEMES[mode]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.07)
    x = day.index

    fig.add_trace(go.Scatter(x=x, y=day.forecast_kw, name="Day-ahead forecast",
                             line=dict(color=t["neutral_light"], width=1.5, dash="dot"),
                             hovertemplate="%{y:,.0f} kW<extra>Forecast</extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=day.actual_kw, name="Actual load", line=dict(color=t["ink2"], width=1.5),
                             hovertemplate="%{y:,.0f} kW<extra>Actual</extra>"), 1, 1)
    # filling to the previous trace (actual) shades whatever the battery did
    fig.add_trace(go.Scatter(x=x, y=day.net_kw, name=net_label, line=dict(color=t["accent"], width=2.5),
                             fill="tonexty", fillcolor=t["accent_wash"],
                             hovertemplate="%{y:,.0f} kW<extra>Net</extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=day.target_kw, name="kW target",
                             line=dict(color=t["neutral_dark"], width=1.5, dash="dash", shape="hv"),
                             hovertemplate="%{y:,.0f} kW<extra>Target</extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=day.soc_kwh, name="State of charge", showlegend=False,
                             line=dict(color=t["neutral_dark"], width=2), fill="tozeroy", fillcolor=t["neutral_wash"],
                             hovertemplate="%{y:,.0f} kWh<extra>State of charge</extra>"), 2, 1)

    # dot + label on the day's actual peak and on the peak after the battery
    for col, label in (("actual_kw", "peak"), ("net_kw", "net peak")):
        i = day[col].to_numpy().argmax()
        dot_color = t["accent"] if col == "net_kw" else t["ink2"]
        fig.add_trace(go.Scatter(x=[x[i]], y=[day[col].iloc[i]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=9, color=dot_color, line=dict(color=t["surface"], width=2))), 1, 1)
        fig.add_annotation(x=x[i], y=day[col].iloc[i], text=f"{label} {day[col].iloc[i]:,.0f} kW", showarrow=False,
                           yshift=14, font=dict(size=11, color=t["ink2"]), row=1, col=1)

    _layout(fig, mode, height)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="kW", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title="kWh stored", tickformat=",.0f", rangemode="tozero", row=2, col=1)
    three_hours = 3 * 3600 * 1000
    fig.update_xaxes(tickformat="%-I %p", dtick=three_hours, row=1, col=1)
    fig.update_xaxes(tickformat="%-I %p", dtick=three_hours, row=2, col=1)
    return fig


def monthly_chart(monthly: pd.DataFrame, mode: str = "light", height: int = 360) -> go.Figure:
    """Demand charge per month, without and with the battery."""
    t = THEMES[mode]
    names = pd.to_datetime(monthly["month"]).dt.strftime("%b")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=monthly.demand_cost_before, name="Without battery",
                         marker=dict(color=t["neutral_bar"], cornerradius=4), customdata=monthly.peak_kw_before,
                         hovertemplate="%{y:$,.0f}, peak %{customdata:,.0f} kW<extra>Without battery</extra>"))
    fig.add_trace(go.Bar(x=names, y=monthly.demand_cost_after, name="With battery",
                         marker=dict(color=t["accent"], cornerradius=4), customdata=monthly.peak_kw_after,
                         hovertemplate="%{y:$,.0f}, peak %{customdata:,.0f} kW<extra>With battery</extra>"))
    _layout(fig, mode, height)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(title="Demand charge per month", tickprefix="$", tickformat="~s", rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    return fig


def error_scatter(daily: pd.DataFrame, policy: str, policy_labels: dict, mode: str = "light", height: int = 420):
    """Each business day: peak forecast error vs dollars lost against perfect foresight.

    Dots for the selected policy, a fitted line for every policy. Returns
    (figure, {policy: dollars lost per kW of error}).
    """
    t = THEMES[mode]
    daily = daily.assign(lost_usd=-daily.vs_oracle_usd)
    fig = go.Figure()
    sel = daily[daily.policy == policy]
    fig.add_trace(
        go.Scatter(
            x=sel.abs_peak_error_kw, y=sel.lost_usd, mode="markers", name=f"Business days ({policy_labels[policy].lower()})",
            marker=dict(size=8, color=policy_color(policy, mode), opacity=0.45, line=dict(color=t["surface"], width=1)),
            customdata=np.c_[_day_label(sel.date), sel.peak_error_kw],
            hovertemplate="%{customdata[0]}<br>peak error %{customdata[1]:+,.0f} kW<br>"
            "%{y:$,.0f} lost vs perfect foresight<extra></extra>",
        )
    )

    slopes, ends = {}, []
    xmax = float(daily.abs_peak_error_kw.quantile(0.99))  # ignore the odd extreme day for the x range
    xs = np.array([0.0, xmax])
    for p in POLICY_ORDER:
        d = daily[daily.policy == p]
        if len(d) < 3:
            continue
        slope, intercept = np.polyfit(d.abs_peak_error_kw, d.lost_usd, 1)
        slopes[p] = float(slope)
        fig.add_trace(
            go.Scatter(
                x=xs, y=intercept + slope * xs, mode="lines", name=f"Fit: {policy_labels[p]}",
                line=dict(color=policy_color(p, mode), width=3 if p == policy else 1.5),
                hovertemplate=f"{money(slope)} per kW of error<extra>{policy_labels[p]}</extra>",
            )
        )
        ends.append((f"{policy_labels[p]}: {money(slope)}/kW", float(intercept + slope * xmax)))

    span = float(daily.lost_usd.max() - daily.lost_usd.min()) or 1.0
    _end_labels(fig, mode, xmax, ends, min_gap=0.06 * span)
    _layout(fig, mode, height)
    fig.update_layout(margin=dict(r=190))
    fig.update_xaxes(title="Day-ahead error in the day's peak (|kW|)", range=[0, xmax * 1.02], tickformat=",.0f")
    fig.update_yaxes(title="Lost vs perfect foresight ($/day)", tickprefix="$", tickformat="~s", rangemode="tozero")
    return fig, slopes
