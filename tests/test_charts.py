import pandas as pd

from src import charts

POLICY_LABELS = {"threshold": "Threshold", "ratchet": "Threshold + ratchet", "window": "Window"}


def test_money_formats():
    assert charts.money(893) == "$893"
    assert charts.money(-3043) == "-$3.0k"
    assert charts.money(9995) == "$10k"
    assert charts.money(1250, signed=True) == "+$1.2k"
    assert charts.usd(402118) == "$402,118"
    assert charts.usd(-922, signed=True) == "-$922"


def test_every_policy_keeps_its_colour_in_both_themes():
    for mode in ("light", "dark"):
        seen = set()
        for p in POLICY_LABELS:
            seen.add(charts.policy_color(p, mode))
        assert len(seen) == 3


def test_money_chart_builds_with_the_gap_brackets():
    rows = []
    for p in POLICY_LABELS:
        for m in (-30, 0, 30):
            rows.append({"policy": p, "shift_minutes": m, "savings_usd": 1000.0 if p != "window" or m == 0 else 200.0})
    fig = charts.money_chart(pd.DataFrame(rows), POLICY_LABELS)
    assert len(fig.data) == 3
    texts = []
    for a in fig.layout.annotations:
        texts.append(a.text)
    assert "<b>$800</b><br>gap" in texts
