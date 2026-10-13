"""Regenerate the result figures in assets/figures from the CSVs in docs/results/data.

Run from the repository root: ``python scripts/make_figures.py``. The SVGs are written
directly with numpy and pandas only. ``carry_equity.csv`` is rebuilt from a runs directory
(``SOFTEDGE_RUNS``, default ``runs/``) when it holds the crypto_basis_1d_prior_v2 run;
otherwise the committed CSV is used unchanged.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "results" / "data"
FIGS = ROOT / "assets" / "figures"
RUN = "crypto_basis_1d_prior_v2"
EQUITY_COLUMNS = ("sac_banded", "prior_only", "buy_and_hold")

# Each figure draws its own light panel, so its colors are fixed light values and do not
# follow the page theme.
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
ACCENT = "#2a78d6"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

ROW_H = 26
HEAD_H = 24
GAP = 10
RULE_H = 20
PANEL_GAP = 16


@dataclass(frozen=True)
class Row:
    label: str
    est: float
    lo: float
    hi: float
    detail: str = ""
    accent: bool = False
    verdict: str = ""


@dataclass(frozen=True)
class Group:
    heading: str
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Panel:
    groups: tuple[Group, ...]
    x_lo: float
    x_hi: float
    ticks: tuple[float, ...]
    zero_label: str
    caption: str


@dataclass(frozen=True)
class Layout:
    width: int
    label_x: float
    plot_x0: float
    plot_x1: float
    value_x: float
    verdict_x: float | None = None


def _text(
    x: float,
    y: float,
    s: str,
    size: float = 13,
    anchor: str = "start",
    fill: str = INK,
    weight: str | None = None,
) -> str:
    extra = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size:g}" '
        f'text-anchor="{anchor}" fill="{fill}"{extra}>{escape(s)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:g}" stroke-linecap="round"/>'
    )


def _svg(
    width: int, height: int, title: str, subtitle: tuple[str, ...], desc: str, body: list[str]
) -> str:
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
        f'<title id="t">{escape(title)}</title>',
        f'<desc id="d">{escape(desc)}</desc>',
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8" '
        f'fill="{SURFACE}" stroke="{GRID}"/>',
        _text(24, 34, title, size=18, weight="600"),
    ]
    head += [_text(24, 56 + 18 * i, line, fill=INK_2) for i, line in enumerate(subtitle)]
    return "\n".join(head + body + ["</svg>"]) + "\n"


def _body_top(subtitle: tuple[str, ...]) -> float:
    return 56 + 18 * (len(subtitle) - 1) + 30


def _draw_row(parts: list[str], row: Row, cy: float, x_of, lay: Layout) -> None:
    color = ACCENT if row.accent else INK_2
    tip = f"{row.label}: {row.est:+.2f}, 95% CI [{row.lo:.2f}, {row.hi:.2f}]"
    if row.detail:
        tip += f" ({row.detail})"
    hit_x = lay.label_x - 4
    parts.append("<g>")
    parts.append(f"<title>{escape(tip)}</title>")
    parts.append(
        f'<rect x="{hit_x:.1f}" y="{cy - 12:.1f}" width="{lay.width - 2 * hit_x:.1f}" '
        'height="24" fill="transparent"/>'
    )
    parts.append(_text(lay.label_x, cy + 4.5, row.label))
    parts.append(_line(x_of(row.lo), cy, x_of(row.hi), cy, color, 2))
    parts.append(
        f'<circle cx="{x_of(row.est):.1f}" cy="{cy:.1f}" r="5" fill="{color}" '
        f'stroke="{SURFACE}" stroke-width="2"/>'
    )
    value = f"{row.est:+.2f} [{row.lo:.2f}, {row.hi:.2f}]"
    parts.append(_text(lay.value_x, cy + 4.5, value, size=12, fill=INK_2))
    if lay.verdict_x is not None and row.verdict:
        parts.append(_text(lay.verdict_x, cy + 4.5, row.verdict, size=12, fill=INK_2))
    parts.append("</g>")


def _draw_panel(parts: list[str], top: float, panel: Panel, lay: Layout, heads: bool) -> float:
    span = lay.plot_x1 - lay.plot_x0

    def x_of(v: float) -> float:
        return lay.plot_x0 + (v - panel.x_lo) / (panel.x_hi - panel.x_lo) * span

    rows_h = sum(HEAD_H + ROW_H * len(g.rows) + GAP for g in panel.groups)
    bottom = top + RULE_H + rows_h
    for t in panel.ticks:
        if t != 0:
            parts.append(_line(x_of(t), top + 19, x_of(t), bottom, GRID))
    parts.append(_line(x_of(0), top + 19, x_of(0), bottom, INK_2))
    parts.append(_text(x_of(0), top + 13, panel.zero_label, size=11, anchor="middle", fill=INK_2))
    if heads and lay.verdict_x is not None:
        parts.append(_text(lay.value_x, top + 13, "Change, 95% CI", size=11, fill=INK_2))
        parts.append(_text(lay.verdict_x, top + 13, "Verdict", size=11, fill=INK_2))

    y = top + RULE_H
    for group in panel.groups:
        parts.append(_text(lay.label_x, y + 16, group.heading, weight="600"))
        y += HEAD_H
        for row in group.rows:
            _draw_row(parts, row, y + ROW_H / 2, x_of, lay)
            y += ROW_H
        y += GAP

    parts.append(_line(lay.plot_x0, bottom, lay.plot_x1, bottom, AXIS))
    for t in panel.ticks:
        parts.append(_text(x_of(t), bottom + 15, f"{t:g}", size=11, anchor="middle", fill=MUTED))
    mid = (lay.plot_x0 + lay.plot_x1) / 2
    parts.append(_text(mid, bottom + 34, panel.caption, size=12, anchor="middle", fill=INK_2))
    return bottom + 40


def forest_svg(
    title: str, subtitle: tuple[str, ...], desc: str, panels: tuple[Panel, ...], lay: Layout
) -> str:
    body: list[str] = []
    top = _body_top(subtitle)
    for i, panel in enumerate(panels):
        top = _draw_panel(body, top, panel, lay, heads=i == 0) + PANEL_GAP
    return _svg(lay.width, math.ceil(top - PANEL_GAP + 8), title, subtitle, desc, body)


def _rows(sub: pd.DataFrame, accent: bool) -> tuple[Row, ...]:
    sub = sub.sort_values("diff", ascending=False, kind="stable")
    return tuple(
        Row(r.market, r.diff, r.lo, r.hi, r.detail, accent) for r in sub.itertuples()
    )


def sac_vs_rule_svg(df: pd.DataFrame) -> str:
    def group(key: str, heading: str, accent: bool) -> Group:
        return Group(heading, _rows(df[df["group"] == key], accent))

    controls = Panel(
        groups=(group("control", "Synthetic markets and controls", False),),
        x_lo=-6,
        x_hi=18,
        ticks=(-5, 0, 5, 10, 15),
        zero_label="no difference",
        caption="SAC minus rule, Sharpe, 95% CI. Own scale, not comparable to the panel below.",
    )
    real = Panel(
        groups=(
            group("directional", "Directional markets, no edge", False),
            group("carry", "Funding-carry markets", True),
        ),
        x_lo=-2,
        x_hi=1.5,
        ticks=(-2, -1, 0, 1),
        zero_label="no difference",
        caption="SAC minus rule, Sharpe, 95% CI. Left: rule better. Right: SAC better.",
    )
    subtitle = (
        "Sharpe of SAC minus its best simple rule, out of sample, with 95% CI.",
        "Two of three carry markets clear zero. The Hyperliquid basis CI crosses it (p = 0.13).",
    )
    lay = Layout(width=760, label_x=24, plot_x0=318, plot_x1=566, value_x=590)
    return forest_svg(
        "On real markets, SAC beats its simple rule only in funding-carry markets",
        subtitle,
        "Forest plot: SAC minus its best simple rule, Sharpe, with 95% intervals.",
        (controls, real),
        lay,
    )


def ablations_svg(df: pd.DataFrame) -> str:
    names = {"hidden-regime": "Hidden-regime market", "crypto-carry": "Crypto carry"}
    groups = []
    for base in dict.fromkeys(df["base"]):
        sub = df[df["base"] == base]
        heading = f"{names[base]} (base Sharpe {float(sub['base_sharpe'].iloc[0]):.2f})"
        groups.append(Group(heading, _rows_with_verdict(sub)))
    panel = Panel(
        groups=tuple(groups),
        x_lo=-8,
        x_hi=4,
        ticks=(-8, -6, -4, -2, 0, 2, 4),
        zero_label="same as base",
        caption="Change in ensemble Sharpe vs base, 95% CI. Left: worse. Right: better.",
    )
    subtitle = (
        "Change in ensemble Sharpe from one change to the base agent, with 95% CI "
        "and the verdict for each change.",
    )
    lay = Layout(width=760, label_x=24, plot_x0=300, plot_x1=500, value_x=520, verdict_x=668)
    return forest_svg(
        "Most add-ons do not beat the base agent",
        subtitle,
        "Forest plot: change in ensemble Sharpe for each ablation against its base.",
        (panel,),
        lay,
    )


def _rows_with_verdict(sub: pd.DataFrame) -> tuple[Row, ...]:
    sub = sub.sort_values("diff", ascending=False, kind="stable")
    return tuple(
        Row(r.change, r.diff, r.lo, r.hi, verdict=r.verdict) for r in sub.itertuples()
    )


SERIES = (
    ("sac_banded", "SAC", ACCENT),
    ("prior_only", "Funding rule (prior alone)", INK_2),
    ("buy_and_hold", "Passive carry (buy and hold)", MUTED),
)


def equity_svg(eq: pd.DataFrame) -> str:
    dates = pd.to_datetime(eq["time"])
    first = dates.iloc[0]
    days = (dates - first).dt.days.to_numpy(dtype=float)
    span = days[-1]
    x0, x1, y0, y1 = 64.0, 600.0, 124.0, 372.0
    values = eq[[col for col, _, _ in SERIES]].to_numpy(dtype=float)
    lo = math.floor(values.min() * 20) / 20
    hi = math.ceil(values.max() * 20) / 20

    def x_of(d: float) -> float:
        return x0 + d / span * (x1 - x0)

    def y_of(v: float) -> float:
        return y1 - (v - lo) / (hi - lo) * (y1 - y0)

    body: list[str] = []
    lx = 24.0
    for _, name, color in SERIES:
        body.append(_line(lx, 96, lx + 18, 96, color, 2))
        body.append(_text(lx + 24, 100, name, size=12, fill=INK_2))
        lx += 24 + 7 * len(name) + 22

    for t in np.arange(lo, hi + 1e-9, 0.05):
        body.append(_line(x0, y_of(t), x1, y_of(t), GRID))
        body.append(_text(x0 - 8, y_of(t) + 4, f"${t:.2f}", size=11, anchor="end", fill=MUTED))

    body.append(_line(x0, y1, x1, y1, AXIS))
    marks = [(0.0, str(first.year), "start")]
    for year in range(first.year + 1, dates.iloc[-1].year + 1):
        d = (pd.Timestamp(year=year, month=1, day=1) - first).days
        marks.append((float(d), str(year), "middle"))
    for d, label, anchor in marks:
        body.append(_text(x_of(d), y1 + 18, label, size=11, anchor=anchor, fill=MUTED))

    for col, name, color in SERIES:
        pts = " ".join(
            f"{x_of(d):.1f},{y_of(v):.1f}"
            for d, v in zip(days, eq[col].to_numpy(float), strict=True)
        )
        body.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"><title>{escape(name)}'
            "</title></polyline>"
        )

    end_y = y_of(float(eq["sac_banded"].iloc[-1]))
    body.append(_line(x1 + 10, end_y, x1 + 24, end_y, ACCENT, 2))
    body.append(_text(x1 + 30, end_y + 4, "SAC", weight="600"))

    subtitle = (
        "Growth of $1, out of sample, 2023 to 2026, daily, after modelled costs.",
        "Exchange, counterparty and intrabar liquidation risk are not modelled. "
        "Compare lines, not levels.",
    )
    return _svg(
        760,
        410,
        "On cash-and-carry, SAC's edge over its funding rule is small but positive",
        subtitle,
        "Line chart: growth of $1 over time for SAC, the funding rule and passive carry.",
        body,
    )


def build_carry_equity(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(source, parse_dates=["time"])
    out = pd.DataFrame({"time": raw["time"].dt.strftime("%Y-%m-%d")})
    for col in EQUITY_COLUMNS:
        out[col] = np.cumprod(1.0 + raw[col].to_numpy(dtype=float)).round(6)
    return out


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    runs = Path(os.environ.get("SOFTEDGE_RUNS", ROOT / "runs"))
    source = runs / RUN / "pooled_returns.csv"
    if source.exists():
        build_carry_equity(source).to_csv(DATA / "carry_equity.csv", index=False)

    rules = pd.read_csv(DATA / "sac_vs_rule.csv")
    ablations = pd.read_csv(DATA / "ablations.csv")
    equity = pd.read_csv(DATA / "carry_equity.csv")
    figures = {
        "sac-vs-rule.svg": sac_vs_rule_svg(rules),
        "ablations.svg": ablations_svg(ablations),
        "carry-equity.svg": equity_svg(equity),
    }
    for name, svg in figures.items():
        ET.fromstring(svg)
        (FIGS / name).write_text(svg, encoding="utf-8")
        print(f"wrote assets/figures/{name}")


if __name__ == "__main__":
    main()
