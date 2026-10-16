"""Real-data check of the parity network on currency deposits held by a US-dollar investor.

The registration is docs/preregistration-parity-network-fx.md. The script reads cached FRED files
under ``$SOFTEDGE_DATA/fred`` (default ``data/fred``) and never downloads. It writes the weekly
panel, per-currency results and the registered claims as JSON under ``--out``.

    python scripts/parity_fx.py --out <folder>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from softedge.parity import ParityConfig
from softedge.parity import backtest as bt

# name: (FX series, quoted in units per dollar, foreign short rate, cost per unit traded)
CURRENCIES = {
    "GBP": ("DEXUSUK", False, "IR3TIB01GBM156N", 0.0002),
    "JPY": ("DEXJPUS", True, "IR3TIB01JPM156N", 0.0002),
    "CHF": ("DEXSZUS", True, "IR3TIB01CHM156N", 0.0002),
    "CAD": ("DEXCAUS", True, "IR3TIB01CAM156N", 0.0002),
    "AUD": ("DEXUSAL", False, "IR3TIB01AUM156N", 0.0002),
    "EUR": ("DEXUSEU", False, "IR3TIB01EZM156N", 0.0002),
    "NZD": ("DEXUSNZ", False, "IR3TIB01NZM156N", 0.0002),
    "ZAR": ("DEXSFUS", True, "IR3TIB01ZAM156N", 0.0010),
    "MXN": ("DEXMXUS", True, "IR3TIB01MXM156N", 0.0010),
    "KRW": ("DEXKOUS", True, "IR3TIB01KRM156N", 0.0010),
    "CNY": ("DEXCHUS", True, "IR3TIB01CNM156N", 0.0010),
    "BRL": ("DEXBZUS", True, "IRSTCI01BRM156N", 0.0010),
    "INR": ("DEXINUS", True, "IRSTCI01INM156N", 0.0010),
}
USD_CASH = "DTB3"
STALE_DAYS = 4
CUTS = (60, 200, 400, 600, 800, 1000)
BLOCK = 52
DRAWS = 10_000
SEED = 20261016


def read_series(root: str, series_id: str) -> pd.Series:
    raw = pd.read_csv(os.path.join(root, f"{series_id}.csv"), na_values=["."])
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy()
    out = pd.Series(values, index=pd.to_datetime(raw.iloc[:, 0]))
    return out.dropna().sort_index()


def as_of(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """The last value dated on or before each date, NaN before the first."""
    pos = series.index.searchsorted(dates, side="right") - 1
    vals = np.where(pos >= 0, series.to_numpy()[np.maximum(pos, 0)], np.nan)
    return pd.Series(vals, index=dates)


def weekly_panel(currency: str, root: str) -> pd.DataFrame:
    """Weekly log returns of the deposit and of US cash, one row per week ending on a Friday.

    The deposit return is the log change of the dollar price of the currency between Friday
    closes plus the foreign rate of the month before the week starts, accrued over 7 days. Cash
    is DTB3 on the Friday the week starts, accrued over 7 days. A Friday takes the last FX
    observation on or before it.
    """
    fx_id, per_usd, rate_id, _ = CURRENCIES[currency]
    fx = read_series(root, fx_id)
    price = 1.0 / fx if per_usd else fx
    fridays = pd.date_range(price.index[0], price.index[-1], freq="W-FRI")
    px = as_of(price, fridays)
    obs = price.index[price.index.searchsorted(fridays, side="right") - 1]
    start = fridays[:-1]
    month_before = (start.to_period("M") - 1).to_timestamp()
    foreign = as_of(read_series(root, rate_id), month_before).to_numpy()
    usd = as_of(read_series(root, USD_CASH), start).to_numpy()
    frame = pd.DataFrame({
        "fx": np.diff(np.log(px.to_numpy())),
        "carry": np.log1p(foreign / 100.0 * 7.0 / 365.0),
        "cash": np.log1p(usd / 100.0 * 7.0 / 365.0),
        "stale": (fridays[1:] - obs[1:]).days > STALE_DAYS,
    }, index=fridays[1:])
    frame["risky"] = frame["fx"] + frame["carry"]
    valid = frame[["fx", "carry", "cash"]].notna().all(axis=1).to_numpy()
    first = int(np.argmax(valid))
    frame = frame.iloc[first:]
    if frame[["fx", "carry", "cash"]].isna().any().any():
        raise RuntimeError(f"{currency}: missing values after the first complete week")
    return frame


def policies(cost: float) -> dict:
    """Every policy run on one currency, as callables of (risky, cash)."""
    base = ParityConfig()
    out = {"parity_network": lambda r, c: bt.run_parity(r, base, cash=c, cost=cost, trace=True)}
    for name, cfg in {"direction_off": replace(base, direction=False),
                      "gate_off": replace(base, gate=False),
                      "governor_off": replace(base, governor=False),
                      "default_one": replace(base, default_exposure=1.0)}.items():
        out[name] = lambda r, c, cfg=cfg: (bt.run_parity(r, cfg, cash=c, cost=cost)[0], None)
    for name, make in bt.SIMPLE_RULES.items():
        out[name] = lambda r, c, make=make: (bt.simulate(r, make(), cash=c, cost=cost), None)
    out["trend_governed"] = lambda r, c: (
        bt.simulate(r, bt.Governed(bt.TrendGate()), cash=c, cost=cost), None)
    return out


def metrics(res: dict, cash: np.ndarray) -> dict:
    g = res["growth"][0, bt.WARMUP:]
    wealth = res["wealth"][0]
    peak = np.maximum.accumulate(wealth)
    gross = wealth[bt.WARMUP:] / wealth[bt.WARMUP - 1:-1]
    excess = gross - np.exp(cash[bt.WARMUP:])
    held = res["held"][0, bt.WARMUP - 1:]
    return {
        "annual_log_growth": float(bt.ANNUAL * g.mean()),
        "cagr": float(math.expm1(bt.ANNUAL * g.mean())),
        "max_drawdown": float(np.max(1.0 - wealth[bt.WARMUP:] / peak[bt.WARMUP:])),
        "breach": bool(res["breach"][0]),
        "sharpe": float(math.sqrt(bt.ANNUAL) * excess.mean() / excess.std(ddof=1)),
        "mean_exposure": float(held[1:].mean()),
        "annual_turnover": float(bt.ANNUAL * np.abs(np.diff(held)).mean()),
    }


def scramble_probe(risky: np.ndarray, cash: np.ndarray, cost: float) -> dict:
    base = bt.run_parity(risky, cash=cash, cost=cost)[0]
    failed = []
    for t0 in CUTS:
        if t0 >= risky.shape[1] - 2:
            continue
        rng = np.random.default_rng(t0)
        flips = np.where(rng.random(risky.shape[1] - t0 - 1) < 0.5, -1.0, 1.0)
        scrambled = risky.copy()
        scrambled[0, t0 + 1:] = flips * risky[0, t0 + 1:]
        other = bt.run_parity(scrambled, cash=cash, cost=cost)[0]
        if not (np.array_equal(base["target"][:, : t0 + 1], other["target"][:, : t0 + 1])
                and np.array_equal(base["held"][:, : t0 + 2], other["held"][:, : t0 + 2])):
            failed.append(t0)
    return {"passed": not failed, "failed_cuts": failed}


def pooled(diffs: dict, calendar: pd.DatetimeIndex) -> dict:
    """Annualised mean of weekly differences over all currency-weeks, with a block bootstrap.

    Weeks are resampled on the shared calendar in circular blocks of 52, so every currency
    keeps the same resampled weeks and their correlation is kept.
    """
    total = np.zeros(calendar.size)
    count = np.zeros(calendar.size)
    for series in diffs.values():
        idx = calendar.get_indexer(series.index)
        np.add.at(total, idx, series.to_numpy())
        np.add.at(count, idx, 1.0)
    mean = bt.ANNUAL * total.sum() / count.sum()
    rng = np.random.default_rng(SEED)
    n = calendar.size
    blocks = math.ceil(n / BLOCK)
    stats = np.empty(DRAWS)
    offsets = np.arange(BLOCK)
    for i in range(DRAWS):
        starts = rng.integers(0, n, blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        stats[i] = bt.ANNUAL * total[idx].sum() / count[idx].sum()
    low, high = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(mean), "low": float(low), "high": float(high),
            "currency_weeks": int(count.sum())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=os.environ.get("SOFTEDGE_DATA") or "data")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    root = os.path.join(args.data, "fred")
    panel_info, results, growth = {}, {}, {}
    for cost_scale in (1.0, 2.0):
        tag = "" if cost_scale == 1.0 else "_2x_cost"
        for cur, (_, _, _, unit_cost) in CURRENCIES.items():
            frame = weekly_panel(cur, root)
            risky = frame["risky"].to_numpy()[None, :]
            cash = frame["cash"].to_numpy()
            cost = unit_cost * cost_scale
            if cost_scale == 1.0:
                panel_info[cur] = {
                    "first_week": str(frame.index[0].date()), "last_week": str(frame.index[-1].date()),
                    "weeks": int(len(frame)), "stale_weeks": int(frame["stale"].sum()),
                    "annual_fx": float(bt.ANNUAL * frame["fx"].mean()),
                    "annual_carry": float(bt.ANNUAL * frame["carry"].mean()),
                    "annual_cash": float(bt.ANNUAL * frame["cash"].mean()),
                    "cost": unit_cost}
                results[cur] = {"scramble_probe": scramble_probe(risky, cash, cost), "policies": {}}
            for name, run in policies(cost).items():
                if cost_scale != 1.0 and name not in ("parity_network", "trend_governed",
                                                      *bt.SIMPLE_RULES):
                    continue
                res, net = run(risky, cash)
                key = name + tag
                results[cur]["policies"][key] = metrics(res, cash)
                growth.setdefault(key, {})[cur] = pd.Series(
                    res["growth"][0, bt.WARMUP:], index=frame.index[bt.WARMUP:])
                if net is not None and cost_scale == 1.0:
                    w = np.stack(net.trace["weight"], axis=1)[0]
                    e = np.stack(net.trace["evidence"], axis=1)[0]
                    results[cur]["network"] = {
                        "deployed_share_by_ledger": [float(v) for v in (w > 0).mean(axis=0)],
                        "final_evidence": [float(v) for v in e[-1]],
                        "max_evidence": [float(v) for v in e.max(axis=0)]}
    calendar = pd.DatetimeIndex(sorted({d for s in growth["parity_network"].values()
                                        for d in s.index}))
    pooled_growth = {k: bt.ANNUAL * float(np.concatenate([s.to_numpy() for s in v.values()]).mean())
                     for k, v in growth.items()}
    feasible = [name for name in bt.SIMPLE_RULES
                if not any(results[c]["policies"][name]["breach"] for c in CURRENCIES)]
    best = max(feasible, key=lambda name: pooled_growth[name]) if feasible else None

    def diff(a, b):
        return pooled({c: growth[a][c] - growth[b][c] for c in CURRENCIES}, calendar)

    comparisons = {ref: diff("parity_network", ref) for ref in
                   list(bt.SIMPLE_RULES) + ["trend_governed", "direction_off", "gate_off",
                                            "governor_off", "default_one"]}
    comparisons["parity_network_2x_cost_vs_" + (best or "none")] = (
        diff("parity_network_2x_cost", best + "_2x_cost") if best else None)
    network_breaches = [c for c in CURRENCIES if results[c]["policies"]["parity_network"]["breach"]]
    rd1 = {"best_rule": best, "feasible_rules": feasible,
           "difference": comparisons[best] if best else None,
           "network_breaches": network_breaches}
    rd1["passed"] = bool(best and comparisons[best]["mean"] > 0.0 and comparisons[best]["low"] > 0.0
                         and not network_breaches)
    rd2 = {"difference": comparisons["direction_off"],
           "passed": bool(comparisons["direction_off"]["low"] > 0.0)}
    probes = {c: results[c]["scramble_probe"]["passed"] for c in CURRENCIES}
    claims = {"RD1": rd1, "RD2": rd2, "RD3": {"scramble_probe": probes,
                                               "passed": all(probes.values())}}
    out = {"panel": panel_info, "pooled_annual_log_growth": pooled_growth,
           "comparisons": comparisons, "claims": claims, "currencies": results}
    with open(os.path.join(args.out, "fx_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(claims, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
