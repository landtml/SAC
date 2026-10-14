"""Run directional hypotheses on one return panel and write results.json and results.md.

    PYTHONPATH=src python scripts/direction_study.py --out DIR
    PYTHONPATH=src python scripts/direction_study.py --out DIR --source csv \
        --csv returns.csv --costs-json costs.json

``--source fred`` loads the long FRED panel from ``softedge.data.long_daily``. ``--source csv``
reads a wide CSV with dates in the first column and one column per asset, plus a JSON object
mapping each asset to its cost in basis points per unit of turnover.

Each hypothesis runs on the panel, on the same panel delayed one extra day, and on
``--n-null`` random-walk panels matched on assets, dates and vols. The false-positive rate is
the share of null panels that pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from softedge.eval import direction as d

ALL_HYPOTHESES = list(d.HYPOTHESES)
NOT_RUN = ["H4a", "H4b"]


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, str]:
    if args.source == "fred":
        from softedge.data.long_daily import load_daily_panel

        returns, costs = load_daily_panel()
        return returns, costs, "fred:long_daily"
    if not args.csv or not args.costs_json:
        raise SystemExit("--source csv needs --csv and --costs-json")
    returns = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    returns.index = pd.DatetimeIndex(returns.index).tz_localize("UTC") \
        if returns.index.tz is None else returns.index
    returns = returns.sort_index().astype(np.float64)
    costs_map = json.loads(Path(args.costs_json).read_text())
    costs = pd.Series({a: float(costs_map[a]) for a in returns.columns}, name="cost_bp")
    return returns, costs, f"csv:{Path(args.csv).name}"


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _run(name: str, panel: pd.DataFrame, costs: pd.Series, delay: int = 0) -> dict:
    weights = d.HYPOTHESES[name](panel)
    return d.evaluate(weights, panel, costs, delay=delay)


def run(returns: pd.DataFrame, costs: pd.Series, hypotheses: list[str], n_null: int,
        seed: int, source: str) -> dict:
    nulls = d.null_random_walk_panels(returns, n_panels=n_null, seed=seed)
    rows = {}
    for name in hypotheses:
        main = _run(name, returns, costs)
        delayed = _run(name, returns, costs, delay=1)
        null_rows = [_run(name, panel, costs) for panel in nulls]
        rows[name] = {
            **main,
            "delay_1": {k: delayed[k] for k in ("sharpe", "nw_t", "dsr", "passes")},
            "null": {
                "n_panels": len(null_rows),
                "pass_count": int(sum(r["passes"] for r in null_rows)),
                "false_positive_rate": float(np.mean([r["passes"] for r in null_rows])),
                "nw_t": [r["nw_t"] for r in null_rows],
            },
        }
    return {
        "source": source,
        "n_trials": d.N_TRIALS,
        "pass_rule": {"nw_t_gt": d.PASS_NW_T, "dsr_gt": d.PASS_DSR,
                      "sharpe_positive_in_both_halves": True},
        "panel": {
            "first_date": str(returns.index[0].date()),
            "last_date": str(returns.index[-1].date()),
            "n_rows": int(returns.shape[0]),
            "assets": list(returns.columns),
            "first_valid": {c: str(returns[c].first_valid_index().date())
                            for c in returns.columns if returns[c].first_valid_index() is not None},
            "cost_bp": {c: float(costs[c]) for c in returns.columns},
        },
        "hypotheses": rows,
        "not_run": NOT_RUN,
    }


def _markdown(result: dict) -> str:
    lines = [
        "# Directional study results",
        "",
        f"Source: {result['source']}. Panel {result['panel']['first_date']} to "
        f"{result['panel']['last_date']}, {result['panel']['n_rows']} rows, "
        f"{len(result['panel']['assets'])} assets. Trials counted: {result['n_trials']}.",
        "",
        "| Hypothesis | Sharpe | NW t | DSR | Sharpe 1st half | Sharpe 2nd half | "
        "Min. detectable Sharpe | Delay-1 Sharpe | Null FPR | Pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"

    for name, r in result["hypotheses"].items():
        lines.append(
            f"| {name} | {fmt(r['sharpe'])} | {fmt(r['nw_t'])} | {fmt(r['dsr'])} | "
            f"{fmt(r['sharpe_first_half'])} | {fmt(r['sharpe_second_half'])} | "
            f"{fmt(r['min_detectable_sharpe'])} | {fmt(r['delay_1']['sharpe'])} | "
            f"{fmt(r['null']['false_positive_rate'])} | {'yes' if r['passes'] else 'no'} |"
        )
    lines += [
        "",
        f"Not run: {', '.join(result['not_run'])} (SAC residual check runs only after a "
        "hypothesis passes).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", choices=["fred", "csv"], default="fred")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--costs-json", type=Path)
    parser.add_argument("--hypotheses", default=",".join(ALL_HYPOTHESES),
                        help="comma-separated subset of " + ", ".join(ALL_HYPOTHESES))
    parser.add_argument("--n-null", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    chosen = [h.strip() for h in args.hypotheses.split(",") if h.strip()]
    unknown = [h for h in chosen if h not in d.HYPOTHESES]
    if unknown:
        parser.error(f"unknown hypotheses {unknown}, expected a subset of {ALL_HYPOTHESES}")

    returns, costs, source = _load(args)
    result = run(returns, costs, chosen, args.n_null, args.seed, source)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(_clean(result), indent=2, allow_nan=False) + "\n")
    (args.out / "results.md").write_text(_markdown(result))
    print(f"wrote {args.out / 'results.json'} and {args.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
