"""Run directional hypotheses on one return panel and write results.json and results.md.

    PYTHONPATH=src python scripts/direction_study.py --out DIR
    PYTHONPATH=src python scripts/direction_study.py --out DIR --panel csv \
        --csv returns.csv --costs-json costs.json

``--panel fred`` loads the long FRED panel from ``softedge.data.long_daily``. ``--panel csv``
reads a wide CSV with dates in the first column and one column per asset, plus a JSON object
mapping each asset to its cost in basis points per unit of turnover. ``--panel crypto`` is not
wired yet and raises ``NotImplementedError``.

Each hypothesis runs on the panel, on the same panel with a one-day delay, and on
``--n-null`` random-walk panels matched on assets, dates and vols. The false-positive rate is
the share of null panels that pass, with a 95% Wilson interval. The DSR is reported twice:
registered (the repo function at the trial count) and sensitivity (the same function with the
variance of the trial Sharpe ratios observed in this run).

On the FRED panel a missing print on a calendar day keeps the previous position.
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
BARS_PER_YEAR = {"fred": 252, "csv": 252, "crypto": 365}
DEFAULT_N_NULL = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--panel", choices=["fred", "csv", "crypto"], default="fred")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--costs-json", type=Path)
    parser.add_argument("--bars-per-year", type=float, default=None,
                        help="annualisation; defaults to 252 for fred and csv, 365 for crypto")
    parser.add_argument("--hypotheses", default=",".join(ALL_HYPOTHESES),
                        help="comma-separated subset of " + ", ".join(ALL_HYPOTHESES))
    parser.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series]:
    if args.panel == "crypto":
        raise NotImplementedError(
            "the crypto panel needs load_crypto_xs, which is not in this checkout yet")
    if args.panel == "fred":
        from softedge.data.long_daily import load_daily_panel

        return load_daily_panel()
    if not args.csv or not args.costs_json:
        raise SystemExit("--panel csv needs --csv and --costs-json")
    returns = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    if returns.index.tz is None:
        returns.index = pd.DatetimeIndex(returns.index).tz_localize("UTC")
    returns = returns.sort_index().astype(np.float64)
    costs_map = json.loads(Path(args.costs_json).read_text())
    costs = pd.Series({a: float(costs_map[a]) for a in returns.columns}, name="cost_bp")
    return returns, costs


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _weights(name: str, panel: pd.DataFrame, bars_per_year: float, carry_gaps: bool) -> pd.DataFrame:
    weights = d.HYPOTHESES[name](panel, bars_per_year=bars_per_year)
    if carry_gaps:
        weights = d.hold_on_missing(weights, panel)
    return weights


def run(returns: pd.DataFrame, costs: pd.Series, hypotheses: list[str], *,
        n_null: int = DEFAULT_N_NULL, seed: int = 0, panel_name: str = "csv",
        bars_per_year: float = 252, carry_gaps: bool = False) -> dict:
    nulls = d.null_random_walk_panels(returns, n_panels=n_null, seed=seed)
    trials = {}
    for name in hypotheses:
        weights = _weights(name, returns, bars_per_year, carry_gaps)
        main = d.trial_returns(weights, returns, costs)
        delayed = d.trial_returns(weights, returns, costs, delay=1)
        null_nets = [d.trial_returns(_weights(name, p, bars_per_year, carry_gaps), p, costs)
                     for p in nulls]
        trials[name] = (main, delayed, null_nets)

    summaries = {name: d.summarize(main, bars_per_year=bars_per_year)
                 for name, (main, _, _) in trials.items()}
    observed_var = d.observed_sr_variance(
        [s["sharpe"] for s in summaries.values()], bars_per_year=bars_per_year)

    rows = {}
    for name, (main, delayed, null_nets) in trials.items():
        row = dict(summaries[name])
        null_rows = [d.summarize(net, bars_per_year=bars_per_year) for net in null_nets]
        passes = [r["passes"] for r in null_rows]
        k, n = int(sum(passes)), len(passes)
        lo, hi = d.wilson_interval(k, n) if n else (float("nan"), float("nan"))
        delay_row = d.summarize(delayed, bars_per_year=bars_per_year)
        row["dsr_sensitivity"] = (d.deflated_sharpe(main, d.N_TRIALS, sr_variance=observed_var)
                                  if main.size else float("nan"))
        row["delay_1"] = {k2: delay_row[k2] for k2 in ("sharpe", "nw_t", "dsr", "passes")}
        row["null"] = {
            "n_panels": n,
            "pass_count": k,
            "pass_share": (k / n) if n else float("nan"),
            "pass_share_ci95": [lo, hi],
            "nw_t": [r["nw_t"] for r in null_rows],
        }
        rows[name] = row

    return {
        "source": panel_name,
        "n_trials": d.N_TRIALS,
        "bars_per_year": bars_per_year,
        "observed_trial_sr_variance": observed_var,
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
        f"{len(result['panel']['assets'])} assets. Trials counted: {result['n_trials']}. "
        f"Annualised at {result['bars_per_year']:g} bars per year.",
        "",
        "| Hypothesis | Sharpe | NW t | DSR | DSR (observed variance) | Sharpe 1st half | "
        "Sharpe 2nd half | Min. detectable Sharpe | Delay-1 Sharpe | Null pass share "
        "[95% CI] | Pass |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(x):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"

    for name, r in result["hypotheses"].items():
        lo, hi = r["null"]["pass_share_ci95"]
        share = r["null"]["pass_share"]
        null_cell = "n/a" if share is None or math.isnan(share) else (
            f"{share:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(
            f"| {name} | {fmt(r['sharpe'])} | {fmt(r['nw_t'])} | {fmt(r['dsr'])} | "
            f"{fmt(r['dsr_sensitivity'])} | {fmt(r['sharpe_first_half'])} | "
            f"{fmt(r['sharpe_second_half'])} | {fmt(r['min_detectable_sharpe'])} | "
            f"{fmt(r['delay_1']['sharpe'])} | {null_cell} | "
            f"{'yes' if r['passes'] else 'no'} |"
        )
    lines += [
        "",
        f"Not run: {', '.join(result['not_run'])} (SAC residual check runs only after a "
        "hypothesis passes).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    chosen = [h.strip() for h in args.hypotheses.split(",") if h.strip()]
    unknown = [h for h in chosen if h not in d.HYPOTHESES]
    if unknown:
        parser.error(f"unknown hypotheses {unknown}, expected a subset of {ALL_HYPOTHESES}")

    returns, costs = _load(args)
    bars = args.bars_per_year or BARS_PER_YEAR[args.panel]
    result = run(returns, costs, chosen, n_null=args.n_null, seed=args.seed,
                 panel_name=args.panel, bars_per_year=bars,
                 carry_gaps=args.panel == "fred")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(_clean(result), indent=2, allow_nan=False) + "\n")
    (args.out / "results.md").write_text(_markdown(result))
    print(f"wrote {args.out / 'results.json'} and {args.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
