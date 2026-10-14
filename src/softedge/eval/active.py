"""Active return of a learned policy over its prior rule, with its deflated Sharpe ratio.

For a run whose ``pooled_returns.csv`` has a ``prior_only`` column, the active return is the
policy's per-bar return minus the prior's on the same bars. Runs without that column are
skipped.

The trial count is the number of SAC variants that ``trials.jsonl`` records for the run's
universe (``trial_stats``). ``dsr`` takes the null floor as the cross-trial variance, which is
what the post-hoc script behind docs/results.md used. ``dsr_cross_trial`` takes the variance
of the registry's per-bar Sharpe ratios, which is what ``recompute_dsr`` uses.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from softedge.eval.stats import dsr_summary
from softedge.experiment import load_config, trial_stats

PRIOR = "prior_only"
COLUMNS = ["run", "universe", "strategy", "n_bars", "sharpe", "psr", "n_trials", "dsr",
           "dsr_cross_trial"]


def active_table(runs: str | Path, strategy: str = "sac_banded") -> pd.DataFrame:
    """One row per run with a ``prior_only`` column: the Sharpe and DSR of ``strategy`` minus it."""
    root = Path(runs)
    rows = []
    for run in sorted(p for p in root.iterdir() if p.is_dir() and (p / "config.yaml").exists()):
        pooled_path = run / "pooled_returns.csv"
        if not pooled_path.exists():
            continue
        if PRIOR not in pd.read_csv(pooled_path, nrows=0).columns:
            continue
        pooled = pd.read_csv(pooled_path, index_col=0, parse_dates=True,
                             float_precision="round_trip")
        active = (pooled[strategy] - pooled[PRIOR]).dropna()
        universe = load_config(run / "config.yaml")["universe"]
        n_trials, registry_variance = trial_stats(root, universe, [])
        bars_per_year = _bars_per_year(active.index)
        null_floor = dsr_summary(active.to_numpy(), bars_per_year, n_trials)
        cross_trial = dsr_summary(active.to_numpy(), bars_per_year, n_trials,
                                  registry_variance)
        rows.append({
            "run": run.name, "universe": universe, "strategy": strategy,
            "n_bars": len(active), "sharpe": null_floor["sharpe"], "psr": null_floor["psr"],
            "n_trials": n_trials, "dsr": null_floor["dsr"],
            "dsr_cross_trial": cross_trial["dsr"],
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    span = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
    return (len(index) - 1) / span
