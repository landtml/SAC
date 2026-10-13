"""Paired comparison of runs that share their out-of-sample bars (ablations, variants).

Deflated Sharpe ratios answer "is this strategy better than luck?". Choosing between two variants
evaluated on the same markets and periods is a sharper question: the paired difference of their
returns cancels the shared market noise. ``compare_runs`` reads each run's ``pooled_returns.csv``,
aligns it with the base run on common timestamps and reports the Sharpe ratio of one strategy,
the mean Sharpe over seeds, the share of the sign oracle's Sharpe when the market has one, and a
paired stationary-bootstrap test of the difference against the base.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from softedge.eval.stats import pbo_cscv, sharpe_diff_test

COLUMNS = ["run", "n_bars", "sharpe", "seed_mean_sharpe", "oracle_share", "diff_vs_base",
           "diff_lo", "diff_hi", "p_value"]


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    years = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
    return (len(index) - 1) / years if years > 0 else float("nan")


def _sharpe(r: np.ndarray, bpy: float) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0


def _load(root: Path, name: str) -> pd.DataFrame:
    path = root / name / "pooled_returns.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; the run has not finished")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def compare_runs(root: str | Path, base: str, names: list[str], strategy: str = "sac_banded",
                 n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per run in ``[base] + names``; differences are against ``base`` on common bars."""
    root = Path(root)
    ref = _load(root, base)
    rows = []
    for name in [base, *[n for n in names if n != base]]:
        df = _load(root, name)
        common = ref.index.intersection(df.index)
        if len(common) < 3:
            raise ValueError(f"{name} shares {len(common)} bars with {base}")
        r = df.loc[common, strategy].to_numpy(np.float64)
        r0 = ref.loc[common, strategy].to_numpy(np.float64)
        bpy = _bars_per_year(common)
        seeds = [c for c in df.columns if c.startswith("sac_seed")]
        row = {"run": name, "n_bars": len(common), "sharpe": _sharpe(r, bpy),
               "seed_mean_sharpe": float(np.mean([_sharpe(df.loc[common, c].to_numpy(), bpy)
                                                  for c in seeds])) if seeds else float("nan"),
               "oracle_share": float("nan")}
        if "oracle" in df.columns:
            oracle = _sharpe(df.loc[common, "oracle"].to_numpy(np.float64), bpy)
            row["oracle_share"] = row["sharpe"] / oracle if oracle > 0 else float("nan")
        if name == base:
            row.update(diff_vs_base=0.0, diff_lo=float("nan"), diff_hi=float("nan"),
                       p_value=float("nan"))
        else:
            test = sharpe_diff_test(r, r0, bpy, n_boot=n_boot, seed=seed)
            row.update(diff_vs_base=test["diff"], diff_lo=test["lo"], diff_hi=test["hi"],
                       p_value=test["p_value"])
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def universe_pbo(root: str | Path, universe: str, n_splits: int = 16) -> dict:
    """Probability of backtest overfitting across every SAC variant tried on ``universe``.

    Collects the ``sac_*`` columns of every finished run of the universe, keeps the timestamps
    all of them share, and runs CSCV on that T x N matrix. Returns the PBO with the number of
    variants and bars it rests on. Raises ``ValueError`` below two variants or 2 * n_splits bars.
    """
    root = Path(root)
    cols = []
    for res in sorted(root.glob("*/result.json")):
        run = res.parent.name
        cfg_path = res.parent / "config.yaml"
        pooled = res.parent / "pooled_returns.csv"
        if not pooled.exists() or not cfg_path.exists():
            continue
        if yaml.safe_load(cfg_path.read_text()).get("universe") != universe:
            continue
        json.loads(res.read_text())  # an unreadable result means an unfinished run
        df = pd.read_csv(pooled, index_col=0, parse_dates=True)
        cols += [df[c].rename(f"{run}:{c}") for c in df.columns if c.startswith("sac_")]
    if len(cols) < 2:
        raise ValueError(f"universe {universe!r} has {len(cols)} SAC variants, need two")
    matrix = pd.concat(cols, axis=1, join="inner")
    return {"universe": universe, "pbo": pbo_cscv(matrix.to_numpy(), n_splits),
            "n_variants": matrix.shape[1], "n_bars": matrix.shape[0]}
