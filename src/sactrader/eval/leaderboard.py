"""Ranking of every saved run by its deflated Sharpe ratio, walk-forward and CPCV alike.

``collect(root)`` returns one row per strategy of each run under ``root``, sorted by ``dsr``
(descending, missing values last). Runs with ``pooled_returns.csv`` (walk-forward and single
split) take their statistics from ``recompute_dsr``, so every row of a universe is deflated by
the current trial count. CPCV runs take their Sharpe, PSR and DSR from ``recompute_dsr`` too (means over backtest
paths, deflated by the same count) and their spread statistics from ``result.json``. A directory without the files its rows need is skipped.
``leaderboard_markdown`` renders the table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sactrader.eval.report import to_markdown
from sactrader.experiment import recompute_dsr

COLUMNS = [
    "run", "universe", "mode", "strategy", "sharpe", "psr", "sr0", "dsr", "n_trials",
    "min_trl_years", "max_drawdown", "cvar_95", "sharpe_diff", "p_value", "deployed_share",
    "sharpe_std", "share_positive", "beats_reference_share",
]


def collect(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows: list[dict] = []
    for rec in recompute_dsr(root).to_dict("records"):
        result = _read_result(root / rec["run"])
        if result is None:
            continue
        if rec["mode"] == "cpcv":
            rows.append(_cpcv_row(rec, result))
        else:
            rows.append(_walk_forward_row(rec, result))
    table = pd.DataFrame(rows, columns=COLUMNS)
    return table.sort_values("dsr", ascending=False, na_position="last", kind="stable",
                             ignore_index=True)


def leaderboard_markdown(df: pd.DataFrame, top: int | None = None) -> str:
    if top is not None and top < 0:
        raise ValueError("top must be non-negative")
    return to_markdown(df if top is None else df.head(top))


def _read_result(run_dir: Path) -> dict | None:
    path = run_dir / "result.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _num(value) -> float:
    # pandas writes a missing summary value as null in result.json.
    return np.nan if value is None else float(value)


def _walk_forward_row(rec: dict, result: dict) -> dict:
    strategy = rec["strategy"]
    summary = result["summary"][strategy]
    return {
        "run": rec["run"], "universe": rec["universe"], "mode": "walk_forward",
        "strategy": strategy, "sharpe": rec["sharpe"], "psr": rec["psr"], "sr0": rec["sr0"],
        "dsr": rec["dsr"], "n_trials": int(rec["n_trials"]),
        "min_trl_years": rec["min_trl_years"], "max_drawdown": _num(summary["max_drawdown"]),
        "cvar_95": _num(summary["cvar_95"]), "sharpe_diff": _num(summary["sharpe_diff"]),
        "p_value": _num(summary["p_value"]),
        "deployed_share": _deployed_share(result) if strategy == "sac_gated" else np.nan,
        "sharpe_std": np.nan, "share_positive": np.nan, "beats_reference_share": np.nan,
    }


def _deployed_share(result: dict) -> float:
    # A skipped fold has no band and is not counted in n_folds either.
    deployed = sum(bool(f["no_trade_band"]["deployed"]) for f in result["folds"]
                   if "no_trade_band" in f)
    return deployed / result["n_folds"]


def _cpcv_row(rec: dict, result: dict) -> dict:
    agg = result["aggregate"][rec["strategy"]]
    return {
        "run": rec["run"], "universe": rec["universe"], "mode": "cpcv",
        "strategy": rec["strategy"], "sharpe": rec["sharpe"], "psr": rec["psr"],
        "sr0": rec["sr0"], "dsr": rec["dsr"], "n_trials": int(rec["n_trials"]),
        "min_trl_years": rec["min_trl_years"], "max_drawdown": _num(agg["max_drawdown_mean"]),
        "cvar_95": _num(agg["cvar_95_mean"]), "sharpe_diff": np.nan, "p_value": np.nan,
        "deployed_share": _cpcv_deployed_share(result) if rec["strategy"] == "sac_gated" else np.nan,
        "sharpe_std": _num(agg["sharpe_std"]), "share_positive": _num(agg["share_positive"]),
        "beats_reference_share": _num(agg["beats_reference_share"]),
    }


def _cpcv_deployed_share(result: dict) -> float:
    splits = result.get("splits", [])
    return float(np.mean([bool(s["no_trade_band"]["deployed"]) for s in splits])) if splits else np.nan
