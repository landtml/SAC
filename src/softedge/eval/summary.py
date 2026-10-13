"""One table over every finished run: the SAC strategies against the baselines of their market.

For each run with a stitched out-of-sample record, ``run_summary`` reports the ungated banded
policy and the validation-gated policy (the predeclared deployable strategies), their Sharpe
ratio, maximum drawdown and deflated Sharpe ratio over every trial of the universe, the best
baseline of the run (by out-of-sample Sharpe, so the comparison is against the strongest simple
rule, chosen with hindsight), and a paired stationary-bootstrap test of the gated policy against
that baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from softedge.eval.stats import sharpe_diff_test

BASELINES = ("buy_and_hold", "trend", "always_no", "prior_only", "flat")
COLUMNS = ["run", "universe", "years", "banded_sharpe", "gated_sharpe", "gated_max_dd", "gated_dsr",
           "n_trials", "best_baseline", "baseline_sharpe", "gated_minus_baseline", "diff_lo",
           "diff_hi", "p_value"]


def run_summary(root: str | Path, dsr_table: pd.DataFrame | None = None,
                n_boot: int = 1000) -> pd.DataFrame:
    """One row per walk-forward or split run in ``root``; ``dsr_table`` is ``recompute_dsr``."""
    root = Path(root)
    if dsr_table is None:
        from softedge.experiment import recompute_dsr

        dsr_table = recompute_dsr(root)
    dsr = dsr_table.set_index(["run", "strategy"])
    rows = []
    for res_path in sorted(root.glob("*/result.json")):
        run_dir = res_path.parent
        pooled = run_dir / "pooled_returns.csv"
        result = json.loads(res_path.read_text())
        if "summary" not in result or not pooled.exists():
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        df = pd.read_csv(pooled, index_col=0, parse_dates=True)
        if "sac_gated" not in df.columns or len(df) < 3:
            continue
        span = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400)
        bpy = (len(df) - 1) / span
        summ = result["summary"]
        base = [b for b in BASELINES if b in df.columns and b != "flat"]
        best = max(base, key=lambda b: summ[b]["sharpe"]) if base else "flat"
        key = (run_dir.name, "sac_gated")
        row = {"run": run_dir.name, "universe": cfg.get("universe"), "years": span,
               "banded_sharpe": summ.get("sac_banded", {}).get("sharpe", np.nan),
               "gated_sharpe": summ["sac_gated"]["sharpe"],
               "gated_max_dd": summ["sac_gated"]["max_drawdown"],
               "gated_dsr": float(dsr.loc[key, "dsr"]) if key in dsr.index else np.nan,
               "n_trials": int(dsr.loc[key, "n_trials"]) if key in dsr.index else 0,
               "best_baseline": best, "baseline_sharpe": summ[best]["sharpe"]}
        test = sharpe_diff_test(df["sac_gated"].to_numpy(), df[best].to_numpy(), bpy,
                                n_boot=n_boot)
        row.update(gated_minus_baseline=test["diff"], diff_lo=test["lo"], diff_hi=test["hi"],
                   p_value=test["p_value"])
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)
