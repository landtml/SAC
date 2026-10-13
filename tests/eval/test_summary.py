import json

import numpy as np
import pandas as pd

from sactrader.eval.summary import COLUMNS, run_summary


def test_summary_tests_gated_against_best_baseline(tmp_path):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=1500, freq="1D", tz="UTC", name="time")
    noise = rng.normal(0, 0.01, len(idx))
    df = pd.DataFrame({"sac_banded": noise + 0.002, "sac_gated": noise + 0.002,
                       "buy_and_hold": noise, "trend": noise + 0.001, "flat": 0.0}, index=idx)
    run = tmp_path / "r1"
    run.mkdir()
    df.to_csv(run / "pooled_returns.csv")
    (run / "config.yaml").write_text("universe: u\n")
    summary = {c: {"sharpe": float(df[c].mean() / (df[c].std() or 1) * np.sqrt(365)),
                   "max_drawdown": -0.1} for c in df.columns}
    (run / "result.json").write_text(json.dumps({"summary": summary}))
    (tmp_path / "unfinished").mkdir()
    dsr = pd.DataFrame([{"run": "r1", "strategy": "sac_gated", "dsr": 0.97, "n_trials": 3}])
    table = run_summary(tmp_path, dsr_table=dsr, n_boot=300)
    assert list(table.columns) == COLUMNS and len(table) == 1
    row = table.iloc[0]
    assert row["best_baseline"] == "trend" and row["gated_dsr"] == 0.97 and row["n_trials"] == 3
    assert row["gated_minus_baseline"] > 0 and row["diff_lo"] <= row["gated_minus_baseline"]
