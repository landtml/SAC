import json

import pandas as pd

from softedge.cpcv_experiment import carve_validation
from softedge.experiment import run_experiment

CFG = {
    "name": "cpcv_tiny",
    "universe": "synthetic",
    "symbols": ["A", "B"],
    "data": {"start": "2021-01", "end": "2021-05"},
    "costs": {"fee_bps": 5.0, "half_spread_bps": 1.0, "borrow_apr": 0.0},
    "cpcv": {"start": "2021-02-01", "end": "2021-05-01", "n_groups": 4, "n_test": 2,
             "purge": "2D", "embargo": "1D", "val": "7D", "pretrain_val": "10D",
             "split_steps": 600},
    "env": {"window": 4, "episode_len": 64, "n_envs": 8},
    "sac": {"hidden": 32, "critic_blocks": 1, "actor_hidden": 16, "batch_size": 64},
    "train": {"total_steps": 1200, "warmup_steps": 400, "utd": 0.25, "eval_every": 50,
              "threads": 1},
    "seeds": [0],
}


def test_carve_validation_leaves_purge_gap():
    t = pd.Timestamp("2021-01-01", tz="UTC")
    d = pd.Timedelta
    wins, val = carve_validation([(t, t + d("30D")), (t + d("40D"), t + d("80D"))], d("7D"), d("2D"))
    assert val == (t + d("73D"), t + d("80D"))
    assert wins[-1] == (t + d("40D"), t + d("71D"))
    _, none = carve_validation([(t, t + d("10D"))], d("7D"), d("2D"))
    assert none is None


def test_cpcv_run_builds_paths_and_counts_trials(tmp_path):
    res = run_experiment(CFG, tmp_path)
    assert res["mode"] == "cpcv" and res["n_splits"] == 6 and res["n_paths"] == 3
    agg = res["aggregate"]
    assert {"sac_seed0", "sac_banded", "sac_gated", "flat", "buy_and_hold"} <= set(agg)
    assert agg["flat"]["sharpe_mean"] == 0.0
    paths = pd.read_csv(tmp_path / "cpcv_tiny" / "paths.csv")
    assert sorted(paths["path"].unique()) == [0, 1, 2]
    p0 = pd.read_csv(tmp_path / "cpcv_tiny" / "path0_returns.csv", index_col=0, parse_dates=True)
    # A path covers the whole CPCV span once: about 89 days of hourly bars minus window warm-up.
    assert p0.index.is_unique and len(p0) > 85 * 24
    rows = [json.loads(line) for line in open(tmp_path / "trials.jsonl")]
    assert rows[-1]["mode"] == "cpcv" and res["n_trials"] == 3
