import json

from sactrader.cli import main
from sactrader.experiment import run_experiment

CFG = {
    "name": "tiny",
    "universe": "synthetic",
    "symbols": ["A", "B"],
    "data": {"start": "2021-01", "end": "2021-03"},
    "costs": {"fee_bps": 5.0, "half_spread_bps": 1.0, "borrow_apr": 0.0},
    "split": {"train": [None, "2021-02-01"], "val": ["2021-02-01", "2021-02-15"],
              "test": ["2021-02-15", None]},
    "env": {"window": 4, "episode_len": 64, "n_envs": 8},
    "sac": {"hidden": 32, "critic_blocks": 1, "actor_hidden": 16, "batch_size": 64},
    "train": {"total_steps": 1200, "warmup_steps": 400, "utd": 0.25, "eval_every": 50,
              "threads": 1},
    "seeds": [0, 1],
}


def test_run_experiment_writes_report_and_counts_trials(tmp_path):
    res = run_experiment(CFG, tmp_path)
    assert {"sac_seed0", "sac_seed1", "sac_ensemble", "sac_banded", "flat", "buy_and_hold",
            "trend"} <= set(res["summary"])
    assert res["no_trade_band"]["chosen"] in res["no_trade_band"]["val_sharpe"]
    assert res["summary"]["flat"]["sharpe"] == 0.0
    out = tmp_path / "tiny"
    assert (out / "agent_seed0.pt").exists() and "## Pooled" in (out / "report.md").read_text()
    res2 = run_experiment({**CFG, "name": "tiny2", "seeds": [0]}, tmp_path)
    assert res2["n_trials"] == 2
    rows = [json.loads(line) for line in open(tmp_path / "trials.jsonl")]
    assert [r["name"] for r in rows] == ["tiny", "tiny2"]


def test_cli_run(tmp_path, capsys):
    import yaml

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({**CFG, "seeds": [0]}))
    assert main(["run", str(path), "--out", str(tmp_path / "runs")]) == 0
    assert "sac_seed0" in json.loads(capsys.readouterr().out)


def test_band_holds_exposure_inside_threshold():
    import numpy as np

    from sactrader.env.vec_env import N_CONTEXT
    from sactrader.experiment import banded

    obs = np.zeros((3, 5 + N_CONTEXT), dtype=np.float32)
    obs[:, -N_CONTEXT] = [0.5, 0.5, -0.2]
    act = banded(lambda o: np.array([0.55, 0.9, -0.2]), 0.1)
    np.testing.assert_allclose(act(obs), [0.5, 0.9, -0.2])
