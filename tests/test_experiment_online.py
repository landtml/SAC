import pandas as pd

from softedge.experiment import run_experiment

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


def test_online_switch_moves_only_the_deployed_banded_run(tmp_path):
    off = run_experiment({**CFG, "name": "off"}, tmp_path)
    on_cfg = {**CFG, "name": "on", "train": {**CFG["train"], "online_every": 25,
                                             "online_steps": 4, "online_window": 300}}
    on = run_experiment(on_cfg, tmp_path)
    off_r = pd.read_csv(tmp_path / "off" / "pooled_returns.csv", index_col=0, parse_dates=True)
    on_r = pd.read_csv(tmp_path / "on" / "pooled_returns.csv", index_col=0, parse_dates=True)
    # Only the deployed banded run may move. This tiny policy holds flat under its band, so its
    # banded column can coincide with the frozen one; test_blocks_update_the_policy_and_are_counted
    # shows that online updates change the executed targets.
    frozen = [c for c in off_r.columns if c not in ("sac_banded", "sac_gated")]
    pd.testing.assert_frame_equal(off_r[frozen], on_r[frozen])
    band_on = on["folds"][0]["no_trade_band"]
    assert band_on["online"]["blocks"] > 0 and band_on["online"]["updates"] > 0
    assert band_on["online"]["restores"] == 0
    assert [f["no_trade_band"]["deployed"] for f in on["folds"]] == [
        f["no_trade_band"]["deployed"] for f in off["folds"]]
    assert "online" not in off["folds"][0]["no_trade_band"]
