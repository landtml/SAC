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
    assert {"sac_seed0", "sac_seed1", "sac_ensemble", "sac_banded", "sac_gated", "flat", "buy_and_hold",
            "trend"} <= set(res["summary"])
    band = res["folds"][0]["no_trade_band"]
    assert band["chosen"] in band["val_sharpe"]
    gated = res["summary"]["sac_gated"]["sharpe"]
    assert gated == (res["summary"]["sac_banded"]["sharpe"] if band["deployed"] else 0.0)
    assert res["summary"]["flat"]["sharpe"] == 0.0
    out = tmp_path / "tiny"
    assert (out / "agent_fold0_seed0.pt").exists() and "## Out-of-sample summary" in (out / "report.md").read_text()
    assert res["n_trials"] == 5  # two seeds, ensemble, banded, gated
    res2 = run_experiment({**CFG, "name": "tiny2", "seeds": [0]}, tmp_path)
    assert res2["n_trials"] == 8  # earlier five plus seed, banded, gated
    rows = [json.loads(line) for line in open(tmp_path / "trials.jsonl")]
    assert [r["name"] for r in rows] == ["tiny", "tiny2"]


def test_recompute_dsr_counts_trials_per_universe(tmp_path):
    from sactrader.experiment import recompute_dsr

    run_experiment({**CFG, "name": "a", "seeds": [0]}, tmp_path)
    run_experiment({**CFG, "name": "b", "seeds": [1]}, tmp_path)
    table = recompute_dsr(tmp_path)
    assert set(table["run"]) == {"a", "b"}
    assert (table["n_trials"] == 6).all()
    sac = table[table["strategy"].str.startswith("sac_")]
    finite = sac.dropna(subset=["dsr", "psr"])  # a gated, never-deployed series is flat
    assert (finite["dsr"] <= finite["psr"] + 1e-12).all()
    assert main(["dsr", "--runs", str(tmp_path)]) == 0
    assert (tmp_path / "dsr.md").exists()


def test_walk_forward_folds_warm_start_and_stitch(tmp_path):
    cfg = {**CFG, "name": "wf", "seeds": [0], "data": {"start": "2021-01", "end": "2021-04"},
           "walk_forward": {"first_test": "2021-02-15", "end": "2021-04-01", "test": "21D",
                            "val": "10D", "embargo": "1D", "warm_start": True, "fold_steps": 600},
           "train": {**CFG["train"], "target_entropy_end_scale": 2.0}}
    del cfg["split"]
    res = run_experiment(cfg, tmp_path)
    assert res["n_folds"] == 3
    tests = [f["test"] for f in res["folds"]]
    assert all(a[1] <= b[0] for a, b in zip(tests, tests[1:], strict=False))
    pooled = __import__("pandas").read_csv(tmp_path / "wf" / "pooled_returns.csv", index_col=0)
    # two full 21-day folds plus a last fold clipped at the data end (2021-04-01)
    assert pooled.index.is_unique and len(pooled) > 2 * 21 * 24
    logs = [json.loads(line) for line in open(tmp_path / "wf" / "train_log.jsonl")]
    assert {r["fold"] for r in logs} == {0, 1, 2}
    assert logs[-1]["target_entropy"] < -1.0


def test_cli_run(tmp_path, capsys):
    import yaml

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({**CFG, "seeds": [0]}))
    assert main(["run", str(path), "--out", str(tmp_path / "runs")]) == 0
    assert "sac_seed0" in json.loads(capsys.readouterr().out)


def test_discrete_agent_trains_through_the_runner(tmp_path):
    cfg = {**CFG, "name": "disc", "seeds": [0],
           "sac": {**CFG["sac"], "discrete_levels": [-1.0, 0.0, 1.0], "smooth_coef": 0.1,
                   "critic_dropout": 0.1, "obs_noise": 0.05}}
    res = run_experiment(cfg, tmp_path)
    assert "sac_seed0" in res["summary"]
    import torch

    state = torch.load(tmp_path / "disc" / "agent_fold0_seed0.pt", weights_only=False)
    assert state["kind"] == "discrete" and state["levels"] == [-1.0, 0.0, 1.0]


def test_sources_concatenate_universes_with_their_own_costs():
    from sactrader.experiment import build_all_tapes

    src = {"data": {"start": "2021-01", "end": "2021-02"}}
    cfg = {"universe": "mixed", "sources": [
        {**src, "universe": "synthetic", "symbols": ["A", "B"], "costs": {"fee_bps": 1.0}},
        {**src, "universe": "synthetic_null", "symbols": ["C"], "costs": {"fee_bps": 9.0}}]}
    tapes = build_all_tapes(cfg)
    assert [tp.symbol for tp in tapes] == ["A", "B", "C"]
    assert tapes[0].cost_long[0] < tapes[2].cost_long[0]


def test_deflated_gate_is_stricter_than_plain_psr(tmp_path):
    res = run_experiment({**CFG, "name": "gate_dsr", "seeds": [0],
                          "deploy": {"gate": "dsr", "gate_psr": 0.8}}, tmp_path)
    for fold in res["folds"]:
        info = fold["no_trade_band"]
        assert info["gate"] == "dsr"
        assert info["val_psr"] <= info["val_psr_plain"] + 1e-12
        assert info["deployed"] == (info["val_psr"] >= 0.8)


def test_prior_action_map_adds_prior_only_baseline(tmp_path):
    cfg = {**CFG, "name": "prior_run", "seeds": [0],
           "env": {**CFG["env"], "prior_feature": "ret_h4", "vol_target": 0.3}}
    res = run_experiment(cfg, tmp_path)
    assert "prior_only" in res["summary"]
    assert "prior_only" not in run_experiment({**CFG, "name": "plain", "seeds": [0]},
                                              tmp_path)["summary"]
