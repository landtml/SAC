import json

import numpy as np
import pytest

from softedge.eval.leaderboard import COLUMNS, collect, leaderboard_markdown
from softedge.experiment import recompute_dsr, run_experiment

BASE = {
    "universe": "synthetic",
    "symbols": ["A", "B"],
    "costs": {"fee_bps": 5.0, "half_spread_bps": 1.0, "borrow_apr": 0.0},
    "env": {"window": 4, "episode_len": 64, "n_envs": 8},
    "sac": {"hidden": 32, "critic_blocks": 1, "actor_hidden": 16, "batch_size": 64},
    "train": {"total_steps": 1200, "warmup_steps": 400, "utd": 0.25, "eval_every": 50,
              "threads": 1},
}
# Two folds: the second is clipped at the end of the walk-forward range.
WALK_FORWARD = {
    **BASE, "name": "walk_forward", "seeds": [0, 1],
    "data": {"start": "2021-01", "end": "2021-04"},
    "walk_forward": {"first_test": "2021-02-15", "end": "2021-03-22", "test": "21D",
                     "val": "10D", "embargo": "1D"},
}
SINGLE = {
    **BASE, "name": "single", "seeds": [0],
    "data": {"start": "2021-01", "end": "2021-03"},
    "split": {"train": [None, "2021-02-01"], "val": ["2021-02-01", "2021-02-15"],
              "test": ["2021-02-15", None]},
}
CPCV = {
    **BASE, "name": "cpcv", "seeds": [0],
    "data": {"start": "2021-01", "end": "2021-05"},
    "cpcv": {"start": "2021-02-01", "end": "2021-05-01", "n_groups": 4, "n_test": 2,
             "purge": "2D", "embargo": "1D", "val": "7D", "pretrain_val": "10D",
             "split_steps": 600},
}


@pytest.fixture(scope="module")
def runs_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    for cfg in (WALK_FORWARD, SINGLE, CPCV):
        run_experiment(cfg, root)
    return root


def _result(root, name):
    return json.loads((root / name / "result.json").read_text())


def _same(got, want):
    # result.json stores a missing statistic as null, which the leaderboard reads as NaN.
    if want is None:
        assert np.isnan(got)
    else:
        assert got == pytest.approx(want)


def test_each_run_contributes_one_row_per_strategy(runs_root):
    table = collect(runs_root)
    assert list(table.columns) == COLUMNS
    assert _result(runs_root, "walk_forward")["n_folds"] == 2
    for name in ("walk_forward", "single"):
        rows = table[table["run"] == name]
        assert sorted(rows["strategy"]) == sorted(_result(runs_root, name)["summary"])
        assert set(rows["mode"]) == {"walk_forward"}
        assert set(rows["universe"]) == {"synthetic"}
    cpcv = table[table["run"] == "cpcv"]
    assert sorted(cpcv["strategy"]) == sorted(_result(runs_root, "cpcv")["aggregate"])
    assert set(cpcv["mode"]) == {"cpcv"}


def test_sorted_by_dsr_with_missing_values_last(runs_root):
    dsr = collect(runs_root)["dsr"].to_numpy()
    finite = dsr[np.isfinite(dsr)]
    assert len(finite) > 0 and np.isnan(dsr).any()  # flat strategies have no dispersion
    assert np.all(np.diff(finite) <= 0)
    assert not np.isnan(dsr[:len(finite)]).any()
    assert np.isnan(dsr[len(finite):]).all()


def test_walk_forward_rows_count_current_trials_per_universe(runs_root):
    table = collect(runs_root)
    # One universe: every SAC variant of every run, CPCV included, is one trial
    # (five in the two-seed walk-forward run, three in the single split, three in CPCV).
    n_sac = int(table["strategy"].str.startswith("sac_").sum())
    assert set(table["n_trials"]) == {n_sac} == {11}
    assert set(recompute_dsr(runs_root)["n_trials"]) == {11}


def test_deployed_share_is_the_deployed_fold_fraction(runs_root):
    table = collect(runs_root)
    gated = table[table["strategy"] == "sac_gated"]
    result = _result(runs_root, "walk_forward")
    deployed = sum(f["no_trade_band"]["deployed"] for f in result["folds"])
    wf_gated = gated[gated["run"] == "walk_forward"]["deployed_share"]
    assert wf_gated.item() == deployed / result["n_folds"]
    assert gated.loc[gated["mode"] == "walk_forward", "deployed_share"].between(0, 1).all()
    assert gated.loc[gated["mode"] == "cpcv", "deployed_share"].between(0, 1).all()
    assert table.loc[table["strategy"] != "sac_gated", "deployed_share"].isna().all()


def test_walk_forward_summary_fields_join_from_result(runs_root):
    table = collect(runs_root).set_index(["run", "strategy"])
    summary = _result(runs_root, "walk_forward")["summary"]
    for strategy, row in summary.items():
        got = table.loc[("walk_forward", strategy)]
        _same(got["max_drawdown"], row["max_drawdown"])
        _same(got["cvar_95"], row["cvar_95"])
        if strategy != "buy_and_hold":  # the reference row has no difference test
            _same(got["sharpe_diff"], row["sharpe_diff"])
            _same(got["p_value"], row["p_value"])
    assert np.isnan(table.loc[("walk_forward", "buy_and_hold"), "sharpe_diff"])


def test_cpcv_rows_take_path_aggregates(runs_root):
    table = collect(runs_root)
    agg = _result(runs_root, "cpcv")["aggregate"]
    rows = table[table["run"] == "cpcv"].set_index("strategy")
    for strategy, a in agg.items():
        row = rows.loc[strategy]
        # Sharpe is recomputed from the saved paths, annualized from their timestamps.
        assert row["sharpe"] == pytest.approx(a["sharpe_mean"], rel=0.05, abs=1e-9)
        if np.isfinite(row["dsr"]):
            assert row["dsr"] <= row["psr"] + 1e-12
        _same(row["max_drawdown"], a["max_drawdown_mean"])
        _same(row["cvar_95"], a["cvar_95_mean"])
        _same(row["share_positive"], a["share_positive"])
        _same(row["beats_reference_share"], a["beats_reference_share"])


def test_collect_skips_directories_without_needed_files(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "trials.jsonl").write_text("")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "result.json").write_text('{"mode": "cpcv", "aggregate": {}}')
    table = collect(tmp_path)
    assert table.empty
    assert list(table.columns) == COLUMNS


def test_markdown_header_rows_and_top(runs_root):
    table = collect(runs_root)
    lines = leaderboard_markdown(table).splitlines()
    assert lines[0].startswith("| run |")
    assert lines[1].startswith("| --- |")
    assert len(lines) == len(table) + 2
    assert any("n/a" in line for line in lines[2:])
    top = leaderboard_markdown(table, top=2).splitlines()
    assert top == lines[:4]
    assert leaderboard_markdown(table, top=0).count("\n") == 1
    with pytest.raises(ValueError):
        leaderboard_markdown(table, top=-1)
