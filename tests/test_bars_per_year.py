import json
import socket

import numpy as np
import pandas as pd
import pytest

from softedge.experiment import (
    _bars_per_year,
    build_all_tapes,
    make_splits,
    recompute_dsr,
    run_experiment,
)
from softedge.pipeline import split_tapes

BASE = {
    "universe": "synthetic",
    "symbols": ["A", "B"],
    "costs": {"fee_bps": 5.0, "half_spread_bps": 1.0, "borrow_apr": 0.0},
    "env": {"window": 4, "episode_len": 64, "n_envs": 8},
    "sac": {"hidden": 32, "critic_blocks": 1, "actor_hidden": 16, "batch_size": 64},
    "train": {"total_steps": 1200, "warmup_steps": 400, "utd": 0.25, "eval_every": 50,
              "threads": 1},
}
# Two folds: the pooled out-of-sample index misses bars at the fold boundary, which is where
# the recomputed bars per year drifts from the median over the test tapes.
WF = {
    **BASE, "name": "wf", "seeds": [0],
    "data": {"start": "2021-01", "end": "2021-04"},
    "walk_forward": {"first_test": "2021-02-15", "end": "2021-03-22", "test": "21D",
                     "val": "10D", "embargo": "1D"},
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
    root = tmp_path_factory.mktemp("bpy_runs")
    run_experiment(WF, root)
    run_experiment(CPCV, root)
    return root


def _result_path(root, name):
    return root / name / "result.json"


def _load(root, name):
    return json.loads(_result_path(root, name).read_text())


def _write(root, name, result):
    _result_path(root, name).write_text(json.dumps(result, indent=2, default=str))


def test_walk_forward_run_writes_its_bars_per_year(runs_root):
    expected_tapes = []
    tapes = build_all_tapes(WF)
    for split in make_splits(WF):
        parts = split_tapes(tapes, split)
        expected_tapes.extend(parts["test"])
    expected = float(np.median([tp.bars_per_year for tp in expected_tapes]))
    assert _load(runs_root, "wf")["bars_per_year"] == expected


def test_cpcv_run_writes_its_bars_per_year(runs_root):
    assert _load(runs_root, "cpcv")["bars_per_year"] > 0


def test_recompute_sharpe_matches_walk_forward_summary(runs_root):
    summary = _load(runs_root, "wf")["summary"]
    table = recompute_dsr(runs_root)
    rows = table[table["run"] == "wf"]
    assert set(rows["strategy"]) == set(summary)
    for _, row in rows.iterrows():
        assert row["sharpe"] == pytest.approx(summary[row["strategy"]]["sharpe"], rel=1e-9)


def test_recompute_sharpe_matches_cpcv_aggregate(runs_root):
    aggregate = _load(runs_root, "cpcv")["aggregate"]
    table = recompute_dsr(runs_root)
    rows = table[table["run"] == "cpcv"]
    assert set(rows["strategy"]) == set(aggregate)
    for _, row in rows.iterrows():
        assert row["sharpe"] == pytest.approx(aggregate[row["strategy"]]["sharpe_mean"], rel=1e-9)


def test_recompute_without_saved_bars_per_year_falls_back_and_warns(tmp_path):
    run_experiment(WF, tmp_path)
    result = _load(tmp_path, "wf")
    saved = result.pop("bars_per_year")
    _write(tmp_path, "wf", result)

    with pytest.warns(UserWarning, match="'wf'"):
        table = recompute_dsr(tmp_path)

    pooled = pd.read_csv(tmp_path / "wf" / "pooled_returns.csv", index_col=0,
                         parse_dates=True)
    fallback = _bars_per_year(pooled.index)
    # Sharpe is linear in sqrt(bars per year), so the fallback is the saved value rescaled.
    for _, row in table[table["run"] == "wf"].iterrows():
        want = result["summary"][row["strategy"]]["sharpe"] * np.sqrt(fallback / saved)
        assert row["sharpe"] == pytest.approx(want, rel=1e-9)


def _backfill_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_bars_per_year.py"
    spec = importlib.util.spec_from_file_location("backfill_bars_per_year", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backfill_recomputes_the_run_time_value_and_changes_nothing_else(runs_root, tmp_path):
    import shutil

    saved = _load(runs_root, "wf")
    shutil.copytree(runs_root / "wf", tmp_path / "wf")
    stripped = {k: v for k, v in saved.items() if k != "bars_per_year"}
    _write(tmp_path, "wf", stripped)

    assert _backfill_module().main(["--runs", str(tmp_path), "--min-age", "0"]) == 0

    filled = _load(tmp_path, "wf")
    assert filled["bars_per_year"] == saved["bars_per_year"]
    assert {k: v for k, v in filled.items() if k != "bars_per_year"} == stripped
    assert json.loads((tmp_path / "wf" / "result.json.bak").read_text()) == stripped


def test_backfill_refuses_network_calls_and_restores_them(tmp_path):
    import urllib.request

    backfill = _backfill_module()
    original = urllib.request.urlopen
    with backfill.no_network():
        with pytest.raises(backfill.NetworkBlocked):
            socket.create_connection(("example.com", 443), timeout=1)
        with pytest.raises(backfill.NetworkBlocked):
            urllib.request.urlopen("https://example.com")
    assert urllib.request.urlopen is original
