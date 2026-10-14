import json

import numpy as np
import pandas as pd

from softedge.experiment import _per_bar_sharpe, recompute_dsr, trial_stats


def _write_run(root, name, universe, series: dict[str, np.ndarray]):
    """A saved walk-forward run: config, pooled per-bar returns, and its trials.jsonl row."""
    d = root / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(f"name: {name}\nuniverse: {universe}\n")
    index = pd.date_range("2021-01-01", periods=len(next(iter(series.values()))), freq="D")
    pd.DataFrame(series, index=index).to_csv(d / "pooled_returns.csv")
    (d / "result.json").write_text(json.dumps({"bars_per_year": 365.0}))
    sr = {s: _per_bar_sharpe(pd.Series(v)) for s, v in series.items() if s.startswith("sac_")}
    with open(root / "trials.jsonl", "a") as f:
        f.write(json.dumps({"name": name, "universe": universe, "sac_sr_bar": sr}) + "\n")
    return sr


def _returns(seed: int, n: int = 200) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0005, 0.01, n)


def test_identical_return_series_count_once(tmp_path):
    base, other = _returns(0), _returns(1)
    sr = _write_run(tmp_path, "a", "u", {"sac_seed0": other, "sac_banded": base, "sac_gated": base.copy()})
    ledger_before = (tmp_path / "trials.jsonl").read_text()

    n, var = trial_stats(tmp_path, "u", [])

    assert n == 2
    assert var == np.var([sr["sac_banded"], sr["sac_seed0"]], ddof=1)
    assert (tmp_path / "trials.jsonl").read_text() == ledger_before  # the ledger keeps every row
    table = recompute_dsr(tmp_path)
    assert (table["n_trials"] == 2).all()


def test_near_identical_return_series_are_both_kept(tmp_path):
    base = _returns(2)
    nudged = base.copy()
    nudged[57] += 1e-12  # one bar differs, far below any Sharpe-visible change
    sr = _write_run(tmp_path, "a", "u", {"sac_banded": base, "sac_gated": nudged})

    n, var = trial_stats(tmp_path, "u", [])

    assert n == 2
    assert var == np.var([sr["sac_banded"], sr["sac_gated"]], ddof=1)


def test_current_run_trials_with_equal_keys_count_once(tmp_path):
    n, var = trial_stats(tmp_path, "u", [0.1, 0.1, 0.3], ["same", "same", "other"])

    assert n == 2
    assert var == np.var([0.1, 0.3], ddof=1)


def test_trial_without_a_key_is_never_merged(tmp_path):
    n, _ = trial_stats(tmp_path, "u", [0.1, 0.1], [None, None])

    assert n == 2
