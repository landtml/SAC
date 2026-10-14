import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import softedge.eval.active as active_mod
from softedge.cli import main
from softedge.eval.active import active_table
from softedge.eval.stats import expected_max_sharpe, null_sr_variance

BARS = 40  # even, so the alternating +1/-1 pattern below has zero mean
FLOOR = null_sr_variance(BARS)


def _alternating(n: int) -> np.ndarray:
    # Two equal-weight values: symmetric, so skew is 0 and excess kurtosis is exactly -2.
    return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def _sr(x) -> float:
    return float(np.mean(x) / np.std(x, ddof=1))


def _write_run(root, name, *, universe="toy", columns=None, env_prior=False):
    run = root / name
    run.mkdir(parents=True)
    text = f"universe: {universe}\n"
    if env_prior:
        text += "env:\n  prior_feature: funding_ewm\n"
    (run / "config.yaml").write_text(text)
    index = pd.date_range("2024-01-01", periods=BARS, freq="D", tz="UTC", name="time")
    pd.DataFrame(columns, index=index).to_csv(run / "pooled_returns.csv")
    return run


def _write_registry(root, rows):
    with open(root / "trials.jsonl", "w") as f:
        for universe, srs in rows:
            f.write(json.dumps({"name": "x", "universe": universe,
                                "sac_sr_bar": {f"s{i}": v for i, v in enumerate(srs)}}) + "\n")


def _toy_universe(root, universe, levels):
    """One run per level. active = level + 0.01 * (+1, -1, ...); sac_banded = prior + active."""
    parts = {}
    for i, level in enumerate(levels):
        prior = 0.0005 * np.sin(np.arange(BARS) + i)
        active = level + 0.01 * _alternating(BARS)
        banded = prior + active
        gated = prior + 0.02 * (i + 1) * _alternating(BARS)
        name = f"{universe}_{i}"
        _write_run(root, name, universe=universe, columns={
            "sac_banded": banded, "sac_gated": gated, "prior_only": prior})
        parts[name] = (active, banded, gated)
    return parts


def _expected_dsr(sr, n_trials, variance):
    # PSR against the expected maximum Sharpe; skew 0 and excess kurtosis -2 give var_term 1.
    benchmark = expected_max_sharpe(n_trials, max(variance, FLOOR))
    return stats.norm.cdf((sr - benchmark) * np.sqrt(BARS - 1))


def _toy_root(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    return root


def test_sharpe_and_psr_match_closed_form(tmp_path):
    root = _toy_root(tmp_path)
    parts = _toy_universe(root, "toy", [0.002])
    active = parts["toy_0"][0]
    row = active_table(root).set_index("run").loc["toy_0"]

    sr_bar = _sr(active)
    assert row["sharpe"] == pytest.approx(sr_bar * np.sqrt(365.25), rel=1e-12)
    assert row["psr"] == pytest.approx(stats.norm.cdf(sr_bar * np.sqrt(BARS - 1)), rel=1e-9)


def test_single_run_has_no_registry_and_dsr_is_psr(tmp_path):
    root = _toy_root(tmp_path)
    _toy_universe(root, "toy", [0.002])
    row = active_table(root).set_index("run").loc["toy_0"]

    assert row["n_trials"] == 1
    assert row["n_with_baseline"] == 1
    assert row["cross_var"] == "floor"
    assert row["dsr"] == pytest.approx(row["psr"], rel=1e-12)
    assert row["dsr_floor"] == pytest.approx(row["psr"], rel=1e-12)


def test_primary_floor_and_abs_dsr_are_three_different_known_values(tmp_path):
    root = _toy_root(tmp_path)
    parts = _toy_universe(root, "toy", [0.0, 0.004, 0.008])
    _write_registry(root, [("toy", [0.1, 0.2, 0.3]), ("other", [0.9] * 4)])
    row = active_table(root).set_index("run").loc["toy_1"]

    active_srs = [_sr(parts[name][0]) for name in sorted(parts)]
    active_var = float(np.var(active_srs, ddof=1))
    abs_srs = [_sr(x) for name in sorted(parts) for x in parts[name][1:]]
    abs_var = float(np.var(abs_srs, ddof=1))
    sr = _sr(parts["toy_1"][0])

    assert row["n_trials"] == 3
    assert row["n_with_baseline"] == 3
    assert row["cross_var"] == "active"
    assert row["dsr"] == pytest.approx(_expected_dsr(sr, 3, active_var), rel=1e-9)
    assert row["dsr_floor"] == pytest.approx(_expected_dsr(sr, 3, FLOOR), rel=1e-9)
    assert row["dsr_abs"] == pytest.approx(_expected_dsr(sr, 3, abs_var), rel=1e-9)
    values = [row["dsr"], row["dsr_floor"], row["dsr_abs"]]
    assert min(abs(a - b) for i, a in enumerate(values) for b in values[i + 1:]) > 1e-3


def test_fewer_than_three_baselines_falls_back_to_floor_and_flags(tmp_path):
    root = _toy_root(tmp_path)
    _toy_universe(root, "pair", [0.0, 0.008])
    _write_registry(root, [("pair", [0.1, 0.2, 0.3, 0.4])])
    table = active_table(root).set_index("run")

    for name in ("pair_0", "pair_1"):
        row = table.loc[name]
        assert row["n_with_baseline"] == 2
        assert row["cross_var"] == "floor"
        assert row["dsr"] == pytest.approx(row["dsr_floor"], rel=1e-12)


def test_run_without_prior_column_or_prior_feature_is_skipped(tmp_path):
    root = _toy_root(tmp_path)
    _toy_universe(root, "toy", [0.002])
    prior = 0.0005 * np.sin(np.arange(BARS))
    _write_run(root, "no_prior", columns={"sac_banded": prior, "flat": prior * 0})

    assert list(active_table(root)["run"]) == ["toy_0"]


def test_run_without_prior_column_gets_rule_when_config_declares_prior_feature(
        tmp_path, monkeypatch):
    root = _toy_root(tmp_path)
    prior = 0.0005 * np.sin(np.arange(BARS))
    active = 0.002 + 0.01 * _alternating(BARS)
    _write_run(root, "carry", universe="carry", env_prior=True, columns={
        "sac_banded": prior + active, "sac_gated": prior})
    rule_index = pd.date_range("2024-01-01", periods=BARS, freq="D", tz="UTC", name="time")
    calls = []

    def fake_rule(run, cfg):
        calls.append(run.name)
        return pd.Series(prior, index=rule_index)

    monkeypatch.setattr(active_mod, "rule_prior", fake_rule)
    row = active_table(root).set_index("run").loc["carry"]

    assert calls == ["carry"]
    assert row["sharpe"] == pytest.approx(_sr(active) * np.sqrt(365.25), rel=1e-12)


def test_strategy_argument_selects_gated_column(tmp_path):
    root = _toy_root(tmp_path)
    prior = 0.0005 * np.sin(np.arange(BARS))
    gated = prior + 0.002 + 0.01 * _alternating(BARS)
    _write_run(root, "gated_run", columns={
        "sac_banded": prior, "sac_gated": gated, "prior_only": prior})

    row = active_table(root, strategy="sac_gated").iloc[0]

    assert row["strategy"] == "sac_gated"
    assert row["sharpe"] == pytest.approx(_sr(gated - prior) * np.sqrt(365.25), rel=1e-9)


def test_cli_active_writes_only_to_out(tmp_path):
    root = _toy_root(tmp_path)
    _toy_universe(root, "toy", [0.0, 0.004, 0.008])
    _write_registry(root, [("toy", [0.1, 0.2, 0.4])])

    def snapshot():
        return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    before = snapshot()
    target = tmp_path / "reports" / "active.md"

    assert main(["active", "--runs", str(root), "--out", str(target)]) == 0

    text = target.read_text()
    assert "toy_1" in text and "dsr_abs" in text and "dsr_floor" in text
    assert snapshot() == before
    assert not (root / "active.md").exists()
