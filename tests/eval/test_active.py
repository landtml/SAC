import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from softedge.cli import main
from softedge.eval.active import active_table
from softedge.eval.stats import expected_max_sharpe, null_sr_variance

BARS = 40  # even, so the alternating +1/-1 pattern below has zero mean


def _alternating(n: int) -> np.ndarray:
    # Two equal-weight values: symmetric, so skew is 0 and excess kurtosis is exactly -2.
    return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def _write_run(root, name, *, universe="toy", columns=None):
    run = root / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(f"universe: {universe}\n")
    index = pd.date_range("2024-01-01", periods=BARS, freq="D", tz="UTC", name="time")
    pd.DataFrame(columns, index=index).to_csv(run / "pooled_returns.csv")
    return run


def _write_registry(root, rows):
    with open(root / "trials.jsonl", "w") as f:
        for universe, srs in rows:
            f.write(json.dumps({"name": "x", "universe": universe,
                                "sac_sr_bar": {f"s{i}": v for i, v in enumerate(srs)}}) + "\n")


def _toy_root(tmp_path):
    """One run whose active return (sac_banded - prior_only) is 0.002 + 0.01 * (+1, -1, ...)."""
    root = tmp_path / "runs"
    root.mkdir()
    prior = 0.0005 * np.sin(np.arange(BARS))
    active = 0.002 + 0.01 * _alternating(BARS)
    _write_run(root, "toy_run", columns={
        "sac_banded": prior + active, "sac_gated": prior, "prior_only": prior})
    return root, active


def _sr_bar(active):
    return active.mean() / active.std(ddof=1)


def test_sharpe_and_psr_match_closed_form(tmp_path):
    root, active = _toy_root(tmp_path)
    row = active_table(root).set_index("run").loc["toy_run"]

    sr_bar = _sr_bar(active)
    # Symmetric two-point returns have var_term 1, so PSR(0) = Phi(SR * sqrt(n - 1)).
    assert row["sharpe"] == pytest.approx(sr_bar * np.sqrt(365.25), rel=1e-12)
    assert row["psr"] == pytest.approx(stats.norm.cdf(sr_bar * np.sqrt(BARS - 1)), rel=1e-9)


def test_dsr_without_registry_is_psr_because_one_trial_has_zero_benchmark(tmp_path):
    root, active = _toy_root(tmp_path)
    row = active_table(root).set_index("run").loc["toy_run"]

    assert row["n_trials"] == 1
    assert row["dsr"] == pytest.approx(row["psr"], rel=1e-12)


def test_dsr_deflates_by_trials_of_its_universe_in_registry(tmp_path):
    root, active = _toy_root(tmp_path)
    _write_registry(root, [("toy", [0.1 * i for i in range(9)]),
                           ("other", [0.3] * 5)])
    row = active_table(root).set_index("run").loc["toy_run"]

    n = 9
    assert row["n_trials"] == n
    benchmark = expected_max_sharpe(n, null_sr_variance(BARS))
    expected = stats.norm.cdf((_sr_bar(active) - benchmark) * np.sqrt(BARS - 1))
    assert row["dsr"] == pytest.approx(expected, rel=1e-9)
    assert row["dsr"] < row["psr"]


def test_cross_trial_dsr_uses_variance_of_registry_sharpes(tmp_path):
    root, active = _toy_root(tmp_path)
    srs = [0.5 * i for i in range(9)]
    _write_registry(root, [("toy", srs)])
    row = active_table(root).set_index("run").loc["toy_run"]

    variance = float(np.var(srs, ddof=1))
    assert variance > null_sr_variance(BARS)  # so the registry variance, not the floor, is used
    benchmark = expected_max_sharpe(len(srs), variance)
    expected = stats.norm.cdf((_sr_bar(active) - benchmark) * np.sqrt(BARS - 1))
    assert row["dsr_cross_trial"] == pytest.approx(expected, rel=1e-9)


def test_run_without_prior_only_is_skipped(tmp_path):
    root, _ = _toy_root(tmp_path)
    prior = 0.0005 * np.sin(np.arange(BARS))
    _write_run(root, "no_prior", columns={"sac_banded": prior, "flat": prior * 0})

    table = active_table(root)

    assert list(table["run"]) == ["toy_run"]


def test_strategy_argument_selects_gated_column(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    prior = 0.0005 * np.sin(np.arange(BARS))
    gated = prior + 0.002 + 0.01 * _alternating(BARS)
    _write_run(root, "gated_run", columns={
        "sac_banded": prior, "sac_gated": gated, "prior_only": prior})

    row = active_table(root, strategy="sac_gated").iloc[0]

    assert row["strategy"] == "sac_gated"
    assert row["sharpe"] == pytest.approx(_sr_bar(gated - prior) * np.sqrt(365.25), rel=1e-9)


def test_cli_active_writes_only_to_out(tmp_path):
    root, _ = _toy_root(tmp_path)
    _write_registry(root, [("toy", [0.1, 0.2, 0.4])])

    def snapshot():
        return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    before = snapshot()
    target = tmp_path / "reports" / "active.md"

    assert main(["active", "--runs", str(root), "--out", str(target)]) == 0

    assert target.exists()
    text = target.read_text()
    assert "toy_run" in text and "dsr_cross_trial" in text
    assert snapshot() == before
    assert not (root / "active.md").exists()
