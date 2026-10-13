import numpy as np
import pytest

from sactrader.eval.stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    sharpe_diff_test,
    stationary_bootstrap_indices,
)


def test_psr_is_half_when_sample_sharpe_equals_benchmark():
    r = np.random.default_rng(1).normal(0.02, 1.0, 300)
    sr = r.mean() / r.std(ddof=1)
    assert probabilistic_sharpe_ratio(r, sr_benchmark=sr) == pytest.approx(0.5, abs=1e-12)


def test_psr_is_nan_without_dispersion():
    assert np.isnan(probabilistic_sharpe_ratio(np.full(10, 0.01)))


def test_expected_max_sharpe_grows_with_trials_and_is_zero_for_one():
    assert expected_max_sharpe(1, 0.01) == 0.0
    values = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert np.all(np.diff(values) > 0)


def test_dsr_decreases_as_n_trials_grows():
    r = np.random.default_rng(2).normal(0.05, 1.0, 250)
    values = [deflated_sharpe_ratio(r, n, sr_variance=0.0025) for n in (1, 2, 10, 100)]
    assert np.all(np.diff(values) < 0)


def test_stationary_bootstrap_continues_blocks_with_probability_one_minus_inverse_block():
    rng = np.random.default_rng(5)
    idx = stationary_bootstrap_indices(1000, 10.0, 50, rng)
    assert idx.shape == (50, 1000)
    assert idx.min() >= 0 and idx.max() < 1000
    nxt = (idx[:, :-1] + 1) % 1000 == idx[:, 1:]
    assert abs(nxt.mean() - 0.9) < 0.02


def test_sharpe_diff_p_value_small_when_a_is_clearly_better():
    rng = np.random.default_rng(6)
    a = rng.normal(0.002, 0.01, 1000)
    b = rng.normal(0.0, 0.01, 1000)
    out = sharpe_diff_test(a, b, bars_per_year=252.0, seed=0)
    assert out["diff"] > 0
    assert out["lo"] < out["diff"] < out["hi"]
    assert out["p_value"] < 0.05


def test_bootstrap_ci_covers_true_sharpe_difference_in_most_seeded_trials():
    bpy = 252.0
    mu_a, mu_b, sigma = 0.001, 0.0, 0.01
    true_diff = np.sqrt(bpy) * (mu_a / sigma - mu_b / sigma)
    covered = 0
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        a = rng.normal(mu_a, sigma, 500)
        b = rng.normal(mu_b, sigma, 500)
        out = sharpe_diff_test(a, b, bpy, seed=seed)
        covered += out["lo"] <= true_diff <= out["hi"]
    assert covered >= 34


def test_pbo_is_low_when_one_column_dominates():
    rng = np.random.default_rng(3)
    perf = rng.normal(0.0, 0.01, size=(320, 10))
    perf[:, 0] += 0.01
    assert pbo_cscv(perf, n_splits=16) < 0.2


def test_pbo_is_near_half_for_iid_noise_columns():
    rng = np.random.default_rng(4)
    perf = rng.normal(0.0, 0.01, size=(320, 10))
    assert 0.3 <= pbo_cscv(perf, n_splits=16) <= 0.7


def test_pbo_rejects_bad_shapes():
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((40, 1)), n_splits=4)
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((40, 3)), n_splits=5)
