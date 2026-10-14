import numpy as np
import pandas as pd
import pytest

from softedge.eval import direction as d
from softedge.eval.stats import deflated_sharpe_ratio, null_sr_variance


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2000-01-03", periods=n, tz="UTC")


def _frame(values, cols=None) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float64)
    names = cols or [f"A{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=_dates(values.shape[0]), columns=names)


def _planted_momentum(seed, n_days=4000, n_assets=10, drift=0.001, vol=0.01, switch=0.01):
    """Returns with a regime-switching drift, so past returns predict the next day's mean."""
    rng = np.random.default_rng(seed)
    state = np.empty((n_days, n_assets))
    s = rng.choice([-1.0, 1.0], size=n_assets)
    for t in range(n_days):
        s = np.where(rng.random(n_assets) < switch, -s, s)
        state[t] = s
    return _frame(drift * state + vol * rng.standard_normal((n_days, n_assets)))


def test_planted_momentum_is_detected():
    panel = _planted_momentum(seed=7)
    cost = pd.Series(5.0, index=panel.columns)
    res = d.evaluate(d.HYPOTHESES["H1b"](panel), panel, cost)
    assert res["nw_t"] > 2.5
    assert res["sharpe"] > 0
    assert res["passes"]


def test_random_walk_is_not_detected():
    rng = np.random.default_rng(11)
    panel = _frame(rng.normal(0.0, 0.01, (4000, 10)))
    cost = pd.Series(5.0, index=panel.columns)
    for name in ("H1a", "H1b", "H1c"):
        res = d.evaluate(d.HYPOTHESES[name](panel), panel, cost)
        assert res["nw_t"] < 2.5, name
        assert not res["passes"], name


def test_random_walk_null_panels_match_assets_dates_and_vols():
    rng = np.random.default_rng(5)
    panel = _frame(rng.normal(0.0, 0.02, (3000, 4)))
    panel.iloc[:500, 2] = np.nan
    nulls = d.null_random_walk_panels(panel, n_panels=2, seed=1)
    assert len(nulls) == 2
    for null in nulls:
        assert null.index.equals(panel.index)
        assert list(null.columns) == list(panel.columns)
        assert null.iloc[:500, 2].isna().all()
        assert null.iloc[500:, 2].notna().all()
        ratio = null.std(ddof=1).to_numpy() / panel.std(ddof=1).to_numpy()
        assert np.allclose(ratio, 1.0, rtol=0.15)
    again = d.null_random_walk_panels(panel, n_panels=2, seed=1)
    pd.testing.assert_frame_equal(nulls[0], again[0])


def _assert_same_on_prefix(make_weights, rng_seed, cut):
    rng = np.random.default_rng(rng_seed)
    panel = _frame(rng.normal(0.0003, 0.01, (3300, 10)))
    base = make_weights(panel)
    altered = panel.copy()
    altered.iloc[cut + 1:] = rng.normal(0.0, 0.5, (panel.shape[0] - cut - 1, 10))
    moved = make_weights(altered)
    np.testing.assert_allclose(base.iloc[: cut + 1].to_numpy(),
                               moved.iloc[: cut + 1].to_numpy(), rtol=0, atol=0)


def test_time_series_positions_ignore_returns_after_day_t():
    _assert_same_on_prefix(lambda r: d.ts_momentum_weights(r, 63), rng_seed=2, cut=2500)


def test_quintile_positions_ignore_returns_after_day_t():
    _assert_same_on_prefix(
        lambda r: d.quintile_weights(d.trailing_return(r, 7), min_assets=5), rng_seed=3, cut=2000
    )


def test_ridge_positions_ignore_returns_after_day_t():
    weights_fn = d.HYPOTHESES["H3a"]
    _assert_same_on_prefix(weights_fn, rng_seed=4, cut=3100)
    rng = np.random.default_rng(4)
    panel = _frame(rng.normal(0.0003, 0.01, (3300, 10)))
    assert np.any(weights_fn(panel).iloc[3100].to_numpy() != 0.0)


def test_portfolio_cost_and_timing_match_hand_calculation():
    r = _frame([[0.0, 0.0], [0.01, -0.02], [0.03, 0.01], [-0.01, 0.02]], cols=["A", "B"])
    w = _frame([[0.5, 0.0], [0.5, -0.5], [0.0, -0.5], [0.0, 0.0]], cols=["A", "B"])
    cost = pd.Series([10.0, 5.0], index=["A", "B"])
    net = d.portfolio_returns(w, r, cost)
    # Day t earns the weights set at t-1. Each change at t is charged to day t+1.
    # Day 1: 0.5 * 0.01 = 0.005, less 0.5 * 10 bp on A at t=0.
    # Day 2: 0.5 * 0.03 - 0.5 * 0.01 = 0.010, less 0.5 * 5 bp on B at t=1.
    # Day 3: -0.5 * 0.02 = -0.010, less 0.5 * 10 bp on A at t=2.
    assert net.iloc[0] == pytest.approx(0.0)
    assert net.iloc[1] == pytest.approx(0.005 - 0.0005)
    assert net.iloc[2] == pytest.approx(0.010 - 0.00025)
    assert net.iloc[3] == pytest.approx(-0.010 - 0.0005)


def test_delayed_portfolio_holds_each_position_one_day_longer():
    r = _frame([[0.0, 0.0], [0.01, -0.02], [0.03, 0.01], [-0.01, 0.02]], cols=["A", "B"])
    w = _frame([[0.5, 0.0], [0.5, -0.5], [0.0, -0.5], [0.0, 0.0]], cols=["A", "B"])
    cost = pd.Series([10.0, 5.0], index=["A", "B"])
    net = d.portfolio_returns(w, r, cost, delay=1)
    # The delayed weight on day t is w_{t-1}, and day t earns the delayed weight of day t-1.
    # Day 0 and day 1 earn nothing. Day 2 earns [0.5, 0] . r_2 = 0.015, less the 10 bp cost
    # of the change at t=1 (0.5 of A). Day 3 earns [0.5, -0.5] . r_3 = -0.015, less the
    # 5 bp cost of the change at t=2 (0.5 of B).
    assert net.iloc[0] == pytest.approx(0.0)
    assert net.iloc[1] == pytest.approx(0.0)
    assert net.iloc[2] == pytest.approx(0.015 - 0.0005)
    assert net.iloc[3] == pytest.approx(-0.015 - 0.00025)


def test_newey_west_matches_ar1_long_run_variance():
    # AR(1) with phi = 0.5 and unit innovations. The Bartlett estimate with 5 lags has
    # population long-run variance sum_j w_j * gamma_j = 3.125, where gamma_j = phi^|j| / (1 - phi^2).
    phi, n = 0.5, 400_000
    rng = np.random.default_rng(3)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    lrv = d.newey_west_variance_of_mean(x, lags=5) * n
    assert lrv == pytest.approx(3.125, rel=0.05)
    assert lrv < 1.0 / (1 - phi) ** 2


def test_newey_west_t_is_mean_over_its_standard_error():
    rng = np.random.default_rng(9)
    x = rng.normal(0.01, 1.0, 2000)
    var = d.newey_west_variance_of_mean(x, lags=5)
    assert d.newey_west_t(x, lags=5) == pytest.approx(x.mean() / np.sqrt(var))


def test_min_detectable_sharpe_matches_hand_calculation():
    # 2520 trading days is 10 years. (1.959964 + 0.841621) / sqrt(10) = 0.885939.
    assert d.min_detectable_sharpe(2520) == pytest.approx(0.885939, abs=1e-5)


def test_deflated_sharpe_uses_the_repo_function_at_the_study_trial_count():
    r = np.random.default_rng(2).normal(0.02, 1.0, 5000)
    expected = deflated_sharpe_ratio(r, d.N_TRIALS, null_sr_variance(r.size))
    assert d.deflated_sharpe(r) == pytest.approx(expected)
    assert d.N_TRIALS == 11
    assert d.deflated_sharpe(r) < d.deflated_sharpe(r, n_trials=1)


def test_quintile_weights_are_dollar_neutral_and_pick_the_extremes():
    signal = _frame(np.arange(10.0)[None, :])
    w = d.quintile_weights(signal, min_assets=5).iloc[0].to_numpy()
    assert w[9] == pytest.approx(0.5) and w[8] == pytest.approx(0.5)
    assert w[0] == pytest.approx(-0.5) and w[1] == pytest.approx(-0.5)
    assert np.allclose(w[2:8], 0.0)
    assert w.sum() == pytest.approx(0.0)


def test_quintile_weights_need_min_assets():
    signal = _frame(np.arange(10.0)[None, :])
    assert np.all(d.quintile_weights(signal, min_assets=30).to_numpy() == 0.0)


def test_time_series_weight_is_ten_percent_vol_over_sixty_three_day_vol():
    rng = np.random.default_rng(6)
    x = _frame(0.001 + 0.01 * rng.standard_normal((200, 1)))
    w = d.ts_momentum_weights(x, 21)
    sd = x["A0"].iloc[-63:].std(ddof=1) * np.sqrt(252)
    assert w.iloc[-1, 0] == pytest.approx(min(2.0, 0.10 / sd))


def test_zscore_is_causal():
    rng = np.random.default_rng(8)
    x = _frame(rng.normal(size=(400, 2)))
    z = d.causal_zscore(x, min_periods=50)
    later = x.copy()
    later.iloc[300:] = 1e6
    z2 = d.causal_zscore(later, min_periods=50)
    pd.testing.assert_frame_equal(z.iloc[:300], z2.iloc[:300])


def test_ridge_closed_form_recovers_a_linear_rule():
    rng = np.random.default_rng(10)
    f = rng.normal(size=(3000, 1, 2))
    y = f[:, 0, 0] * 0.5 - f[:, 0, 1] * 0.25
    pred = d.walk_forward_ridge(f, y[:, None], _dates(3000), burn_in_years=5, alpha=1e-6)
    assert np.allclose(pred.iloc[-1000:, 0].to_numpy(), y[-1000:], atol=1e-3)
