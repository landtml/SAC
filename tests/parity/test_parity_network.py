import math
from dataclasses import replace

import numpy as np
import pytest

from softedge.parity import ParityConfig, ParityNetwork
from softedge.parity import backtest as bt

CASH = ParityConfig().cash


def symmetric_paths(n_paths, n_steps, seed):
    """Excess log returns with GARCH-like volatility and t tails, symmetric given the past."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_paths, n_steps))
    var = np.full(n_paths, 1e-4)
    for t in range(n_steps):
        eps = rng.standard_t(4, n_paths) / math.sqrt(2.0)
        x[:, t] = np.sqrt(var) * eps
        var = 1e-6 + 0.85 * var + 0.1 * x[:, t] ** 2
    return CASH + x


def run_network(logrets, config):
    net = ParityNetwork(logrets.shape[0], logrets.shape[1], config, trace=True)
    ones = np.ones(logrets.shape[0])
    out = []
    for t in range(logrets.shape[1]):
        net.observe(logrets[:, t])
        out.append(net.target(ones, ones))
    return net, np.stack(out, axis=1)


def test_threshold_is_log_k_over_alpha():
    cfg = ParityConfig()
    assert cfg.n_ledgers == 4
    assert cfg.threshold == pytest.approx(math.log(4 / 0.1))


def test_default_exposure_is_one_half_without_direction():
    r = symmetric_paths(8, 200, 0)
    _, e = run_network(r, ParityConfig(direction=False, governor=False))
    assert np.all(e == 0.5)


def test_sign_flip_mirrors_the_exposure_exactly():
    r = symmetric_paths(16, 400, 1)
    mirror = 2.0 * CASH - r
    cfg = ParityConfig(governor=False, gate=False)
    net, e = run_network(r, cfg)
    net_m, e_m = run_network(mirror, cfg)
    np.testing.assert_allclose(e_m, 1.0 - e, atol=1e-12)
    np.testing.assert_allclose(net_m.pred, -net.pred, atol=1e-12)
    np.testing.assert_allclose(net_m.evidence, net.evidence, atol=1e-12)


def test_a_directional_prior_breaks_the_mirror_identity():
    r = symmetric_paths(16, 400, 1)
    mirror = 2.0 * CASH - r
    cfg = ParityConfig(governor=False, gate=False, default_exposure=1.0)
    _, e = run_network(r, cfg)
    _, e_m = run_network(mirror, cfg)
    assert np.max(np.abs(e_m - (1.0 - e))) > 0.1


def test_ledgers_respect_ville_under_a_symmetric_null():
    n_paths = 4000
    r = symmetric_paths(n_paths, 520, 2)
    net = ParityNetwork(n_paths, 520, ParityConfig())
    for t in range(520):
        net.observe(r[:, t])
    alpha = 0.05
    exceed = (net.max_log_capital >= math.log(1.0 / alpha)).mean(axis=0)
    limit = alpha + 3.0 * math.sqrt(alpha * (1 - alpha) / n_paths)
    assert np.all(exceed <= limit), exceed


def test_deployment_is_rare_under_a_symmetric_null():
    n_paths = 2000
    r = symmetric_paths(n_paths, 520, 3)
    net, _ = run_network(r, ParityConfig(governor=False))
    weights = np.stack(net.trace["weight"], axis=1)
    assert (weights > 0.0).mean() < 0.02


def test_learns_and_deploys_a_persistent_drift():
    rng = np.random.default_rng(4)
    n_paths, n_steps = 200, 1040
    sig = np.zeros((n_paths, n_steps))
    for t in range(1, n_steps):
        sig[:, t] = 0.95 * sig[:, t - 1] + math.sqrt(1 - 0.95 ** 2) * rng.standard_normal(n_paths)
    drift = np.zeros_like(sig)
    drift[:, 1:] = 0.3 * 0.01 * sig[:, :-1]
    r = CASH + drift + 0.01 * rng.standard_normal((n_paths, n_steps))
    net, _ = run_network(r, ParityConfig(governor=False))
    weights = np.stack(net.trace["weight"], axis=1)
    assert (weights[:, -1, 1:].max(axis=1) > 0.0).mean() > 0.5
    z_dep = np.stack(net.trace["z_deployed"], axis=1)[:, 520:-1]
    assert np.corrcoef(z_dep.ravel(), drift[:, 521:].ravel())[0, 1] > 0.1
    on = bt.simulate(r, ParityNetwork(n_paths, n_steps, ParityConfig()))
    off = bt.simulate(r, ParityNetwork(n_paths, n_steps, ParityConfig(direction=False)))
    assert bt.paired(on, off)["low"] > 0.0


def test_future_returns_do_not_move_past_decisions():
    r = symmetric_paths(32, 600, 5)
    rng = np.random.default_rng(6)
    t0 = 300
    scrambled = r.copy()
    flips = np.where(rng.random((32, 600 - t0 - 1)) < 0.5, -1.0, 1.0)
    scrambled[:, t0 + 1:] = flips * r[:, t0 + 1:]
    a = bt.simulate(r, ParityNetwork(32, 600, ParityConfig()))
    b = bt.simulate(scrambled, ParityNetwork(32, 600, ParityConfig()))
    np.testing.assert_array_equal(a["target"][:, : t0 + 1], b["target"][:, : t0 + 1])
    np.testing.assert_array_equal(a["held"][:, : t0 + 2], b["held"][:, : t0 + 2])


def test_governor_caps_the_exposure():
    r = symmetric_paths(64, 800, 7) - 0.002
    res, net = bt.run_parity(r, trace=True)
    e_star = np.stack(net.trace["e_star"], axis=1)
    cap = np.stack(net.trace["cap"], axis=1)
    targets = res["target"][:, bt.FIRST_DECISION:]
    np.testing.assert_allclose(targets, np.minimum(e_star, cap))
    assert np.all(targets <= cap + 1e-15)


def test_buy_and_hold_wealth_is_the_compounded_return():
    r = symmetric_paths(4, 120, 8)
    res = bt.simulate(r, bt.BuyAndHold())
    held = np.zeros(120)
    held[bt.WARMUP:] = 1.0
    gross = np.exp(bt.CASH) * np.ones(120)
    gross[bt.WARMUP:] = np.exp(r[0, bt.WARMUP:])
    cost = np.zeros(120)
    cost[bt.WARMUP] = bt.COST
    np.testing.assert_allclose(res["wealth"][0], np.cumprod((1.0 - cost) * gross))


def test_paired_difference_of_a_policy_with_itself_is_zero():
    r = symmetric_paths(10, 200, 9)
    res = bt.simulate(r, bt.TrendGate())
    out = bt.paired(res, res)
    assert out["mean"] == 0.0 and out["low"] == 0.0 and out["high"] == 0.0


def test_ablation_switches_change_the_path():
    r = symmetric_paths(16, 520, 10) + 0.0005
    base = bt.run_parity(r)[0]["held"]
    for change in ({"gate": False}, {"governor": False}, {"direction": False},
                   {"default_exposure": 1.0}):
        other = bt.run_parity(r, replace(ParityConfig(), **change))[0]["held"]
        assert not np.array_equal(base, other), change


def test_explicit_cash_and_cost_reproduce_the_defaults():
    r = symmetric_paths(8, 300, 11)
    a = bt.run_parity(r)[0]
    b = bt.run_parity(r, cash=np.full(300, bt.CASH), cost=np.full(8, bt.COST))[0]
    np.testing.assert_array_equal(a["growth"], b["growth"])
    np.testing.assert_array_equal(a["held"], b["held"])


def test_cash_series_moves_the_default_exposure_only_through_direction():
    r = symmetric_paths(8, 300, 12)
    cash = np.linspace(0.0, 0.001, 300)
    res = bt.run_parity(r, replace(ParityConfig(), direction=False, governor=False), cash=cash)[0]
    assert np.all(res["target"][:, bt.FIRST_DECISION:] == 0.5)
