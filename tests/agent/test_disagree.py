import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from softedge.agent.disagree import (
    ShrinkPolicy,
    deployed_action,
    disagree_shrink_on,
    ensemble_stats,
    multiplier_matched_policy,
    shrink_multiplier,
    validation_tau,
)
from softedge.agent.sac import TQCSAC, SACConfig
from softedge.data.synthetic import binary_market, gbm, with_funding
from softedge.env.vec_env import TapeSet, rollout
from softedge.experiment import _test_runs, ensemble_policy
from softedge.features.causal import FeatureConfig, compute_features
from softedge.tape import build_tape

FEATS = FeatureConfig(horizons=(1, 4, 16), ema_spans=(16,), vol_long_span=64)
ROOT = Path(__file__).resolve().parents[2]
# Tiny networks run faster on one thread than on the default pool.
torch.set_num_threads(1)


class _Fixed:
    """Stands in for a member: a fixed deterministic action per observation row."""

    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float64)

    def act(self, obs, deterministic=False):
        return self.actions[: len(obs)].reshape(-1, 1)


def _tapes(shift=0):
    spot = gbm(400, sigma=0.01, seed=1 + shift, symbol="S")
    perp = with_funding(gbm(400, sigma=0.01, seed=2 + shift, symbol="P"), seed=3 + shift)
    binary = binary_market(120, seed=4 + shift, symbol="B")
    return [build_tape(f, compute_features(f, FEATS)) for f in (spot, perp, binary)]


def _agents(obs_dim, seeds=(0, 1, 2)):
    rng = np.random.default_rng(0)
    cfg = SACConfig(hidden=32, critic_blocks=1, actor_hidden=16)
    out = []
    for s in seeds:
        agent = TQCSAC(obs_dim, 1, cfg, seed=s)
        agent.obs_norm.update(torch.as_tensor(rng.normal(size=(64, obs_dim)), dtype=torch.float32))
        out.append(agent)
    return out


def test_multiplier_on_a_toy_case():
    m = shrink_multiplier(np.array([0.0, 1.0, 2.0, 3.0]), tau=1.0)
    np.testing.assert_allclose(m, [1.0, 0.5, 0.2, 0.1])
    assert np.all(np.diff(m) < 0)


def test_ensemble_stats_are_mean_and_population_std_per_row():
    agents = [_Fixed([0.2, -0.5]), _Fixed([0.6, -0.1]), _Fixed([0.4, -0.3])]
    rbar, sigma = ensemble_stats(agents, np.zeros((2, 3)))
    assert rbar.shape == sigma.shape == (2,)
    np.testing.assert_allclose(rbar, [0.4, -0.3])
    # Row 0 has members 0.2, 0.6, 0.4: mean 0.4, squared deviations 0.04 + 0.04 + 0, population
    # variance 0.08 / 3. Row 1 is the mirror image, with the same spread.
    np.testing.assert_allclose(sigma, [np.sqrt(0.08 / 3), np.sqrt(0.08 / 3)])


def test_deployed_action_is_mean_times_multiplier_on_a_toy_case():
    agents = [_Fixed([0.5, 0.0, -1.0]), _Fixed([0.5, 1.0, 0.0])]
    obs = np.zeros((3, 2))
    # Row 0 has no disagreement (m = 1). Rows 1 and 2 have sigma 0.5 and tau 0.5 (m = 0.5).
    action, m = deployed_action(agents, obs, tau=0.5)
    np.testing.assert_allclose(m, [1.0, 0.5, 0.5])
    np.testing.assert_allclose(action, [0.5, 0.25, -0.25])


def test_constant_mode_multiplies_the_ensemble_mean_by_a_constant():
    agents = [_Fixed([0.5, 0.0, -1.0]), _Fixed([0.5, 1.0, 0.0])]
    action, m = deployed_action(agents, np.zeros((3, 2)), tau=None, const=0.3)
    np.testing.assert_allclose(m, [0.3, 0.3, 0.3])
    np.testing.assert_allclose(action, 0.3 * np.array([0.5, 0.5, -0.5]))


def test_shrink_policy_reports_mean_multiplier_over_all_calls():
    agents = [_Fixed([0.5, 0.0, -1.0]), _Fixed([0.5, 1.0, 0.0])]
    policy = ShrinkPolicy(agents, tau=0.5)
    policy(np.zeros((3, 2)))
    policy(np.zeros((1, 2)))  # a later call with one live row
    assert policy.mean_multiplier == pytest.approx((1.0 + 0.5 + 0.5 + 1.0) / 4)


def test_real_agents_give_one_action_per_row_and_do_not_move():
    agents = _agents(obs_dim=5)
    obs = np.random.default_rng(1).normal(size=(7, 5)).astype(np.float32)
    weights = [p.detach().clone() for a in agents for p in a.actor.parameters()]
    action, m = deployed_action(agents, obs, tau=0.1)
    assert action.shape == m.shape == (7,)
    assert np.all((m > 0) & (m <= 1))
    assert all(not isinstance(x, torch.Tensor) for x in (action, m))
    after = [p.detach() for a in agents for p in a.actor.parameters()]
    for w, p in zip(weights, after, strict=True):
        torch.testing.assert_close(p, w)


def test_identical_members_have_no_disagreement_and_no_shrink():
    ts = TapeSet(_tapes(), window=4)
    agent = _agents(ts.obs_dim, seeds=(0,))[0]
    twin = TQCSAC.from_state_dict(agent.state_dict())
    obs = ts.observe(np.array([0, 1]), np.array([30, 30]), np.zeros(2), np.ones(2))
    rbar, sigma = ensemble_stats([agent, twin], obs)
    np.testing.assert_allclose(sigma, 0.0, atol=1e-7)
    np.testing.assert_allclose(rbar, agent.act(obs, deterministic=True)[:, 0], rtol=1e-6)
    with pytest.raises(ValueError, match="tau"):
        validation_tau([agent, twin], ts)


def test_tau_is_the_median_disagreement_over_validation_bars():
    ts = TapeSet(_tapes(), window=4)
    agents = _agents(ts.obs_dim)
    seen = []

    def ensemble(obs):
        rbar, sigma = ensemble_stats(agents, obs)
        seen.append(sigma)
        return rbar

    rollout(ts, ensemble)
    assert validation_tau(agents, ts) == pytest.approx(np.median(np.concatenate(seen)))


def test_matched_control_holds_the_shrink_mean_multiplier_over_validation():
    ts = TapeSet(_tapes(), window=4)
    agents = _agents(ts.obs_dim)
    tau = validation_tau(agents, ts)
    shrink = ShrinkPolicy(agents, tau=tau)
    rollout(ts, shrink)
    matched = multiplier_matched_policy(agents, ts, tau)
    assert matched.tau is None
    assert matched.const == pytest.approx(shrink.mean_multiplier)


def test_multiplier_at_bar_t_ignores_bars_after_t():
    ts = TapeSet(_tapes(), window=4)
    agents = _agents(ts.obs_dim)
    k, t, w = np.array([0]), np.array([60]), np.zeros(1)
    obs = ts.observe(k, t, w, np.ones(1))
    action, m = deployed_action(agents, obs, tau=0.05)
    assert m[0] < 1.0  # the check is not vacuous: the members disagree here
    g = ts.offsets[0] + 60
    later = copy.deepcopy(ts)
    later.features[g + 1:] += 5.0
    for arr in later.step.values():
        arr[g + 1:] *= 3.0
    later_obs = later.observe(k, t, w, np.ones(1))
    later_action, later_m = deployed_action(agents, later_obs, tau=0.05)
    np.testing.assert_array_equal(later_obs, obs)
    np.testing.assert_array_equal(later_m, m)
    np.testing.assert_array_equal(later_action, action)


def test_switch_takes_only_zero_or_one():
    assert disagree_shrink_on({}) is False
    assert disagree_shrink_on({"disagree_shrink": 0}) is False
    assert disagree_shrink_on({"disagree_shrink": 1.0}) is True
    for bad in (0.5, 2, -1):
        with pytest.raises(ValueError, match="disagree_shrink"):
            disagree_shrink_on({"disagree_shrink": bad})


def _test_runs_inputs():
    tapes = _tapes()
    ts = TapeSet(tapes, window=4)
    cfg = {"env": {"window": 4, "episode_len": 64, "n_envs": 8},
           "sac": {"hidden": 32, "critic_blocks": 1, "actor_hidden": 16}, "seeds": [0, 1]}
    parts = {"train": tapes, "val": tapes, "test": tapes}
    return cfg, parts, ts.feature_names


def test_switch_absent_and_switch_off_give_bit_identical_runs():
    # Both calls run the same code. The check is that the off switch adds no run and changes no
    # output. Equality with code that has no switch is not tested here.
    cfg, parts, names = _test_runs_inputs()
    agents = _agents(TapeSet(parts["test"], window=4, feature_names=names).obs_dim, seeds=(0, 1))
    base, _ = _test_runs(cfg, agents, [0, 1], parts, 4, names)
    off, _ = _test_runs({**cfg, "deploy": {"disagree_shrink": 0}}, agents, [0, 1], parts, 4, names)
    assert set(off) == set(base)
    assert not any("disagree" in k or "matched" in k for k in base)
    for key in base:
        for a, b in zip(base[key], off[key], strict=True):
            for field in a:
                np.testing.assert_array_equal(a[field], b[field])
    test_ts = TapeSet(parts["test"], window=4, feature_names=names)
    direct = rollout(test_ts, ensemble_policy(agents))
    for a, b in zip(base["sac_ensemble"], direct, strict=True):
        np.testing.assert_array_equal(a["ret"], b["ret"])


def test_switch_on_adds_the_shrunk_run_and_the_matched_control():
    cfg, parts, names = _test_runs_inputs()
    agents = _agents(TapeSet(parts["test"], window=4, feature_names=names).obs_dim, seeds=(0, 1))
    on, _ = _test_runs({**cfg, "deploy": {"disagree_shrink": 1.0}}, agents, [0, 1], parts, 4, names)
    assert "sac_disagree_shrink" in on and "sac_multiplier_matched" in on
    shrunk = np.concatenate([r["target"] for r in on["sac_disagree_shrink"]])
    ens = np.concatenate([r["target"] for r in on["sac_ensemble"]])
    assert not np.allclose(shrunk, ens)
    assert np.all(np.isfinite(shrunk))
    with pytest.raises(ValueError, match="two seeds"):
        _test_runs({**cfg, "deploy": {"disagree_shrink": 1.0}}, agents[:1], [0], parts, 4, names)


def test_matched_constant_ignores_test_fold_tapes():
    cfg, parts, names = _test_runs_inputs()
    cfg = {**cfg, "deploy": {"disagree_shrink": 1.0}}
    agents = _agents(TapeSet(parts["test"], window=4, feature_names=names).obs_dim, seeds=(0, 1))
    _, a = _test_runs(cfg, agents, [0, 1], {**parts, "test": _tapes(10)}, 4, names)
    _, b = _test_runs(cfg, agents, [0, 1], {**parts, "test": _tapes(20)}, 4, names)
    # The test-fold multiplier moves with the test tapes, so the check is not vacuous.
    assert a["disagree"]["test_mean_multiplier"] != pytest.approx(b["disagree"]["test_mean_multiplier"])
    assert a["disagree"]["matched_const"] == b["disagree"]["matched_const"]


def test_bench_switch_configs_differ_only_in_the_switch():
    off = yaml.safe_load((ROOT / "configs/bench/disagree-shrink_off.yaml").read_text())
    on = yaml.safe_load((ROOT / "configs/bench/disagree-shrink_on.yaml").read_text())
    assert off["deploy"] == {"gate_psr": 0.8, "disagree_shrink": 0}
    assert on["deploy"] == {"gate_psr": 0.8, "disagree_shrink": 1.0}
    assert off["data"]["seed_offset"] == on["data"]["seed_offset"] == 3000
    assert off["data"]["family"] == on["data"]["family"] == "all"
    assert off["name"] != on["name"]
    off.pop("deploy"), on.pop("deploy"), off.pop("name"), on.pop("name")
    assert off == on
