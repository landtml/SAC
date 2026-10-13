import numpy as np
import pytest
import torch

from softedge.agent.pathwise import PathwiseObjective, step_portfolio_torch
from softedge.agent.sac import TQCSAC, SACConfig
from softedge.data.synthetic import binary_market, gbm, with_funding
from softedge.env.vec_env import ActionMap, TapeSet
from softedge.features.causal import FeatureConfig, compute_features
from softedge.portfolio import log_reward, step_portfolio
from softedge.tape import build_tape

CFG = FeatureConfig(horizons=(1, 4, 16), ema_spans=(16,), vol_long_span=64)
GAMMA = 0.9
SCALE = 3.0


def _tapes():
    spot = gbm(300, sigma=0.02, seed=1, symbol="S")
    perp = with_funding(gbm(300, sigma=0.02, seed=2, symbol="P"), seed=3)
    binary = binary_market(120, seed=4, symbol="B")
    return [build_tape(f, compute_features(f, CFG)) for f in (spot, perp, binary)]


class _FixedAgent:
    """Constant action, zero log-probability and constant Q: the value is then a plain sum."""

    def __init__(self, action: float, q: float):
        self.action, self.q = action, q
        self.obs_norm = torch.nn.Identity()

    def actor(self, obs):
        n = obs.shape[0]
        return torch.full((n, 1), self.action), torch.zeros(n)

    def q_value(self, obs, a):
        return torch.full((obs.shape[0],), self.q)


def _reference(ts, k, t, w, action, q, horizon):
    """Numpy rollout through TapeSet.transition with the same stopping rules."""
    total, disc = 0.0, 1.0
    for _ in range(horizon):
        if t > ts.last_step_row[k]:
            return total + disc * q
        res, reward, terminal = ts.transition(np.array([k]), np.array([t]), np.array([w]),
                                              np.array([action]), np.ones(1))
        total += disc * SCALE * float(reward[0])
        if terminal[0]:
            return total
        w, t, disc = float(res.w_next[0]), t + 1, disc * GAMMA
    return total + disc * q


def test_torch_step_matches_numpy_step():
    rng = np.random.default_rng(0)
    n = 512
    m = {"long_ret": rng.normal(0, 0.05, n), "short_ret": rng.normal(0, 0.05, n),
         "long_worst": -np.abs(rng.normal(0, 0.3, n)), "short_worst": -np.abs(rng.normal(0, 0.3, n)),
         "carry_long": rng.normal(0, 1e-4, n), "carry_short": rng.normal(0, 1e-4, n),
         "cost_long": rng.uniform(0, 1e-3, n), "cost_short": rng.uniform(0, 1e-3, n)}
    m["short_growth"] = 1.0 + m["long_ret"]
    w, target = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    impact, maint = np.where(rng.random(n) < 0.5, 0.0, 1e-3), np.full(n, 0.05)
    ref = step_portfolio(w, target, m["long_ret"], m["short_ret"], m["short_growth"],
                         m["long_worst"], m["short_worst"], m["carry_long"], m["carry_short"],
                         m["cost_long"], m["cost_short"], impact, maint)
    assert ref.liquidated.any() and not ref.liquidated.all()
    t64 = {key: torch.as_tensor(v) for key, v in m.items()}
    ret, w_next = step_portfolio_torch(torch.as_tensor(w), torch.as_tensor(target), t64,
                                       torch.as_tensor(impact), torch.as_tensor(maint))
    np.testing.assert_allclose(ret.numpy(), ref.ret, atol=1e-9)
    np.testing.assert_allclose(w_next.numpy(), ref.w_next, atol=1e-9)


@pytest.mark.parametrize("amap", [None, ActionMap(prior_feature="ret_h4", vol_target=0.5)])
@pytest.mark.parametrize("horizon", [1, 3])
def test_value_matches_numpy_rollout(amap, horizon):
    ts = TapeSet(_tapes(), window=4, action_map=amap)
    obj = PathwiseObjective(ts, horizon, GAMMA, SCALE)
    agent = _FixedAgent(action=0.6, q=0.25)
    # Mid-tape rows, a non-terminal tape's last steppable row, and a binary market's settlement.
    cases = [(0, 50, 0.3), (1, 80, -0.4), (0, int(ts.last_step_row[0]), 0.1),
             (2, int(ts.last_step_row[2]) - 1, 0.2), (2, int(ts.last_step_row[2]), 0.0)]
    k = np.array([c[0] for c in cases])
    t = np.array([c[1] for c in cases])
    w = np.array([c[2] for c in cases])
    value, _ = obj.value(agent, k, t, w, np.ones(len(k)), torch.tensor(0.0))
    want = [_reference(ts, *c, 0.6, 0.25, horizon) for c in cases]
    np.testing.assert_allclose(value.numpy(), want, rtol=1e-4, atol=1e-6)


def test_pathwise_gradient_reaches_actor_and_training_runs():
    from softedge.env.vec_env import EnvConfig
    from softedge.train import TrainConfig, train

    ts = TapeSet(_tapes(), window=4)
    agent = TQCSAC(ts.obs_dim, 1, SACConfig(hidden=32, critic_blocks=1, actor_hidden=16), seed=0)
    obj = PathwiseObjective(ts, 2, GAMMA, SCALE)
    k, t = np.array([0, 1, 2]), np.array([40, 40, 40])
    value, logp = obj.value(agent, k, t, np.zeros(3), np.ones(3), torch.tensor(0.1))
    value.mean().backward()
    grads = [p.grad for p in agent.actor.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    assert logp.shape == (3,)
    res = train(_tapes()[:2], _tapes()[:1], TrainConfig(
        total_steps=640, warmup_steps=160, utd=0.25, eval_every=40, threads=1,
        env=EnvConfig(window=4, episode_len=32, n_envs=8),
        sac=SACConfig(hidden=32, critic_blocks=1, actor_hidden=16, batch_size=32,
                      pathwise_horizon=2)))
    assert res.history and np.isfinite(res.history[-1]["actor_loss"])
    assert float(log_reward(np.array([0.0]))[0]) == 0.0
