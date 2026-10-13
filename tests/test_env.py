import numpy as np
import pytest

from sactrader.agent.replay import IndexReplay, materialize, relabel
from sactrader.data.synthetic import binary_market, gbm, with_funding
from sactrader.env.vec_env import N_CONTEXT, EnvConfig, TapeSet, VecTradingEnv, rollout
from sactrader.features.causal import FeatureConfig, compute_features
from sactrader.portfolio import log_reward, step_portfolio
from sactrader.tape import build_tape

CFG = FeatureConfig(horizons=(1, 4, 16), ema_spans=(16,), vol_long_span=64)


def _tapes():
    spot = gbm(400, sigma=0.01, seed=1, symbol="S")
    perp = with_funding(gbm(400, sigma=0.01, seed=2, symbol="P"), seed=3)
    binary = binary_market(120, seed=4, symbol="B")
    return [build_tape(f, compute_features(f, CFG)) for f in (spot, perp, binary)]


@pytest.fixture(scope="module")
def tapeset():
    return TapeSet(_tapes(), window=4)


def test_union_feature_space_and_obs_layout(tapeset):
    names = tapeset.feature_names
    assert "funding_last" in names and "logit" in names
    obs = tapeset.observe(np.array([0, 2]), np.array([10, 10]), np.array([0.5, -0.5]),
                          np.ones(2))
    assert obs.shape == (2, 4 * len(names) + N_CONTEXT)
    # spot tape has no logit column: zeros in its window
    j = names.index("logit")
    assert np.all(obs[0, : 4 * len(names)].reshape(4, -1)[:, j] == 0)
    ctx = obs[:, -N_CONTEXT:]
    assert ctx[0, 0] == pytest.approx(0.5) and ctx[1, 0] == pytest.approx(-0.5)
    assert ctx[0, 6 + 0] == 1 and ctx[1, 6 + 3] == 1


def test_transition_matches_portfolio(tapeset):
    k, t = np.array([1]), np.array([20])
    res, reward, terminal = tapeset.transition(k, t, np.array([0.3]), np.array([-0.5]), np.ones(1))
    tp = tapeset.tapes[1]
    ref = step_portfolio(0.3, -0.5 * tp.max_leverage, tp.long_ret[20], tp.short_ret[20],
                         tp.short_growth[20], tp.long_worst[20], tp.short_worst[20],
                         tp.carry_long[20], tp.carry_short[20], tp.cost_long[20],
                         tp.cost_short[20], tp.impact, tp.maint_margin)
    assert res.ret[0] == pytest.approx(float(ref.ret))
    assert reward[0] == pytest.approx(float(log_reward(ref.ret)))
    assert not terminal[0]


def test_binary_last_row_is_terminal(tapeset):
    last = tapeset.lengths[2] - 1
    _, _, terminal = tapeset.transition(np.array([2]), np.array([last]), np.zeros(1),
                                        np.ones(1), np.ones(1))
    assert terminal[0]


def test_env_episodes_end_and_reset():
    ts = TapeSet(_tapes(), window=4)
    env = VecTradingEnv(ts, EnvConfig(window=4, episode_len=16, n_envs=8), seed=0)
    ends = 0
    for _ in range(40):
        b = env.step(np.zeros(8))
        assert (b.t <= ts.last_step_row[b.k]).all()
        assert not (b.terminal & b.truncated).any()
        ends += int((b.terminal | b.truncated).sum())
    assert ends >= 8 * 2
    assert len(env.completed_returns) == ends


def test_rollout_covers_each_tape(tapeset):
    out = rollout(tapeset, lambda o: np.ones(len(o)))
    for o, n_last, tp in zip(out, tapeset.last_step_row, tapeset.tapes, strict=True):
        assert len(o["ret"]) == n_last - tapeset.first_row + 1
        assert np.allclose(o["target"], tp.max_leverage)


def test_relabel_recomputes_exact_rewards(tapeset):
    env = VecTradingEnv(tapeset, EnvConfig(window=4, episode_len=32, n_envs=8), seed=1)
    rb = IndexReplay(1000)
    rng = np.random.default_rng(0)
    for _ in range(20):
        rb.add(env.step(rng.uniform(-1, 1, 8)))
    s = relabel(rb.sample(64, rng), tapeset, 1.0, rng)
    res, reward, terminal = tapeset.transition(s["k"], s["t"], s["w_prev"], s["action"], s["cost_mult"])
    np.testing.assert_allclose(s["reward"], reward)
    np.testing.assert_allclose(s["w_next"], res.w_next)
    assert (np.abs(s["w_prev"]) <= tapeset.max_leverage[s["k"]] + 1e-12).all()
    obs, nxt = materialize(s, tapeset)
    assert obs.shape == nxt.shape == (64, tapeset.obs_dim)
    assert np.isfinite(obs).all() and np.isfinite(nxt).all()
