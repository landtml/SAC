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


def test_recency_weighting_skews_episode_starts_late():
    ts = TapeSet(_tapes()[:1], window=4)
    flat_env = VecTradingEnv(ts, EnvConfig(window=4, episode_len=8, n_envs=512), seed=0)
    late_env = VecTradingEnv(ts, EnvConfig(window=4, episode_len=8, n_envs=512, recency_halflife=30),
                             seed=0)
    hi = ts.last_step_row[0]
    assert (late_env.t >= ts.first_row).all() and (late_env.t <= hi).all()
    assert np.median(hi - late_env.t) < 40 < np.median(hi - flat_env.t)


def test_observed_carry_is_the_interval_that_just_ended(tapeset):
    perp = tapeset.tapes[1]
    j = int(np.flatnonzero(perp.carry_long[10:] != 0)[0]) + 10
    # Row j's own carry is next bar's funding, which the close at j does not know yet.
    obs = tapeset.observe(np.array([1, 1]), np.array([j, j + 1]), np.zeros(2), np.ones(2))
    ctx = obs[:, -N_CONTEXT:]
    assert ctx[0, 3] == pytest.approx(np.clip(perp.carry_long[j - 1] * 1e4, -5, 5))
    assert ctx[1, 3] == pytest.approx(np.clip(perp.carry_long[j] * 1e4, -5, 5))
    assert tapeset.known_carry["long"][tapeset.offsets[1]] == 0.0


def test_tape_sigma_is_causal_and_sliced():
    frame = gbm(400, sigma=0.01, seed=1, symbol="S")
    tape = build_tape(frame, compute_features(frame, CFG))
    bumped = frame.copy()
    bumped.iloc[300:, :4] *= 1.5
    tape2 = build_tape(bumped, compute_features(bumped, CFG))
    cut = 300 - (len(frame) - len(tape))
    np.testing.assert_array_equal(tape.extra["sigma"][:cut], tape2.extra["sigma"][:cut])
    assert tape2.extra["sigma"][cut] > tape.extra["sigma"][cut]
    np.testing.assert_array_equal(tape.slice(50, 90).extra["sigma"], tape.extra["sigma"][50:90])


def test_action_map_targets_follow_prior_and_vol_target():
    from sactrader.env.vec_env import N_MAP_CONTEXT, ActionMap

    tapes = _tapes()
    amap = ActionMap(prior_feature="ret_h16", prior_gain=0.5, residual=0.5, vol_target=0.2)
    ts = TapeSet(tapes, window=4, action_map=amap)
    assert ts.obs_dim == TapeSet(tapes, window=4).obs_dim + N_MAP_CONTEXT
    k, t = np.array([0, 1, 2]), np.array([30, 30, 30])
    a = np.array([0.4, -1.0, 0.6])
    g = ts.offsets[k] + t
    j = ts.feature_names.index("ret_h16")
    b = np.clip(0.5 * ts.features[g, j], -1, 1)
    for i in range(2):
        tp = tapes[i]
        m = min(0.2 / (tp.extra["sigma"][30] * np.sqrt(tp.bars_per_year)), tp.max_leverage)
        want = np.clip(m * (b[i] + 0.5 * a[i]), -tp.max_leverage, tp.max_leverage)
        assert ts.targets(k[i:i + 1], t[i:i + 1], a[i:i + 1])[0] == pytest.approx(want)
    # Binary markets ignore the volatility target.
    assert ts.scale[g[2]] == tapes[2].max_leverage
    obs = ts.observe(k, t, np.zeros(3), np.ones(3))
    np.testing.assert_allclose(obs[:, -2], b, atol=1e-6)
    np.testing.assert_allclose(obs[:, -1], ts.scale[g] / ts.max_leverage[k], rtol=1e-6)
    res, reward, _ = ts.transition(k, t, np.zeros(3), a, np.ones(3))
    ref, ref_reward, _ = ts.transition_target(k, t, np.zeros(3), ts.targets(k, t, a), np.ones(3))
    np.testing.assert_allclose(reward, ref_reward)
    with pytest.raises(ValueError):
        TapeSet(tapes, window=4, action_map=ActionMap(prior_feature="nope"))


def test_identity_action_map_is_inactive():
    from sactrader.env.vec_env import ActionMap

    ts = TapeSet(_tapes(), window=4, action_map=ActionMap())
    assert ts.action_map is None and EnvConfig().action_map is None
    assert EnvConfig(vol_target=0.3).action_map == ActionMap(vol_target=0.3)


def test_rollout_band_holds_small_rebalances(tapeset):
    plan = iter([0.5, 0.55] + [0.9] * 1000)
    out = rollout(tapeset, lambda o: np.full(len(o), next(plan)), band=0.1)
    for o, tp in zip(out, tapeset.tapes, strict=True):
        lev = tp.max_leverage
        assert o["target"][0] == pytest.approx(0.5 * lev)
        # Row 1 asks for 0.55 L from an exposure that drifted near 0.5 L: inside the band.
        assert o["target"][1] == pytest.approx(o["w_prev"][1])
        assert o["target"][2] == pytest.approx(0.9 * lev)


def test_relabel_is_exact_under_an_action_map():
    from sactrader.env.vec_env import ActionMap

    ts = TapeSet(_tapes(), window=4, action_map=ActionMap(prior_feature="ret_h4", vol_target=0.3))
    env = VecTradingEnv(ts, EnvConfig(window=4, episode_len=32, n_envs=8), seed=2)
    rb = IndexReplay(1000)
    rng = np.random.default_rng(1)
    for _ in range(20):
        rb.add(env.step(rng.uniform(-1, 1, 8)))
    s = relabel(rb.sample(64, rng), ts, 1.0, rng)
    res, reward, _ = ts.transition(s["k"], s["t"], s["w_prev"], s["action"], s["cost_mult"])
    np.testing.assert_allclose(s["reward"], reward)
    np.testing.assert_allclose(s["w_next"], res.w_next)


def test_train_warm_start_shrink_and_redo_log_dormancy():
    import json  # noqa: F401

    from sactrader.agent.sac import SACConfig
    from sactrader.train import TrainConfig, train

    tapes = _tapes()[:2]
    cfg = TrainConfig(total_steps=640, warmup_steps=160, utd=0.25, eval_every=40, threads=1,
                      redo_every=50, env=EnvConfig(window=4, episode_len=32, n_envs=8),
                      sac=SACConfig(hidden=32, critic_blocks=1, actor_hidden=16, batch_size=32))
    first = train(tapes, tapes[:1], cfg)
    assert all(0.0 <= h["critic_dormant"] <= 1.0 for h in first.history)
    warm = train(tapes, tapes[:1], replace_cfg(cfg, warm_shrink=0.8), init_state=first.best_state,
                 feature_names=first.feature_names)
    assert warm.history and np.isfinite(warm.history[-1]["critic_loss"])


def replace_cfg(cfg, **kw):
    from dataclasses import replace

    return replace(cfg, **kw)


def test_several_priors_sum_and_skip_missing_features():
    from sactrader.env.vec_env import ActionMap

    tapes = _tapes()
    cfg = EnvConfig(prior_features={"ret_h4": 0.5, "funding_last": -0.2, "absent": 3.0})
    ts = TapeSet(tapes, window=4, action_map=cfg.action_map)
    names = ts.feature_names
    f = ts.features.astype(np.float64)
    raw = 0.5 * f[:, names.index("ret_h4")] - 0.2 * f[:, names.index("funding_last")]
    np.testing.assert_allclose(ts.prior, np.clip(raw, -1, 1), atol=1e-12)
    assert (ts.prior != 0).any()
    assert ActionMap(prior_features=(("ret_h4", 1.0),)).has_prior
