import hashlib

import numpy as np
import pytest
import torch

from softedge.agent.drawdown import (
    DD_EXCESS_CAP,
    DrawdownShaping,
    DrawdownSwitch,
    drawdown_switch,
    next_drawdown,
)
from softedge.agent.replay import IndexReplay, materialize, relabel
from softedge.agent.sac import TQCSAC, SACConfig
from softedge.data.synthetic import gbm
from softedge.env.vec_env import EnvConfig, TapeSet, VecTradingEnv, rollout
from softedge.experiment import build_train_config
from softedge.pipeline import to_tapes
from softedge.tape import MarketTape
from softedge.train import TrainConfig, train

torch.set_num_threads(1)
WINDOW = 4
STEP_ARRAYS = (
    "long_ret",
    "short_ret",
    "short_growth",
    "long_worst",
    "short_worst",
    "carry_long",
    "carry_short",
    "fin_long",
    "cost_long",
    "cost_short",
)


def _tapes(n_tapes: int = 2, n: int = 600, seed: int = 0) -> list[MarketTape]:
    return to_tapes(
        [
            gbm(n, sigma=0.01, seed=seed + i, symbol=f"S{i}", start="2021-01-01")
            for i in range(n_tapes)
        ]
    )


def _perturb_after(tapes: list[MarketTape], row: int) -> list[MarketTape]:
    """Copies of the tapes with every step value from ``row`` on changed."""
    out = []
    for tp in tapes:
        changed = {}
        for name in STEP_ARRAYS:
            arr = getattr(tp, name).copy()
            arr[row:] = arr[row:] * 1.5 + 0.02
            changed[name] = arr
        out.append(MarketTape(**{**tp.__dict__, **changed}))
    return out


def _small_cfg() -> dict:
    return {
        "env": {"window": WINDOW, "episode_len": 128, "n_envs": 4},
        "sac": {"hidden": 32, "critic_blocks": 1, "actor_hidden": 16, "batch_size": 32},
        "train": {
            "total_steps": 300,
            "warmup_steps": 100,
            "utd": 0.5,
            "eval_every": 50,
            "threads": 1,
        },
    }


def _filled_replay(ts: TapeSet, seed: int = 1, steps: int = 20, n_envs: int = 8) -> IndexReplay:
    env = VecTradingEnv(ts, EnvConfig(window=WINDOW, episode_len=200, n_envs=n_envs), seed=seed)
    replay = IndexReplay(1000, drawdown=True)
    rng = np.random.default_rng(seed + 1)
    for _ in range(steps):
        replay.add(env.step(rng.uniform(-1.0, 1.0, n_envs)))
    return replay


def test_next_drawdown_is_the_log_drawdown_of_the_equity_path():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.02, 300)
    log_equity = np.concatenate([[0.0], np.cumsum(r)])
    expected = np.maximum.accumulate(log_equity) - log_equity  # log(P_t / E_t)
    got = np.zeros_like(expected)
    for t in range(len(r)):
        got[t + 1] = next_drawdown(got[t], r[t])
    assert np.allclose(got, expected, atol=1e-12)


def test_next_drawdown_toy_cases():
    assert next_drawdown(0.1, 0.03) == pytest.approx(0.07)  # equity rose: drawdown shrinks
    assert next_drawdown(0.1, -0.02) == pytest.approx(0.12)  # equity fell: drawdown deepens
    assert next_drawdown(0.1, 0.5) == 0.0  # a new peak: no drawdown


def test_shaping_charges_only_drawdown_beyond_tolerance():
    values = np.array([0.01, -0.01, 0.03, -0.03])
    shaping = DrawdownShaping(penalty=2.0, tolerance=0.05)
    shaping.fit(values)
    sd = np.std(values, ddof=1)
    assert shaping.sd == pytest.approx(sd)
    # Drawdown after the step is 0.04, inside the tolerance: no charge.
    assert shaping.shape(np.array([0.0]), np.array([0.04])) == pytest.approx([0.0])
    # Drawdown 0.10, excess 0.05 over the tolerance.
    assert shaping.shape(np.array([0.0]), np.array([0.10])) == pytest.approx([-2.0 * sd * 0.05])
    # A gain of 0.02 takes the drawdown to 0.08, excess 0.03.
    assert shaping.shape(np.array([0.02]), np.array([0.10])) == pytest.approx(
        [0.02 - 2.0 * sd * 0.03]
    )


def test_shaping_caps_the_charge_so_a_flat_book_after_a_wipe_out_stays_bounded():
    values = np.array([0.01, -0.01, 0.03, -0.03])
    shaping = DrawdownShaping(penalty=1.0, tolerance=0.05)
    shaping.fit(values)
    sd = np.std(values, ddof=1)
    # A wipe-out floors the step at log(1e3) of loss and leaves the book flat, so every later bar
    # carries this drawdown with no growth; the charge must stop growing with it.
    charge = shaping.shape(np.zeros(1), np.array([np.log(1e3)]))
    assert charge == pytest.approx([-1.0 * sd * DD_EXCESS_CAP])


def test_shaping_needs_a_fitted_positive_scale():
    with pytest.raises(ValueError):
        DrawdownShaping(1.0, 0.05).shape(np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError):
        DrawdownShaping(1.0, 0.05).fit(np.zeros(5))


def test_switch_is_off_by_default_and_the_penalty_needs_the_observation():
    assert drawdown_switch({}) == DrawdownSwitch(0.0, 0.05, False)
    assert drawdown_switch({"obs": {"drawdown": True}}).obs
    with pytest.raises(ValueError, match="obs.drawdown"):
        drawdown_switch({"reward": {"dd_penalty": 1.0}})
    with pytest.raises(ValueError):
        drawdown_switch({"reward": {"dd_penalty": -0.5}, "obs": {"drawdown": True}})
    with pytest.raises(ValueError):
        drawdown_switch(
            {"reward": {"dd_penalty": 1.0, "dd_tolesance": 0.1}, "obs": {"drawdown": True}}
        )


def test_train_config_refuses_the_penalty_without_state_or_with_online_updates():
    with pytest.raises(ValueError):
        TrainConfig(dd_penalty=1.0)
    env = EnvConfig(window=WINDOW, drawdown_obs=True)
    with pytest.raises(ValueError):
        TrainConfig(dd_penalty=1.0, env=env, online_every=5)
    with pytest.raises(ValueError):
        TrainConfig(env=env, sac=SACConfig(pathwise_horizon=4))


def test_env_drawdown_matches_the_equity_path_of_its_own_rewards():
    ts = TapeSet(_tapes(1), WINDOW, drawdown=True)
    env = VecTradingEnv(ts, EnvConfig(window=WINDOW, episode_len=10_000, n_envs=1), seed=0)
    rng = np.random.default_rng(3)
    log_equity, peak = 0.0, 0.0
    for _ in range(200):
        expected = peak - log_equity
        assert env.u[0] == pytest.approx(expected, abs=1e-9)
        assert env.observe()[0, -1] == pytest.approx(np.log1p(expected), abs=1e-6)
        batch = env.step(rng.uniform(-1.0, 1.0, 1))
        log_equity += batch.reward[0]
        peak = max(peak, log_equity)
        if batch.terminal[0] or batch.truncated[0]:
            break


def test_observation_is_log1p_of_the_drawdown_state_and_stays_finite():
    ts = TapeSet(_tapes(1), WINDOW, drawdown=True)
    u = np.array([0.0, 0.05, 1.0, 7.0, 50.0])
    n = len(u)
    obs = ts.observe(np.zeros(n, dtype=np.int64), np.full(n, 50), np.zeros(n), np.ones(n), u)
    assert np.isfinite(obs).all()
    assert obs[:, -1] == pytest.approx(np.log1p(u), abs=1e-6)


def test_drawdown_observation_at_bar_t_ignores_later_bars():
    tapes = _tapes(2, n=900)
    row = len(tapes[0]) // 2  # features warm up over the first bars, so the tape is shorter

    def last_column(ts: TapeSet) -> np.ndarray:
        seen = []

        def act(obs):
            seen.append(obs[:, -1].copy())
            return np.full(len(obs), 0.5)

        rollout(ts, act)
        return np.array(seen)

    before = last_column(TapeSet(tapes, WINDOW, drawdown=True))
    after = last_column(TapeSet(_perturb_after(tapes, row), WINDOW, drawdown=True))
    decided = row - (WINDOW - 1) + 1  # steps whose bar is at or before ``row``
    assert np.allclose(before[:decided], after[:decided])
    assert not np.allclose(before[decided:], after[decided:])  # the change is visible later


def test_relabel_keeps_the_stored_state_and_recomputes_the_next_drawdown():
    ts = TapeSet(_tapes(), WINDOW, drawdown=True)
    replay = _filled_replay(ts)
    rng = np.random.default_rng(5)
    sample = replay.sample(64, rng)
    stored_u = sample["u"].copy()
    relabel(sample, ts, 1.0, rng)
    assert np.array_equal(sample["u"], stored_u)
    obs, next_obs = materialize(sample, ts)
    assert obs.shape == (64, ts.obs_dim)
    assert obs[:, -1] == pytest.approx(np.log1p(stored_u), abs=1e-6)
    next_u = next_drawdown(stored_u, sample["reward"])
    assert next_obs[:, -1] == pytest.approx(np.log1p(next_u), abs=1e-6)
    _, reward, _ = ts.transition(
        sample["k"], sample["t"], sample["w_prev"], sample["action"], sample["cost_mult"]
    )
    assert reward == pytest.approx(sample["reward"])


def test_actor_and_critic_take_the_drawdown_feature_with_finite_gradients():
    ts = TapeSet(_tapes(), WINDOW, drawdown=True)
    replay = _filled_replay(ts)
    rng = np.random.default_rng(7)
    sample = replay.sample(32, rng)
    obs, next_obs = materialize(sample, ts)
    agent = TQCSAC(
        ts.obs_dim, 1, SACConfig(hidden=32, critic_blocks=1, actor_hidden=16, batch_size=32), seed=0
    )
    net = getattr(agent.critic, "net", agent.critic)
    before = net.inp.weight.detach().clone()
    stats = agent.update(obs, sample["action"], sample["reward"], next_obs, sample["terminal"])
    assert all(np.isfinite(v) for v in stats.values() if np.ndim(v) == 0)
    assert torch.isfinite(net.inp.weight).all()
    # The drawdown is the last observation column: the critic's first layer must learn from it.
    drawdown_col = (net.inp.weight.detach() - before)[:, ts.obs_dim - 1]
    assert float(drawdown_col.abs().sum()) > 0.0


def test_training_with_the_drawdown_reward_runs_and_is_finite():
    cfg = {
        **_small_cfg(),
        "reward": {"dd_penalty": 1.0, "dd_tolerance": 0.05},
        "obs": {"drawdown": True},
    }
    res = train(_tapes(), _tapes(seed=10), build_train_config(cfg, seed=3))
    assert res.agent.cfg.hidden == 32
    assert np.isfinite(res.best_score)
    assert res.history and all(np.isfinite(r["val_sharpe"]) for r in res.history)


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if torch.is_tensor(a):
        return torch.equal(a, b)
    return np.array_equal(np.asarray(a), np.asarray(b))


def _feed(h, obj) -> None:
    if isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=repr):
            h.update(repr(k).encode())
            _feed(h, obj[k])
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for x in obj:
            _feed(h, x)
        h.update(b"]")
    elif torch.is_tensor(obj):
        arr = obj.detach().cpu().numpy()
        h.update(f"T{arr.dtype}{arr.shape}".encode())
        h.update(arr.tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(f"A{obj.dtype}{obj.shape}".encode())
        h.update(obj.tobytes())
    else:
        h.update(repr(obj).encode())


def _digest(res) -> str:
    """Weights, scores and history (without wall-clock times) of a training result."""
    h = hashlib.sha256()
    _feed(h, res.best_state)
    _feed(h, [res.best_score, res.reward_scale])
    _feed(h, [{k: v for k, v in r.items() if k != "time"} for r in res.history])
    return h.hexdigest()


# Digest of the training run with the base agent (no drawdown switch) under ``_small_cfg``.
# A change to the base training path changes it, which is the point of pinning it.
BASE_DIGEST = "fe5b777b26abcb6006fc0941de3c826731faaf913cdcf8f1089ef81bf52c7813"


def test_switch_off_matches_the_pinned_digest_of_the_base_agent():
    cfg = {
        **_small_cfg(),
        "reward": {"dd_penalty": 0.0, "dd_tolerance": 0.05},
        "obs": {"drawdown": False},
    }
    res = train(_tapes(), _tapes(seed=10), build_train_config(cfg, seed=3))
    assert _digest(res) == BASE_DIGEST


def test_absent_switch_equals_explicit_switch_off():
    base_cfg = _small_cfg()
    off_cfg = {
        **_small_cfg(),
        "reward": {"dd_penalty": 0.0, "dd_tolerance": 0.05},
        "obs": {"drawdown": False},
    }
    runs = []
    for cfg in (base_cfg, off_cfg):
        runs.append(train(_tapes(), _tapes(seed=10), build_train_config(cfg, seed=3)))
    base, off = runs
    assert _same(base.best_state, off.best_state)
    assert base.reward_scale == off.reward_scale and base.best_score == off.best_score
    # repr compares NaN values (the train-episode mean before any episode ends) as equal.
    strip = [{k: v for k, v in r.items() if k != "time"} for r in base.history]
    assert repr(strip) == repr([{k: v for k, v in r.items() if k != "time"} for r in off.history])
