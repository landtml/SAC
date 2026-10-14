import numpy as np
import pytest
import torch

from softedge.agent import online
from softedge.agent.online import causal_update, online_rollout
from softedge.agent.replay import IndexReplay
from softedge.agent.sac import TQCSAC, SACConfig
from softedge.data.synthetic import gbm
from softedge.env.vec_env import Batch, TapeSet, rollout
from softedge.pipeline import to_tapes
from softedge.tape import MarketTape
from softedge.train import TrainConfig

torch.set_num_threads(1)  # as the training configs; the test networks are tiny

WINDOW = 4
STEP_ARRAYS = ("long_ret", "short_ret", "short_growth", "long_worst", "short_worst",
               "carry_long", "carry_short", "fin_long", "cost_long", "cost_short")


def _tapes(n_tapes: int = 2, n: int = 600) -> list[MarketTape]:
    return to_tapes([gbm(n, sigma=0.01, seed=i, symbol=f"S{i}", start="2021-01-01")
                     for i in range(n_tapes)])


def _agent(obs_dim: int, seed: int = 0) -> TQCSAC:
    cfg = SACConfig(hidden=32, critic_blocks=1, actor_hidden=16, batch_size=32)
    return TQCSAC(obs_dim, 1, cfg, seed=seed)


def _batch(tags) -> Batch:
    n = len(tags)
    z = np.zeros(n)
    return Batch(k=np.zeros(n, np.int64), t=np.asarray(tags, np.int64), w_prev=z, action=z,
                 reward=z, w_next=z, terminal=np.zeros(n, bool), truncated=np.zeros(n, bool),
                 cost_mult=np.ones(n), ret=z)


def _perturb_after(tapes: list[MarketTape], row: int) -> list[MarketTape]:
    """Copies of the tapes with every value from ``row`` on changed (features and step arrays)."""
    out = []
    for tp in tapes:
        changed = {}
        for name in STEP_ARRAYS:
            arr = getattr(tp, name).copy()
            arr[row:] = arr[row:] * 1.5 + 0.02
            changed[name] = arr
        feats = tp.features.copy()
        feats[row:] = feats[row:] * 0.5 + 0.3
        out.append(MarketTape(**{**tp.__dict__, **changed, "features": feats}))
    return out


# Replay: causal sampling mask (toy case, exact).

def test_causal_range_selects_transitions_realized_by_the_current_bar():
    replay = IndexReplay(40)
    for s in range(10):  # step s records two transitions, tagged with s in the t field
        replay.add(_batch([s, s]), step=s)
    # At bar 5, a transition recorded at step s is realized when s + 1 <= 5, so s <= 4. The
    # last 3 bars keep s in {2, 3, 4}: insertion indices 4 to 9.
    assert replay.causal_range(now=5, window=3) == (4, 10)
    assert replay.causal_range(now=5, window=None) == (0, 10)
    rng = np.random.default_rng(0)
    tags = replay.sample_causal(500, rng, now=5, window=3)["t"]
    assert set(np.unique(tags)) == {2, 3, 4}


def test_causal_sampling_refuses_an_empty_past():
    replay = IndexReplay(8)
    replay.add(_batch([0, 0]), step=0)
    with pytest.raises(ValueError):
        replay.sample_causal(4, np.random.default_rng(0), now=0, window=5)


def test_wrapped_replay_refuses_causal_sampling():
    replay = IndexReplay(4)
    for s in range(3):
        replay.add(_batch([s, s]), step=s)  # the third batch wraps the ring
    with pytest.raises(ValueError):
        replay.causal_range(now=3, window=2)


def test_steps_must_not_decrease():
    replay = IndexReplay(8)
    replay.add(_batch([3, 3]), step=3)
    with pytest.raises(ValueError):
        replay.add(_batch([2, 2]), step=2)


# Loop mechanics.

def test_default_switch_is_off():
    cfg = TrainConfig()
    assert cfg.online_every == 0 and cfg.online_steps == 16 and cfg.online_window == 5000


def test_without_blocks_the_loop_is_the_frozen_rollout():
    ts = TapeSet(_tapes(), WINDOW)
    agent = _agent(ts.obs_dim)
    ref = rollout(ts, lambda o: agent.act(o, deterministic=True), band=0.1)
    got, info = online_rollout(ts, [agent], [1.0], every=10**9, steps=4, window=50,
                               batch_size=32, relabel_frac=0.5, band=0.1)
    assert info["blocks"] == 0 and info["updates"] == 0
    for r, g in zip(ref, got, strict=True):
        for key in r:
            np.testing.assert_array_equal(g[key], r[key])


def test_blocks_update_the_policy_and_are_counted():
    ts = TapeSet(_tapes(), WINDOW)
    agent = _agent(ts.obs_dim)
    frozen = rollout(ts, lambda o: agent.act(o, deterministic=True))
    got, info = online_rollout(ts, [agent], [1.0], every=20, steps=3, window=50,
                               batch_size=32, relabel_frac=0.5)
    horizon = int((ts.last_step_row - ts.first_row + 1).max())
    assert info["blocks"] == (horizon - 1) // 20
    assert info["updates"] == info["blocks"] * 3
    assert info["restores"] == 0
    assert any(not np.array_equal(g["target"], f["target"]) for g, f in zip(got, frozen, strict=True))


def test_stored_transitions_are_exact_samples_of_the_environment():
    # With a no-trade band the executed exposure differs from the policy target. The replay
    # keeps the policy's own transition, so reward and next exposure must match the tape.
    ts = TapeSet(_tapes(), WINDOW)
    agent = _agent(ts.obs_dim)
    _, info = online_rollout(ts, [agent], [1.0], every=15, steps=2, window=40, batch_size=32,
                             relabel_frac=0.0, band=0.2)
    rep = info["replay"]
    n = rep.size
    k, t = rep.data["k"][:n], rep.data["t"][:n]
    w, a, cm = rep.data["w_prev"][:n], rep.data["action"][:n], rep.data["cost_mult"][:n]
    res, reward, terminal = ts.transition(k, t, w, a, cm)
    np.testing.assert_array_equal(rep.data["reward"][:n], reward)
    np.testing.assert_array_equal(rep.data["w_next"][:n], res.w_next)
    np.testing.assert_array_equal(rep.data["terminal"][:n], terminal)


def test_causal_update_moves_actor_critic_and_temperature():
    ts = TapeSet(_tapes(), WINDOW)
    seed_agent = _agent(ts.obs_dim)
    _, info = online_rollout(ts, [seed_agent], [1.0], every=10**9, steps=1, window=50,
                             batch_size=32, relabel_frac=0.5)
    agent = _agent(ts.obs_dim, seed=1)
    before = {name: p.detach().clone() for name, p in agent.actor.named_parameters()}
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]
    alpha_before = float(agent.log_alpha.detach())
    causal_update(agent, info["replay"], ts, now=60, reward_scale=1.0, window=50, steps=3,
                  batch_size=32, relabel_frac=0.5, rng=np.random.default_rng(0),
                  snapshot=agent.state_dict())
    assert any(not np.array_equal(before[n].numpy(), p.detach().numpy())
               for n, p in agent.actor.named_parameters())
    assert any(not np.array_equal(b.numpy(), p.detach().numpy())
               for b, p in zip(critic_before, agent.critic.parameters(), strict=True))
    assert float(agent.log_alpha.detach()) != alpha_before
    assert agent.updates == 3


def test_divergence_guard_restores_the_snapshot():
    ts = TapeSet(_tapes(), WINDOW)
    agent = _agent(ts.obs_dim)
    initial = online._digest(agent)
    trace: list = []
    _, info = online_rollout(ts, [agent], [1.0], every=10, steps=2, window=50, batch_size=32,
                             relabel_frac=0.5, q_abs_limit=0.0, trace=trace)
    assert info["restores"] == info["updates"] > 0
    # Every update trips the guard, so the policy never leaves its checkpoint.
    assert all(digests == [initial] for _, digests in trace)


# Look-ahead tests.

@pytest.mark.parametrize("block", [10, 20])
def test_update_at_bar_c_is_unchanged_by_tape_data_after_c(block):
    tapes = _tapes()
    ts = TapeSet(tapes, WINDOW)
    row = ts.first_row + block  # the bar of the update at step `block`
    ts_late = TapeSet(_perturb_after(tapes, row + 1), WINDOW)
    kwargs = dict(every=10, steps=2, window=50, batch_size=32, relabel_frac=0.5, seed=3)
    trace_a, trace_b = [], []
    out_a, _ = online_rollout(ts, [_agent(ts.obs_dim, 0)], [1.0], trace=trace_a, **kwargs)
    out_b, _ = online_rollout(ts_late, [_agent(ts.obs_dim, 0)], [1.0], trace=trace_b, **kwargs)
    digests_a = dict(trace_a)
    digests_b = dict(trace_b)
    assert digests_a[block] == digests_b[block]
    for ra, rb in zip(out_a, out_b, strict=True):
        np.testing.assert_array_equal(ra["target"][: block + 1], rb["target"][: block + 1])
    # The perturbation is visible later, so the check is not vacuous.
    later = [i for i in digests_a if i > block]
    assert later and any(digests_a[i] != digests_b[i] for i in later)


def test_planted_look_ahead_in_next_observation_is_detected(monkeypatch):
    tapes = _tapes()
    ts = TapeSet(tapes, WINDOW)
    row = ts.first_row + 10
    ts_late = TapeSet(_perturb_after(tapes, row + 1), WINDOW)
    real = online.materialize

    def leaky(sample, tapes_):  # next observation read two bars ahead
        obs, _ = real(sample, tapes_)
        t2 = np.minimum(sample["t"] + 2, tapes_.lengths[sample["k"]] - 1)
        return obs, tapes_.observe(sample["k"], t2, sample["w_next"], sample["cost_mult"])

    monkeypatch.setattr(online, "materialize", leaky)
    kwargs = dict(every=10, steps=2, window=50, batch_size=32, relabel_frac=0.5, seed=3)
    trace_a, trace_b = [], []
    online_rollout(ts, [_agent(ts.obs_dim, 0)], [1.0], trace=trace_a, **kwargs)
    online_rollout(ts_late, [_agent(ts.obs_dim, 0)], [1.0], trace=trace_b, **kwargs)
    assert dict(trace_a)[10] != dict(trace_b)[10]


def test_unsupported_agents_are_refused():
    ts = TapeSet(_tapes(), WINDOW)
    agent = TQCSAC(ts.obs_dim, 1, SACConfig(hidden=32, critic_blocks=1, actor_hidden=16,
                                            batch_size=32, pathwise_horizon=4), seed=0)
    with pytest.raises(ValueError):
        online_rollout(ts, [agent], [1.0], every=10, steps=1, window=50, batch_size=32,
                       relabel_frac=0.5)
