"""Causal online updates during deployment (continual test-time training).

A walk-forward fold deploys its agent frozen for the whole test window. Here the deployed agents
keep learning from what the deployment itself realizes. Before the decision at step c, and every
``every`` steps, each agent takes ``steps`` gradient steps of the ordinary critic, actor and
temperature losses on transitions recorded at earlier steps, sampled uniformly from the last
``window`` steps.

Tapes advance in lockstep by row, but the tapes of a market set can start on different dates, so
one step may sit at different bars on different tapes. A transition of tape k at row t realizes
its reward and next exposure at the bar of row t + 1. It is usable at step c only when that bar
is at or before the earliest bar among the live tapes at step c, so no update reads a bar later
than a decision it serves. Counterfactual relabelling stays on; the values it reads belong to the
transition's own row.

Exposures still come from the live policy. The executed exposure is the policy target after the
no-trade band, when one is set. The replay keeps the policy's own transition (its action with
the reward and next exposure the tape gives that action), so every stored transition is an exact
sample of the environment dynamics even when the band holds the executed exposure.

Losses, learning rates and the divergence guard are the training ones, except that Adam moments
start from zero: checkpoints hold no optimizer state. The target-entropy annealing and the
learning-rate schedule are not carried, and TrainConfig refuses them with online updates. A guard
trip restores the checkpoint the deployment started from, as training restores its best one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from softedge.agent.factory import agent_from_state
from softedge.agent.replay import IndexReplay, materialize, relabel
from softedge.agent.sac import TQCSAC
from softedge.env.vec_env import Batch, TapeSet

if TYPE_CHECKING:
    from softedge.train import TrainConfig

KEYS = ("ret", "target", "w_prev", "cost", "turnover")


class CausalReplay(IndexReplay):
    """Replay that also records, per transition, the decision step that recorded it and the bar
    time at which its reward and next exposure are realized. Insertion order need not follow
    either, so usable transitions are selected by mask, not by a contiguous range."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.step_of = np.zeros(capacity, dtype=np.int64)
        self.realized_ns = np.zeros(capacity, dtype=np.int64)

    def add(self, batch: Batch, step: int, realized_ns: np.ndarray) -> None:
        n = len(batch.k)
        if self.size + n > self.capacity:
            raise ValueError("the online replay must hold every transition of the rollout")
        at = slice(self.size, self.size + n)
        super().add(batch)
        self.step_of[at] = step
        self.realized_ns[at] = realized_ns

    def pool(self, now: int, now_ns: int, window: int | None) -> np.ndarray:
        """Insertion indices of the transitions usable at step ``now`` with bar time ``now_ns``:
        realized by that bar and recorded within the last ``window`` steps."""
        n = self.size
        ok = self.realized_ns[:n] <= now_ns
        if window is not None:
            ok &= self.step_of[:n] >= now - window
        return np.flatnonzero(ok)

    def sample_from(self, pool: np.ndarray, n: int,
                    rng: np.random.Generator) -> dict[str, np.ndarray]:
        if pool.size == 0:
            raise ValueError("no transition is realized by this bar")
        idx = pool[rng.integers(0, pool.size, size=n)]
        return {f: a[idx].copy() for f, a in self.data.items()}


def causal_update(agent: TQCSAC, replay: CausalReplay, tapes: TapeSet, *, now: int, now_ns: int,
                  reward_scale: float, window: int | None, steps: int, batch_size: int,
                  relabel_frac: float, rng: np.random.Generator, snapshot: dict,
                  q_abs_limit: float = 100.0) -> dict[str, int]:
    """``steps`` gradient steps of ``agent`` on transitions realized by bar ``now_ns``, the bar of
    step ``now``. Returns the number of updates and guard restores. A guard trip restores
    ``snapshot``."""
    pool = replay.pool(now, now_ns, window)
    restores = 0
    for _ in range(steps):
        sample = replay.sample_from(pool, batch_size, rng)
        relabel(sample, tapes, relabel_frac, rng)
        obs, next_obs = materialize(sample, tapes)
        stats = agent.update(obs, sample["action"], sample["reward"] * reward_scale, next_obs,
                             sample["terminal"])
        if _diverged(agent, stats, q_abs_limit):
            _restore(agent, snapshot)
            restores += 1
    return {"updates": steps, "restores": restores}


def _restore(agent: TQCSAC, snapshot: dict) -> None:
    agent.load_state_dict(snapshot)
    agent._make_optimizers()


def _diverged(agent: TQCSAC, stats: dict[str, float], q_abs_limit: float) -> bool:
    """True on a non-finite loss, temperature, critic or actor weight, or |mean Q| past the limit."""
    values = (stats["critic_loss"], stats["actor_loss"], stats["alpha"], stats["q_mean"])
    if not all(np.isfinite(v) for v in values) or abs(stats["q_mean"]) > q_abs_limit:
        return True
    if not np.isfinite(float(agent.log_alpha.detach())):
        return True
    weights = [*agent.critic.parameters(), *agent.actor.parameters()]
    return not all(bool(torch.isfinite(p).all()) for p in weights)


def online_rollout(tapes: TapeSet, agents: Sequence[TQCSAC], reward_scales: Sequence[float], *,
                   every: int, steps: int, window: int, batch_size: int, relabel_frac: float,
                   seed: int = 0, band: float = 0.0, q_abs_limit: float = 100.0,
                   trace: list | None = None) -> tuple[list[dict[str, np.ndarray]], dict]:
    """Deploy ``agents`` as one ensemble on every tape, updating them online.

    The ensemble acts with the mean of the agents' deterministic actions, as ``ensemble_policy``
    does, and the tapes advance in lockstep as in ``rollout``. Before the bar of step i, for every
    positive i that is a multiple of ``every``, each agent runs ``causal_update``. With no block
    the loop is ``rollout`` exactly. The agents are changed in place. A non-finite exposure
    restores the agent that produced it and acts again from the checkpoint.

    Returns per-tape arrays as ``rollout`` does, and the counts ``blocks``, ``updates`` and
    ``restores``, plus the replay (``replay``) for inspection. ``trace``, when a list, receives
    ``(step, [digest of each agent])`` after every block.
    """
    if every < 1 or steps < 1 or window < 1:
        raise ValueError("every, steps and window must be at least 1")
    if not agents or len(agents) != len(reward_scales):
        raise ValueError("one reward scale per agent is required")
    for agent in agents:
        plain = isinstance(agent, TQCSAC) and agent.cfg.pathwise_horizon == 0
        if not plain or agent.cfg.discrete_levels:
            raise ValueError("online updates need a tanh-Gaussian TQC agent without the pathwise "
                             "objective")

    rng = np.random.default_rng(seed)
    n = len(tapes.tapes)
    k = np.arange(n)
    t = np.full(n, tapes.first_row, dtype=np.int64)
    w = np.zeros(n)
    cm = np.ones(n)
    span = tapes.last_step_row - tapes.first_row + 1
    horizon = int(span.max())
    grid = {key: np.zeros((n, horizon)) for key in KEYS}
    bar_ns = np.concatenate([tp.times for tp in tapes.tapes])  # indexed like the tape rows
    replay = CausalReplay(int(span.sum()))
    snapshots = [agent.state_dict() for agent in agents]
    info = {"blocks": 0, "updates": 0, "restores": 0, "replay": replay}

    for i in range(horizon):
        live = np.flatnonzero(i < span)
        if i > 0 and i % every == 0:
            now_ns = int(bar_ns[tapes.offsets[live] + t[live]].min())
            for j, agent in enumerate(agents):
                counts = causal_update(agent, replay, tapes, now=i, now_ns=now_ns,
                                       reward_scale=reward_scales[j], window=window, steps=steps,
                                       batch_size=batch_size, relabel_frac=relabel_frac, rng=rng,
                                       snapshot=snapshots[j], q_abs_limit=q_abs_limit)
                info["updates"] += counts["updates"]
                info["restores"] += counts["restores"]
            info["blocks"] += 1
            if trace is not None:
                trace.append((i, [_digest(agent) for agent in agents]))

        obs = tapes.observe(k[live], t[live], w[live], cm[live])
        acts = [agent.act(obs, deterministic=True) for agent in agents]
        for j, agent in enumerate(agents):
            if not np.all(np.isfinite(acts[j])):
                _restore(agent, snapshots[j])
                info["restores"] += 1
                acts[j] = agent.act(obs, deterministic=True)
                if not np.all(np.isfinite(acts[j])):
                    raise FloatingPointError(f"agent {j} acts non-finitely at its checkpoint")
        act = np.asarray(np.mean(acts, axis=0), dtype=np.float64).reshape(-1)
        policy_target = tapes.targets(k[live], t[live], act)
        policy, reward, terminal = tapes.transition_target(k[live], t[live], w[live],
                                                           policy_target, cm[live])
        target, res = policy_target, policy
        if band > 0:
            hold = np.abs(policy_target - w[live]) <= band * tapes.max_leverage[live]
            target = np.where(hold, w[live], policy_target)
            res, _, _ = tapes.transition_target(k[live], t[live], w[live], target, cm[live])

        # Reward and next exposure of the policy's transition are realized at the next row.
        next_row = np.minimum(t[live] + 1, tapes.lengths[live] - 1)
        replay.add(Batch(k=k[live].copy(), t=t[live].copy(), w_prev=w[live].copy(),
                         action=act.copy(), reward=reward, w_next=policy.w_next,
                         terminal=terminal, truncated=np.zeros(len(live), dtype=bool),
                         cost_mult=cm[live].copy(), ret=policy.ret), step=i,
                   realized_ns=bar_ns[tapes.offsets[live] + next_row])
        grid["ret"][live, i] = res.ret
        grid["target"][live, i] = target
        grid["w_prev"][live, i] = w[live]
        grid["cost"][live, i] = res.cost
        grid["turnover"][live, i] = res.turnover
        w[live] = res.w_next
        t[live] += 1

    out = [{key: grid[key][j, : span[j]] for key in KEYS} for j in range(n)]
    return out, info


def deploy_online(agents: Sequence[TQCSAC], tapes: TapeSet, reward_scales: dict[int, float] | None,
                  seeds: Sequence[int], tc: TrainConfig, band: float):
    """Walk-forward deployment of the banded ensemble with online updates (``tc.online_*``).

    The learners are rebuilt from the checkpoints, so the agents behind the validation gate stay
    frozen. Returns the per-tape arrays and the counts ``blocks``, ``updates`` and ``restores``.
    """
    if reward_scales is None:
        raise ValueError("online deployment is supported in walk-forward runs only")
    learners = [agent_from_state(agent.state_dict()) for agent in agents]
    out, info = online_rollout(tapes, learners, [reward_scales[s] for s in seeds],
                               every=tc.online_every, steps=tc.online_steps,
                               window=tc.online_window, batch_size=tc.sac.batch_size,
                               relabel_frac=tc.relabel_frac, seed=tc.seed, band=band,
                               q_abs_limit=tc.q_abs_limit)
    return out, {key: info[key] for key in ("blocks", "updates", "restores")}


def _digest(agent: TQCSAC) -> str:
    """Hash of the learned state (statistics, actor, critics, temperature)."""
    state = agent.state_dict()
    h = hashlib.sha256()
    for part in ("obs_norm", "actor", "critic", "critic_target"):
        for name, v in state[part].items():
            arr = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            h.update(name.encode())
            h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.float64(state["log_alpha"]).tobytes())
    return h.hexdigest()
