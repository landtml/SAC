"""Causal online updates during deployment (continual test-time training).

A walk-forward fold deploys its agent frozen for the whole test window. Here the deployed agent
keeps learning from what the deployment itself realizes. Before acting on bar c, and every
``every`` bars, each agent takes ``steps`` gradient steps of the ordinary critic, actor and
temperature losses on transitions recorded at earlier steps. A transition recorded at step s is
usable at step c only when s + 1 <= c (its reward and next exposure are realized at bar s + 1),
and sampling is uniform over the last ``window`` steps. Counterfactual relabelling stays on,
since every tape value it reads lies at or before bar c.

Exposures still come from the live policy. The executed exposure is the policy target after the
no-trade band, when one is set. The replay keeps the policy's own transition (its action with
the reward and next exposure the tape gives that action), so every stored transition is an exact
sample of the environment dynamics even when the band holds the executed exposure.

Learning rates, losses and the divergence guard are the training ones. A guard trip restores the
checkpoint the deployment started from, as training restores its best checkpoint.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch

from softedge.agent.replay import IndexReplay, materialize, relabel
from softedge.agent.sac import TQCSAC
from softedge.env.vec_env import Batch, TapeSet

KEYS = ("ret", "target", "w_prev", "cost", "turnover")


def causal_update(agent: TQCSAC, replay: IndexReplay, tapes: TapeSet, *, now: int,
                  reward_scale: float, window: int | None, steps: int, batch_size: int,
                  relabel_frac: float, rng: np.random.Generator, snapshot: dict,
                  q_abs_limit: float = 100.0) -> dict[str, int]:
    """``steps`` gradient steps of ``agent`` on transitions realized by step ``now``.

    Returns the number of updates and guard restores. A guard trip restores ``snapshot``.
    """
    restores = 0
    for _ in range(steps):
        sample = replay.sample_causal(batch_size, rng, now=now, window=window)
        relabel(sample, tapes, relabel_frac, rng)
        obs, next_obs = materialize(sample, tapes)
        stats = agent.update(obs, sample["action"], sample["reward"] * reward_scale, next_obs,
                             sample["terminal"])
        if not np.isfinite(stats["critic_loss"]) or abs(stats["q_mean"]) > q_abs_limit:
            agent.load_state_dict(snapshot)
            agent._make_optimizers()
            restores += 1
    return {"updates": steps, "restores": restores}


def online_rollout(tapes: TapeSet, agents: Sequence[TQCSAC], reward_scales: Sequence[float], *,
                   every: int, steps: int, window: int, batch_size: int, relabel_frac: float,
                   seed: int = 0, band: float = 0.0, q_abs_limit: float = 100.0,
                   trace: list | None = None) -> tuple[list[dict[str, np.ndarray]], dict]:
    """Deploy ``agents`` as one ensemble on every tape, updating them online.

    The ensemble acts with the mean of the agents' deterministic actions, as ``ensemble_policy``
    does, and the tapes advance in lockstep as in ``rollout``. Before the bar of step i, for every
    positive i that is a multiple of ``every``, each agent runs ``causal_update``. With no block
    the loop is ``rollout`` exactly. The agents are changed in place.

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
    replay = IndexReplay(int(span.sum()))
    snapshots = [agent.state_dict() for agent in agents]
    info = {"blocks": 0, "updates": 0, "restores": 0, "replay": replay}

    for i in range(horizon):
        if i > 0 and i % every == 0:
            for j, agent in enumerate(agents):
                counts = causal_update(agent, replay, tapes, now=i, reward_scale=reward_scales[j],
                                       window=window, steps=steps, batch_size=batch_size,
                                       relabel_frac=relabel_frac, rng=rng,
                                       snapshot=snapshots[j], q_abs_limit=q_abs_limit)
                info["updates"] += counts["updates"]
                info["restores"] += counts["restores"]
            info["blocks"] += 1
            if trace is not None:
                trace.append((i, [_digest(agent) for agent in agents]))

        live = np.flatnonzero(i < span)
        obs = tapes.observe(k[live], t[live], w[live], cm[live])
        act = np.asarray(np.mean([agent.act(obs, deterministic=True) for agent in agents], axis=0),
                         dtype=np.float64).reshape(-1)
        policy_target = tapes.targets(k[live], t[live], act)
        policy, reward, terminal = tapes.transition_target(k[live], t[live], w[live],
                                                           policy_target, cm[live])
        target, res = policy_target, policy
        if band > 0:
            hold = np.abs(policy_target - w[live]) <= band * tapes.max_leverage[live]
            target = np.where(hold, w[live], policy_target)
            res, _, _ = tapes.transition_target(k[live], t[live], w[live], target, cm[live])

        replay.add(Batch(k=k[live].copy(), t=t[live].copy(), w_prev=w[live].copy(),
                         action=act.copy(), reward=reward, w_next=policy.w_next,
                         terminal=terminal, truncated=np.zeros(len(live), dtype=bool),
                         cost_mult=cm[live].copy(), ret=policy.ret), step=i)
        grid["ret"][live, i] = res.ret
        grid["target"][live, i] = target
        grid["w_prev"][live, i] = w[live]
        grid["cost"][live, i] = res.cost
        grid["turnover"][live, i] = res.turnover
        w[live] = res.w_next
        t[live] += 1

    out = [{key: grid[key][j, : span[j]] for key in KEYS} for j in range(n)]
    return out, info


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
