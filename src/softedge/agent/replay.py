"""Index replay with counterfactual position relabeling.

A small trader does not move the market, so the market half of a transition depends only on
(tape, row). Each stored transition is ``(k, t, w_prev, action, reward, w_next, terminal,
cost_mult)``; observations are rebuilt on sampling. ``relabel`` replaces ``w_prev`` and the action
of a share of the batch with fresh draws and recomputes reward, next exposure and termination
exactly through the tape, which spreads training over the whole position range instead of the
positions the behaviour policy happened to visit.
"""

from __future__ import annotations

import numpy as np

from softedge.agent.drawdown import next_drawdown
from softedge.env.vec_env import Batch, TapeSet

FIELDS = {
    "k": np.int64, "t": np.int64, "w_prev": np.float64, "action": np.float64,
    "reward": np.float64, "w_next": np.float64, "terminal": np.bool_, "cost_mult": np.float64,
}
# Stored only with the drawdown observation: the log drawdown u_t before each transition.
DRAWDOWN_FIELDS = {"u": np.float64}


class IndexReplay:
    def __init__(self, capacity: int, drawdown: bool = False):
        self.capacity = capacity
        fields = {**FIELDS, **DRAWDOWN_FIELDS} if drawdown else FIELDS
        self.data = {f: np.zeros(capacity, dtype=d) for f, d in fields.items()}
        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        return self.size

    def add(self, batch: Batch) -> None:
        n = len(batch.k)
        idx = (self.ptr + np.arange(n)) % self.capacity
        for f in self.data:
            value = getattr(batch, f)
            if value is None:
                raise ValueError(f"the batch has no {f}, which the replay stores")
            self.data[f][idx] = value
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=n)
        return {f: a[idx].copy() for f, a in self.data.items()}


def relabel(sample: dict[str, np.ndarray], tapes: TapeSet, frac: float,
            rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Resample ``w_prev`` and ``action`` for a ``frac`` share of the batch, in place.

    Exposure draws: half uniform on [-L, L], a quarter flat, a quarter at +L or -L, where L is the
    tape's max leverage. Actions are uniform on [-1, 1].
    """
    n = len(sample["k"])
    m = int(round(frac * n))
    if m == 0:
        return sample
    sel = rng.choice(n, size=m, replace=False)
    k = sample["k"][sel]
    lev = tapes.max_leverage[k]
    u = rng.random(m)
    w = np.where(u < 0.5, rng.uniform(-1.0, 1.0, m) * lev,
                 np.where(u < 0.75, 0.0, np.sign(rng.random(m) - 0.5) * lev))
    a = rng.uniform(-1.0, 1.0, m)
    res, reward, terminal = tapes.transition(k, sample["t"][sel], w, a, sample["cost_mult"][sel])
    sample["w_prev"][sel] = w
    sample["action"][sel] = a
    sample["reward"][sel] = reward
    sample["w_next"][sel] = res.w_next
    sample["terminal"][sel] = terminal
    return sample


def materialize(sample: dict[str, np.ndarray], tapes: TapeSet) -> tuple[np.ndarray, np.ndarray]:
    """Observation and next observation for a sampled batch.

    With the drawdown observation the next drawdown follows the batch's (possibly relabelled)
    reward, so call this after ``relabel``.
    """
    k, t = sample["k"], sample["t"]
    u = sample.get("u")
    obs = tapes.observe(k, t, sample["w_prev"], sample["cost_mult"], u)
    # Terminal steps leave the tape; their next observation is masked out of the target.
    t_next = np.minimum(t + 1, tapes.lengths[k] - 1)
    u_next = None if u is None else next_drawdown(u, sample["reward"])
    next_obs = tapes.observe(k, t_next, sample["w_next"], sample["cost_mult"], u_next)
    return obs, next_obs
