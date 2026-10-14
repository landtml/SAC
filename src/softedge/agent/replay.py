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

from softedge.env.vec_env import Batch, TapeSet

FIELDS = {
    "k": np.int64, "t": np.int64, "w_prev": np.float64, "action": np.float64,
    "reward": np.float64, "w_next": np.float64, "terminal": np.bool_, "cost_mult": np.float64,
}


class IndexReplay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = {f: np.zeros(capacity, dtype=d) for f, d in FIELDS.items()}
        # Decision step that recorded each transition. Steps never decrease, so the transitions of
        # any range of steps are a contiguous range of insertion indices (``causal_range``).
        self.step_of = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self.ptr = 0
        self.total = 0  # transitions ever added; above capacity the ring has overwritten rows
        self.last_step = 0

    def __len__(self) -> int:
        return self.size

    def add(self, batch: Batch, step: int = 0) -> None:
        if step < self.last_step:
            raise ValueError(f"step {step} is earlier than the last recorded step {self.last_step}")
        n = len(batch.k)
        idx = (self.ptr + np.arange(n)) % self.capacity
        for f in FIELDS:
            self.data[f][idx] = getattr(batch, f)
        self.step_of[idx] = step
        self.last_step = step
        self.total += n
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=n)
        return {f: a[idx].copy() for f, a in self.data.items()}

    def causal_range(self, now: int, window: int | None = None) -> tuple[int, int]:
        """Insertion range ``[lo, hi)`` of the transitions usable at decision step ``now``.

        A transition recorded at step s reads the tape at bar s and realizes its reward and next
        exposure at bar s + 1, so it is usable only when s + 1 <= now (s < now). With a window, it
        must also lie in the last ``window`` steps (s >= now - window).
        """
        if self.total > self.capacity:
            raise ValueError("replay wrapped; insertion order no longer follows the steps")
        stamps = self.step_of[: self.size]
        hi = int(np.searchsorted(stamps, now, side="left"))
        lo = 0 if window is None else int(np.searchsorted(stamps, now - window, side="left"))
        return lo, hi

    def sample_causal(self, n: int, rng: np.random.Generator, now: int,
                      window: int | None = None) -> dict[str, np.ndarray]:
        """Uniform sample of ``n`` transitions realized by step ``now`` (see ``causal_range``)."""
        lo, hi = self.causal_range(now, window)
        if hi <= lo:
            raise ValueError(f"no transition is realized by step {now}")
        idx = rng.integers(lo, hi, size=n)
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
    """Observation and next observation for a sampled batch."""
    k, t = sample["k"], sample["t"]
    obs = tapes.observe(k, t, sample["w_prev"], sample["cost_mult"])
    # Terminal steps leave the tape; their next observation is masked out of the target.
    t_next = np.minimum(t + 1, tapes.lengths[k] - 1)
    next_obs = tapes.observe(k, t_next, sample["w_next"], sample["cost_mult"])
    return obs, next_obs
