"""Vectorized trading environment over a set of market tapes.

State of one sub-environment is ``(tape k, row t, exposure w, cost multiplier c)``. The market part
is read from the tape, so observations can be rebuilt from those four numbers at any time. The
replay buffer stores them instead of observation vectors, and counterfactual replay relabels
``w`` and the action, then recomputes the reward with ``step_portfolio``.

Observation layout: ``[feature window (window x F) | w / max_leverage | log cost long | log cost
short | carry long bps | carry short bps | max_leverage / 5 | market-kind one-hot (4)]``.
Tapes may carry different feature columns; the observation uses the union of names and fills
missing columns with zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sactrader.portfolio import log_reward, step_portfolio
from sactrader.tape import KIND_INDEX, MarketTape

N_CONTEXT = 6 + len(KIND_INDEX)

_STEP_FIELDS = (
    "long_ret", "short_ret", "short_growth", "long_worst", "short_worst",
    "carry_long", "carry_short", "cost_long", "cost_short",
)


@dataclass(frozen=True)
class EnvConfig:
    window: int = 8
    episode_len: int = 1024
    n_envs: int = 16
    cost_mult_low: float = 1.0
    cost_mult_high: float = 1.0
    tape_weighting: str = "length"  # "length" or "uniform"
    # Half-life in bars for sampling episode starts toward the end of each tape. Markets drift,
    # so recent history is weighted up; None samples starts uniformly.
    recency_halflife: float | None = None


class TapeSet:
    """Concatenated tapes with a shared feature space and vectorized lookups."""

    def __init__(self, tapes: list[MarketTape], window: int, feature_names: tuple[str, ...] | None = None):
        if not tapes:
            raise ValueError("need at least one tape")
        self.tapes = tapes
        self.window = window
        names = feature_names or _union_names(tapes)
        self.feature_names = tuple(names)
        col = {n: i for i, n in enumerate(self.feature_names)}
        lengths = np.array([len(tp) for tp in tapes], dtype=np.int64)
        self.lengths = lengths
        self.offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        total = int(lengths.sum())
        feats = np.zeros((total, len(self.feature_names)), dtype=np.float32)
        for tp, off in zip(tapes, self.offsets, strict=True):
            idx = [col[n] for n in tp.feature_names if n in col]
            src = [i for i, n in enumerate(tp.feature_names) if n in col]
            feats[off:off + len(tp), idx] = tp.features[:, src]
        self.features = feats
        self.step = {f: np.concatenate([getattr(tp, f) for tp in tapes]) for f in _STEP_FIELDS}
        self.max_leverage = np.array([tp.max_leverage for tp in tapes])
        self.maint_margin = np.array([tp.maint_margin for tp in tapes])
        self.impact = np.array([tp.impact for tp in tapes])
        self.terminal_at_end = np.array([tp.terminal_at_end for tp in tapes])
        self.kind = np.array([tp.kind_index for tp in tapes])
        # Last row an episode may step from: T-1 for terminal tapes, T-2 otherwise.
        self.last_step_row = lengths - 1 - (~self.terminal_at_end).astype(np.int64)
        self.first_row = window - 1
        if (self.last_step_row < self.first_row).any():
            short = [tp.symbol for tp, ok in zip(tapes, self.last_step_row >= self.first_row, strict=True) if not ok]
            raise ValueError(f"tapes shorter than the window: {short}")

    @property
    def obs_dim(self) -> int:
        return self.window * len(self.feature_names) + N_CONTEXT

    def observe(self, k: np.ndarray, t: np.ndarray, w: np.ndarray, cost_mult: np.ndarray) -> np.ndarray:
        """Observation rows for arrays of (tape, row, exposure, cost multiplier)."""
        k = np.asarray(k, dtype=np.int64)
        t = np.asarray(t, dtype=np.int64)
        g = self.offsets[k] + t
        rows = g[:, None] + np.arange(-self.window + 1, 1)[None, :]
        win = self.features[rows].reshape(len(k), -1)
        lev = self.max_leverage[k]
        cl = self.step["cost_long"][g] * cost_mult
        cs = self.step["cost_short"][g] * cost_mult
        ctx = np.empty((len(k), N_CONTEXT), dtype=np.float32)
        ctx[:, 0] = np.asarray(w) / lev
        ctx[:, 1] = (np.log10(cl + 1e-6) + 3.0) / 2.0
        ctx[:, 2] = (np.log10(cs + 1e-6) + 3.0) / 2.0
        ctx[:, 3] = np.clip(self.step["carry_long"][g] * 1e4, -5, 5)
        ctx[:, 4] = np.clip(self.step["carry_short"][g] * 1e4, -5, 5)
        ctx[:, 5] = lev / 5.0
        ctx[:, 6:] = 0.0
        ctx[np.arange(len(k)), 6 + self.kind[k]] = 1.0
        return np.concatenate([win, ctx], axis=1).astype(np.float32)

    def transition(self, k: np.ndarray, t: np.ndarray, w_prev: np.ndarray, action: np.ndarray,
                   cost_mult: np.ndarray):
        """Reward and next exposure for action in [-1, 1] taken at (k, t) from exposure w_prev."""
        k = np.asarray(k, dtype=np.int64)
        g = self.offsets[k] + np.asarray(t, dtype=np.int64)
        lev = self.max_leverage[k]
        target = np.clip(action, -1.0, 1.0) * lev
        s = self.step
        res = step_portfolio(
            w_prev, target, s["long_ret"][g], s["short_ret"][g], s["short_growth"][g],
            s["long_worst"][g], s["short_worst"][g], s["carry_long"][g], s["carry_short"][g],
            s["cost_long"][g] * cost_mult, s["cost_short"][g] * cost_mult,
            self.impact[k], self.maint_margin[k],
        )
        terminal = self.terminal_at_end[k] & (np.asarray(t) == self.lengths[k] - 1)
        return res, log_reward(res.ret), terminal


def _union_names(tapes: list[MarketTape]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for tp in tapes:
        for n in tp.feature_names:
            if n not in seen:
                seen.add(n)
                names.append(n)
    return names


@dataclass
class Batch:
    """Transitions from one vector step, stored by index (see module docstring)."""

    k: np.ndarray
    t: np.ndarray
    w_prev: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    w_next: np.ndarray
    terminal: np.ndarray
    truncated: np.ndarray
    cost_mult: np.ndarray
    ret: np.ndarray


class VecTradingEnv:
    """``n_envs`` independent episodes on random windows of the tape set, auto-reset on end."""

    def __init__(self, tapes: TapeSet, cfg: EnvConfig, seed: int = 0):
        self.ts = tapes
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        n = cfg.n_envs
        span = (tapes.last_step_row - tapes.first_row + 1).astype(np.float64)
        if cfg.tape_weighting == "length":
            self.tape_p = span / span.sum()
        elif cfg.tape_weighting == "uniform":
            self.tape_p = np.full(len(span), 1.0 / len(span))
        else:
            raise ValueError(cfg.tape_weighting)
        self.k = np.zeros(n, dtype=np.int64)
        self.t = np.zeros(n, dtype=np.int64)
        self.end = np.zeros(n, dtype=np.int64)
        self.w = np.zeros(n)
        self.cost_mult = np.ones(n)
        self.ep_return = np.zeros(n)
        self.completed_returns: list[float] = []
        self._reset(np.arange(n))

    @property
    def obs_dim(self) -> int:
        return self.ts.obs_dim

    def _reset(self, idx: np.ndarray) -> None:
        if len(idx) == 0:
            return
        ts = self.ts
        k = self.rng.choice(len(self.tape_p), size=len(idx), p=self.tape_p)
        lo = ts.first_row
        hi = ts.last_step_row[k]
        u = self.rng.random(len(idx))
        if self.cfg.recency_halflife:
            # Inverse CDF of a truncated exponential over the distance back from ``hi``.
            lam = np.log(2.0) / self.cfg.recency_halflife
            span = (hi - lo + 1).astype(np.float64)
            back = -np.log1p(-u * (1.0 - np.exp(-lam * span))) / lam
            t0 = np.clip(hi - np.floor(back).astype(np.int64), lo, hi)
        else:
            t0 = lo + (u * (hi - lo + 1)).astype(np.int64)
        self.k[idx] = k
        self.t[idx] = t0
        # Stepping from row `end` is the episode's last step.
        self.end[idx] = np.minimum(t0 + self.cfg.episode_len - 1, hi)
        self.w[idx] = 0.0
        c_lo, c_hi = self.cfg.cost_mult_low, self.cfg.cost_mult_high
        self.cost_mult[idx] = np.exp(self.rng.uniform(np.log(c_lo), np.log(c_hi), len(idx)))
        self.ep_return[idx] = 0.0

    def observe(self) -> np.ndarray:
        return self.ts.observe(self.k, self.t, self.w, self.cost_mult)

    def step(self, action: np.ndarray) -> Batch:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        res, reward, terminal = self.ts.transition(self.k, self.t, self.w, action, self.cost_mult)
        last = self.t >= self.end
        truncated = last & ~terminal
        batch = Batch(
            k=self.k.copy(), t=self.t.copy(), w_prev=self.w.copy(), action=action.copy(),
            reward=reward, w_next=res.w_next, terminal=terminal, truncated=truncated,
            cost_mult=self.cost_mult.copy(), ret=res.ret,
        )
        self.ep_return += reward
        self.w = res.w_next
        self.t = self.t + 1
        done = np.flatnonzero(terminal | truncated)
        self.completed_returns.extend(self.ep_return[done].tolist())
        self._reset(done)
        return batch


def rollout(tapes: TapeSet, act_fn, cost_mult: float = 1.0) -> list[dict[str, np.ndarray]]:
    """Run ``act_fn(obs) -> actions`` over every tape from ``first_row`` to its end, all tapes in
    lockstep. Returns per-tape arrays of step returns, targets, previous exposures and costs."""
    n = len(tapes.tapes)
    k = np.arange(n)
    t = np.full(n, tapes.first_row, dtype=np.int64)
    w = np.zeros(n)
    cm = np.full(n, cost_mult)
    steps = tapes.last_step_row - tapes.first_row + 1
    keys = ("ret", "target", "w_prev", "cost", "turnover")
    grid = {key: np.zeros((n, int(steps.max()))) for key in keys}
    for i in range(int(steps.max())):
        live = np.flatnonzero(i < steps)
        obs = tapes.observe(k[live], t[live], w[live], cm[live])
        act = np.asarray(act_fn(obs), dtype=np.float64).reshape(-1)
        res, _, _ = tapes.transition(k[live], t[live], w[live], act, cm[live])
        grid["ret"][live, i] = res.ret
        grid["target"][live, i] = np.clip(act, -1, 1) * tapes.max_leverage[live]
        grid["w_prev"][live, i] = w[live]
        grid["cost"][live, i] = res.cost
        grid["turnover"][live, i] = res.turnover
        w[live] = res.w_next
        t[live] += 1
    return [{key: grid[key][j, : steps[j]] for key in keys} for j in range(n)]
