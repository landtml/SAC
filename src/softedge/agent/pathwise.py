"""Pathwise actor objective: short differentiable rollouts along the recorded market path.

A small trader does not move prices, so along a recorded tape the next H rewards are exact,
differentiable functions of the actions: exposure drift, trading costs, carry and liquidation are
all known once the market path is fixed. From a replayed state (tape, row, exposure, cost
multiplier) the actor maximizes

    J = sum_{h<H} gamma^h (r_h - alpha log pi(a_h | s_h)) + gamma^H V(s_H),
    V(s) = Q(s, a) - alpha log pi(a | s),  a ~ pi(. | s)

with gradients flowing through the portfolio dynamics into every action, as in short-horizon
actor-critic (SHAC, Xu et al., 2022) and direct reinforcement for trading (Moody and Saffell,
2001). The critic only covers what lies beyond H steps, so less of the policy gradient rests on a
value estimate fit to noisy returns. H = 0 is the usual SAC actor objective.

A rollout that reaches a binary market's settlement stops there; one that reaches the last row
of a tape it cannot step from bootstraps from the critic at that row.
"""

from __future__ import annotations

import numpy as np
import torch

from softedge.env.vec_env import TapeSet
from softedge.portfolio import MIN_EQUITY

_FIELDS = ("long_ret", "short_ret", "short_growth", "long_worst", "short_worst",
           "carry_long", "carry_short", "fin_long", "cost_long", "cost_short")


def step_portfolio_torch(w_prev: torch.Tensor, target: torch.Tensor, m: dict[str, torch.Tensor],
                         impact: torch.Tensor, maint: torch.Tensor):
    """Torch twin of ``softedge.portfolio.step_portfolio``: (simple return, next exposure)."""
    lp0, sp0 = w_prev.clamp_min(0.0), (-w_prev).clamp_min(0.0)
    lp, sp = target.clamp_min(0.0), (-target).clamp_min(0.0)
    dl, ds = (lp - lp0).abs(), (sp - sp0).abs()
    turnover = dl + ds
    # The impact term's gradient is infinite at zero turnover, so it is only formed when used.
    imp = torch.where(impact > 0, impact * (turnover + 1e-12) ** 1.5, torch.zeros_like(turnover))
    cost = m["cost_long"] * dl + m["cost_short"] * ds + imp
    carry = lp * m["carry_long"] + sp * m["carry_short"] + m["fin_long"] * (lp - 1.0).clamp_min(0.0)
    ret = lp * m["long_ret"] + sp * m["short_ret"] - carry - cost
    gross = lp + sp
    worst = lp * m["long_worst"] + sp * m["short_worst"] - carry - cost
    liquidated = (gross > 0) & (1.0 + worst <= maint * gross)
    ret = torch.where(liquidated, (worst - maint * gross).clamp_min(-1.0), ret).clamp_min(-1.0)
    equity = 1.0 + ret
    alive = equity > MIN_EQUITY
    safe = torch.where(alive, equity, torch.ones_like(equity))
    w_next = (lp * (1.0 + m["long_ret"]) - sp * m["short_growth"]) / safe
    w_next = torch.where(liquidated | ~alive, torch.zeros_like(w_next), w_next)
    return ret, w_next


class PathwiseObjective:
    """Builds the H-step pathwise value of the current policy for a replayed batch."""

    def __init__(self, tapes: TapeSet, horizon: int, gamma: float, reward_scale: float):
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        self.ts = tapes
        self.horizon = horizon
        self.gamma = gamma
        self.reward_scale = reward_scale
        f32 = torch.float32
        self.step = {f: torch.as_tensor(tapes.step[f], dtype=f32) for f in _FIELDS}
        self.prior = torch.as_tensor(tapes.prior, dtype=f32)
        self.scale = torch.as_tensor(tapes.scale, dtype=f32)
        self.residual = tapes.action_map.residual if tapes.action_map is not None else 1.0
        self.lev = torch.as_tensor(tapes.max_leverage, dtype=f32)
        self.impact = torch.as_tensor(tapes.impact, dtype=f32)
        self.maint = torch.as_tensor(tapes.maint_margin, dtype=f32)
        # Column of the observation that holds exposure / max leverage.
        self.w_col = tapes.window * len(tapes.feature_names)

    def _target(self, g: torch.Tensor, k: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        lev = self.lev[k]
        if self.ts.action_map is None:
            return a * lev
        raw = self.scale[g] * (self.prior[g] + self.residual * a)
        return torch.maximum(torch.minimum(raw, lev), -lev)

    def value(self, agent, k: np.ndarray, t: np.ndarray, w0: np.ndarray, cost_mult: np.ndarray,
              alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample pathwise soft value J and the log-probability of the first action."""
        ts = self.ts
        n = len(k)
        k_t = torch.as_tensor(k, dtype=torch.int64)
        lev = self.lev[k_t]
        cm = torch.as_tensor(cost_mult, dtype=torch.float32)
        w = torch.as_tensor(w0, dtype=torch.float32)
        last_step = ts.last_step_row[k]
        settle_row = np.where(ts.terminal_at_end[k], ts.lengths[k] - 1, -1)
        alive = torch.ones(n, dtype=torch.bool)
        total = torch.zeros(n)
        logp0 = None
        disc = 1.0
        for h in range(self.horizon + 1):
            row = np.minimum(t + h, ts.lengths[k] - 1)
            base = torch.as_tensor(ts.observe(k, row, np.zeros(n), cost_mult))
            onehot = torch.zeros(base.shape[1])
            onehot[self.w_col] = 1.0
            obs = agent.obs_norm(base + (w / lev)[:, None] * onehot)
            a, logp = agent.actor(obs)
            if logp0 is None:
                logp0 = logp
            stop = alive & torch.as_tensor(t + h > last_step) if h < self.horizon else alive
            if stop.any():
                v = agent.q_value(obs, a) - alpha * logp
                total = total + disc * torch.where(stop, v, torch.zeros_like(v))
                alive = alive & ~stop
            if h == self.horizon or not alive.any():
                break
            g_np = ts.offsets[k] + row
            g = torch.as_tensor(g_np, dtype=torch.int64)
            m = {f: arr[g] for f, arr in self.step.items()}
            m["cost_long"] = m["cost_long"] * cm
            m["cost_short"] = m["cost_short"] * cm
            ret, w_next = step_portfolio_torch(w, self._target(g, k_t, a[:, 0]), m,
                                               self.impact[k_t], self.maint[k_t])
            r = torch.log((1.0 + ret).clamp_min(MIN_EQUITY)) * self.reward_scale
            total = total + disc * torch.where(alive, r - alpha * logp, torch.zeros_like(r))
            alive = alive & ~torch.as_tensor(row == settle_row)
            w = torch.where(alive, w_next, torch.zeros_like(w_next))
            disc *= self.gamma
        return total, logp0
