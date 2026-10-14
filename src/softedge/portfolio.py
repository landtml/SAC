"""Vectorized single-step portfolio accounting over market tapes.

The same function serves the environment and counterfactual replay: the market
part of a transition is exogenous, so the return of any (previous exposure,
target exposure) pair at row ``t`` is a pure function of the tape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Equity below this fraction after a step counts as a wipe-out.
MIN_EQUITY = 1e-3


@dataclass
class StepResult:
    ret: np.ndarray  # simple return of equity over the step
    w_next: np.ndarray  # exposure after the price move, before the next trade
    cost: np.ndarray  # trading cost paid (fraction of equity)
    carry: np.ndarray  # holding cost paid (fraction of equity)
    turnover: np.ndarray  # |change| in long plus short exposure
    liquidated: np.ndarray  # bool


def step_portfolio(
    w_prev: np.ndarray,
    target: np.ndarray,
    long_ret: np.ndarray,
    short_ret: np.ndarray,
    short_growth: np.ndarray,
    long_worst: np.ndarray,
    short_worst: np.ndarray,
    carry_long: np.ndarray,
    carry_short: np.ndarray,
    cost_long: np.ndarray,
    cost_short: np.ndarray,
    impact: np.ndarray | float,
    maint_margin: np.ndarray | float,
    fin_long: np.ndarray | float = 0.0,
) -> StepResult:
    """Trade from ``w_prev`` to ``target`` at row ``t`` and hold to ``t + 1``.

    Every argument is an array broadcastable to the batch shape. ``impact`` scales
    a ``turnover ** 1.5`` term (square-root market impact times traded size).
    A position whose worst intra-bar equity falls to its maintenance margin is
    liquidated: it loses that margin as well and ends the step flat.
    """
    lp0 = np.maximum(w_prev, 0.0)
    sp0 = np.maximum(-w_prev, 0.0)
    lp = np.maximum(target, 0.0)
    sp = np.maximum(-target, 0.0)
    dl = np.abs(lp - lp0)
    ds = np.abs(sp - sp0)
    turnover = dl + ds
    cost = cost_long * dl + cost_short * ds + impact * turnover**1.5
    # Cash financing on long exposure above 1x; a leveraged spot long borrows the excess.
    carry = lp * carry_long + sp * carry_short + fin_long * np.maximum(lp - 1.0, 0.0)
    ret = lp * long_ret + sp * short_ret - carry - cost

    gross = lp + sp
    worst = lp * long_worst + sp * short_worst - carry - cost
    liquidated = (gross > 0) & (1.0 + worst <= maint_margin * gross)
    liq_ret = np.maximum(worst - maint_margin * gross, -1.0)
    ret = np.where(liquidated, liq_ret, ret)
    ret = np.maximum(ret, -1.0)

    equity = 1.0 + ret
    alive = equity > MIN_EQUITY
    safe_equity = np.where(alive, equity, 1.0)
    w_next = (lp * (1.0 + long_ret) - sp * short_growth) / safe_equity
    w_next = np.where(liquidated | ~alive, 0.0, w_next)
    return StepResult(ret=ret, w_next=w_next, cost=cost, carry=carry,
                      turnover=turnover, liquidated=liquidated)


def log_reward(ret: np.ndarray) -> np.ndarray:
    """Log growth of equity, floored so a wipe-out stays finite."""
    return np.log(np.maximum(1.0 + ret, MIN_EQUITY))
