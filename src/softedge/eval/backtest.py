"""Backtest a target-exposure policy over a market tape.

Each step calls ``softedge.portfolio.step_portfolio`` on one row of the tape, so
the accounting matches the environment. A policy receives ``(t, w_prev)`` and
returns a target exposure. The target is clipped to the tape's leverage limit
before it trades. Rows ``0 .. T-2`` are steps. Row ``T-1`` is a step too when
``tape.terminal_at_end`` is set, which is how binary settlement is charged.
The run starts flat.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from softedge.portfolio import step_portfolio
from softedge.tape import MarketTape


@dataclass
class BacktestResult:
    returns: np.ndarray  # simple equity return per step
    equity: np.ndarray  # cumulative product of (1 + returns)
    exposure: np.ndarray  # target exposure after clipping, per step
    w_prev: np.ndarray  # exposure before the trade, after the previous move
    cost: np.ndarray  # trading cost paid, fraction of equity
    carry: np.ndarray  # holding cost paid, fraction of equity
    turnover: np.ndarray  # change in long plus short exposure
    liquidated: np.ndarray  # bool, position closed by the margin check


def _n_steps(tape: MarketTape) -> int:
    return len(tape) if tape.terminal_at_end else len(tape) - 1


def backtest(tape: MarketTape, policy: Callable[[int, float], float]) -> BacktestResult:
    """Run ``policy`` over ``tape`` one step at a time."""
    n = _n_steps(tape)
    lim = float(tape.max_leverage)
    returns = np.empty(n)
    exposure = np.empty(n)
    w_prev_hist = np.empty(n)
    cost = np.empty(n)
    carry = np.empty(n)
    turnover = np.empty(n)
    liquidated = np.empty(n, dtype=bool)

    w = 0.0
    for t in range(n):
        target = min(max(float(policy(t, w)), -lim), lim)
        s = slice(t, t + 1)
        step = step_portfolio(
            np.array([w]), np.array([target]),
            tape.long_ret[s], tape.short_ret[s], tape.short_growth[s],
            tape.long_worst[s], tape.short_worst[s],
            tape.carry_long[s], tape.carry_short[s],
            tape.cost_long[s], tape.cost_short[s],
            tape.impact, tape.maint_margin, fin_long=tape.fin_long[s],
        )
        returns[t] = step.ret[0]
        exposure[t] = target
        w_prev_hist[t] = w
        cost[t] = step.cost[0]
        carry[t] = step.carry[0]
        turnover[t] = step.turnover[0]
        liquidated[t] = step.liquidated[0]
        w = float(step.w_next[0])

    return BacktestResult(
        returns=returns,
        equity=np.cumprod(1.0 + returns),
        exposure=exposure,
        w_prev=w_prev_hist,
        cost=cost,
        carry=carry,
        turnover=turnover,
        liquidated=liquidated,
    )


def backtest_exposures(tape: MarketTape, exposures: np.ndarray) -> BacktestResult:
    """Backtest a precomputed exposure path with one value per tape row."""
    exp = np.asarray(exposures, dtype=np.float64)
    if exp.shape != (len(tape),):
        raise ValueError(f"exposures must have shape ({len(tape)},), got {exp.shape}")
    return backtest(tape, lambda t, _w: exp[t])
