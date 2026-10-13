"""Reference exposure paths for ``backtest_exposures``.

Each function returns one exposure per tape row. The value at row ``t`` uses rows
``<= t`` only. The tape's ``long_ret`` and related step arrays describe the move
after row ``t``, so they never feed a signal here. Trend and volatility read
``tape.close``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from softedge.tape import MarketTape


def flat(tape: MarketTape) -> np.ndarray:
    return np.zeros(len(tape))


def buy_and_hold(tape: MarketTape, leverage: float = 1.0) -> np.ndarray:
    return np.full(len(tape), float(leverage))


def vol_target_trend(
    tape: MarketTape,
    lookback: int = 24 * 30,
    vol_span: int = 24 * 7,
    target_vol: float = 0.4,
) -> np.ndarray:
    """Sign of the trailing ``lookback`` log return, scaled to ``target_vol``.

    The scale is ``target_vol`` over the annualized EWMA volatility of simple
    returns. The result is clipped to the tape's leverage limit and is zero for the
    first ``lookback`` rows.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    close = np.asarray(tape.close, dtype=np.float64)
    n = close.size

    ret = np.zeros(n)
    ret[1:] = close[1:] / close[:-1] - 1.0
    ewm_var = pd.Series(ret**2).ewm(span=vol_span).mean().to_numpy()
    ann_vol = np.sqrt(ewm_var * tape.bars_per_year)

    trend = np.zeros(n)
    if lookback < n:
        trend[lookback:] = np.sign(np.log(close[lookback:]) - np.log(close[:-lookback]))

    scale = np.zeros(n)
    np.divide(target_vol, ann_vol, out=scale, where=ann_vol > 0.0)
    exposure = trend * scale
    return np.clip(exposure, -tape.max_leverage, tape.max_leverage)


def random_exposures(tape: MarketTape, seed: int = 0, hold: int = 24) -> np.ndarray:
    """Unit long or short exposure drawn at random for each block of ``hold`` rows."""
    if hold < 1:
        raise ValueError("hold must be at least 1")
    n = len(tape)
    rng = np.random.default_rng(seed)
    sides = rng.choice(np.array([-1.0, 1.0]), size=-(-n // hold))
    exposure = np.repeat(sides, hold)[:n]
    return np.clip(exposure, -tape.max_leverage, tape.max_leverage)
