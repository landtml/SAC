"""Variance risk premium on the S&P 500 as a tradable short-variance index.

``load_vrp_frame`` returns a daily bar frame (symbol ``SPXVRP``) whose close is the cumulative
P&L of a short variance position, in variance units per unit of notional. Row ``t`` moves to
row ``t + 1`` by

    g_{t+1} = (iv_t - rv_{t+1}) / 252

where ``iv_t = (VIXCLS_t / 100) ** 2`` is the implied variance known at the close of day ``t``
and ``rv_t = 252 * mean`` of the 21 squared daily log returns of the S&P 500 (FRED ``SP500``)
ending on day ``t``. As for every spot frame, a position decided at row ``t`` earns the move
from its close to the next close.

With ``prior=True`` the frame also carries ``vrp_prior = iv_t - rv_t``, zeroed while the SPXVRP
index is below its trailing 63-bar high and during the 21-day warm-up. The index is the premium
itself: the P&L of an always-on short-variance position, with no policy in it. The gate is a
drawdown gate on that premium, not on the strategy's P&L. A positive ``prior_gain`` in a config (the shipped configs use 50)
sets the position to clip(c * vrp_prior, -1, 1). Positive exposure holds the index long, which
is short variance, so the position is long when implied exceeds realized.

FRED's ``SP500`` series starts in October 2016, so the sample is about ten years. The payoff
is a proxy built from VIX and trailing 21-day realized variance, not from variance-swap quotes.
Costs come from each config and are placeholders (see configs/README.md). Roll is not modelled. The 21-day window smooths crash days and the
daily close-to-close variance understates intraday variance, so tail losses are understated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from softedge.data.fred_markets import load_fred_market
from softedge.data.schema import make_frame
from softedge.data.vol_indices import attach_fred_implied_vol

SYMBOL = "SPXVRP"
TRADING_DAYS = 252
RV_WINDOW = 21
GATE_WINDOW = 63


def realized_variance(close: np.ndarray) -> np.ndarray:
    """252 times the mean of the 21 squared log returns ending at each close.

    The first value sits at row 21, the first row with 21 returns.
    """
    log_ret = np.diff(np.log(close), prepend=np.nan)
    sq = pd.Series(log_ret * log_ret)
    return TRADING_DAYS * sq.rolling(RV_WINDOW, min_periods=RV_WINDOW).mean().to_numpy()


def short_variance_closes(iv: np.ndarray, rv: np.ndarray) -> np.ndarray:
    """Index closes from 100 at row 0: row ``t`` times ``1 + (iv[t-1] - rv[t]) / 252``.

    ``iv`` is implied variance. Rows ``1`` to ``RV_WINDOW - 1`` move by zero while the realized
    window fills. Any other missing input raises, so a day's P&L is never dropped silently.
    """
    rows = np.arange(1, len(rv))
    missing = ~np.isfinite(iv[:-1]) | (~np.isfinite(rv[1:]) & (rows >= RV_WINDOW))
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} rows after warm-up lack iv or rv, first at row {int(rows[missing][0])}"
        )
    ret = np.zeros(len(rv))
    ret[1:] = (iv[:-1] - rv[1:]) / TRADING_DAYS
    ret = np.nan_to_num(ret, nan=0.0)
    return 100.0 * np.cumprod(1.0 + ret)


def _window_end(end: str) -> pd.Timestamp:
    stamp = pd.Timestamp(end, tz="UTC")
    return stamp + pd.offsets.MonthEnd(0) if len(end) == 7 else stamp


def load_vrp_frame(start: str = "2000-01", end: str = "2026-09",
                   prior: bool = False) -> pd.DataFrame:
    """Daily short-variance bars from ``start`` to ``end``, both inclusive.

    The index and the prior are built on the full FRED history, so the warm-up of the realized
    variance comes from before ``start``. ``prior=True`` adds the ``vrp_prior`` column.
    """
    spx = load_fred_market("SPX", "1900-01", "2100-01")
    days = spx.index
    close = spx["close"].to_numpy(dtype=np.float64)
    iv = attach_fred_implied_vol(spx)["implied_vol"].to_numpy(dtype=np.float64) ** 2
    rv = realized_variance(close)
    index = short_variance_closes(iv, rv)
    open_ = np.empty_like(index)
    open_[1:] = index[:-1]
    open_[:1] = index[:1]
    high = np.maximum(open_, index)
    low = np.minimum(open_, index)
    raw = iv - rv
    trailing_high = pd.Series(index).rolling(GATE_WINDOW, min_periods=1).max().to_numpy()
    gated = np.where((index >= trailing_high) & np.isfinite(raw), raw, 0.0)

    keep = (days >= pd.Timestamp(start, tz="UTC")) & (days <= _window_end(end))
    out = make_frame(days[keep], open_[keep], high[keep], low[keep], index[keep],
                     symbol=SYMBOL, kind="spot", bar="1D")
    if prior:
        out["vrp_prior"] = gated[keep]
    return out
