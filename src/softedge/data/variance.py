"""Variance risk premium on BTC and ETH as a daily bar frame: a short 30-day variance position.

Holding one unit of a frame over day t earns the daily variance-swap carry proxy

    x_t = DVOL_t^2 / 365 - r_t^2

per unit of equity. DVOL_t is the Deribit DVOL candle that ended at or before the open of day t,
annualized. r_t is the log return of day t, from its open to the next day's open. The proxy
omits gamma path, hedging, and option bid and ask, so only differences between strategies on
these frames mean anything.

The frame's close on day t is 100 times the product of (1 + x_k) for k before t, so it is known
at the open of day t. Row t also carries ``dvol_open`` (annualized DVOL at that open) and
``rv30``, the root of 365 times the mean squared daily log return over the 30 days before t.
Both are known at the open of day t. ``compute_features`` turns them into ``vrp``, the
annualized variance premium ``dvol_open^2 - rv30^2``.

The frame's open, high and low bound the previous close and this close. A position on row t earns
the move from the close of row t to the close of row t + 1, which is x_t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from softedge.data.schema import make_frame

RV_WINDOW = 30
DAYS_PER_YEAR = 365.0
BAR = "1D"


def variance_frame(opens: pd.Series, dvol: pd.Series, symbol: str) -> pd.DataFrame:
    """Daily variance-carry frame of one underlying. See the module docstring.

    ``opens`` holds the underlying's daily open prices indexed by UTC day open. ``dvol`` holds
    DVOL closes in annualized decimal indexed by candle end time. Days before the first DVOL
    candle are dropped, since their payoff is undefined. Raises ``ValueError`` when fewer than
    three days remain or when DVOL is missing on a day that needs it.
    """
    days = opens.index
    px = opens.to_numpy(dtype=np.float64)
    n = len(px)
    if n < 3:
        raise ValueError("need at least 3 daily opens")
    # Log return of each day, open to next open. The last day has no next open.
    sq = np.full(n, np.nan)
    sq[:-1] = np.diff(np.log(px)) ** 2
    # Row t averages the squares of days t-30 to t-1, all of which closed before its open.
    mean_sq = pd.Series(sq).rolling(RV_WINDOW).mean().shift(1).to_numpy()
    rv30 = np.sqrt(DAYS_PER_YEAR * mean_sq)
    dvol_open = _dvol_at(dvol, days)
    known = np.flatnonzero(np.isfinite(dvol_open))
    if len(known) == 0:
        raise ValueError("no DVOL candle ends at or before the first open")
    first = int(known[0])
    last = n - 1
    if not np.isfinite(dvol_open[first:last]).all():
        raise ValueError("DVOL is missing on a day inside the frame")
    payoff = dvol_open[first:last] ** 2 / DAYS_PER_YEAR - sq[first:last]
    close = 100.0 * np.concatenate(([1.0], np.cumprod(1.0 + payoff)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    frame = make_frame(
        days[first:], open_, np.maximum(open_, close), np.minimum(open_, close), close,
        symbol=f"{symbol}-VRP", kind="spot", bar=BAR,
    )
    frame["dvol_open"] = dvol_open[first:]
    frame["rv30"] = rv30[first:]
    return frame


def vrp_frames(start: str = "2021-03", end: str = "2026-09",
               symbols: tuple[str, ...] = ("BTC", "ETH")) -> list[pd.DataFrame]:
    """One variance-carry frame per coin, from Binance spot opens and Deribit DVOL."""
    from softedge.data import binance, deribit

    frames = []
    for coin in symbols:
        hourly = binance.load_spot(f"{coin.upper()}USDT", "1h", start, end)
        opens = hourly["open"].resample(BAR).first().dropna()
        dvol = deribit.load_dvol(coin, start, end, "1h")
        frames.append(variance_frame(opens, dvol, coin.upper()))
    return frames


def _dvol_at(dvol: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    """DVOL of the last candle that ended at or before each time; NaN before the first candle."""
    ends = dvol.index.as_unit("ns").asi8
    pos = np.searchsorted(ends, days.as_unit("ns").asi8, side="right") - 1
    padded = np.concatenate(([np.nan], dvol.to_numpy(dtype=np.float64)))
    return padded[pos + 1]
