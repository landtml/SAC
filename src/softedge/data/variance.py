"""Variance risk premium on BTC and ETH as a daily bar frame: a short variance carry rolled daily.

Holding one unit of a frame over day t earns the one-day variance-swap carry proxy

    x_t = DVOL_t^2 / 365 - r_t^2

per unit of equity. DVOL_t is the Deribit DVOL candle that ended at or before the open of day t,
annualized, and no more than one day before it. DVOL is a 30-day implied volatility index, so
the daily swap is priced from the 30-day rate. r_t is the log return of day t, from its open to
the next day's open. Each day rolls a fresh one-day swap; no 30-day position is held. The proxy
omits gamma path, hedging, option bid and ask, and vega. A short 30-day variance book loses when
implied vol rises, and this proxy charges none of that loss, so results on these frames overstate
a real 30-day position in volatility spikes. Only differences between strategies on these frames
mean anything.

Timing exception: row t is stamped at the open of day t, not at its bar close as
``softedge.data.schema`` requires. Its close is the index value at that open, and its open is the
previous row's close, so the bar spans the open of day t-1 to the open of day t. Code that gates
by bar close, such as ``deribit.attach_implied_vol`` and ``premium.attach_premium``, is one day
off on this frame and must not be applied to it.

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
DAY_NS = 86_400_000_000_000


def variance_frame(opens: pd.Series, dvol: pd.Series, symbol: str) -> pd.DataFrame:
    """Daily variance-carry frame of one underlying. See the module docstring.

    ``opens`` holds the underlying's daily open prices indexed by UTC day open, on consecutive
    days; ``vrp_frames`` splits at gaps first. ``dvol`` holds DVOL closes in annualized decimal
    indexed by candle end time. Days before the first DVOL candle are dropped, since their payoff
    is undefined. A candle that ended more than one day before an open counts as missing. Raises ``ValueError`` when the days are not consecutive, when fewer than three
    daily opens are given, or when DVOL is missing on a day that needs it.
    """
    days = opens.index
    px = opens.to_numpy(dtype=np.float64)
    n = len(px)
    if n < 3:
        raise ValueError("need at least 3 daily opens")
    if (np.diff(days.as_unit("ns").asi8) != DAY_NS).any():
        raise ValueError("daily opens must be consecutive days; split at gaps first")
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
    """Variance-carry frames per coin, from Binance spot opens and Deribit DVOL.

    A missing day in the opens starts a new frame, so each frame covers consecutive days. A coin
    with more than one run gets frames labeled ``<COIN>-VRP-1``, ``<COIN>-VRP-2`` and so on.
    """
    from softedge.data import binance, deribit

    frames = []
    for coin in symbols:
        hourly = binance.load_spot(f"{coin.upper()}USDT", "1h", start, end)
        opens = hourly["open"].resample(BAR).first().dropna()
        dvol = deribit.load_dvol(coin, start, end, "1h")
        runs = _consecutive_runs(opens)
        for k, run in enumerate(runs, start=1):
            fr = variance_frame(run, dvol, coin.upper())
            if len(runs) > 1:
                fr.attrs["symbol"] = f"{coin.upper()}-VRP-{k}"
            frames.append(fr)
    return frames


def _consecutive_runs(opens: pd.Series) -> list[pd.Series]:
    """Runs of daily opens on consecutive days. Runs shorter than three days carry no payoff."""
    ns = opens.index.as_unit("ns").asi8
    cuts = [0, *(np.flatnonzero(np.diff(ns) != DAY_NS) + 1), len(opens)]
    runs = [opens.iloc[a:b] for a, b in zip(cuts[:-1], cuts[1:], strict=True)]
    return [run for run in runs if len(run) >= 3]


def _dvol_at(dvol: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    """DVOL of the last candle that ended at or before each time, NaN when that candle is more
    than one day old or when none exists."""
    ends = dvol.index.as_unit("ns").asi8
    times = days.as_unit("ns").asi8
    if len(ends) == 0:
        return np.full(len(times), np.nan)
    pos = np.searchsorted(ends, times, side="right") - 1
    padded = np.concatenate(([np.nan], dvol.to_numpy(dtype=np.float64)))
    value = padded[pos + 1]
    age = times - ends[np.clip(pos, 0, None)]
    return np.where((pos < 0) | (age > DAY_NS), np.nan, value)
