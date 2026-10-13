"""Crypto cash-and-carry as one tradable bar frame: long spot, short perpetual.

``basis_frame`` joins a spot frame and a perp frame of the same symbol and bar, and returns a
frame of kind ``"perp"`` with symbol ``f"{symbol}-BASIS"``. One unit of the frame is long one
unit of spot and short one unit of perp. The close starts at 100 on the first common bar and
compounds the leg spread of close-to-close simple returns:

    close_t = close_{t-1} * (1 + r_spot_t - r_perp_t)

Each bar opens at the previous close. High and low bound the open and close only, because the
intrabar spread extremes are not in bar data. Volume is the smaller of the two legs' volumes.
Returns run between consecutive common bars, so a bar missing from one leg widens the return
that spans it.

``funding_rate`` is the negated perp funding rate. The short perp leg receives funding when the
rate is positive, and a perp frame charges a positive rate to its long side.

Approximations: the frame assumes equal leg notionals at every bar open. Between rebalances the
notionals drift apart, and the P&L of that drift is second order in the leg returns; the frame
ignores it. Intrabar liquidation of the perp leg is not modelled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sactrader.data.schema import make_frame


def basis_frame(spot: pd.DataFrame, perp: pd.DataFrame) -> pd.DataFrame:
    """Cash-and-carry bar frame from a spot frame and a perp frame. See the module docstring.

    Raises ``ValueError`` when the frames are not a spot frame then a perp frame, when their
    symbols or bars differ, or when fewer than two bars are common to both.
    """
    if spot.attrs.get("kind") != "spot" or perp.attrs.get("kind") != "perp":
        raise ValueError("basis_frame takes a spot frame, then a perp frame")
    symbol, bar = spot.attrs["symbol"], spot.attrs["bar"]
    if perp.attrs.get("symbol") != symbol or perp.attrs.get("bar") != bar:
        raise ValueError(
            f"legs differ: spot {symbol!r} {bar!r}, perp "
            f"{perp.attrs.get('symbol')!r} {perp.attrs.get('bar')!r}"
        )
    return spread_frame(spot, perp, f"{symbol}-BASIS")


def spread_frame(long_leg: pd.DataFrame, short_leg: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """One unit long ``long_leg`` and one unit short ``short_leg`` as a perp-kind frame.

    The legs may be any linear frames with the same bar, for example two venues' perpetuals on
    one coin. ``funding_rate`` is the long leg's funding minus the short leg's (a leg without a
    ``funding_rate`` column pays none), so a positive value is paid by a long spread position.
    The approximations of the module docstring apply.
    """
    bar = long_leg.attrs["bar"]
    if short_leg.attrs.get("bar") != bar:
        raise ValueError(f"legs have different bars: {bar!r} and {short_leg.attrs.get('bar')!r}")
    common = long_leg.index.intersection(short_leg.index)
    if len(common) < 2:
        raise ValueError(f"legs share {len(common)} bars, need at least 2")

    a, b = long_leg.loc[common], short_leg.loc[common]
    a_close = a["close"].to_numpy(dtype=np.float64)
    b_close = b["close"].to_numpy(dtype=np.float64)
    spread = np.diff(a_close) / a_close[:-1] - np.diff(b_close) / b_close[:-1]
    close = 100.0 * np.concatenate(([1.0], np.cumprod(1.0 + spread)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    volume = np.minimum(a["volume"].to_numpy(dtype=np.float64),
                        b["volume"].to_numpy(dtype=np.float64))
    zero = np.zeros(len(common))
    funding = (a["funding_rate"].to_numpy(dtype=np.float64) if "funding_rate" in a else zero) - (
        b["funding_rate"].to_numpy(dtype=np.float64) if "funding_rate" in b else zero)
    frame = make_frame(
        common, open_, np.maximum(open_, close), np.minimum(open_, close), close, volume,
        symbol=symbol, kind="perp", bar=bar, funding_rate=funding,
    )
    # The perp premium index leads funding; it nets across the legs with funding's sign.
    if "premium" in a or "premium" in b:
        frame["premium"] = (a["premium"].to_numpy(dtype=np.float64) if "premium" in a else zero) - (
            b["premium"].to_numpy(dtype=np.float64) if "premium" in b else zero)
    # Shorting the spread shorts the long leg; a spot leg must then be borrowed.
    frame.attrs["short_borrow"] = long_leg.attrs.get("kind") == "spot"
    return frame
