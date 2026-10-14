"""Pegged spot pairs: stablecoin and liquid-staking pairs and their deviation from a peg.

A pegged pair should trade near a reference value. A stablecoin pair (``USDCUSDT``) has the
peg 1.0. A liquid-staking token priced in its base asset (``WBETHETH``, ``BNSOLSOL``) drifts up
slowly as its redemption ratio rises, so its reference is the trailing 30-day median of its own
market price. The archive does not hold the redemption ratio, so the median stands in for it.
Both kinds stay spot frames of kind ``"spot"``, and ``attrs["peg"]`` names the reference: a float
for a fixed peg, or ``TRAILING`` for the trailing median.

``peg_deviation`` returns ``log(price / peg_t)`` per bar. Row ``t`` reads bars ``<= t`` only. The
feature ``peg_dev`` is the one price feature that is not scale-free: a fixed peg is an absolute
price, so multiplying every price by a constant changes it.

The archive lacks some monthly files, and the binance loader bridges them with one return across
the gap. The loader cuts each series at every gap longer than ``GAP_BARS`` bars. A series with
gaps yields one frame per piece, named ``<SYMBOL>-<n>`` by the piece's position in the series;
pieces shorter than ``PIECE_MIN`` are dropped. A series without gaps keeps its symbol.

Costs and borrowing come from the config, not from this module. Binance's zero-fee promotions on
stablecoin pairs are not modelled; the configs charge the normal spot fee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAILING = "trailing_median"
PEG_WINDOW = "30D"
PEG_MIN_HISTORY = pd.Timedelta("7D")
GAP_BARS = 2
PIECE_MIN = pd.Timedelta("30D")
STABLECOIN_PAIRS = ("USDCUSDT", "FDUSDUSDT", "TUSDUSDT")
STAKING_PAIRS = ("WBETHETH", "BNSOLSOL")
# Symbol -> reference. DAIUSDT is absent from the Binance spot archive and is not listed.
PEG_RULES: dict[str, float | str] = {
    **{s: 1.0 for s in STABLECOIN_PAIRS},
    **{s: TRAILING for s in STAKING_PAIRS},
}


def attach_peg(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of a spot frame with ``attrs["peg"]`` set from ``PEG_RULES`` for its symbol."""
    symbol = frame.attrs["symbol"]
    if symbol not in PEG_RULES:
        raise ValueError(f"{symbol!r} is not a pegged pair; known: {sorted(PEG_RULES)}")
    out = frame.copy()
    out.attrs = {**frame.attrs, "peg": PEG_RULES[symbol]}
    return out


def load_peg_frames(start: str = "2020-01", end: str = "2026-09",
                    symbols=None) -> list[pd.DataFrame]:
    """Hourly spot frames of the pegged pairs, each with its peg attached, cut at archive gaps."""
    from softedge.data import binance

    names = tuple(symbols) if symbols is not None else (*STABLECOIN_PAIRS, *STAKING_PAIRS)
    out = []
    for s in names:
        out.extend(_pieces(attach_peg(binance.load_spot(s, "1h", start, end))))
    return out


def _pieces(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Cut a frame at each gap longer than ``GAP_BARS`` bars and drop pieces shorter than
    ``PIECE_MIN``. A frame without a gap comes back as a one-item list.

    Each piece is its own frame, so its trailing median starts at its own first bar."""
    step = pd.Timedelta(frame.attrs["bar"])
    breaks = np.flatnonzero((frame.index.to_series().diff() > GAP_BARS * step).to_numpy())
    if breaks.size == 0:
        return [frame]
    bounds = [0, *breaks.tolist(), len(frame)]
    pieces = []
    for n, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True), start=1):
        part = frame.iloc[a:b]
        if part.index[-1] - part.index[0] + step < PIECE_MIN:
            continue
        piece = part.copy()
        piece.attrs = {**frame.attrs, "symbol": f"{frame.attrs['symbol']}-{n}"}
        pieces.append(piece)
    return pieces


def peg_deviation(frame: pd.DataFrame) -> pd.Series:
    """``log(close / peg_t)`` per bar. ``peg_t`` is the frame's ``attrs["peg"]``.

    For ``TRAILING`` the peg is the median close over the trailing ``PEG_WINDOW``, the current
    bar included. It is NaN during the first ``PEG_MIN_HISTORY`` of the frame.
    """
    peg = frame.attrs["peg"]
    close = frame["close"].astype("float64")
    if peg == TRAILING:
        ref = close.rolling(PEG_WINDOW, min_periods=1).median()
        young = np.asarray(frame.index - frame.index[0] < PEG_MIN_HISTORY)
        ref = ref.mask(young)
    elif isinstance(peg, (int, float)) and not isinstance(peg, bool):
        ref = pd.Series(float(peg), index=frame.index)
    else:
        raise ValueError(f"unknown peg {peg!r}")
    return pd.Series(np.log(close.to_numpy() / ref.to_numpy()), index=frame.index, name="peg_dev")
