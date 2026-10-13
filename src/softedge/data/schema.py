"""Standard bar frame shared by every data source.

A bar frame is a pandas DataFrame with:

* index: tz-aware UTC ``DatetimeIndex`` named ``time``, strictly increasing, no
  duplicates. The timestamp is the bar *open* time; the bar covers
  ``[time, time + bar)``.
* columns ``open, high, low, close`` (float64, > 0) and ``volume`` (float64, >= 0
  or NaN when the source has none).
* perps only: ``funding_rate`` (float64). The rate charged to a long position at a
  funding event that falls inside the bar, 0.0 for bars without one. A positive
  rate means longs pay shorts.
* fx, optional: ``rate_diff`` (float64), annualized short-rate differential base minus quote
  known at the bar close. Longs earn it, shorts pay it.
* binary only: prices are the YES token price in ``(0, 1)``. ``df.attrs["resolution"]``
  holds the settlement value of the YES token (1.0 or 0.0).

``df.attrs`` carries ``symbol`` (str), ``kind`` (one of ``KINDS``) and ``bar``
(pandas offset alias such as ``"1h"`` or ``"1D"``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KINDS = ("spot", "perp", "fx", "binary")
PRICE_COLUMNS = ("open", "high", "low", "close")
BASE_COLUMNS = (*PRICE_COLUMNS, "volume")


class FrameError(ValueError):
    """Raised when a frame breaks the bar frame contract."""


def make_frame(
    index: pd.DatetimeIndex,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray | None = None,
    *,
    symbol: str,
    kind: str,
    bar: str,
    funding_rate: np.ndarray | None = None,
    resolution: float | None = None,
) -> pd.DataFrame:
    """Build a bar frame from arrays and validate it."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    idx.name = "time"
    n = len(idx)
    vol = np.full(n, np.nan) if volume is None else np.asarray(volume, dtype=np.float64)
    df = pd.DataFrame(
        {
            "open": np.asarray(open_, dtype=np.float64),
            "high": np.asarray(high, dtype=np.float64),
            "low": np.asarray(low, dtype=np.float64),
            "close": np.asarray(close, dtype=np.float64),
            "volume": vol,
        },
        index=idx,
    )
    if funding_rate is not None:
        df["funding_rate"] = np.asarray(funding_rate, dtype=np.float64)
    df.attrs.update(symbol=symbol, kind=kind, bar=bar)
    if resolution is not None:
        df.attrs["resolution"] = float(resolution)
    validate_frame(df)
    return df


def validate_frame(df: pd.DataFrame) -> None:
    """Raise ``FrameError`` if ``df`` breaks the bar frame contract."""
    kind = df.attrs.get("kind")
    if kind not in KINDS:
        raise FrameError(f"attrs['kind'] must be one of {KINDS}, got {kind!r}")
    for key in ("symbol", "bar"):
        if not isinstance(df.attrs.get(key), str):
            raise FrameError(f"attrs[{key!r}] must be a str")
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or str(idx.tz) != "UTC":
        raise FrameError("index must be a tz-aware UTC DatetimeIndex")
    if len(idx) < 2:
        raise FrameError("frame needs at least 2 bars")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise FrameError("index must be strictly increasing")
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise FrameError(f"missing columns {missing}")
    prices = df[list(PRICE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise FrameError("prices must be finite and > 0")
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    oc_max = np.maximum(df["open"].to_numpy(), df["close"].to_numpy())
    oc_min = np.minimum(df["open"].to_numpy(), df["close"].to_numpy())
    tol = 1e-9 * np.abs(oc_max)
    if (hi < oc_max - tol).any() or (lo > oc_min + tol).any():
        raise FrameError("high/low must bound open/close")
    vol = df["volume"].to_numpy(dtype=np.float64)
    if (vol[np.isfinite(vol)] < 0).any():
        raise FrameError("volume must be >= 0")
    if kind == "perp":
        if "funding_rate" not in df.columns:
            raise FrameError("perp frames need a funding_rate column")
        if not np.isfinite(df["funding_rate"].to_numpy(dtype=np.float64)).all():
            raise FrameError("funding_rate must be finite (use 0.0 for bars without funding)")
    if kind == "binary":
        if (prices >= 1).any():
            raise FrameError("binary prices must lie in (0, 1)")
        if df.attrs.get("resolution") not in (0.0, 1.0):
            raise FrameError("binary frames need attrs['resolution'] in {0.0, 1.0}")
