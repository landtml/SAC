"""Causal premium features: time-series momentum, carry confirmed by trend, Treasury curve slope.

``premia_features`` returns the columns below for a linear bar frame (spot, perp, fx). Binary
frames get no columns. Every column is causal: row ``t`` reads only bars ``<= t`` and the
Treasury yields stamped at or before bar ``t``.

* ``tsmom_12m``: sign of the log return from bar ``t - 21`` to bar ``t - 252``, the 12-month
  return that skips the last month, times ``TSMOM_TARGET / sigma_t``. ``sigma_t`` is the EWMA
  standard deviation of one-bar log returns, span 32 as in ``causal.py``. ``TSMOM_TARGET`` is
  the 10% annual volatility target of the configs, per daily bar. The lookbacks are
  in bars, so the feature assumes daily bars. Scaling prices by a constant leaves it unchanged.
* ``carry_trend``: the rate-differential feature of ``causal.py`` (``rate_diff`` times 20),
  kept only when the 63-bar log return has the same sign as the differential, else zero.
  Present only on frames with a ``rate_diff`` column.
* ``curve_slope``: the 10-year minus 2-year Treasury yield in percentage points, on the
  ``UST10`` and ``UST2`` frames of ``macro``. Every other frame holds zeros.

Leading rows without a full lookback hold zeros, which is no prior. Every column is clipped
to ``[-5, 5]`` and stored as float32 on the frame index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from softedge.data.fred_markets import _observations
from softedge.data.schema import validate_frame

TSMOM_LOOKBACK = 252
TSMOM_SKIP = 21
# The 10% annual volatility target of the configs, per daily bar.
TSMOM_TARGET = 0.10 / np.sqrt(252.0)
CARRY_WINDOW = 63
CARRY_SCALE = 20.0  # the rate-differential scaling of causal.py: 5% maps to 1.0
TREASURY_SYMBOLS = ("UST10", "UST2")
_VOL_SPAN = 32
_FLOOR = 1e-8
_CLIP = 5.0


def premia_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Premium feature table for ``frame``, one row per bar."""
    validate_frame(frame)
    if frame.attrs["kind"] == "binary":
        return pd.DataFrame(index=frame.index)
    log_close = np.log(frame["close"].astype("float64"))
    cols: dict[str, pd.Series | np.ndarray] = {"tsmom_12m": _tsmom(log_close)}
    if "rate_diff" in frame.columns:
        carry = frame["rate_diff"].astype("float64") * CARRY_SCALE
        cols["carry_trend"] = _carry_trend(log_close, carry)
    if frame.attrs["symbol"] in TREASURY_SYMBOLS:
        cols["curve_slope"] = slope_on_index(frame.index, *_treasury_yields())
    else:
        cols["curve_slope"] = np.zeros(len(frame))
    out = pd.DataFrame(cols, index=frame.index)
    return out.fillna(0.0).clip(-_CLIP, _CLIP).astype(np.float32)


def slope_on_index(index: pd.DatetimeIndex, ten: pd.Series, two: pd.Series) -> np.ndarray:
    """10-year minus 2-year yield in percentage points, each bar reading the last value of each
    tenor stamped at or before it. NaN before both tenors have a value."""
    stamps = ten.index.union(two.index).sort_values()
    yields = pd.DataFrame({"ten": ten, "two": two}, index=stamps).ffill()
    aligned = yields.reindex(pd.DatetimeIndex(index), method="ffill")
    return (aligned["ten"] - aligned["two"]).to_numpy(np.float64)


def _tsmom(log_close: pd.Series) -> pd.Series:
    ret = log_close.diff()
    sigma = np.sqrt((ret * ret).ewm(span=_VOL_SPAN, adjust=False).mean()).clip(lower=_FLOOR)
    lookback = log_close.shift(TSMOM_SKIP) - log_close.shift(TSMOM_LOOKBACK)
    return np.sign(lookback) * (TSMOM_TARGET / sigma)


def _carry_trend(log_close: pd.Series, carry: pd.Series) -> pd.Series:
    trend = log_close - log_close.shift(CARRY_WINDOW)
    agree = np.sign(trend) == np.sign(carry)
    return carry.where(agree, 0.0)


def _treasury_yields() -> tuple[pd.Series, pd.Series]:
    return _fred_yield("DGS10"), _fred_yield("DGS2")


def _fred_yield(series_id: str) -> pd.Series:
    return _observations(series_id)
