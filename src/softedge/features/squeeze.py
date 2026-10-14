"""Open-interest squeeze gate on the funding prior of a cash-and-carry bar frame.

The gate closes the carry while open interest grows fast. On each daily bar t:

    growth_t = log(OI_t) - log(OI_{t-7})           seven-day growth in USD open interest
    limit_t  = 80th percentile of growth over the 180 daily values ending at t
    gate_t   = 0 if growth_t > limit_t, else 1

``OI_t`` is the last sample at or before day t, held over days without one. The gate is 0
until both growth and a full 180-day distribution exist. The gated prior is the funding prior
times the gate: ``funding_ewm_gated = funding_ewm * gate``.

The thresholds (7 days, 180 days, 0.8) are the pre-registered values of the carry framing.
Row t depends on samples at or before t only.

Gate and prior rows are computed on daily bars, so a frame must be resampled to ``1D`` first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_DAYS = 7
HISTORY_DAYS = 180
QUANTILE = 0.8


def squeeze_gate(oi_value: pd.Series, bar: str) -> pd.Series:
    """Gate per bar (1.0 open, 0.0 closed) from daily USD open interest. See the module docstring."""
    if pd.Timedelta(bar) != pd.Timedelta("1D"):
        raise ValueError(f"the squeeze gate runs on daily bars, got {bar!r}")
    oi = oi_value.astype(np.float64).where(oi_value > 0.0).ffill()
    log_oi = np.log(oi)
    growth = log_oi - log_oi.shift(WINDOW_DAYS)
    limit = growth.rolling(HISTORY_DAYS, min_periods=HISTORY_DAYS).quantile(QUANTILE)
    known = growth.notna() & limit.notna()
    return ((growth <= limit) & known).astype(np.float64)


def gated_funding(funding_ewm: pd.Series, oi_value: pd.Series, bar: str) -> pd.Series:
    """Funding prior times the squeeze gate, on the same index as ``funding_ewm``."""
    return funding_ewm * squeeze_gate(oi_value, bar)
