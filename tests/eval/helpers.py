"""Tape builders shared by the eval tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from softedge.data.schema import make_frame
from softedge.tape import CostSpec, MarketTape, build_tape


def make_tape(
    close,
    *,
    kind: str = "spot",
    costs: CostSpec | None = None,
    bars_per_year: float = 8760.0,
    low=None,
    high=None,
    funding=None,
    resolution: float | None = None,
) -> MarketTape:
    """Hourly tape from closes. Open is the previous close, high and low bracket it."""
    close = np.asarray(close, dtype=np.float64)
    idx = pd.date_range("2024-01-01", periods=close.size, freq="1h", tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * 1.001 if high is None else np.asarray(high, dtype=np.float64)
    lo = np.minimum(open_, close) * 0.999 if low is None else np.asarray(low, dtype=np.float64)
    frame = make_frame(
        idx, open_, hi, lo, close,
        symbol="TEST", kind=kind, bar="1h",
        funding_rate=funding, resolution=resolution,
    )
    features = pd.DataFrame({"x": np.zeros(close.size)}, index=frame.index)
    return build_tape(frame, features, costs, bars_per_year=bars_per_year)
