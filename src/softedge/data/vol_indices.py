"""Daily implied-volatility indices from FRED, as a causal series.

``load_vol_index`` returns one FRED index as a decimal (20 becomes 0.20), indexed by the UTC
instant each close becomes known: 00:00 UTC on the day after the close.
``attach_fred_implied_vol`` joins it onto a bar frame (see ``softedge.data.schema``). Each bar
gets the last value known at or before its close, and NaN before the first value. Only frames whose
symbol is a key of ``VOL_INDEX_SERIES`` get the column.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

from softedge.data.cache import cache_dir, http_get

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Symbol -> FRED series id. OVXCLS is the CBOE Crude Oil ETF Volatility Index, so Brent reads a
# crude oil proxy, not a Brent index.
VOL_INDEX_SERIES: dict[str, str] = {
    "SPX": "VIXCLS",
    "NDX": "VXNCLS",
    "WTI": "OVXCLS",
    "BRENT": "OVXCLS",
}

MAX_CACHE_AGE_S = 86400
# Minimum gap between requests made by this module.
REQUEST_INTERVAL_S = 0.2

_last_request = 0.0


def _pace() -> None:
    global _last_request
    wait = _last_request + REQUEST_INTERVAL_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _fred_csv(series_id: str) -> bytes:
    path = cache_dir("fred") / f"{series_id}.csv"
    if path.exists() and time.time() - path.stat().st_mtime <= MAX_CACHE_AGE_S:
        return path.read_bytes()
    _pace()
    body = http_get(FRED_CSV_URL, params={"id": series_id}).content
    # Write under a temporary name so an interrupted download never leaves a short cache file.
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return body


def load_vol_index(series_id: str) -> pd.Series:
    """Closes of a FRED index as a decimal, indexed by the UTC instant each becomes known.

    Missing values are dropped. The index is named ``time`` and the series ``series_id``.
    """
    raw = pd.read_csv(io.BytesIO(_fred_csv(series_id)), na_values=["."])
    # Columns are read by position, so the header names do not matter.
    dates = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0], utc=True))
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(np.float64) / 100.0
    # A close is known by the end of its daily bar, which ends at the midnight after the day.
    known = (dates + pd.Timedelta(days=1)).as_unit("ns").rename("time")
    return pd.Series(values, index=known, name=series_id).dropna().sort_index()


def attach_fred_implied_vol(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of a bar frame with a float64 ``implied_vol`` column when its symbol has an index.

    A bar closes at its open plus ``frame.attrs["bar"]``. Frames for other symbols are copied
    without the column and no series is fetched.
    """
    out = frame.copy()
    symbol = frame.attrs.get("symbol")
    if symbol not in VOL_INDEX_SERIES:
        return out
    close = frame.index + pd.Timedelta(frame.attrs["bar"])
    out["implied_vol"] = _as_of(load_vol_index(VOL_INDEX_SERIES[symbol]), close)
    return out


def _as_of(series: pd.Series, times: pd.DatetimeIndex) -> np.ndarray:
    known = series.index.as_unit("ns").asi8
    pos = np.searchsorted(known, times.as_unit("ns").asi8, side="right") - 1
    # Position -1, before the first value, reads the leading NaN.
    padded = np.concatenate(([np.nan], series.to_numpy(dtype=np.float64)))
    return padded[pos + 1]
