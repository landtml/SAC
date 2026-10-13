"""Daily forex bars from FRED reference rates.

``load_fx_daily`` returns a bar frame with kind ``"fx"``, bar ``"1D"`` and the pair as
symbol. The source holds one value per date. Each bar opens at the previous value and
closes at its own, and high and low bound the two. Missing source values are dropped
before the bars are built.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

from softedge.data.cache import cache_dir, http_get
from softedge.data.schema import make_frame

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# pair -> (FRED series id, invert). Invert replaces each value with its reciprocal, for a
# series quoted opposite to the pair.
FRED_FX_SERIES: dict[str, tuple[str, bool]] = {
    "EURUSD": ("DEXUSEU", False),
    "GBPUSD": ("DEXUSUK", False),
    "AUDUSD": ("DEXUSAL", False),
    "NZDUSD": ("DEXUSNZ", False),
    "USDJPY": ("DEXJPUS", False),
    "USDCAD": ("DEXCAUS", False),
    "USDCHF": ("DEXSZUS", False),
    # Emerging-market series are quoted as units of currency per US dollar, like DEXJPUS,
    # so none is inverted.
    "USDMXN": ("DEXMXUS", False),
    "USDZAR": ("DEXSFUS", False),
    "USDBRL": ("DEXBZUS", False),
    "USDINR": ("DEXINUS", False),
    "USDKRW": ("DEXKOUS", False),
    "USDCNY": ("DEXCHUS", False),
    "USDTHB": ("DEXTHUS", False),
    "USDSGD": ("DEXSIUS", False),
}

# The emerging-market pairs of FRED_FX_SERIES.
EM_FX_PAIRS: tuple[str, ...] = (
    "USDMXN", "USDZAR", "USDBRL", "USDINR", "USDKRW", "USDCNY", "USDTHB", "USDSGD",
)

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


def _closes(series_id: str, invert: bool) -> pd.Series:
    raw = pd.read_csv(io.BytesIO(_fred_csv(series_id)), na_values=["."])
    # Columns are read by position, so the header names do not matter.
    frame = pd.DataFrame({
        "date": pd.to_datetime(raw.iloc[:, 0], utc=True),
        "value": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
    }).dropna()
    close = frame.set_index("date")["value"].sort_index()
    return 1.0 / close if invert else close


def load_fx_daily(pair: str, start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily bars for ``pair`` with dates from ``start`` to ``end``, both inclusive.

    The open of the first bar in the window is the close of the last value before it, so a
    window agrees with the full history. Raises ``ValueError`` for an unknown pair and
    ``FrameError`` when fewer than two bars remain.
    """
    if pair not in FRED_FX_SERIES:
        raise ValueError(f"unknown pair {pair!r}, expected one of {sorted(FRED_FX_SERIES)}")
    series_id, invert = FRED_FX_SERIES[pair]
    close = _closes(series_id, invert)
    values = close.to_numpy(dtype=np.float64)
    open_ = np.empty_like(values)
    open_[1:] = values[:-1]
    open_[:1] = values[:1]
    high = np.maximum(open_, values)
    low = np.minimum(open_, values)

    days = close.index
    keep = days >= pd.Timestamp(start, tz="UTC")
    if end is not None:
        keep &= days <= pd.Timestamp(end, tz="UTC")
    return make_frame(
        days[keep], open_[keep], high[keep], low[keep], values[keep],
        symbol=pair, kind="fx", bar="1D",
    )
