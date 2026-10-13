"""Daily index, oil and Treasury bar frames from FRED, for markets with no price feed here.

``load_fred_market`` returns a bar frame with kind ``"spot"``, bar ``"1D"`` and the market
name as symbol. Price markets build bars the way ``sactrader.data.fx`` does: each bar opens
at the previous value and closes at its own, and high and low bound the two. Missing and
non-positive values are dropped before the bars are built, so a dropped day never feeds the
next open. FRED's WTI series holds -36.98 on 2020-04-20.

Bond markets turn a constant-maturity yield in percent into a total-return price index that
starts at 100 on the first FRED observation. For consecutive observation dates d0 < d1 with
yields y0 and y1 (decimals) and dt = calendar days / 365, the index moves by

    ret = y0 * dt - D * (y1 - y0) + 0.5 * C * (y1 - y0) ** 2

where D is the modified duration and C the convexity of the bond. This is a
duration-convexity approximation of the return of a constant-maturity bond fund.

Volume is NaN for every FRED market.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

from sactrader.data.cache import cache_dir, http_get
from sactrader.data.schema import make_frame

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# name -> FRED series id and quote. "price" series are index or spot levels. "yield" series are
# percent yields, with the modified duration and convexity of the bond they stand for.
FRED_MARKETS: dict[str, dict] = {
    "SPX": {"series": "SP500", "quote": "price"},
    "NDX": {"series": "NASDAQCOM", "quote": "price"},
    "NKY": {"series": "NIKKEI225", "quote": "price"},
    "WTI": {"series": "DCOILWTICO", "quote": "price"},
    "BRENT": {"series": "DCOILBRENTEU", "quote": "price"},
    "UST10": {"series": "DGS10", "quote": "yield", "duration": 8.0, "convexity": 80.0},
    "UST2": {"series": "DGS2", "quote": "yield", "duration": 1.9, "convexity": 4.5},
}

MAX_CACHE_AGE_S = 86400
# Minimum gap between requests made by this module.
REQUEST_INTERVAL_S = 0.2

_last_request = 0.0


def load_fred_market(name: str, start: str = "2000-01", end: str = "2026-09") -> pd.DataFrame:
    """Daily bars for ``name`` from ``start`` to ``end``, both inclusive.

    ``start`` and ``end`` take a date or a ``YYYY-MM`` month, and an ``end`` month covers all
    of that month. Bars are built on the full FRED history, so the open of the first bar in
    the window is the close of the last value before it. Raises ``ValueError`` for an unknown
    name and ``FrameError`` when fewer than two bars remain.
    """
    if name not in FRED_MARKETS:
        raise ValueError(f"unknown market {name!r}, expected one of {sorted(FRED_MARKETS)}")
    spec = FRED_MARKETS[name]
    if spec["quote"] == "price":
        close = _price_closes(spec["series"])
    else:
        close = _bond_closes(spec["series"], spec["duration"], spec["convexity"])
    days = close.index
    values = close.to_numpy(dtype=np.float64)
    open_ = np.empty_like(values)
    open_[1:] = values[:-1]
    open_[:1] = values[:1]
    high = np.maximum(open_, values)
    low = np.minimum(open_, values)

    keep = (days >= pd.Timestamp(start, tz="UTC")) & (days <= _window_end(end))
    return make_frame(
        days[keep], open_[keep], high[keep], low[keep], values[keep],
        symbol=name, kind="spot", bar="1D",
    )


def _window_end(end: str) -> pd.Timestamp:
    stamp = pd.Timestamp(end, tz="UTC")
    # A month as the end runs to its last day, so the default end keeps all of September.
    return stamp + pd.offsets.MonthEnd(0) if len(end) == 7 else stamp


def _price_closes(series_id: str) -> pd.Series:
    values = _observations(series_id)
    # A bar needs a positive close, so non-positive prints are dropped rather than clipped.
    return values[values > 0]


def _bond_closes(series_id: str, duration: float, convexity: float) -> pd.Series:
    yields = _observations(series_id) / 100.0
    dt = yields.index.to_series().diff().dt.total_seconds().to_numpy()[1:] / (365 * 86400)
    y = yields.to_numpy(dtype=np.float64)
    dy = np.diff(y)
    ret = y[:-1] * dt - duration * dy + 0.5 * convexity * dy**2
    close = 100.0 * np.concatenate(([1.0], np.cumprod(1.0 + ret)))
    return pd.Series(close, index=yields.index)


def _observations(series_id: str) -> pd.Series:
    raw = pd.read_csv(io.BytesIO(_fred_csv(series_id)), na_values=["."])
    # Columns are read by position, so the header names do not matter.
    frame = pd.DataFrame({
        "date": pd.to_datetime(raw.iloc[:, 0], utc=True),
        "value": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
    }).dropna()
    return frame.set_index("date")["value"].sort_index()


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
