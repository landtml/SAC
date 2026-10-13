"""Deribit DVOL, the 30-day implied volatility index of BTC and ETH options, as a causal series.

``load_dvol`` returns the DVOL candle close in annualized decimal (55.0 becomes 0.55), indexed by
candle END time in UTC: the instant the value becomes known. ``attach_implied_vol`` joins it onto
a spot or perp bar frame (see ``sactrader.data.schema``). Each bar gets the last value known at or
before its close, and NaN before the first candle.

The public endpoint serves candles newest first, one page at a time, with ``continuation`` as the
end bound of the next older page. Raw pages are cached as JSON per currency, resolution and
calendar month under ``$SACTRADER_DATA/deribit/dvol``. A month that has not ended yet is fetched on
every call and not cached.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sactrader.data.cache import cache_dir, http_get

URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
CURRENCIES = ("BTC", "ETH")
# Resolution name -> (value the endpoint takes, candle length). The endpoint reads a plain
# number as seconds and "1D" as daily candles.
RESOLUTIONS = {"1h": ("3600", pd.Timedelta(hours=1)), "1D": ("1D", pd.Timedelta(days=1))}
# Two requests per second at most.
MIN_INTERVAL_S = 0.5
_last_request = float("-inf")


def load_dvol(currency: str, start: str = "2021-03", end: str = "2026-09",
              resolution: str = "1h") -> pd.Series:
    """DVOL closes in annualized decimal over the inclusive months ``start`` to ``end``.

    The index is the candle end time in UTC. The series is named ``implied_vol``.
    """
    cur = _currency(currency)
    api_resolution, candle = _resolution(resolution)
    months = pd.period_range(start, end, freq="M")
    if len(months) == 0:
        raise ValueError(f"end month {end} is before start month {start}")
    rows: list[list[float]] = []
    for month in months:
        rows.extend(_month_rows(cur, resolution, api_resolution, month))
    opened = np.array([row[0] for row in rows], dtype=np.int64)
    close = np.array([row[4] for row in rows], dtype=np.float64)
    ends = (pd.to_datetime(opened, unit="ms", utc=True) + candle).as_unit("ns").rename("time")
    return pd.Series(close / 100.0, index=ends, name="implied_vol").sort_index()


def attach_implied_vol(frame: pd.DataFrame, currency: str) -> pd.DataFrame:
    """Copy of a spot or perp bar frame with a float64 ``implied_vol`` column.

    A bar closes at its open plus ``frame.attrs["bar"]``. It takes the last DVOL candle that
    ended at or before that close.
    """
    kind = frame.attrs.get("kind")
    if kind not in ("spot", "perp"):
        raise ValueError(f"implied vol attaches to spot or perp frames, not {kind!r}")
    close = frame.index + pd.Timedelta(frame.attrs["bar"])
    # Hourly candles: a daily candle is known only at the next midnight, too late for intraday bars.
    dvol = load_dvol(currency, frame.index[0].strftime("%Y-%m"),
                     close[-1].strftime("%Y-%m"), "1h")
    out = frame.copy()
    out["implied_vol"] = _as_of(dvol, close)
    return out


def _currency(currency: str) -> str:
    cur = currency.upper()
    if cur not in CURRENCIES:
        raise ValueError(f"currency must be one of {CURRENCIES}, got {currency!r}")
    return cur


def _resolution(resolution: str) -> tuple[str, pd.Timedelta]:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {tuple(RESOLUTIONS)}, got {resolution!r}")
    return RESOLUTIONS[resolution]


def _month_rows(currency: str, resolution: str, api_resolution: str,
                month: pd.Period) -> list[list[float]]:
    path = cache_dir("deribit", "dvol", currency, resolution) / f"{month}.json"
    if path.exists():
        pages = json.loads(path.read_text(encoding="utf-8"))["pages"]
    else:
        first_ms, last_ms = _bounds_ms(month)
        pages = _fetch_pages(currency, api_resolution, first_ms, last_ms)
        # A month in progress gets new candles on every call, so caching it would go stale.
        if (month + 1).start_time.tz_localize("UTC") <= pd.Timestamp.now(tz="UTC"):
            _write_json(path, {"pages": pages})
    return [row for page in pages for row in page]


def _bounds_ms(month: pd.Period) -> tuple[int, int]:
    first = month.start_time.tz_localize("UTC")
    after = (month + 1).start_time.tz_localize("UTC")
    return round(first.timestamp() * 1000), round(after.timestamp() * 1000) - 1


def _fetch_pages(currency: str, api_resolution: str, first_ms: int,
                 last_ms: int) -> list[list[list[float]]]:
    pages: list[list[list[float]]] = []
    upper = last_ms
    while True:
        result = _get_result({"currency": currency, "resolution": api_resolution,
                              "start_timestamp": first_ms, "end_timestamp": upper})
        pages.append(result["data"])
        older = result.get("continuation")
        # Stop when no older candles remain in the window, or when the bound stops moving back.
        if older is None or older < first_ms or older >= upper:
            return pages
        upper = older


def _get_result(params: dict) -> dict:
    _pace()
    body = http_get(URL, params=params).json()
    if "error" in body:
        raise RuntimeError(f"Deribit returned an error: {body['error']}")
    return body["result"]


def _pace() -> None:
    global _last_request
    wait = _last_request + MIN_INTERVAL_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _write_json(path: Path, payload: dict) -> None:
    # Rename within the directory, so a reader never sees a half-written file.
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _as_of(series: pd.Series, times: pd.DatetimeIndex) -> np.ndarray:
    known = series.index.as_unit("ns").asi8
    pos = np.searchsorted(known, times.as_unit("ns").asi8, side="right") - 1
    # Position -1, before the first candle, reads the leading NaN.
    padded = np.concatenate(([np.nan], series.to_numpy(dtype=np.float64)))
    return padded[pos + 1]
