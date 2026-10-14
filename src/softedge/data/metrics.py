"""USD-M perpetual open interest from the daily metrics files of data.binance.vision.

Source: one zip per symbol and day under
https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{YYYY-MM-DD}.zip,
each holding one CSV of snapshots every five minutes. ``sum_open_interest_value`` is the open
interest of the symbol in USD. A day the archive answers with 404 is skipped. A symbol listed
later has no earlier days.

``load_open_interest`` returns that value per snapshot, indexed by the snapshot's UTC time.
Months before ``OI_START`` are not requested, since each day costs one request.
``attach_open_interest`` adds a float64 ``oi_value`` column to a bar frame. A bar takes the
last snapshot strictly before its close, so a bar that closes at 00:00 UTC takes the 23:55
snapshot and nothing later. A bar with no earlier snapshot gets NaN. Hourly bars resampled to
daily keep the last hourly value, which is the 23:55 snapshot of the day.

Parsed series are cached as parquet under ``$SOFTEDGE_DATA/binance/metrics`` once the end month
has passed. Raw zips are cached by ``fetch_bytes``.
"""

from __future__ import annotations

import io
import warnings
import zipfile
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from softedge.data.cache import cache_dir, fetch_bytes, load_frame, save_frame

BASE_URL = "https://data.binance.vision/data"
METRICS_DIR = "futures/um/daily/metrics"
OI_START = "2022-01"


def load_open_interest(symbol: str, start: str = OI_START, end: str = "2026-09") -> pd.Series:
    """USD open interest per snapshot, indexed by UTC snapshot time and named ``oi_value``.

    ``start`` and ``end`` are inclusive ``YYYY-MM`` months. Months before ``OI_START`` are not
    requested.
    """
    sym = symbol.upper()
    start = max(start, OI_START)
    path = cache_dir("binance", "metrics") / f"{sym}-{start}-{end}.parquet"
    if path.exists():
        return load_frame(path)["oi_value"]
    series = _oi_series(sym, start, end)
    if _month_passed(end):
        save_frame(series.to_frame(), path)
    return series


def attach_open_interest(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Copy of a bar frame with a float64 ``oi_value`` column: the last snapshot before each
    bar's close. ``symbol`` is the Binance symbol of the coin, such as ``BTCUSDT``."""
    width = pd.Timedelta(frame.attrs["bar"])
    series = load_open_interest(symbol, _month(frame.index[0]), _month(frame.index[-1]))
    out = frame.copy()
    out["oi_value"] = _last_before(series, frame.index + width)
    return out


def _last_before(series: pd.Series, closes: pd.DatetimeIndex) -> np.ndarray:
    """Per bar close, the value of the last snapshot strictly before it, else NaN."""
    times = series.index.as_unit("ns").asi8
    values = series.to_numpy(dtype=np.float64)
    pos = np.searchsorted(times, closes.as_unit("ns").asi8, side="left") - 1
    out = np.full(len(closes), np.nan)
    found = pos >= 0
    out[found] = values[pos[found]]
    return out


def _oi_series(symbol: str, start: str, end: str) -> pd.Series:
    parts, days = [], []
    for day in _days(start, end):
        body = _fetch_day(symbol, day)
        if body is not None:
            parts.append(_parse_metrics(body))
            days.append(day)
    if not parts:
        raise ValueError(f"no open interest for {symbol} in {start}..{end}")
    _warn_missing_days(symbol, days)
    series = pd.concat(parts)
    return series[~series.index.duplicated(keep="last")].sort_index()


def _warn_missing_days(symbol: str, days: list[str]) -> None:
    """Warn about days between the first and last archive day with no file. Gate features hold
    the last sample over such a day, so a warning marks a stretch that is stale."""
    have = pd.DatetimeIndex(pd.to_datetime(days))
    span = pd.date_range(have[0], have[-1], freq="D")
    missing = span.difference(have)
    if len(missing):
        warnings.warn(
            f"{symbol} open interest has {len(missing)} missing day(s) between {have[0].date()} "
            f"and {have[-1].date()}; the squeeze gate holds the last sample over them",
            UserWarning, stacklevel=3)


def _fetch_day(symbol: str, day: str) -> bytes | None:
    """One daily metrics zip, or None when the archive has no file for that day."""
    name = f"{symbol}-metrics-{day}.zip"
    dest = cache_dir("binance", "metrics", symbol) / name
    try:
        return fetch_bytes(f"{BASE_URL}/{METRICS_DIR}/{symbol}/{name}", dest)
    except FileNotFoundError:
        return None


def _parse_metrics(data: bytes) -> pd.Series:
    raw = pd.read_csv(io.BytesIO(_read_zip(data)), usecols=["create_time",
                                                             "sum_open_interest_value"])
    index = pd.DatetimeIndex(pd.to_datetime(raw["create_time"], utc=True)).rename("time")
    return pd.Series(raw["sum_open_interest_value"].to_numpy(dtype=np.float64), index=index,
                     name="oi_value")


def _read_zip(body: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        return archive.read(archive.namelist()[0])


def _days(start: str, end: str) -> list[str]:
    first = pd.Timestamp(f"{start}-01")
    last = pd.Timestamp(f"{end}-01") + pd.offsets.MonthEnd(0)
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(first, last, freq="D")]


def _month_passed(month: str) -> bool:
    """True once ``month`` lies entirely before the current UTC month."""
    now = datetime.now(UTC)
    year, mon = map(int, month.split("-"))
    return (year, mon) < (now.year, now.month)


def _month(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")
