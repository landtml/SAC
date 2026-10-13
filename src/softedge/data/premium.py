"""USD-M perpetual premium index, and its alignment to perp bar frames.

Source: monthly zips of premium index klines under https://data.binance.vision, in the kline
CSV layout. A premium is the perp's price over its index as a decimal, so 0.0001 is 1 bp.
``load_premium`` returns the close of each premium kline indexed by the kline's UTC open time.
That close is known only when the kline closes, which is when features are computed.

``attach_premium`` adds the premium to a perp bar frame as a float64 ``premium`` column. A bar
of one hour or shorter takes the premium kline with the same open time. A longer bar takes the
last hourly premium whose kline closes at or before the bar's close, so a daily bar takes the
23:00 hourly kline and nothing later. Bars without a premium get NaN.

Parsed series are cached as parquet under ``$SOFTEDGE_DATA/binance/premium`` once the end
month has passed. Raw zips are cached by ``fetch_bytes``.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from softedge.data.cache import cache_dir, fetch_bytes, load_frame, save_frame

BASE_URL = "https://data.binance.vision/data"
PREMIUM_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
    "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
)
PREMIUM_INTERVALS = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")
HOUR = pd.Timedelta("1h")
# The archive may write microseconds instead of milliseconds. A millisecond value stays below
# this bound until the year 5138, so magnitude tells the two units apart.
MICROSECOND_FLOOR = 10**14
# pandas rejects "1m" and deprecates "1d", so bar aliases use the pandas spellings.
_PANDAS_UNITS = {"s": "s", "m": "min", "h": "h", "d": "D"}


def load_premium(symbol: str, interval: str = "1h", start: str = "2020-01",
                 end: str = "2026-09") -> pd.Series:
    """Premium close per kline, indexed by UTC kline open time and named ``premium``.

    ``start`` and ``end`` are inclusive ``YYYY-MM`` months. A month the archive answers with
    404 is skipped, which covers months before the premium series was listed.
    """
    if interval not in PREMIUM_INTERVALS:
        raise ValueError(f"unsupported premium interval {interval!r}: {PREMIUM_INTERVALS}")
    sym = symbol.upper()
    path = cache_dir("binance", "premium") / f"{sym}-{interval}-{start}-{end}.parquet"
    if path.exists():
        return load_frame(path)["premium"]
    series = _premium_series(sym, interval, start, end)
    if _month_passed(end):
        save_frame(series.to_frame(), path)
    return series


def attach_premium(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of a perp bar frame with a float64 ``premium`` column aligned to its bars."""
    kind = frame.attrs.get("kind")
    if kind != "perp":
        raise ValueError(f"attach_premium needs a perp frame, got kind {kind!r}")
    symbol, bar = frame.attrs["symbol"], frame.attrs["bar"]
    width = pd.Timedelta(bar)
    out = frame.copy()
    if width <= HOUR:
        interval = _premium_interval(bar)
        premium = load_premium(symbol, interval, _month(frame.index[0]), _month(frame.index[-1]))
        out["premium"] = premium.reindex(frame.index).to_numpy(dtype=np.float64)
    else:
        last_hour = frame.index[-1] + width - HOUR
        hourly = load_premium(symbol, "1h", _month(frame.index[0]), _month(last_hour))
        out["premium"] = _last_closed(hourly, frame.index + width)
    return out


def _premium_series(symbol: str, interval: str, start: str, end: str) -> pd.Series:
    rel_dir = f"futures/um/monthly/premiumIndexKlines/{symbol}/{interval}"
    parts = []
    for month in _months(start, end):
        body = _fetch_zip(rel_dir, f"{symbol}-{interval}-{month}.zip")
        if body is not None:
            parts.append(_parse_premium(_read_zip(body)))
    if not parts:
        raise ValueError(f"no premium klines for {symbol} {interval} in {start}..{end}")
    series = pd.concat(parts)
    return series[~series.index.duplicated(keep="last")].sort_index()


def _last_closed(hourly: pd.Series, bar_close: pd.DatetimeIndex) -> np.ndarray:
    """Per bar close, the last hourly premium whose kline closes at or before it, else NaN."""
    closes = (hourly.index + HOUR).as_unit("ns").asi8
    pos = np.searchsorted(closes, bar_close.as_unit("ns").asi8, side="right") - 1
    values = hourly.to_numpy(dtype=np.float64)
    out = np.full(len(bar_close), np.nan)
    found = pos >= 0
    out[found] = values[pos[found]]
    return out


def _parse_premium(data: bytes) -> pd.Series:
    raw = _read_csv(data, PREMIUM_COLUMNS)
    ts = raw["open_time"].to_numpy(dtype=np.int64)
    millis = np.where(ts > MICROSECOND_FLOOR, ts // 1000, ts)
    return pd.Series(raw["close"].to_numpy(dtype=np.float64),
                     index=_utc_index(millis * 1_000_000), name="premium")


def _read_csv(data: bytes, columns: tuple[str, ...]) -> pd.DataFrame:
    # Some monthly files start with a header row and others do not, so the first field decides.
    first_field = data.split(b"\n", 1)[0].split(b",", 1)[0].strip()
    header = not _is_number(first_field.decode(errors="replace"))
    return pd.read_csv(io.BytesIO(data), header=None, names=list(columns),
                       skiprows=int(header))


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _read_zip(body: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        return archive.read(archive.namelist()[0])


def _fetch_zip(rel_dir: str, name: str) -> bytes | None:
    """One monthly zip from the archive, or None when the archive has no such month."""
    dest = cache_dir("binance", *rel_dir.split("/")) / name
    try:
        return fetch_bytes(f"{BASE_URL}/{rel_dir}/{name}", dest)
    except FileNotFoundError:
        return None


def _months(start: str, end: str) -> list[str]:
    year, month = map(int, start.split("-"))
    last = tuple(map(int, end.split("-")))
    months = []
    while (year, month) <= last:
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    if not months:
        raise ValueError(f"empty month range {start}..{end}")
    return months


def _month_passed(month: str) -> bool:
    """True once ``month`` lies entirely before the current UTC month."""
    now = datetime.now(UTC)
    year, mon = map(int, month.split("-"))
    return (year, mon) < (now.year, now.month)


def _month(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")


def _premium_interval(bar: str) -> str:
    """Premium interval whose bar alias is ``bar``, such as ``15min`` for ``15m``."""
    for interval in PREMIUM_INTERVALS:
        if _bar_alias(interval) == bar:
            return interval
    raise ValueError(f"no premium klines at bar {bar!r}")


def _bar_alias(interval: str) -> str:
    """Pandas offset alias of a fixed-length Binance interval, such as ``15m`` to ``15min``."""
    count, unit = interval[:-1], interval[-1:]
    if not count.isdigit() or unit not in _PANDAS_UNITS:
        raise ValueError(f"unsupported interval {interval!r}: fixed lengths only, such as 1h")
    return f"{int(count)}{_PANDAS_UNITS[unit]}"


def _utc_index(ns: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(ns.astype("datetime64[ns]")).tz_localize("UTC").rename("time")
