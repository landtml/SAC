"""Binance spot and USD-M perpetual klines, and perpetual funding, from the public archive.

Source: monthly zips under https://data.binance.vision, each holding one CSV. A month the
archive answers with 404 is skipped when it lies before the first or after the last listed
month. A 404 between two listed months raises, since the bars would bridge the gap.
``start`` and ``end`` are inclusive ``YYYY-MM`` months.

``load_spot`` and ``load_perp`` return bar frames (see ``softedge.data.schema``) with the
extra float64 columns ``quote_volume``, ``trades`` and ``taker_buy_base``. Bars are indexed by
open time in UTC, sorted, de-duplicated with the last row winning, stripped of rows with a
non-positive price, and have high and low widened to bound open and close. A perp frame's
``funding_rate`` is the sum of funding events inside ``[bar open, bar open + interval)``.

``load_funding`` returns the funding events as a Series. Parsed frames are cached as parquet
under ``$SOFTEDGE_DATA/binance/frames``. Raw zips are cached by ``fetch_bytes``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from softedge.data.cache import cache_dir, fetch_bytes, load_frame, save_frame
from softedge.data.schema import make_frame

BASE_URL = "https://data.binance.vision/data"
KLINE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
    "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
)
FUNDING_COLUMNS = ("calc_time", "funding_interval_hours", "last_funding_rate")
# Binance writes microseconds from 2025 on and milliseconds before. A millisecond value
# stays below this bound until the year 5138, so magnitude tells the unit apart.
MICROSECOND_FLOOR = 10**14
# pandas rejects "1m" and deprecates "1d", so bars carry the pandas spelling of Binance's
# fixed-length intervals.
_PANDAS_UNITS = {"s": "s", "m": "min", "h": "h", "d": "D"}


def load_spot(symbol: str, interval: str = "1h", start: str = "2020-01",
              end: str = "2026-09") -> pd.DataFrame:
    """Spot klines as a bar frame of kind ``spot``."""
    sym = symbol.upper()
    path = _frame_path("spot", sym, interval, start, end)
    if path.exists():
        return load_frame(path)
    frame = _bar_frame(_klines("spot", sym, interval, start, end), "spot", sym, interval)
    save_frame(frame, path)
    return frame


def load_perp(symbol: str, interval: str = "1h", start: str = "2020-01",
              end: str = "2026-09") -> pd.DataFrame:
    """USD-M perpetual klines as a bar frame of kind ``perp``, with ``funding_rate`` per bar."""
    sym = symbol.upper()
    path = _frame_path("perp", sym, interval, start, end)
    if path.exists():
        return load_frame(path)
    bars = _klines("futures/um", sym, interval, start, end)
    funding = _funding_events(sym, start, end)
    frame = _bar_frame(bars, "perp", sym, interval,
                       funding_rate=_funding_per_bar(bars.index, funding, interval))
    save_frame(frame, path)
    return frame


def load_funding(symbol: str, start: str = "2020-01", end: str = "2026-09") -> pd.Series:
    """Funding events of a USD-M perpetual: ``last_funding_rate`` indexed by UTC calc time."""
    return _funding_events(symbol.upper(), start, end)


def _klines(market: str, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    _bar_alias(interval)
    rel_dir = f"{market}/monthly/klines/{symbol}/{interval}"
    months = _months(start, end)
    parts, listed = [], []
    for month in months:
        body = _fetch_zip(rel_dir, f"{symbol}-{interval}-{month}.zip")
        if body is not None:
            parts.append(_parse_klines(_read_zip(body)))
            listed.append(month)
    if not parts:
        raise ValueError(f"no {market} klines for {symbol} {interval} in {start}..{end}")
    # Months outside the first and last listing are before the listing or after the data ends.
    # A month missing between two listed months would make the bars jump across it.
    present = set(listed)
    gap = [m for m in months if listed[0] < m < listed[-1] and m not in present]
    if gap:
        raise ValueError(f"no {market} klines archive for {symbol} {interval} in "
                         f"{', '.join(gap)}; bars would bridge the gap")
    return _clean(pd.concat(parts))


def _funding_events(symbol: str, start: str, end: str) -> pd.Series:
    rel_dir = f"futures/um/monthly/fundingRate/{symbol}"
    parts = []
    for month in _months(start, end):
        body = _fetch_zip(rel_dir, f"{symbol}-fundingRate-{month}.zip")
        if body is not None:
            parts.append(_parse_funding(_read_zip(body)))
    if not parts:
        raise ValueError(f"no funding rates for {symbol} in {start}..{end}")
    events = pd.concat(parts)
    return events[~events.index.duplicated(keep="last")].sort_index()


def _funding_per_bar(index: pd.DatetimeIndex, events: pd.Series, interval: str) -> np.ndarray:
    opens = index.as_unit("ns").asi8
    width = pd.Timedelta(_bar_alias(interval)).value
    times = events.index.as_unit("ns").asi8
    pos = np.searchsorted(opens, times, side="right") - 1
    inside = (pos >= 0) & (times < opens[np.clip(pos, 0, None)] + width)
    return np.bincount(pos[inside], weights=events.to_numpy(dtype=np.float64)[inside],
                       minlength=len(opens))


def _bar_frame(bars: pd.DataFrame, kind: str, symbol: str, interval: str,
               funding_rate: np.ndarray | None = None) -> pd.DataFrame:
    frame = make_frame(
        bars.index,
        bars["open"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy(),
        bars["close"].to_numpy(), bars["volume"].to_numpy(),
        symbol=symbol, kind=kind, bar=_bar_alias(interval), funding_rate=funding_rate,
    )
    for col in ("quote_volume", "trades", "taker_buy_base"):
        frame[col] = bars[col].to_numpy()
    return frame


def _clean(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(axis=1)]
    oc_high = np.maximum(bars["open"].to_numpy(), bars["close"].to_numpy())
    oc_low = np.minimum(bars["open"].to_numpy(), bars["close"].to_numpy())
    return bars.assign(high=np.maximum(bars["high"].to_numpy(), oc_high),
                       low=np.minimum(bars["low"].to_numpy(), oc_low))


def _parse_klines(data: bytes) -> pd.DataFrame:
    raw = _read_csv(data, KLINE_COLUMNS)
    ts = raw["open_time"].to_numpy(dtype=np.int64)
    millis = np.where(ts > MICROSECOND_FLOOR, ts // 1000, ts)
    return pd.DataFrame(
        {
            "open": raw["open"].to_numpy(dtype=np.float64),
            "high": raw["high"].to_numpy(dtype=np.float64),
            "low": raw["low"].to_numpy(dtype=np.float64),
            "close": raw["close"].to_numpy(dtype=np.float64),
            "volume": raw["volume"].to_numpy(dtype=np.float64),
            "quote_volume": raw["quote_volume"].to_numpy(dtype=np.float64),
            "trades": raw["count"].to_numpy(dtype=np.float64),
            "taker_buy_base": raw["taker_buy_volume"].to_numpy(dtype=np.float64),
        },
        index=_utc_index(millis * 1_000_000),
    )


def _parse_funding(data: bytes) -> pd.Series:
    raw = _read_csv(data, FUNDING_COLUMNS)
    calc_ms = raw["calc_time"].to_numpy(dtype=np.int64)
    return pd.Series(raw["last_funding_rate"].to_numpy(dtype=np.float64),
                     index=_utc_index(calc_ms * 1_000_000), name="funding_rate")


def _read_csv(data: bytes, columns: tuple[str, ...]) -> pd.DataFrame:
    # Futures files start with a header row and spot files do not, so the first field
    # decides whether to skip one line.
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


def _bar_alias(interval: str) -> str:
    """Pandas offset alias of a fixed-length Binance interval, such as ``15m`` to ``15min``."""
    count, unit = interval[:-1], interval[-1:]
    if not count.isdigit() or unit not in _PANDAS_UNITS:
        raise ValueError(f"unsupported interval {interval!r}: fixed lengths only, such as 1h")
    return f"{int(count)}{_PANDAS_UNITS[unit]}"


def _utc_index(ns: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(ns.astype("datetime64[ns]")).tz_localize("UTC")


def _frame_path(kind: str, symbol: str, interval: str, start: str, end: str) -> Path:
    return cache_dir("binance", "frames") / f"{kind}-{symbol}-{interval}-{start}-{end}.parquet"
