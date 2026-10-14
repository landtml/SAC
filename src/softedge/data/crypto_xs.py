"""Binance spot USDT cross-section: daily close-to-close returns for every listed coin.

The universe is every USDT-quoted spot symbol the public archive lists, delisted symbols
included, minus stablecoin pairs and leveraged tokens. Closes come from the 1d klines that
``softedge.data.binance`` reads and caches.

A return is NaN before a coin lists, after it delists, and on any day whose previous bar is
not the day before. A missing day inside a coin's series is warned about and never bridged.
A day is usable only when at least ``MIN_COINS`` coins have a return on it.

Listing: the archive's S3 bucket, prefix ``data/spot/monthly/klines/``, read page by page.
"""

from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from softedge.data import binance
from softedge.data.cache import http_get

S3_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
KLINES_PREFIX = "data/spot/monthly/klines/"
INTERVAL = "1d"
MIN_COINS = 30
LISTING_WORKERS = 8
QUOTE = "USDT"
ONE_DAY = pd.Timedelta("1D")
_MONTHLY_ZIP = re.compile(r"-1d-(\d{4}-\d{2})\.zip$")

# Base assets of stablecoin and fiat-pegged pairs. The USDS family is listed as a stablecoin
# name; USDSOLD has no 1d files in the archive, so it changes no count.
STABLECOIN_BASES = frozenset({
    "USDC", "BUSD", "TUSD", "USDP", "DAI", "FDUSD", "USDS", "USDSB", "USDSOLD", "USDE",
    "UST", "USTC", "USD1", "RLUSD", "XUSD", "BFUSD", "USDD", "FRAX", "LUSD", "GUSD", "SUSD",
    "VAI", "EURI", "AEUR", "EUR", "GBP", "AUD", "TRY", "BRL", "NGN", "ZAR", "UAH", "RUB",
    "IDRT", "BIDR", "PAX",
})
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
# Coins whose tickers end in a leveraged-token suffix but are not leveraged tokens.
NOT_LEVERAGED = frozenset({"JUP", "SYRUP"})
# A ticker the archive reuses for another coin: bars before the date are the old coin, bars
# from the date on are the new coin under the second name. LUNAUSDT's last old-coin bar is
# 2022-05-13 and its first new-coin bar is 2022-05-31.
TICKER_REUSE = {"LUNAUSDT": ("2022-05-31", "LUNA2USDT")}


def is_excluded(symbol: str) -> bool:
    """True for a stablecoin pair or a leveraged-token pair."""
    base = symbol[: -len(QUOTE)]
    if base in STABLECOIN_BASES:
        return True
    return base not in NOT_LEVERAGED and base.endswith(LEVERAGED_SUFFIXES)


def universe(symbols: Iterable[str]) -> list[str]:
    """USDT spot symbols that are neither stablecoin nor leveraged-token pairs, sorted."""
    return sorted(s for s in symbols
                  if s.endswith(QUOTE) and len(s) > len(QUOTE) and not is_excluded(s))


def list_archive() -> dict[str, list[str]]:
    """Each coin in the universe on the archive, with the ``YYYY-MM`` months holding a 1d zip."""
    _, prefixes = _s3_list(KLINES_PREFIX, delimiter="/")
    symbols = universe(p[len(KLINES_PREFIX):].rstrip("/") for p in prefixes)
    with ThreadPoolExecutor(LISTING_WORKERS) as pool:
        months = list(pool.map(_listed_months, symbols))
    return {sym: m for sym, m in zip(symbols, months, strict=True) if m}


def load_closes(start: str, end: str, archive: Mapping[str, Iterable[str]] | None = None,
                ) -> dict[str, pd.Series]:
    """Daily closes by coin for inclusive ``YYYY-MM-DD`` days, reused tickers split in two.

    ``archive`` maps each symbol to the months that hold its 1d zips; the default is
    ``list_archive()``. Each value is a close Series indexed by UTC bar open.
    """
    if archive is None:
        archive = list_archive()
    closes = {}
    for sym in universe(archive):
        months = sorted(archive[sym])
        if not months:
            continue
        lo, hi = max(months[0], start[:7]), min(months[-1], end[:7])
        if lo > hi:
            continue
        try:
            bars = _read_bars(sym, lo, hi)
        except ValueError as exc:  # no klines in the window, or fewer than two bars
            warnings.warn(f"{sym} skipped: {exc}", UserWarning, stacklevel=2)
            continue
        for name, part in _split_reused(sym, bars).items():
            closes[name] = part["close"]
    return closes


def load_crypto_xs(start: str, end: str, archive: Mapping[str, Iterable[str]] | None = None,
                   ) -> tuple[pd.DataFrame, pd.Series]:
    """Daily close-to-close returns and the day mask, for inclusive ``YYYY-MM-DD`` days.

    ``archive`` maps each symbol to the months that hold its 1d zips; the default is
    ``list_archive()``. Returns ``(returns, mask)``: ``returns`` is indexed by UTC day with one
    column per coin (NaN where the coin has no return), and ``mask[day]`` is True when at least
    ``MIN_COINS`` coins have a return that day.
    """
    columns = {}
    for name, close in load_closes(start, end, archive).items():
        rets = daily_returns(close, name)
        if rets.notna().any():
            columns[name] = rets
    if not columns:
        return pd.DataFrame(), pd.Series(dtype=bool)
    days = pd.date_range(start, end, freq="D")
    returns = pd.DataFrame(columns).reindex(days)
    mask = returns.notna().sum(axis=1) >= MIN_COINS
    return returns, mask


def daily_returns(close: pd.Series, symbol: str = "") -> pd.Series:
    """Close-to-close simple returns by UTC day; NaN where the previous bar is not the day before.

    A missing day produces a ``UserWarning`` naming the first gap, and the return across it is
    left NaN rather than measured over the gap.
    """
    index = close.index
    if index.tz is not None:
        index = index.tz_convert(None)
    days = pd.DatetimeIndex(index.normalize())
    px = pd.Series(close.to_numpy(dtype=np.float64), index=days)
    step = px.index.to_series().diff()
    gaps = step[step > ONE_DAY]
    if not gaps.empty:
        warnings.warn(
            f"{symbol or 'series'} has {len(gaps)} gap(s) in its 1d closes, first from "
            f"{gaps.index[0] - gaps.iloc[0]:%Y-%m-%d} to {gaps.index[0]:%Y-%m-%d}; "
            "returns across a gap are NaN, not bridged.",
            UserWarning, stacklevel=2)
    rets = px / px.shift(1) - 1.0
    rets[step != ONE_DAY] = np.nan
    return rets


def _read_bars(symbol: str, lo: str, hi: str) -> pd.DataFrame:
    """Cached 1d bars for months ``lo``..``hi``, read as the binance loader reads them.

    The loader rejects any frame with negative volume, and this module needs only closes. So
    the same zips are read here, and negative volumes are set to NaN with a warning that names
    the rows. Raises ``ValueError`` when the months hold no usable bars.
    """
    rel_dir = f"spot/monthly/klines/{symbol}/{INTERVAL}"
    parts = []
    for month in binance._months(lo, hi):
        body = binance._fetch_zip(rel_dir, f"{symbol}-{INTERVAL}-{month}.zip")
        if body is not None:
            parts.append(binance._parse_klines(binance._read_zip(body)))
    if not parts:
        raise ValueError(f"no spot klines in {lo}..{hi}")
    bars = binance._clean(pd.concat(parts).sort_index())
    if len(bars) < 2:
        raise ValueError("fewer than two bars")
    binance._warn_gaps(bars, symbol, INTERVAL)
    negative = bars.index[bars["volume"].to_numpy() < 0]
    if len(negative):
        days = ", ".join(d.strftime("%Y-%m-%d") for d in negative[:5])
        more = " and more" if len(negative) > 5 else ""
        warnings.warn(
            f"{symbol} has {len(negative)} row(s) with negative volume ({days}{more}); "
            "volume set to NaN, closes kept.", UserWarning, stacklevel=2)
        bars = bars.assign(volume=bars["volume"].where(bars["volume"] >= 0))
    return bars


def _split_reused(symbol: str, bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Bars by coin name: one entry, unless ``symbol`` is a reused ticker that splits in two."""
    if symbol not in TICKER_REUSE:
        return {symbol: bars}
    reuse_day, new_name = TICKER_REUSE[symbol]
    cut = pd.Timestamp(reuse_day, tz="UTC")
    parts = {symbol: bars[bars.index < cut], new_name: bars[bars.index >= cut]}
    return {name: part for name, part in parts.items() if not part.empty}


def _listed_months(symbol: str) -> list[str]:
    keys, _ = _s3_list(f"{KLINES_PREFIX}{symbol}/{INTERVAL}/")
    found = (_MONTHLY_ZIP.search(key) for key in keys)
    return sorted(m[1] for m in found if m)


def _s3_list(prefix: str, delimiter: str | None = None) -> tuple[list[str], list[str]]:
    """Keys and common prefixes under ``prefix``, following S3's marker pagination."""
    keys: list[str] = []
    prefixes: list[str] = []
    marker = ""
    while True:
        params = {"prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if marker:
            params["marker"] = marker
        root = ET.fromstring(http_get(S3_URL, params=params).content)
        page_keys = [e.text for e in root.iterfind("s:Contents/s:Key", S3_NS)]
        page_prefixes = [e.text for e in root.iterfind("s:CommonPrefixes/s:Prefix", S3_NS)]
        keys += page_keys
        prefixes += page_prefixes
        truncated = root.findtext("s:IsTruncated", default="false", namespaces=S3_NS)
        if truncated != "true":
            return keys, prefixes
        marker = max(page_keys + page_prefixes)
