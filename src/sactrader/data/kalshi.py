"""Kalshi settled binary markets as bar frames.

``list_settled_markets`` lists settled YES/NO markets of a set of series by traded volume.
``load_candles`` returns the hourly binary frame of one listed market from the candlestick
endpoint, and ``load_market`` resamples it into a binary bar frame whose
``attrs["resolution"]`` is the settlement value of the YES side.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sactrader.data.cache import cache_dir, http_get
from sactrader.data.schema import make_frame

API_URL = "https://api.elections.kalshi.com/trade-api/v2"
PAGE_SIZE = 1000
# Multi-day macro series whose settled markets returned candles when probed.
DEFAULT_SERIES = ("KXFED", "KXFEDDECISION", "KXCPI", "KXCPIYOY", "KXPAYROLLS", "KXU3")
PERIOD_MINUTES = 60
# The candlestick endpoint answers 400 above 5000 candles, so each request covers 30 days.
MAX_WINDOW_S = 30 * 24 * 3600
MIN_BARS = 48
PRICE_FLOOR = 0.001
PRICE_CEIL = 0.999
# Kalshi answers 429 to bursts faster than two requests per second.
REQUEST_INTERVAL_S = 0.5
COLUMNS = ("ticker", "series_ticker", "event_ticker", "title", "start", "end", "scheduled_end",
           "resolution", "volume")

_last_request = 0.0


def _pace() -> None:
    global _last_request
    wait = _last_request + REQUEST_INTERVAL_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _get_json(url: str, params: dict | None = None):
    _pace()
    return http_get(url, params=params).json()


def _write_atomic(path: Path, data: bytes) -> None:
    # A temporary name keeps an interrupted write from leaving a truncated cache file.
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _optional_utc(value) -> pd.Timestamp | None:
    if not value:
        return None
    return _utc(value)


def _latest(*values) -> pd.Timestamp | None:
    stamps = [s for s in (_optional_utc(v) for v in values) if s is not None]
    return max(stamps) if stamps else None


def _iso(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None else ts.isoformat()


def _dollars(value) -> float | None:
    # Prices arrive as decimal strings in dollars, with "" or null where there is no quote.
    if value in (None, ""):
        return None
    return float(value)


def _clip(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), PRICE_FLOOR, PRICE_CEIL)


def _scheduled_end(market: dict) -> pd.Timestamp | None:
    # close_time moves to the actual close when a market closes early, so it can reveal the
    # outcome timing. The expiration times stay at the scheduled dates.
    for key in ("latest_expiration_time", "expiration_time"):
        ts = _optional_utc(market.get(key))
        if ts is not None:
            return ts
    return None


def _settled_row(market: dict, series: str) -> dict | None:
    """Stored fields of a settled YES/NO binary market, or None for any other market."""
    if market.get("market_type") != "binary" or market.get("result") not in ("yes", "no"):
        return None
    return {
        "ticker": str(market["ticker"]),
        "series_ticker": series,
        "event_ticker": str(market.get("event_ticker") or ""),
        "title": str(market.get("title") or ""),
        "start": _iso(_optional_utc(market.get("open_time"))),
        # Settlement stands in for the last candle, which would cost one request per market.
        "end": _iso(_latest(market.get("close_time"), market.get("settlement_ts"))),
        "scheduled_end": _iso(_scheduled_end(market)),
        "resolution": 1.0 if market["result"] == "yes" else 0.0,
        "volume": float(market.get("volume_fp") or 0.0),
    }


def _scan_settled(series: tuple[str, ...], min_volume: float, max_markets: int) -> list[dict]:
    # Pages are not ordered by volume, so every page of every series is read before the cut.
    rows: list[dict] = []
    for ticker in series:
        cursor: str | None = None
        while True:
            params: dict = {"series_ticker": ticker, "status": "settled", "limit": PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            body = _get_json(f"{API_URL}/markets", params)
            for market in body.get("markets", []):
                row = _settled_row(market, ticker)
                if row is not None and row["volume"] >= min_volume:
                    rows.append(row)
            cursor = body.get("cursor")
            if not cursor or not body.get("markets"):
                break
    rows.sort(key=lambda r: (-r["volume"], r["ticker"]))
    return rows[:max_markets]


def _markets_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    for column in ("start", "end", "scheduled_end"):
        df[column] = pd.to_datetime(df[column], utc=True)
    df["resolution"] = df["resolution"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def list_settled_markets(series: tuple[str, ...] = DEFAULT_SERIES, min_volume: float = 1e5,
                         max_markets: int = 300) -> pd.DataFrame:
    """Settled YES/NO markets of ``series``, largest volume first, at most ``max_markets`` rows.

    ``start`` is the open time and ``end`` the later of the scheduled close and settlement, which
    bounds the candle history. ``scheduled_end`` is the latest scheduled expiration, never the
    actual close. The result is cached per argument set and never expires; delete the cache file
    to refresh it.
    """
    series = tuple(series)
    key = hashlib.sha256(json.dumps(series).encode()).hexdigest()[:16]
    path = cache_dir("kalshi") / f"settled-{key}-{max_markets}-{min_volume:g}.json"
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        rows = _scan_settled(series, min_volume, max_markets)
        _write_atomic(path, json.dumps(rows).encode())
    return _markets_frame(rows)


def _reduce_candle(candle: dict) -> dict:
    return {
        "t": int(candle["end_period_ts"]),
        "bid": _dollars((candle.get("yes_bid") or {}).get("close_dollars")),
        "ask": _dollars((candle.get("yes_ask") or {}).get("close_dollars")),
        "last": _dollars((candle.get("price") or {}).get("close_dollars")),
    }


def _fetch_candles(ticker: str, series: str, start_ts: int, end_ts: int,
                   period_minutes: int) -> list[dict]:
    path = cache_dir("kalshi", "candles") / f"{ticker}-{period_minutes}-{start_ts}-{end_ts}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows: list[dict] = []
    for lo in range(start_ts, end_ts, MAX_WINDOW_S):
        hi = min(lo + MAX_WINDOW_S, end_ts)
        body = _get_json(f"{API_URL}/series/{series}/markets/{ticker}/candlesticks", {
            "start_ts": lo, "end_ts": hi, "period_interval": period_minutes,
        })
        rows.extend(_reduce_candle(c) for c in body.get("candlesticks", []))
    _write_atomic(path, json.dumps(rows).encode())
    return rows


def _candle_prices(rows: list[dict], period_minutes: int) -> pd.Series:
    """One price per candle, indexed by the candle open time."""
    df = pd.DataFrame(rows, columns=["t", "bid", "ask", "last"]).astype(float)
    mid = (df["bid"] + df["ask"]) / 2
    # A one-sided quote is not a price, so the last trade stands in for it.
    price = mid.where(df["bid"].notna() & df["ask"].notna(), df["last"])
    index = pd.to_datetime(df["t"] - period_minutes * 60, unit="s", utc=True)
    series = pd.Series(price.to_numpy(), index=index, name="price", dtype=float)
    series = series.sort_index(kind="stable")
    return series[~series.index.duplicated(keep="first")]


def _binary_frame(row: dict, index, open_, high, low, close, bar: str) -> pd.DataFrame:
    frame = make_frame(
        index, _clip(open_), _clip(high), _clip(low), _clip(close),
        symbol=str(row["ticker"])[:60], kind="binary", bar=bar,
        resolution=float(row["resolution"]),
    )
    # Only the scheduled end may feed features; the settlement time would leak the outcome.
    scheduled = row.get("scheduled_end")
    if scheduled is not None and not pd.isna(scheduled):
        frame.attrs["end_time"] = _utc(scheduled).isoformat()
    return frame


def load_candles(market: pd.Series | dict, period_minutes: int = PERIOD_MINUTES) -> pd.DataFrame:
    """Binary bar frame of a listed market's YES price, one bar per candle.

    ``market`` is a row of ``list_settled_markets``. A candle contributes the mid of its yes bid
    and ask closes, or its last trade price when a quote is missing. Candles without any price
    repeat the previous close, and bars before the first price are dropped. Raises
    ``ValueError`` when the market lacks start or end, or when fewer than 48 bars remain.
    """
    row = dict(market)
    ticker = str(row["ticker"])
    start, end = _utc(row["start"]), _utc(row["end"])
    if pd.isna(start) or pd.isna(end) or end <= start:
        raise ValueError(f"market {ticker!r} needs start and end times")
    rows = _fetch_candles(ticker, str(row["series_ticker"]), int(start.timestamp()),
                          int(end.timestamp()), period_minutes)
    prices = _candle_prices(rows, period_minutes) if rows else pd.Series(dtype=float)
    if prices.empty:
        raise ValueError(f"market {ticker!r} has no candles")
    grid = pd.date_range(prices.index[0], prices.index[-1], freq=f"{period_minutes}min")
    close = prices.reindex(grid)
    close = close[close.notna().cumsum() > 0].ffill()
    if len(close) < MIN_BARS:
        raise ValueError(f"market {ticker!r} has {len(close)} bars, need {MIN_BARS}")
    return _binary_frame(row, close.index, close, close, close, close,
                         bar=f"{period_minutes}min")


def load_market(market: pd.Series | dict, bar: str = "1h") -> pd.DataFrame:
    """Binary bar frame of a listed market's YES price, resampled from hourly candles.

    ``bar`` must be at least one candle period (one hour). Raises ``ValueError`` when ``bar`` is
    shorter than that or when fewer than 48 bars remain at ``bar``.
    """
    if pd.Timedelta(bar) < pd.Timedelta(minutes=PERIOD_MINUTES):
        raise ValueError(f"bar must be at least {PERIOD_MINUTES} minutes, got {bar!r}")
    row = dict(market)
    hourly = load_candles(row)
    ohlc = hourly["close"].resample(bar, closed="left", label="left").ohlc()
    if len(ohlc) < MIN_BARS:
        raise ValueError(f"market {row['ticker']!r} has {len(ohlc)} bars, need {MIN_BARS}")
    return _binary_frame(row, ohlc.index, ohlc["open"], ohlc["high"], ohlc["low"],
                         ohlc["close"], bar=bar)
