"""Polymarket binary markets as bar frames.

``list_resolved_markets`` lists settled YES/NO markets by volume. ``load_price_history``
returns the CLOB price history of one token. ``load_market`` resamples the YES price of a
listed market into a binary bar frame whose ``attrs["resolution"]`` is the settlement value
of the YES token.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from softedge.data.cache import cache_dir, http_get
from softedge.data.schema import make_frame

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
PAGE_SIZE = 100
# Longer windows are rejected by the prices-history endpoint (not verified here).
MAX_WINDOW_S = 14 * 24 * 3600
MIN_BARS = 48
PRICE_FLOOR = 0.001
PRICE_CEIL = 0.999
# Minimum gap between requests made by this module.
REQUEST_INTERVAL_S = 0.2
COLUMNS = ("id", "slug", "question", "start", "end", "scheduled_end", "yes_token", "resolution",
           "volume")

_last_request = 0.0


def _pace() -> None:
    global _last_request
    wait = _last_request + REQUEST_INTERVAL_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


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


def _json_list(value) -> list:
    # Gamma encodes list fields as JSON strings.
    if isinstance(value, str):
        value = json.loads(value)
    return list(value or [])


def _resolved_row(market: dict) -> dict | None:
    """Stored fields of a settled YES/NO market, or None for any other market."""
    try:
        outcomes = _json_list(market.get("outcomes"))
        prices = _json_list(market.get("outcomePrices"))
        tokens = _json_list(market.get("clobTokenIds"))
    except (TypeError, ValueError):
        return None
    if outcomes != ["Yes", "No"] or not tokens:
        return None
    if prices == ["1", "0"]:
        resolution = 1.0
    elif prices == ["0", "1"]:
        resolution = 0.0
    else:
        return None
    return {
        "id": str(market["id"]),
        "slug": market.get("slug") or "",
        "question": market.get("question") or "",
        "start": _iso(_optional_utc(market.get("startDate"))),
        "end": _iso(_latest(market.get("endDate"), market.get("closedTime"))),
        "scheduled_end": _iso(_optional_utc(market.get("endDate"))),
        "yes_token": str(tokens[0]),
        "resolution": resolution,
        "volume": float(market.get("volumeNum") or 0.0),
    }


def _scan_resolved(min_volume: float, max_markets: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < max_markets:
        _pace()
        page = http_get(GAMMA_URL, params={
            "closed": "true", "limit": PAGE_SIZE, "offset": offset,
            "order": "volumeNum", "ascending": "false",
        }).json()
        for market in page:
            # Pages are sorted by volume, so the first market below the floor ends the scan.
            if float(market.get("volumeNum") or 0.0) < min_volume:
                return rows
            row = _resolved_row(market)
            if row is not None:
                rows.append(row)
                if len(rows) == max_markets:
                    return rows
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def _markets_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    df["start"] = pd.to_datetime(df["start"], format="ISO8601", utc=True)
    df["end"] = pd.to_datetime(df["end"], format="ISO8601", utc=True)
    df["scheduled_end"] = pd.to_datetime(df["scheduled_end"], format="ISO8601", utc=True)
    df["resolution"] = df["resolution"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def list_resolved_markets(min_volume: float = 1e5, max_markets: int = 500) -> pd.DataFrame:
    """Settled YES/NO markets, largest volume first, at most ``max_markets`` rows.

    ``end`` is the later of the end date and the closing time and bounds the price history.
    ``scheduled_end`` is the end date alone: it was public while the market traded, whereas the
    closing time reveals when (and often how) the market resolved. The result is cached per
    argument pair and never expires; delete the cache file to refresh it.
    """
    path = cache_dir("polymarket") / f"resolved-{max_markets}-{min_volume:g}.json"
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        rows = _scan_resolved(min_volume, max_markets)
        _write_atomic(path, json.dumps(rows).encode())
    return _markets_frame(rows)


def _fetch_windows(token_id: str, start_ts: int, end_ts: int,
                   fidelity: int) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for lo in range(start_ts, end_ts, MAX_WINDOW_S):
        hi = min(lo + MAX_WINDOW_S, end_ts)
        _pace()
        body = http_get(CLOB_PRICES_URL, params={
            "market": token_id, "startTs": lo, "endTs": hi, "fidelity": fidelity,
        }).json()
        points.extend((int(h["t"]), float(h["p"])) for h in body.get("history", []))
    return points


def _read_cache(path: Path) -> tuple[int, int, list[tuple[int, float]]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return int(data["start"]), int(data["end"]), [(int(t), float(p)) for t, p in data["points"]]


def _series(points: list[tuple[int, float]]) -> pd.Series:
    times = pd.to_datetime([t for t, _ in points], unit="s", utc=True)
    s = pd.Series([p for _, p in points], index=times, dtype=float, name="price")
    s = s.sort_index(kind="stable")
    s = s[~s.index.duplicated(keep="first")]
    s.index.name = "time"
    return s


def load_price_history(token_id: str, start_ts: int, end_ts: int, fidelity: int = 60) -> pd.Series:
    """YES price history of ``token_id`` between two unix times, both inclusive.

    The cache holds one span per token and fidelity. A request inside the cached span makes
    no network call. Otherwise the requested span is fetched in windows of at most 14 days
    and merged into the cache when it overlaps or touches the cached span.
    """
    if end_ts <= start_ts:
        raise ValueError("end_ts must be greater than start_ts")
    path = cache_dir("polymarket", "prices") / f"{token_id}-{fidelity}.json"
    cached = _read_cache(path)
    if cached is None or not (cached[0] <= start_ts and end_ts <= cached[1]):
        fetched = _fetch_windows(token_id, start_ts, end_ts, fidelity)
        lo, hi, points = start_ts, end_ts, fetched
        if cached is not None and cached[0] <= end_ts and start_ts <= cached[1]:
            lo, hi = min(cached[0], start_ts), max(cached[1], end_ts)
            points = cached[2] + fetched
        body = json.dumps({"start": lo, "end": hi, "points": points})
        _write_atomic(path, body.encode())
        cached = (lo, hi, points)
    series = _series(cached[2])
    lo_ts = pd.to_datetime(start_ts, unit="s", utc=True)
    hi_ts = pd.to_datetime(end_ts, unit="s", utc=True)
    return series[(series.index >= lo_ts) & (series.index <= hi_ts)]


def _clip(values: pd.Series) -> np.ndarray:
    return np.clip(values.to_numpy(dtype=np.float64), PRICE_FLOOR, PRICE_CEIL)


def load_market(market: pd.Series | dict, bar: str = "1h") -> pd.DataFrame:
    """Binary bar frame of a listed market's YES price, from its start to its end.

    ``market`` is a row of ``list_resolved_markets``. A bar with no price point repeats the
    previous close in all four prices. Bars before the first point are dropped. Raises
    ``ValueError`` when the market lacks start or end, or when fewer than 48 bars remain.
    """
    row = dict(market)
    start, end = _utc(row["start"]), _utc(row["end"])
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"market {row['slug']!r} needs start and end times")
    prices = load_price_history(str(row["yes_token"]), int(start.timestamp()),
                                int(end.timestamp()))
    ohlc = prices.resample(bar, closed="left", label="left").ohlc()
    ohlc = ohlc.loc[ohlc["close"].notna().cumsum() > 0]
    close = ohlc["close"].ffill()
    open_ = ohlc["open"].fillna(close)
    high = ohlc["high"].fillna(close)
    low = ohlc["low"].fillna(close)
    if len(close) < MIN_BARS:
        raise ValueError(f"market {row['slug']!r} has {len(close)} bars, need {MIN_BARS}")
    frame = make_frame(
        close.index, _clip(open_), _clip(high), _clip(low), _clip(close),
        symbol=str(row["slug"])[:60], kind="binary", bar=bar,
        resolution=float(row["resolution"]),
    )
    # Only the scheduled end may feed features; the closing time would leak early resolutions.
    scheduled = row.get("scheduled_end")
    if scheduled is not None and not pd.isna(scheduled):
        frame.attrs["end_time"] = _utc(scheduled).isoformat()
    return frame
