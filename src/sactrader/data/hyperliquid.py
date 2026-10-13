"""Hyperliquid perpetuals: the market list, candles and funding, from the public info API.

Source: ``POST https://api.hyperliquid.xyz/info`` with a JSON body and no keys. Requests go out
at most two per second. A reply with 429 or 5xx is retried with backoff, five tries in all.

``list_perps`` returns the meta universe as ``coin``, ``max_leverage`` and ``sz_decimals``.
Delisted coins are included; the reply flags them with ``isDelisted``, which this table leaves out.

``load_funding`` returns the funding events of one coin as a float64 Series named
``funding_rate``, indexed by UTC event time. Events arrive hourly with millisecond jitter. The
sign is the API's ``fundingRate`` sign, unchanged.

``load_perp`` returns a perp bar frame (see ``sactrader.data.schema``) with ``funding_rate`` per
bar: the sum of the events inside ``[bar open, bar open + interval)``, 0.0 for a bar without one.
It adds the float64 column ``trades``. Bars are indexed by candle open time in UTC.

History cap: the API serves only recent candles and nothing older. Measured on 2026-10-13, 1h
candles reach back to 2026-03-19 09:00 UTC (5001 bars), 4h to 2024-07-02 12:00 UTC (5000 bars)
and 1d to 2020-08-19 00:00 UTC (2247 bars, the full history served).

Cache under ``$SACTRADER_DATA/hyperliquid``. Each calendar month that has ended is kept as raw
JSON rows in ``raw/<coin>/<kind>/<YYYY-MM>.json``, written atomically. The parsed perp frame is
kept as parquet in ``frames`` once its end month has ended. The month in progress is fetched on
every call and never cached.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from sactrader.data.cache import RETRY_STATUS, USER_AGENT, cache_dir, load_frame, save_frame
from sactrader.data.schema import make_frame

INFO_URL = "https://api.hyperliquid.xyz/info"
TRIES = 5
MIN_GAP_S = 0.5
BACKOFF_S = 2.0
# A calendar month of candles has to fit in one reply, which holds about 5000 candles. A
# shorter interval would be cut short without an error, so it is refused up front.
MIN_INTERVAL = pd.Timedelta("15min")
_PANDAS_UNITS = {"m": "min", "h": "h", "d": "D"}
_COIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_:.-]*")
_last_request = 0.0


def list_perps() -> pd.DataFrame:
    """Perpetuals in the meta universe: ``coin``, ``max_leverage`` and ``sz_decimals``."""
    universe = _post_json({"type": "meta"})["universe"]
    return pd.DataFrame({
        "coin": [u["name"] for u in universe],
        "max_leverage": np.array([u["maxLeverage"] for u in universe], dtype=np.int64),
        "sz_decimals": np.array([u["szDecimals"] for u in universe], dtype=np.int64),
    })


def load_funding(coin: str, start: str, end: str) -> pd.Series:
    """Funding events of ``coin`` over the inclusive months ``start``..``end``, as ``YYYY-MM``."""
    _check_coin(coin)
    return _funding_events(coin, _months(start, end), _now_ms())


def load_perp(coin: str, interval: str = "1h", start: str = "2023-06",
              end: str = "2026-09") -> pd.DataFrame:
    """Perp bar frame of ``coin`` over the inclusive months ``start``..``end``.

    ``interval`` is a fixed length of 15 minutes or more, such as ``1h``, ``4h`` or ``1d``;
    anything else raises ``ValueError`` before any request. The history cap is in the module
    docstring. A 1h frame starts at the earliest candle the API serves, whatever ``start`` says,
    and a range with no candles at all raises ``ValueError``.
    """
    _check_coin(coin)
    alias = _bar_alias(interval)
    months = _months(start, end)
    path = _frame_path(coin, interval, start, end)
    if path.exists():
        return load_frame(path)
    now = _now_ms()
    bars = _candles(coin, interval, months, now)
    if bars.empty:
        raise ValueError(f"no candles for {coin} {interval} in {start}..{end}")
    rate = _funding_per_bar(bars.index, _funding_events(coin, months, now), pd.Timedelta(alias))
    frame = make_frame(
        bars.index,
        bars["open"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy(),
        bars["close"].to_numpy(), bars["volume"].to_numpy(),
        symbol=coin, kind="perp", bar=alias, funding_rate=rate,
    )
    frame["trades"] = bars["trades"].to_numpy()
    if _month_bounds(end)[1] <= now:
        save_frame(frame, path)
    return frame


def _post_json(body: dict[str, Any]) -> Any:
    """POST one info request and return the decoded reply, retrying 429 and 5xx."""
    last: Exception | None = None
    for attempt in range(TRIES):
        _pace()
        try:
            resp = requests.post(INFO_URL, json=body, timeout=60,
                                 headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            last = exc
        else:
            if resp.status_code not in RETRY_STATUS:
                resp.raise_for_status()
                return resp.json()
            last = requests.HTTPError(f"{resp.status_code} for {INFO_URL}", response=resp)
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                time.sleep(float(retry_after))
                continue
        if attempt < TRIES - 1:
            time.sleep(BACKOFF_S * 2**attempt)
    raise RuntimeError(f"POST failed after {TRIES} tries: {INFO_URL}") from last


def _pace() -> None:
    global _last_request
    wait = MIN_GAP_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _fetch_candles(coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    req = {"coin": coin, "interval": interval, "startTime": start_ms, "endTime": end_ms}
    return _post_json({"type": "candleSnapshot", "req": req})


def _fetch_funding(coin: str, start_ms: int, end_ms: int) -> list[dict]:
    """Every funding event in ``[start_ms, end_ms]``, paged forward from the last event time."""
    rows: list[dict] = []
    cursor = start_ms
    while cursor <= end_ms:
        page = _post_json({"type": "fundingHistory", "coin": coin, "startTime": cursor,
                           "endTime": end_ms})
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1]["time"]) + 1
        # A reply that does not move the cursor forward would page the same events forever.
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return rows


def _raw_month(coin: str, kind: str, month: str, fetch: Callable[[int, int], list[dict]],
               now_ms: int) -> list[dict]:
    """Raw rows of one calendar month. An ended month is read from and written to JSON."""
    start_ms, end_ms = _month_bounds(month)
    if end_ms > now_ms:
        return fetch(start_ms, end_ms - 1)
    dest = cache_dir("hyperliquid", "raw", coin, kind) / f"{month}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    rows = fetch(start_ms, end_ms - 1)
    _write_json(dest, rows)
    return rows


def _write_json(dest: Path, rows: list[dict]) -> None:
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_text(json.dumps(rows))
    tmp.replace(dest)


def _candles(coin: str, interval: str, months: list[str], now_ms: int) -> pd.DataFrame:
    fetch = partial(_fetch_candles, coin, interval)
    rows: list[dict] = []
    for month in months:
        rows.extend(_raw_month(coin, f"candles-{interval}", month, fetch, now_ms))
    return _clean_bars(_parse_candles(rows, now_ms))


def _funding_events(coin: str, months: list[str], now_ms: int) -> pd.Series:
    fetch = partial(_fetch_funding, coin)
    rows: list[dict] = []
    for month in months:
        rows.extend(_raw_month(coin, "funding", month, fetch, now_ms))
    return _parse_funding(rows)


def _parse_candles(rows: list[dict], now_ms: int) -> pd.DataFrame:
    """Candle rows as float columns indexed by open time, without the candle still open."""
    closed = [r for r in rows if int(r["T"]) < now_ms]
    opens_ms = np.array([int(r["t"]) for r in closed], dtype=np.int64)
    return pd.DataFrame(
        {
            "open": _floats(closed, "o"),
            "high": _floats(closed, "h"),
            "low": _floats(closed, "l"),
            "close": _floats(closed, "c"),
            "volume": _floats(closed, "v"),
            "trades": _floats(closed, "n"),
        },
        index=_utc_index(opens_ms * 1_000_000),
    )


def _parse_funding(rows: list[dict]) -> pd.Series:
    times_ms = np.array([int(r["time"]) for r in rows], dtype=np.int64)
    events = pd.Series(_floats(rows, "fundingRate"), index=_utc_index(times_ms * 1_000_000),
                       name="funding_rate")
    return events[~events.index.duplicated(keep="last")].sort_index()


def _clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(axis=1)]
    # The schema requires high and low to bound open and close, so the wicks are widened.
    oc_high = np.maximum(bars["open"].to_numpy(), bars["close"].to_numpy())
    oc_low = np.minimum(bars["open"].to_numpy(), bars["close"].to_numpy())
    return bars.assign(high=np.maximum(bars["high"].to_numpy(), oc_high),
                       low=np.minimum(bars["low"].to_numpy(), oc_low))


def _funding_per_bar(index: pd.DatetimeIndex, events: pd.Series,
                     width: pd.Timedelta) -> np.ndarray:
    opens = index.as_unit("ns").asi8
    times = events.index.as_unit("ns").asi8
    pos = np.searchsorted(opens, times, side="right") - 1
    inside = (pos >= 0) & (times < opens[np.clip(pos, 0, None)] + width.value)
    return np.bincount(pos[inside], weights=events.to_numpy(dtype=np.float64)[inside],
                       minlength=len(opens))


def _bar_alias(interval: str) -> str:
    """Pandas offset alias of a supported interval: ``4h`` stays ``4h``, ``1d`` becomes ``1D``."""
    count, unit = interval[:-1], interval[-1:]
    if not count.isdigit() or unit not in _PANDAS_UNITS:
        raise ValueError(f"unsupported interval {interval!r}: fixed lengths only, such as 1h")
    alias = f"{int(count)}{_PANDAS_UNITS[unit]}"
    if pd.Timedelta(alias) < MIN_INTERVAL:
        raise ValueError(f"interval {interval!r} is below 15m: a month of candles would not fit")
    return alias


def _check_coin(coin: str) -> None:
    # The name becomes a directory name in the cache, so separators and dots are refused.
    if not _COIN.fullmatch(coin):
        raise ValueError(f"unsupported coin name {coin!r}")


def _frame_path(coin: str, interval: str, start: str, end: str) -> Path:
    return cache_dir("hyperliquid", "frames") / f"perp-{coin}-{interval}-{start}-{end}.parquet"


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


def _month_bounds(month: str) -> tuple[int, int]:
    """Epoch milliseconds of the first instant of ``month`` and of the instant after it."""
    year, mon = map(int, month.split("-"))
    nxt_year, nxt_mon = (year + 1, 1) if mon == 12 else (year, mon + 1)
    return (_epoch_ms(dt.datetime(year, mon, 1, tzinfo=dt.UTC)),
            _epoch_ms(dt.datetime(nxt_year, nxt_mon, 1, tzinfo=dt.UTC)))


def _epoch_ms(moment: dt.datetime) -> int:
    # Month boundaries fall on whole seconds, so the float conversion is exact.
    return int(moment.timestamp()) * 1000


def _now_ms() -> int:
    return int(time.time() * 1000)


def _floats(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def _utc_index(ns: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(ns.astype("datetime64[ns]")).tz_localize("UTC")
