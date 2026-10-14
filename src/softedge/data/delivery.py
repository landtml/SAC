"""Dated (quarterly) USD-M futures against spot: a cash-and-carry spread that converges at expiry.

A quarterly future expires at a fixed date and settles to spot, so its premium over spot
decays to zero. Holding spot against a short future earns that premium as it decays. This
module builds one bar frame per coin with the same contract as ``softedge.data.basis``: long one
unit of spot and short one unit of the front quarterly future, rolled to the next contract
``roll_days`` before expiry.

Contracts are named ``<BASE>_<YYMMDD>`` and expire at 08:00 UTC on the last Friday of March,
June, September or December. Their daily klines come from the public archive under
``futures/um/monthly/klines/<SYMBOL>/1d``. Spot comes from the hourly spot cache, resampled to
daily bars so both legs close at 00:00 UTC.

The chained future price moves with each held contract's own return. A roll is traded at the
close that decides it, so its cost, ``roll_cost`` (two trades of one futures unit at the normal
perp fee and half spread), lands on the step that follows the roll. The frame carries that cost
on the short-future leg only, so it hits the long side of the spread; the tape gives the short
side the mirror return. A symmetric roll cost needs a tape change and is not modelled here.
When the successor has no bar yet, the expiring contract is held until it does. The roll
window and roll cost are this module's defaults: the ``costs`` and roll settings of a config do
not reach the frame builder yet.

``basis_ann`` is the prior feature: the held future's close over spot, minus one, annualized over
the days left to expiry. It uses only the close of its own bar and the expiry date.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from softedge.data.basis import spread_frame
from softedge.data.binance import _clean, _fetch_zip, _parse_klines, _read_zip
from softedge.data.cache import cache_dir
from softedge.data.schema import make_frame
from softedge.tape import DEFAULT_COSTS

DELIVERY_SYMBOLS = ("BTCUSDT", "ETHUSDT")
DEFAULT_ROLL_DAYS = 7
# Two trades of one futures unit (close the old contract, open the new) at the normal perp fee.
DEFAULT_ROLL_COST = 2 * (DEFAULT_COSTS["perp"].fee_bps + DEFAULT_COSTS["perp"].half_spread_bps) * 1e-4
EXPIRY_HOUR_UTC = 8
_QUARTER_MONTHS = (3, 6, 9, 12)
_KLINES = "futures/um/monthly/klines"


@dataclass(frozen=True)
class Contract:
    """One quarterly contract and the archive months that hold its daily klines."""

    symbol: str
    expiry: pd.Timestamp
    months: tuple[str, ...]


def expiry_of(symbol: str) -> pd.Timestamp:
    """Expiry of a ``<BASE>_<YYMMDD>`` contract: that date at 08:00 UTC."""
    tag = symbol.rsplit("_", 1)[-1] if "_" in symbol else ""
    if len(tag) != 6 or not tag.isdigit():
        raise ValueError(f"{symbol!r} has no YYMMDD expiry suffix")
    day = pd.Timestamp(f"20{tag[:2]}-{tag[2:4]}-{tag[4:]}", tz="UTC")
    return day + pd.Timedelta(hours=EXPIRY_HOUR_UTC)


def quarterly_expiries(first: str, last: str) -> list[pd.Timestamp]:
    """Quarterly expiries whose month lies in ``first..last`` (``YYYY-MM``, inclusive)."""
    lo, hi = pd.Period(first, "M"), pd.Period(last, "M")
    out = []
    for year in range(lo.year, hi.year + 1):
        for month in _QUARTER_MONTHS:
            m = pd.Period(f"{year}-{month:02d}", "M")
            if not lo <= m <= hi:
                continue
            month_end = m.to_timestamp(how="end").normalize()
            friday = month_end - pd.Timedelta(days=(month_end.weekday() - 4) % 7)
            out.append(friday.tz_localize("UTC") + pd.Timedelta(hours=EXPIRY_HOUR_UTC))
    return out


def front_chain(bars: dict[str, pd.DataFrame], expiries: dict[str, pd.Timestamp], *,
                roll_days: int = DEFAULT_ROLL_DAYS, roll_cost: float = DEFAULT_ROLL_COST
                ) -> pd.DataFrame:
    """Chained price of the front contract, with the held contract on each bar.

    ``bars`` maps each contract to a daily frame with ``close`` and ``volume``; ``expiries``
    gives each contract's expiry. The front on a bar is the earliest-expiring contract with a
    close on that bar and more than ``roll_days`` left. Its return over the next bar is applied
    to ``close`` (starting at 1.0). A roll adds ``roll_cost`` to that next step. Returns columns
    ``close``, ``volume``, ``held``, ``held_close`` and ``days_left``, from the first bar with a
    front contract.
    """
    if set(bars) != set(expiries):
        raise ValueError("bars and expiries must name the same contracts")
    syms = sorted(bars, key=lambda s: expiries[s])
    index = bars[syms[0]].index
    for s in syms[1:]:
        index = index.union(bars[s].index)
    n = len(index)
    close = np.column_stack([bars[s]["close"].reindex(index).to_numpy(np.float64) for s in syms])
    volume = np.column_stack([bars[s]["volume"].reindex(index).to_numpy(np.float64) for s in syms])
    close_time = index.as_unit("ns").asi8 + pd.Timedelta("1D").value
    expiry_ns = np.array([expiries[s].as_unit("ns").value for s in syms], dtype=np.int64)
    left = (expiry_ns[None, :] - close_time[:, None]) / pd.Timedelta("1D").value

    listed = np.isfinite(close)
    primary = listed & (left > roll_days)
    # A successor can list after its roll window opens. Then the expiring contract is held
    # until the successor has a bar: it still converges to spot.
    spare = listed & (left > 0.0)
    held = np.where(primary.any(axis=1), primary.argmax(axis=1),
                    np.where(spare.any(axis=1), spare.argmax(axis=1), -1))
    live = np.flatnonzero(held >= 0)
    if live.size == 0:
        raise ValueError("no front contract in the window")
    first = int(live[0])
    changed = np.zeros(n, dtype=bool)
    changed[1:] = held[1:] != held[:-1]

    price = np.full(n, np.nan)
    price[first] = 1.0
    for i in range(first, n - 1):
        h = int(held[i])
        if h < 0:
            raise ValueError(f"no front contract on {index[i].date()}")
        nxt = close[i + 1, h]
        if not np.isfinite(nxt):
            raise ValueError(f"no bar of {syms[h]} on {index[i + 1].date()}: cannot carry "
                             "the position")
        drag = roll_cost if i > first and changed[i] else 0.0
        price[i + 1] = price[i] * (nxt / close[i, h] + drag)

    rows = np.arange(n)
    pick = np.where(held >= 0, held, 0)
    names = np.array(syms, dtype=object)[pick]
    out = pd.DataFrame(
        {
            "close": price,
            "volume": volume[rows, pick],
            "held": names,
            "held_close": close[rows, pick],
            "days_left": left[rows, pick],
        },
        index=index,
    )
    keep = np.isfinite(price) & (held >= 0)
    return out[keep]


def delivery_basis_frame(spot: pd.DataFrame, bars: dict[str, pd.DataFrame],
                         expiries: dict[str, pd.Timestamp], *, symbol: str,
                         roll_days: int = DEFAULT_ROLL_DAYS,
                         roll_cost: float = DEFAULT_ROLL_COST) -> pd.DataFrame:
    """Long spot, short the chained front future, as a perp-kind frame with ``basis_ann``.

    See the module docstring. ``basis_ann`` is known at each bar's close.
    """
    chain = front_chain(bars, expiries, roll_days=roll_days, roll_cost=roll_cost)
    price = chain["close"].to_numpy(np.float64)
    fut = make_frame(chain.index, price, price, price, price,
                     chain["volume"].to_numpy(np.float64), symbol=f"{symbol}-FUTURE",
                     kind="perp", bar="1D", funding_rate=np.zeros(len(chain)))
    frame = spread_frame(spot, fut, symbol)
    spot_close = spot["close"].reindex(frame.index).to_numpy(np.float64)
    held_close = chain["held_close"].reindex(frame.index).to_numpy(np.float64)
    days_left = chain["days_left"].reindex(frame.index).to_numpy(np.float64)
    frame["basis_ann"] = (held_close / spot_close - 1.0) * 365.0 / days_left
    return frame


def contract_symbol(base: str, expiry: pd.Timestamp) -> str:
    return f"{base}_{expiry.strftime('%y%m%d')}"


def list_contracts(base: str, start: str, end: str) -> list[Contract]:
    """Quarterly contracts of ``base`` with daily klines in the archive for ``start..end``.

    Probes the archive one month at a time, newest first, and stops at the first month the
    archive lacks. The result is cached as a manifest so later runs do not probe again.
    """
    manifest = cache_dir("binance", "delivery") / f"{base}-{start}-{end}.json"
    if manifest.exists():
        rows = json.loads(manifest.read_text())
        return [Contract(r["symbol"], pd.Timestamp(r["expiry"]), tuple(r["months"]))
                for r in rows]
    first = pd.Period(start, "M")
    last = pd.Period(end, "M")
    found = []
    for expiry in quarterly_expiries(start, str(last + 12)):
        sym = contract_symbol(base, expiry)
        m = min(pd.Period(expiry.strftime("%Y-%m"), "M"), last)
        months = []
        while m >= first:
            if _fetch_zip(f"{_KLINES}/{sym}/1d", f"{sym}-1d-{m}.zip") is None:
                break
            months.append(str(m))
            m -= 1
        if months:
            found.append(Contract(sym, expiry, tuple(sorted(months))))
    manifest.write_text(json.dumps([
        {"symbol": c.symbol, "expiry": c.expiry.isoformat(), "months": list(c.months)}
        for c in found
    ]))
    return found


def load_contract(contract: Contract) -> pd.DataFrame:
    """Daily klines of one contract, from its archive months."""
    rel = f"{_KLINES}/{contract.symbol}/1d"
    parts = []
    for month in contract.months:
        body = _fetch_zip(rel, f"{contract.symbol}-1d-{month}.zip")
        if body is not None:
            parts.append(_parse_klines(_read_zip(body)))
    if not parts:
        raise ValueError(f"no daily klines for {contract.symbol}")
    return _clean(pd.concat(parts))


def load_delivery_basis(symbols=DELIVERY_SYMBOLS, start: str = "2020-01", end: str = "2026-09",
                        *, roll_days: int = DEFAULT_ROLL_DAYS,
                        roll_cost: float = DEFAULT_ROLL_COST) -> list[pd.DataFrame]:
    """One long-spot, short-front-quarterly frame per spot symbol (``BTCUSDT`` gives ``BTC-DELIV``)."""
    from softedge.data import binance
    from softedge.pipeline import resample_frame

    frames = []
    for sym in symbols:
        contracts = list_contracts(sym, start, end)
        if not contracts:
            raise ValueError(f"no quarterly contracts of {sym} in {start}..{end}")
        bars = {c.symbol: load_contract(c) for c in contracts}
        expiries = {c.symbol: c.expiry for c in contracts}
        spot = resample_frame(binance.load_spot(sym, "1h", start, end), "1D")
        frames.append(delivery_basis_frame(spot, bars, expiries,
                                           symbol=f"{sym.removesuffix('USDT')}-DELIV",
                                           roll_days=roll_days, roll_cost=roll_cost))
    return frames
