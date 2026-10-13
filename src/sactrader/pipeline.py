"""From named universes to train / validation / test tapes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd

from sactrader.features.causal import FeatureConfig, compute_features, warmup_bars
from sactrader.tape import DEFAULT_COSTS, CostSpec, MarketTape, build_tape

CRYPTO_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
)
FX_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF")
# Binary markets live for weeks, not years: shorter horizons keep more of each market after warm-up.
BINARY_FEATURES = FeatureConfig(horizons=(1, 4, 16, 64), ema_spans=(16, 64), vol_long_span=128)


@dataclass(frozen=True)
class Period:
    start: str | None
    stop: str | None


@dataclass(frozen=True)
class SplitSpec:
    train: Period
    val: Period
    test: Period


def load_frames(universe: str, start: str = "2020-01", end: str = "2026-09",
                symbols: Iterable[str] | None = None, max_markets: int = 300,
                min_volume: float = 1e6) -> list[pd.DataFrame]:
    """Bar frames for ``crypto_perp``, ``crypto_spot``, ``fx``, ``polymarket`` or ``synthetic``.

    ``synthetic`` is a set of hourly spot markets with a planted, observable return signal
    (column ``signal``) from ``start``; ``synthetic_null`` the same markets as random walks with no
    edge. They need no network and check that the agent finds an edge only where one exists.
    """
    if universe in ("synthetic", "synthetic_null"):
        from sactrader.data.synthetic import gbm, planted_signal

        n = int((pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta("1h"))
        names = list(symbols or [f"SIG{i}" for i in range(4)])
        first = f"{start}-01" if len(start) == 7 else start
        if universe == "synthetic_null":
            return [gbm(n, sigma=0.01, seed=i, symbol=name, start=first)
                    for i, name in enumerate(names)]
        return [planted_signal(n, strength=0.1, sigma=0.01, rho=0.9, seed=i, symbol=name,
                               start=first)
                for i, name in enumerate(names)]
    if universe in ("crypto_perp", "crypto_spot"):
        from sactrader.data import binance

        load = binance.load_perp if universe == "crypto_perp" else binance.load_spot
        return [load(s, "1h", start, end) for s in (symbols or CRYPTO_SYMBOLS)]
    if universe == "fx":
        from sactrader.data import fx, rates

        return [rates.attach_rate_diff(fx.load_fx_daily(p, start=f"{start}-01"))
                for p in (symbols or FX_PAIRS)]
    if universe == "polymarket":
        from sactrader.data import polymarket

        markets = polymarket.list_resolved_markets(min_volume=min_volume, max_markets=max_markets)
        frames = []
        for _, row in markets.iterrows():
            try:
                frames.append(polymarket.load_market(row))
            except (ValueError, FileNotFoundError, RuntimeError):
                continue  # too short, no start date, or no price history
        return frames
    raise ValueError(f"unknown universe {universe!r}")


def to_tapes(frames: list[pd.DataFrame], costs: CostSpec | Callable[[pd.DataFrame], CostSpec] | None = None,
             feature_cfg: FeatureConfig | None = None) -> list[MarketTape]:
    tapes = []
    for fr in frames:
        kind = fr.attrs["kind"]
        cfg = feature_cfg or (BINARY_FEATURES if kind == "binary" else FeatureConfig())
        if len(fr) <= warmup_bars(cfg) + 16:
            continue
        spec = costs(fr) if callable(costs) else (costs or DEFAULT_COSTS[kind])
        tapes.append(build_tape(fr, compute_features(fr, cfg), spec))
    return tapes


def split_tapes(tapes: list[MarketTape], split: SplitSpec, min_rows: int = 64) -> dict[str, list[MarketTape]]:
    """Cut each tape by time. Binary markets go whole into the period that holds their last bar,
    so a market never straddles train and test."""
    out: dict[str, list[MarketTape]] = {"train": [], "val": [], "test": []}
    for tp in tapes:
        for name in out:
            period: Period = getattr(split, name)
            if tp.kind == "binary":
                last = pd.Timestamp(int(tp.times[-1]), tz="UTC")
                lo = pd.Timestamp(period.start, tz="UTC") if period.start else None
                hi = pd.Timestamp(period.stop, tz="UTC") if period.stop else None
                if (lo is None or last >= lo) and (hi is None or last < hi):
                    out[name].append(tp)
                continue
            try:
                part = tp.time_slice(period.start, period.stop)
            except ValueError:
                continue
            if len(part) >= min_rows:
                out[name].append(part)
    return out


def window_tapes(tapes: list[MarketTape], windows, min_rows: int = 64) -> list[MarketTape]:
    """Tapes restricted to a list of ``(start, stop)`` windows (UTC, half-open).

    A linear tape yields one slice per window it overlaps by at least ``min_rows`` rows. A binary
    market is kept whole when its last bar falls in one of the windows, as in ``split_tapes``.
    """
    out: list[MarketTape] = []
    for tp in tapes:
        if tp.kind == "binary":
            last = pd.Timestamp(int(tp.times[-1]), tz="UTC")
            if any(_utc(a) <= last < _utc(b) for a, b in windows):
                out.append(tp)
            continue
        for a, b in windows:
            try:
                part = tp.time_slice(a, b)
            except ValueError:
                continue
            if len(part) >= min_rows:
                out.append(part)
    return out


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


_SUM_COLUMNS = ("volume", "funding_rate", "quote_volume", "trades", "taker_buy_base")


def resample_frame(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate a linear bar frame to coarser bars: first open, max high, min low, last close,
    summed volume-like and funding columns, last value of other extras. Bars are labelled by
    their open time; a coarse bar with no fine bar is dropped."""
    if frame.attrs["kind"] == "binary":
        raise ValueError("resample binary markets at download time instead")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    for col in frame.columns:
        if col not in agg:
            agg[col] = "sum" if col in _SUM_COLUMNS else "last"
    out = frame.resample(rule, closed="left", label="left").agg(agg)
    out = out.dropna(subset=["close"])
    if "volume" in out and frame["volume"].isna().all():
        out["volume"] = float("nan")
    out.attrs = {**frame.attrs, "bar": rule}
    return out
