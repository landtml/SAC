"""From named universes to train / validation / test tapes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd

from softedge.features.calibration import CalibConfig, calib_edge
from softedge.features.causal import FeatureConfig, compute_features, warmup_bars
from softedge.tape import DEFAULT_COSTS, CostSpec, MarketTape, build_tape

CRYPTO_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
)
FX_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF")
FRED_MARKET_NAMES = ("SPX", "NDX", "NKY", "WTI", "BRENT", "UST10", "UST2")
# Emerging-market pairs with a short-rate series on both sides (THB and SGD have none on FRED).
EM_CARRY_PAIRS = ("USDMXN", "USDZAR", "USDBRL", "USDINR", "USDKRW", "USDCNY")
# Hyperliquid funding history starts in May 2023.
HYPERLIQUID_START = "2023-06"
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
                min_volume: float = 1e6, implied_vol: bool = False,
                premium: bool = False, roll_days: int | None = None,
                roll_cost: float | None = None, open_interest: bool = False,
                family: str = "all", seed_offset: int = 0) -> list[pd.DataFrame]:
    """Bar frames for ``crypto_perp``, ``crypto_spot``, ``crypto_basis``, ``fx``, ``macro``,
    ``polymarket``, ``kalshi``, ``pegs`` or ``synthetic``.
    ``polymarket``, ``kalshi``, ``vrp`` or ``synthetic``. ``vrp`` holds daily short-variance
    carry frames on BTC and ETH (``softedge.data.variance``).

    ``fx_em`` holds emerging-market dollar pairs with their short-rate differentials, where the
    carry premium is larger than among G10 pairs. ``crypto_basis`` is the cash-and-carry spread of each symbol (long spot, short perp), a
    carry trade whose return is mostly funding. ``hl_basis`` is the same trade against
    Hyperliquid's perp (long Binance spot, short Hyperliquid perp) and ``perp_spread`` the
    cross-venue funding spread (long Binance perp, short Hyperliquid perp), both daily from
    mid-2023. ``delivery_basis`` is long spot against the front quarterly future of Binance
    (BTC and ETH), daily from 2021 (see ``softedge.data.delivery``). ``macro`` holds daily equity indices, oil and Treasury excess-return indices from
    FRED. ``implied_vol`` joins Deribit's DVOL onto crypto frames (ETH's for ETH, BTC's for
    every other coin, the crypto market's implied volatility; for ``vrp_spx`` it selects the
    variance risk premium prior of ``softedge.data.vrp``). ``premium`` joins the Binance perp
    mid-2023. ``macro`` holds daily equity indices, oil and Treasury excess-return indices from
    FRED. ``macro_fomc`` holds the equity indices (SPX, NDX, NKY by default) with the column
    ``fomc_next``, the scheduled-announcement flag of ``softedge.data.fomc``. ``implied_vol``
    joins Deribit's DVOL onto crypto frames (ETH's for ETH, BTC's for
    every other coin, the crypto market's implied volatility). ``premium`` joins the Binance perp
    premium index, which leads the next funding payment, onto Binance perp legs.
    ``open_interest`` attaches Binance's USD open interest (``softedge.data.metrics``) to
    ``crypto_basis`` frames, for the squeeze gate of ``softedge.features.squeeze``.

    ``synthetic`` is a set of hourly spot markets with a planted, observable return signal
    (column ``signal``) from ``start``; ``synthetic_null`` the same markets as random walks with no
    edge; ``synthetic_hard`` markets with hidden regimes, GARCH volatility, Student-t shocks and a
    weak nonlinear edge (``nonlinear_regime_market``). They need no network and check that the agent finds an edge only where one exists.
    ``synthetic_bench`` holds the benchmark markets of ``softedge.data.bench``: ``family`` picks
    one family or ``"all"``, and market ``i`` is seeded ``seed_offset + i``.
    """
    if universe == "synthetic_bench":
        from softedge.data.bench import bench_frames

        n = int((pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta("1h"))
        first = f"{start}-01" if len(start) == 7 else start
        names = list(symbols or [f"B{i}" for i in range(8)])
        return bench_frames(names, n, family=family, seed_offset=seed_offset, start=first)
    if universe in ("synthetic", "synthetic_null", "synthetic_hard"):
        from softedge.data.synthetic import gbm, nonlinear_regime_market, planted_signal

        n = int((pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta("1h"))
        names = list(symbols or [f"SIG{i}" for i in range(4)])
        first = f"{start}-01" if len(start) == 7 else start
        if universe == "synthetic_null":
            return [gbm(n, sigma=0.01, seed=i, symbol=name, start=first)
                    for i, name in enumerate(names)]
        if universe == "synthetic_hard":
            return [nonlinear_regime_market(n, seed=i, symbol=name, start=first)
                    for i, name in enumerate(names)]
        return [planted_signal(n, strength=0.1, sigma=0.01, rho=0.9, seed=i, symbol=name,
                               start=first)
                for i, name in enumerate(names)]
    if universe in ("crypto_perp", "crypto_spot"):
        from softedge.data import binance

        load = binance.load_perp if universe == "crypto_perp" else binance.load_spot
        frames = [load(s, "1h", start, end) for s in (symbols or CRYPTO_SYMBOLS)]
        if premium and universe == "crypto_perp":
            frames = [_with_premium(f) for f in frames]
        if implied_vol:
            from softedge.data.deribit import attach_implied_vol

            frames = [attach_implied_vol(f, "ETH" if f.attrs["symbol"].startswith("ETH") else "BTC")
                      for f in frames]
        return frames
    if universe in ("hl_basis", "perp_spread"):
        from softedge.data import binance, hyperliquid
        from softedge.data.basis import spread_frame

        hl_start = max(start, HYPERLIQUID_START)
        listed = set(hyperliquid.list_perps()["coin"])
        frames = []
        for s in symbols or CRYPTO_SYMBOLS:
            coin = s.removesuffix("USDT")
            if coin not in listed:
                continue
            short = hyperliquid.load_perp(coin, "1d", hl_start, end)
            long_load = binance.load_spot if universe == "hl_basis" else binance.load_perp
            long = long_load(s, "1h", start, end)
            if premium and universe == "perp_spread":
                long = _with_premium(long)
            long = resample_frame(long, "1D")
            tag = "HLBASIS" if universe == "hl_basis" else "XVENUE"
            frames.append(spread_frame(long, short, f"{coin}-{tag}"))
        return frames
    if universe == "crypto_basis":
        from softedge.data import binance
        from softedge.data.basis import basis_frame

        perp = binance.load_perp
        frames = []
        for s in (symbols or CRYPTO_SYMBOLS):
            perp_leg = perp(s, "1h", start, end)
            basis = basis_frame(binance.load_spot(s, "1h", start, end),
                                _with_premium(perp_leg) if premium else perp_leg)
            # Open interest is attached hourly, so a daily resample keeps the last 23:55 sample.
            frames.append(_with_open_interest(basis, s) if open_interest else basis)
        return frames
    if universe == "pegs":
        from softedge.data.pegs import load_peg_frames

        return load_peg_frames(start, end, symbols)
    if universe == "delivery_basis":
        from softedge.data.delivery import DELIVERY_SYMBOLS, load_delivery_basis

        # Unset roll settings keep the loader's defaults.
        rolls = {}
        if roll_days is not None:
            rolls["roll_days"] = roll_days
        if roll_cost is not None:
            rolls["roll_cost"] = roll_cost
        return load_delivery_basis(symbols or DELIVERY_SYMBOLS, start, end, **rolls)
    if universe == "vrp_spx":
        from softedge.data.vrp import load_vrp_frame

        return [load_vrp_frame(start, end, prior=implied_vol)]
    if universe == "macro":
        from softedge.data.fred_markets import load_fred_market

        frames = [load_fred_market(m, start, end) for m in (symbols or FRED_MARKET_NAMES)]
        if implied_vol:
            from softedge.data.vol_indices import attach_fred_implied_vol

            frames = [attach_fred_implied_vol(f) for f in frames]
        return frames
    if universe == "macro_fomc":
        from softedge.data.fomc import attach_fomc

        names = symbols or ("SPX", "NDX", "NKY")
        return [attach_fomc(f) for f in load_frames("macro", start, end, symbols=names,
                                                    implied_vol=implied_vol)]
    if universe == "fx_em":
        from softedge.data import fx, rates

        return [rates.attach_rate_diff(fx.load_fx_daily(p, start=f"{start}-01"))
                for p in (symbols or EM_CARRY_PAIRS)]
    if universe == "fx":
        from softedge.data import fx, rates

        return [rates.attach_rate_diff(fx.load_fx_daily(p, start=f"{start}-01"))
                for p in (symbols or FX_PAIRS)]
    if universe == "vrp":
        from softedge.data.variance import vrp_frames

        return vrp_frames(start, end, tuple(symbols or ("BTC", "ETH")))
    if universe == "kalshi":
        from softedge.data import kalshi

        markets = kalshi.list_settled_markets(min_volume=min_volume, max_markets=max_markets)
        frames = []
        for _, row in markets.iterrows():
            try:
                frames.append(_with_resolved_at(kalshi.load_market(row), row))
            except (ValueError, FileNotFoundError, RuntimeError):
                continue  # too short or no candles
        return frames
    if universe == "polymarket":
        from softedge.data import polymarket

        markets = polymarket.list_resolved_markets(min_volume=min_volume, max_markets=max_markets)
        frames = []
        for _, row in markets.iterrows():
            try:
                frames.append(_with_resolved_at(polymarket.load_market(row), row))
            except (ValueError, FileNotFoundError, RuntimeError):
                continue  # too short, no start date, or no price history
        return frames
    raise ValueError(f"unknown universe {universe!r}")


def _with_resolved_at(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    # The actual close is when the outcome became public. Calibration may use it at later bars;
    # features of the market itself never see it (the market is unresolved at its own bars).
    frame.attrs["resolved_at"] = pd.Timestamp(row["end"]).isoformat()
    return frame


def _with_premium(frame: pd.DataFrame) -> pd.DataFrame:
    from softedge.data.premium import attach_premium

    return attach_premium(frame)


def _with_open_interest(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    from softedge.data.metrics import attach_open_interest

    return attach_open_interest(frame, symbol)


def to_tapes(frames: list[pd.DataFrame], costs: CostSpec | Callable[[pd.DataFrame], CostSpec] | None = None,
             feature_cfg: FeatureConfig | None = None, cross: bool = False,
             regime: bool = False, calib: CalibConfig | None = None,
             premia: bool = False) -> list[MarketTape]:
    """Tapes from frames. ``cross`` adds cross-asset context (``features.cross``) to every
    non-binary frame when there are at least three of them; ``regime`` adds the regime
    indicators of ``features.regime`` to every frame; ``calib`` adds the ``calib_edge`` column
    of ``features.calibration`` to every binary frame; ``premia`` adds the premium features of
    ``features.premia`` to every non-binary frame."""
    kept, tables = [], []
    for fr in frames:
        kind = fr.attrs["kind"]
        cfg = feature_cfg or (BINARY_FEATURES if kind == "binary" else FeatureConfig())
        if len(fr) <= warmup_bars(cfg) + 16:
            continue
        kept.append(fr)
        table = compute_features(fr, cfg)
        if regime:
            from softedge.features.regime import regime_features

            table = pd.concat([table, regime_features(fr)], axis=1)
        if premia:
            from softedge.features.premia import premia_features

            table = pd.concat([table, premia_features(fr)], axis=1)
        tables.append(table)
    if calib is not None:
        binary = [i for i, fr in enumerate(kept) if fr.attrs["kind"] == "binary"]
        for i, edge in zip(binary, calib_edge([kept[i] for i in binary], calib), strict=True):
            tables[i] = pd.concat([tables[i], edge], axis=1)
    linear = [i for i, fr in enumerate(kept) if fr.attrs["kind"] != "binary"]
    if cross and len(linear) >= 3:
        from softedge.features.cross import cross_features

        ctx = cross_features({str(i): tables[i] for i in linear})
        for i in linear:
            tables[i] = pd.concat([tables[i], ctx[str(i)]], axis=1).clip(-5.0, 5.0)
    tapes = []
    for fr, feats in zip(kept, tables, strict=True):
        spec = costs(fr) if callable(costs) else (costs or DEFAULT_COSTS[fr.attrs["kind"]])
        tapes.append(build_tape(fr, feats, spec))
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
