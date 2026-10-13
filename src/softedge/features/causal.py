"""Causal, scale-free features for bar frames of every market kind.

Row ``t`` depends only on frame rows ``<= t``. Each price-based column divides a
log return or a log price gap by a volatility estimate, so multiplying all prices
by a constant leaves it unchanged. Bar shape and time codes are bounded by
construction. Every column is clipped to ``[-clip, clip]``.

The leading rows may hold NaN, at most ``warmup_bars(cfg)`` of them. Every later
row is finite. The output keeps the frame index, uses float32 columns and a fixed
column order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from softedge.data.schema import validate_frame

_FLOOR = 1e-8
_BINARY_LAGS = (1, 4, 24)
_FUNDING_EWM_SPAN = 64
_TAKER_SPAN = 8
_PREMIUM_EWM_SPAN = 8
_PARKINSON = 4.0 * np.log(2.0)


@dataclass(frozen=True)
class FeatureConfig:
    vol_span: int = 32  # EWMA span of squared log returns for sigma_t
    vol_long_span: int = 512
    horizons: tuple[int, ...] = (1, 4, 16, 64, 256)
    ema_spans: tuple[int, ...] = (16, 64, 256)
    volume_span: int = 64
    clip: float = 5.0


def warmup_bars(cfg: FeatureConfig | None = None) -> int:
    """Longest lookback across kinds, which bounds the leading NaN rows."""
    cfg = cfg or FeatureConfig()
    return max(1, *cfg.horizons, *_BINARY_LAGS)


def _is_intraday(bar: str) -> bool:
    try:
        return to_offset(bar).nanos < pd.Timedelta("1D").value
    except ValueError:  # calendar offsets such as months have no fixed length
        return False


def compute_features(frame: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Return the causal feature table for ``frame``, one row per bar."""
    cfg = cfg or FeatureConfig()
    validate_frame(frame)
    kind = frame.attrs["kind"]
    index = frame.index
    open_ = frame["open"].astype("float64")
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    close = frame["close"].astype("float64")
    volume = frame["volume"].astype("float64")

    log_close = np.log(close)
    ret1 = log_close.diff()
    sq_ret = ret1 * ret1
    sigma = np.sqrt(sq_ret.ewm(span=cfg.vol_span, adjust=False).mean()).clip(lower=_FLOOR)
    sigma_long = np.sqrt(sq_ret.ewm(span=cfg.vol_long_span, adjust=False).mean())
    sigma_long = sigma_long.clip(lower=_FLOOR)

    cols: dict[str, pd.Series | np.ndarray] = {}
    for h in cfg.horizons:
        cols[f"ret_h{h}"] = (log_close - log_close.shift(h)) / (sigma * np.sqrt(h))
    for s in cfg.ema_spans:
        ema = log_close.ewm(span=s, adjust=False).mean()
        cols[f"ema_gap_{s}"] = (log_close - ema) / (sigma * np.sqrt(s))
    cols["vol_ratio"] = np.log(sigma / sigma_long)

    # A flat bar (high == low) has a zero range; the floor keeps its log finite.
    log_range = np.log(high / low).clip(lower=_FLOOR)
    range_var = (log_range * log_range / _PARKINSON).ewm(span=cfg.vol_span, adjust=False).mean()
    cols["range_vol_ratio"] = np.log(np.sqrt(range_var) / sigma)

    bar_range = high - low
    cols["body"] = ((close - open_) / bar_range.where(bar_range > 0.0)).fillna(0.0)

    if np.isfinite(volume).any():
        x = np.log1p(volume)
        vol_ewm = x.ewm(span=cfg.volume_span, adjust=False, ignore_na=True)
        # The first observation has no spread yet, so its z-score is zero.
        vol_std = vol_ewm.std().fillna(0.0).clip(lower=_FLOOR)
        cols["volume_z"] = ((x - vol_ewm.mean()) / vol_std).where(volume.notna(), 0.0)

    if "taker_buy_base" in frame.columns:
        taker = frame["taker_buy_base"].astype("float64")
        # A bar without volume has no buy share, so it leaves the average unchanged.
        share = taker / volume.where(volume > 0.0) - 0.5
        avg = share.ewm(span=_TAKER_SPAN, adjust=False, ignore_na=True).mean()
        cols["taker_imbalance"] = avg.fillna(0.0) * 10.0

    if kind == "perp":
        funding_bps = frame["funding_rate"].astype("float64") * 1e4
        # Zero marks a bar with no funding event, so the last nonzero rate is in force.
        carried = funding_bps.where(funding_bps != 0.0).ffill().fillna(0.0)
        cols["funding_last"] = carried
        cols["funding_ewm"] = carried.ewm(span=_FUNDING_EWM_SPAN, adjust=False).mean()

    if _is_intraday(frame.attrs["bar"]):
        seconds = np.asarray(index.hour * 3600 + index.minute * 60 + index.second, dtype=float)
        day_turn = 2.0 * np.pi * seconds / 86400.0
        week_turn = 2.0 * np.pi * np.asarray(index.dayofweek, dtype=float) / 7.0
        cols["hour_sin"] = np.sin(day_turn)
        cols["hour_cos"] = np.cos(day_turn)
        cols["dow_sin"] = np.sin(week_turn)
        cols["dow_cos"] = np.cos(week_turn)

    if kind == "binary":
        logit = np.log(close / (1.0 - close)) / 4.0
        cols["logit"] = logit
        for k in _BINARY_LAGS:
            cols[f"dlogit_{k}"] = logit - logit.shift(k)
        end = frame.attrs.get("end_time")
        if end is not None:
            end_ts = pd.Timestamp(end)
            end_ts = (end_ts.tz_localize("UTC") if end_ts.tzinfo is None
                      else end_ts.tz_convert("UTC"))
            bar_close = index + to_offset(frame.attrs["bar"])
            hours = np.asarray((end_ts - bar_close) / pd.Timedelta(hours=1), dtype=float)
            # A last bar that closes after the scheduled end has no time left to run.
            cols["log_hours_to_end"] = np.log1p(np.maximum(hours, 0.0)) / 5.0

    if "implied_vol" in frame.columns:
        # Implied minus realized volatility (the variance risk premium) and the implied trend.
        # Values before the implied series starts are neutral zeros.
        # Bars per year from the trailing mean bar spacing, so weekday-only daily data
        # annualizes with about 260 bars and 24/7 markets with 365 per day.
        spacing = pd.Series(index.as_unit("ns").asi8, index=index).diff() / 1e9
        spacing = spacing.fillna(pd.Timedelta(frame.attrs["bar"]).total_seconds())
        bars_per_year = 365.25 * 86400.0 / spacing.rolling(256, min_periods=1).mean()
        iv = frame["implied_vol"].astype("float64").clip(lower=_FLOOR)
        cols["iv_rv"] = np.log(iv / (sigma * np.sqrt(bars_per_year))).fillna(0.0)
        cols["iv_change"] = np.log(iv / iv.shift(cfg.horizons[1])).fillna(0.0)

    if "premium" in frame.columns:
        # The perp premium over its index sets the next funding payment, so it leads funding.
        prem_bps = frame["premium"].astype("float64").fillna(0.0) * 1e4
        cols["premium_bps"] = prem_bps
        cols["premium_ewm"] = prem_bps.ewm(span=_PREMIUM_EWM_SPAN, adjust=False).mean()

    if "rate_diff" in frame.columns:
        # 5% differential maps to 1.0, the carry signal behind forex carry trades.
        cols["rate_diff"] = frame["rate_diff"].astype("float64").fillna(0.0) * 20.0

    if "signal" in frame.columns:
        cols["signal"] = frame["signal"].astype("float64")

    out = pd.DataFrame(cols, index=index)
    return out.clip(lower=-cfg.clip, upper=cfg.clip).astype(np.float32)
