"""Calibration edge of binary markets: the outcome rate a price has earned, minus the price.

Contract. For each bar ``t`` of a binary frame, ``calib_edge[t] = f_t(price[t]) - price[t]``.
``f_t`` maps a price to the share of YES outcomes among markets resolved strictly before the
open of bar ``t``. Each market counts once, whatever its length. The price axis is cut into equal
bins, and each bin's outcome share is shrunk toward the query price, so a bin with few markets
stays near zero. With fewer than ``min_markets`` resolved markets the feature is zero.

Resolution time is ``attrs["resolved_at"]``, the market's actual close, which the loaders set.
Outcomes are ``attrs["resolution"]``. A market is unresolved at each of its own bars, so its
outcome never enters its own feature.

Row ``t`` depends only on the prices and outcomes of markets resolved before ``t``, so the output
is causal. Output is float32, on each frame's own index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibConfig:
    min_markets: int = 50  # resolved markets needed before the feature leaves zero
    bins: int = 10  # equal-width price bins on [0, 1]
    shrink: float = 5.0  # pseudo-markets pulling a bin's outcome share toward the price

    def __post_init__(self) -> None:
        if self.min_markets < 1:
            raise ValueError("min_markets must be at least 1")
        if self.bins < 2:
            raise ValueError("bins must be at least 2")
        if self.shrink < 0.0:
            raise ValueError("shrink must be non-negative")


def _resolved_ns(frame: pd.DataFrame) -> np.int64:
    if "resolved_at" not in frame.attrs:
        raise ValueError("binary frame lacks attrs['resolved_at'], its resolution time")
    ts = pd.Timestamp(frame.attrs["resolved_at"])
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return np.int64(ts.as_unit("ns").value)


def _bin_of(price: np.ndarray, bins: int) -> np.ndarray:
    return np.clip(np.floor(price * bins).astype(np.int64), 0, bins - 1)


def calib_edge(frames: list[pd.DataFrame], cfg: CalibConfig | None = None) -> list[pd.Series]:
    """Calibration edge of each binary frame, one float32 series per frame on its index."""
    cfg = cfg or CalibConfig()
    for fr in frames:
        if fr.attrs.get("kind") != "binary":
            raise ValueError("calib_edge takes binary frames only")
    if not frames:
        return []
    bins = cfg.bins
    resolved = np.array([_resolved_ns(fr) for fr in frames], dtype=np.int64)
    # Each market's bars in each price bin, weighted so the market totals one.
    mass = np.zeros((len(frames), bins))
    yes = np.zeros((len(frames), bins))
    for j, fr in enumerate(frames):
        price = fr["close"].to_numpy(dtype=np.float64)
        mass[j] = np.bincount(_bin_of(price, bins), minlength=bins) / len(price)
        yes[j] = mass[j] * float(fr.attrs["resolution"])
    order = np.argsort(resolved, kind="stable")
    times = resolved[order]
    # Row k holds the totals of the k earliest-resolving markets.
    cum_mass = np.vstack([np.zeros((1, bins)), np.cumsum(mass[order], axis=0)])
    cum_yes = np.vstack([np.zeros((1, bins)), np.cumsum(yes[order], axis=0)])
    out = []
    for fr in frames:
        ns = fr.index.as_unit("ns").asi8
        price = fr["close"].to_numpy(dtype=np.float64)
        # side="left" counts only markets with resolution strictly before each bar.
        known = np.searchsorted(times, ns, side="left")
        b = _bin_of(price, bins)
        s_mass = cum_mass[known, b]
        s_yes = cum_yes[known, b]
        denom = s_mass + cfg.shrink
        # (s_yes + shrink * price) / denom - price, written so it stays finite with no markets.
        edge = np.divide(s_yes - s_mass * price, denom, out=np.zeros_like(price), where=denom > 0)
        edge = np.where(known >= cfg.min_markets, edge, 0.0)
        out.append(pd.Series(edge.astype(np.float32), index=fr.index, name="calib_edge"))
    return out
