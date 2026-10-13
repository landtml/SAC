"""Causal regime indicators: whether recent price action trends, mean-reverts, jumps or skews.

``regime_features`` returns, for each window ``n`` in ``RegimeConfig.windows``, five columns
over the last ``n`` one-bar log returns ``r`` of ``close``:

* ``vr_{n}``: log variance ratio of overlapping 4-bar sums to one-bar returns. Above zero
  trends, below zero mean-reverts.
* ``er_{n}``: Kaufman efficiency ratio ``|sum r| / sum |r|``, mapped from [0, 1] to [-1, 1].
* ``ac1_{n}``: lag-1 autocorrelation times ``sqrt(n)``, a z-score under no autocorrelation.
* ``volvol_{n}``: std of ``log(max(|r|, 0.1 * mean |r|))`` over the window, minus its Gaussian
  value. The floor scales with the window, so exact-zero returns stay finite and do not dominate.
  Zero is approximately Gaussian-like, with a small offset; positive means jumps.
* ``skew_{n}``: adjusted sample skewness divided by ``sqrt(6 / n)``, a z-score under normality.

Row ``t`` depends only on closes up to bar ``t``. Returns and ratios of returns do not change
when prices are multiplied by a constant. Binary frames use the logit of the price in place of
the log price, since their prices lie in ``(0, 1)``. A window with zero variance gives 0 in every
column. Leading rows may be NaN, at most ``regime_warmup_bars(cfg)`` of them, and every later
row is finite. Columns are clipped to ``[-clip, clip]``, float32, on the frame index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from sactrader.data.schema import validate_frame

# Lower bound on the variance ratio before its log, so a degenerate ratio stays finite.
_FLOOR = 1e-8
# A window counts as flat when its return variance is at most this share of its mean square.
# A share, not an absolute value, so the test does not depend on price level or bar size.
_FLAT_SHARE = 1e-12
# Volvol floor as a share of the window's mean |r|. Scaling with the window keeps the statistic
# scale-free, and an exact zero return gets a finite log instead of log(0).
_VOLVOL_FLOOR_SHARE = 0.1
# Windows are processed in blocks of about this many entries, which bounds peak memory.
_BLOCK_ELEMENTS = 1 << 20
# A 4-bar variance needs at least two overlapping sums, so shorter windows are rejected.
_MIN_WINDOW = 8


@dataclass(frozen=True)
class RegimeConfig:
    windows: tuple[int, ...] = (16, 64)  # rolling windows in bars
    clip: float = 5.0


def _gaussian_floored_log_std() -> float:
    """Std of log(max(|Z|, s * E|Z|)) for Z ~ N(0, 1), with s the volvol floor share.

    Integrating in u = log|Z| keeps the log singularity at zero off the grid. The density of u is
    sqrt(2 / pi) exp(u - exp(2u) / 2), which is negligible outside [-60, 6]. A window floors at its
    own mean |r|, which is close to E|Z| on Gaussian returns, so E|Z| stands in for it here.
    """
    u = np.linspace(-60.0, 6.0, 400_001)
    du = u[1] - u[0]
    w = np.sqrt(2.0 / np.pi) * np.exp(u - 0.5 * np.exp(2.0 * u))
    x = np.maximum(u, np.log(_VOLVOL_FLOOR_SHARE * np.sqrt(2.0 / np.pi)))
    mass = w.sum() * du
    mean = (x * w).sum() * du / mass
    second = (x * x * w).sum() * du / mass
    return float(np.sqrt(second - mean * mean))


_GAUSS_FLOORED_LOG_STD = _gaussian_floored_log_std()


def _floored_log_std(abs_r: np.ndarray, n: int) -> np.ndarray:
    """Std of log(max(|r|, 0.1 * mean |r|)) over each window of ``n`` returns.

    ``abs_r`` holds |r| for bars 1 onward, one entry fewer than the frame. Element ``t`` of the
    result uses the returns ending at bar ``t``, and is NaN before the first full window.
    """
    out = np.full(abs_r.size + 1, np.nan)
    if abs_r.size < n:
        return out
    windows = sliding_window_view(abs_r, n)  # row k covers bars k + 1 through k + n
    stats = np.empty(len(windows))
    step = max(1, _BLOCK_ELEMENTS // n)
    for start in range(0, len(windows), step):
        block = windows[start:start + step]
        # The tiny floor only matters for an all-zero window, which the caller zeroes as flat.
        floor = np.maximum(_VOLVOL_FLOOR_SHARE * block.mean(axis=1, keepdims=True),
                           np.finfo(float).tiny)
        stats[start:start + len(block)] = np.log(np.maximum(block, floor)).std(axis=1, ddof=1)
    out[n:] = stats
    return out


def regime_warmup_bars(cfg: RegimeConfig | None = None) -> int:
    """Longest window, which bounds the leading NaN rows."""
    cfg = cfg or RegimeConfig()
    return max(cfg.windows)


def regime_features(frame: pd.DataFrame, cfg: RegimeConfig | None = None) -> pd.DataFrame:
    """Return the regime table for ``frame``, one row per bar, float32 columns."""
    cfg = cfg or RegimeConfig()
    validate_frame(frame)
    if not cfg.windows or min(cfg.windows) < _MIN_WINDOW:
        raise ValueError(f"regime windows must be non-empty and at least {_MIN_WINDOW} bars")

    close = frame["close"].astype("float64")
    if frame.attrs["kind"] == "binary":
        level = np.log(close / (1.0 - close))
    else:
        level = np.log(close)
    r = level.diff()
    abs_r = r.abs()
    abs_valid = abs_r.to_numpy()[1:]  # bar 0 has no return
    sum4 = r.rolling(4).sum()

    cols: dict[str, pd.Series] = {}
    for n in cfg.windows:
        var_r = r.rolling(n).var()
        mean_sq = (r * r).rolling(n).mean()
        ready = var_r.notna()
        flat = (var_r <= _FLAT_SHARE * mean_sq) | (mean_sq <= 0.0)
        volvol_raw = pd.Series(_floored_log_std(abs_valid, n), index=frame.index)
        stats = {
            "vr": np.log((sum4.rolling(n - 3).var() / (4.0 * var_r)).clip(lower=_FLOOR)),
            "er": 2.0 * (r.rolling(n).sum().abs() / abs_r.rolling(n).sum() - 0.5),
            "ac1": r.rolling(n - 1).corr(r.shift(1)) * np.sqrt(n),
            "volvol": volvol_raw - _GAUSS_FLOORED_LOG_STD,
            "skew": r.rolling(n).skew() / np.sqrt(6.0 / n),
        }
        for name, x in stats.items():
            # Inside a full window, a flat window or an undefined statistic becomes 0.
            # Warm-up rows stay NaN.
            bad = flat | (ready & ~np.isfinite(x))
            cols[f"{name}_{n}"] = x.mask(bad, 0.0).clip(-cfg.clip, cfg.clip).astype(np.float32)
    return pd.DataFrame(cols, index=frame.index)
