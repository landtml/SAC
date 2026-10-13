"""Causal regime indicators: whether recent price action trends, mean-reverts, jumps or skews.

``regime_features`` returns, for each window ``n`` in ``RegimeConfig.windows``, five columns
over the last ``n`` one-bar log returns ``r`` of ``close``:

* ``vr_{n}``: log variance ratio of overlapping 4-bar sums to one-bar returns. Above zero
  trends, below zero mean-reverts.
* ``er_{n}``: Kaufman efficiency ratio ``|sum r| / sum |r|``, mapped from [0, 1] to [-1, 1].
* ``ac1_{n}``: lag-1 autocorrelation times ``sqrt(n)``, a z-score under no autocorrelation.
* ``volvol_{n}``: std of ``log(|r| + floor)`` minus its Gaussian value. Zero is Gaussian-like,
  positive means jumps.
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

from sactrader.data.schema import validate_frame

# Added inside the log of |r| so an exact zero return stays finite, as in causal.py.
_FLOOR = 1e-8
# A window counts as flat when its return variance is at most this share of its mean square.
# A share, not an absolute value, so the test does not depend on price level or bar size.
_FLAT_SHARE = 1e-12
# A 4-bar variance needs at least two overlapping sums, so shorter windows are rejected.
_MIN_WINDOW = 8


@dataclass(frozen=True)
class RegimeConfig:
    windows: tuple[int, ...] = (16, 64)  # rolling windows in bars
    clip: float = 5.0


def _gaussian_log_abs_std() -> float:
    """Std of log|Z| for Z ~ N(0, 1), by quadrature.

    Integrating in u = log|Z| keeps the log singularity at zero off the grid. The density of u is
    sqrt(2 / pi) exp(u - exp(2u) / 2), which is negligible outside [-60, 6].
    """
    u = np.linspace(-60.0, 6.0, 400_001)
    du = u[1] - u[0]
    w = np.sqrt(2.0 / np.pi) * np.exp(u - 0.5 * np.exp(2.0 * u))
    mass = w.sum() * du
    mean = (u * w).sum() * du / mass
    second = (u * u * w).sum() * du / mass
    return float(np.sqrt(second - mean * mean))


_GAUSS_LOG_ABS_STD = _gaussian_log_abs_std()


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
    sum4 = r.rolling(4).sum()
    log_abs = np.log(abs_r + _FLOOR)

    cols: dict[str, pd.Series] = {}
    for n in cfg.windows:
        var_r = r.rolling(n).var()
        mean_sq = (r * r).rolling(n).mean()
        ready = var_r.notna()
        flat = (var_r <= _FLAT_SHARE * mean_sq) | (mean_sq <= 0.0)
        stats = {
            "vr": np.log((sum4.rolling(n - 3).var() / (4.0 * var_r)).clip(lower=_FLOOR)),
            "er": 2.0 * (r.rolling(n).sum().abs() / abs_r.rolling(n).sum() - 0.5),
            "ac1": r.rolling(n - 1).corr(r.shift(1)) * np.sqrt(n),
            "volvol": log_abs.rolling(n).std() - _GAUSS_LOG_ABS_STD,
            "skew": r.rolling(n).skew() / np.sqrt(6.0 / n),
        }
        for name, x in stats.items():
            # Inside a full window, a flat window or an undefined statistic becomes 0.
            # Warm-up rows stay NaN.
            bad = flat | (ready & ~np.isfinite(x))
            cols[f"{name}_{n}"] = x.mask(bad, 0.0).clip(-cfg.clip, cfg.clip).astype(np.float32)
    return pd.DataFrame(cols, index=frame.index)
