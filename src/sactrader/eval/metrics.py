"""Performance summary for a return series or a backtest.

Sharpe and Sortino divide the arithmetic per-bar mean by a per-bar dispersion,
then scale by sqrt(bars_per_year). Standard deviations use ddof=1. A series with
no dispersion gives 0.0 for every ratio, so a flat strategy still yields a full
row instead of inf or nan.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from sactrader.eval.backtest import BacktestResult

# A standard deviation below this multiple of the largest absolute return is
# rounding residue of a constant series, not dispersion.
DEGENERATE_REL = 1e-12


def dispersion_ok(x: np.ndarray, sd: np.ndarray, axis: int = -1) -> np.ndarray:
    """True where ``sd`` is large enough to divide by."""
    return sd > DEGENERATE_REL * np.abs(x).max(axis=axis)


def sharpe(returns, bars_per_year: float, axis: int = -1) -> np.ndarray:
    """Annualized mean over annualized std along ``axis``. Zero where dispersion is absent."""
    x = np.asarray(returns, dtype=np.float64)
    mean = x.mean(axis=axis)
    sd = x.std(axis=axis, ddof=1)
    ok = dispersion_ok(x, sd, axis=axis)
    ratio = np.zeros_like(mean)
    np.divide(mean, sd, out=ratio, where=ok)
    return ratio * np.sqrt(bars_per_year)


def summarize(res_or_returns, bars_per_year: float) -> dict[str, float]:
    """Metric row for a ``BacktestResult`` or a 1-D array of per-bar returns."""
    is_result = isinstance(res_or_returns, BacktestResult)
    r = np.asarray(res_or_returns.returns if is_result else res_or_returns, dtype=np.float64)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("summarize needs a 1-D series of at least 2 returns")
    if not np.isfinite(r).all():
        raise ValueError("returns must be finite")

    bpy = float(bars_per_year)
    n = r.size
    eq = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    final = float(eq[-1])
    years = n / bpy
    cagr = final ** (1.0 / years) - 1.0 if final > 0.0 else -1.0

    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / peak - 1.0))
    calmar = cagr / abs(max_dd) if max_dd < 0.0 else 0.0

    sd = r.std(ddof=1)
    ann_vol = float(sd * np.sqrt(bpy))

    # Sortino uses the downside deviation below zero over all bars, not the std.
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    if dispersion_ok(r, np.asarray(downside)):
        sortino = float(r.mean() / downside * np.sqrt(bpy))
    else:
        sortino = 0.0

    nonzero = r[r != 0.0]
    hit_rate = float(np.mean(nonzero > 0.0)) if nonzero.size else 0.0

    if dispersion_ok(r, np.asarray(sd)):
        skew = float(stats.skew(r))
        kurt = float(stats.kurtosis(r))
    else:
        skew = kurt = 0.0

    out = {
        "total_return": final - 1.0,
        "cagr": float(cagr),
        "ann_vol": ann_vol,
        "sharpe": float(sharpe(r, bpy)),
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "hit_rate": hit_rate,
        "skew": skew,
        "kurtosis": kurt,
        "n_bars": float(n),
    }
    if is_result:
        exp = res_or_returns.exposure
        out.update(
            avg_exposure=float(np.mean(exp)),
            avg_abs_exposure=float(np.mean(np.abs(exp))),
            turnover_per_year=float(np.mean(res_or_returns.turnover) * bpy),
            cost_per_year=float(np.mean(res_or_returns.cost) * bpy),
            carry_per_year=float(np.mean(res_or_returns.carry) * bpy),
            n_liquidations=float(np.sum(res_or_returns.liquidated)),
        )
    return out
