"""Directional signals, costs and statistics for daily return panels.

Inputs are wide frames of daily simple returns (dates by assets). A weights frame holds the
position in each asset decided at the close of day t. Day t+1 earns those weights, so every
signal reads data up to and including t. Costs are charged per unit of turnover, in basis
points, on the day after the trade.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from softedge.eval.metrics import sharpe
from softedge.eval.stats import deflated_sharpe_ratio, null_sr_variance

TRADING_DAYS = 252
N_TRIALS = 11
TARGET_VOL = 0.10
MAX_EXPOSURE = 2.0
VOL_WINDOW = 63
ZSCORE_MIN_PERIODS = 252
BURN_IN_YEARS = 10
RIDGE_ALPHA = 1.0
NW_LAGS = 5
MIN_ASSETS = 30
PASS_NW_T = 2.5
PASS_DSR = 0.95


def ex_ante_vol(returns: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """Trailing standard deviation over ``window`` days, annualised. NaN until a full window."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def trailing_return(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Compound return over the ``lookback`` days ending at t, NaN until the asset has that many returns.

    A missing day adds no return, so the compound return covers the observed days only.
    """
    growth = (1.0 + returns.fillna(0.0)).cumprod()
    past = growth / growth.shift(lookback) - 1.0
    observed = returns.notna().cumsum() > lookback
    return past.where(observed)


def ts_momentum_weights(returns: pd.DataFrame, lookback: int, *,
                        vol_window: int = VOL_WINDOW) -> pd.DataFrame:
    """Time-series momentum: sign of the trailing return times a 10% vol target, capped at 2x.

    Each asset's exposure is ``sign * TARGET_VOL / ex-ante vol``. The weight is that exposure
    divided by the number of assets, so the panel holds equal risk per asset.
    """
    past = trailing_return(returns, lookback)
    vol = ex_ante_vol(returns, vol_window)
    exposure = (np.sign(past) * TARGET_VOL / vol).clip(-MAX_EXPOSURE, MAX_EXPOSURE)
    return exposure.fillna(0.0) / returns.shape[1]


def combine_equal_weight(*weights: pd.DataFrame) -> pd.DataFrame:
    """Mean of several weights frames on a common index and columns."""
    stacked = sum(weights[1:], weights[0].copy())
    return stacked / len(weights)


def quintile_weights(signal: pd.DataFrame, *, min_assets: int = MIN_ASSETS) -> pd.DataFrame:
    """Dollar-neutral long-short on the top and bottom fifth of each day's cross-section.

    The top ``n // 5`` assets by ``signal`` get weight ``+1/k`` and the bottom ``k`` get
    ``-1/k``, so each leg sums to one. A day with fewer than ``min_assets`` valid signals holds
    no position. Ties resolve by column order.
    """
    values = signal.to_numpy(dtype=np.float64)
    out = np.zeros(values.shape)
    for i in range(values.shape[0]):
        row = values[i]
        valid = np.flatnonzero(np.isfinite(row))
        n = valid.size
        if n < min_assets:
            continue
        k = n // 5
        order = valid[np.argsort(row[valid], kind="stable")]
        out[i, order[-k:]] = 1.0 / k
        out[i, order[:k]] = -1.0 / k
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)


def causal_zscore(frame: pd.DataFrame, min_periods: int = ZSCORE_MIN_PERIODS) -> pd.DataFrame:
    """Each column standardised by its expanding mean and std up to and including t."""
    mean = frame.expanding(min_periods=min_periods).mean()
    sd = frame.expanding(min_periods=min_periods).std(ddof=1)
    return (frame - mean) / sd


def ridge_features(returns: pd.DataFrame) -> np.ndarray:
    """Per-asset features shaped (dates, assets, features), each z-scored causally per asset.

    Features are compound returns over 1, 5, 21, 63 and 252 days, and daily standard deviations
    over 21 and 63 days.
    """
    parts = [trailing_return(returns, n) for n in (1, 5, 21, 63, 252)]
    parts.append(returns.rolling(21, min_periods=21).std(ddof=1))
    parts.append(returns.rolling(63, min_periods=63).std(ddof=1))
    z = [causal_zscore(p) for p in parts]
    return np.stack([p.to_numpy(dtype=np.float64) for p in z], axis=-1)


def vol_scaled_next_return(returns: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """Next-day return divided by the daily std of the last ``window`` days, stamped on day t."""
    daily_vol = returns.rolling(window, min_periods=window).std(ddof=1)
    return returns.shift(-1) / daily_vol


def next_day_rank(returns: pd.DataFrame) -> pd.DataFrame:
    """Next-day cross-sectional percentile rank, centred on zero, stamped on day t."""
    return returns.shift(-1).rank(axis=1, pct=True) - 0.5


def refit_positions(dates: pd.DatetimeIndex, burn_in_years: int = BURN_IN_YEARS) -> list[int]:
    """Row positions where the ridge is refitted: the first row after burn-in, then each new year."""
    burn = dates[0] + pd.DateOffset(years=burn_in_years)
    first = int(dates.searchsorted(burn))
    positions = [first]
    for i in range(first + 1, len(dates)):
        if dates[i].year != dates[i - 1].year:
            positions.append(i)
    return positions


def walk_forward_ridge(features: np.ndarray, target: pd.DataFrame | np.ndarray,
                       dates: pd.DatetimeIndex, *, burn_in_years: int = BURN_IN_YEARS,
                       alpha: float = RIDGE_ALPHA) -> pd.DataFrame:
    """Expanding-window ridge predictions, refitted yearly after the burn-in.

    Training at row ``d`` uses rows ``s < d``. The target at row ``s`` is the return of day
    ``s + 1``, so every training target is known at the close of day ``d``. The intercept and
    feature means come from the training rows and are not penalised. Rows before the first
    refit are NaN.
    """
    columns = target.columns if isinstance(target, pd.DataFrame) else None
    y_all = target.to_numpy(dtype=np.float64) if isinstance(target, pd.DataFrame) else target
    t_len, n_assets, n_feat = features.shape
    pred = np.full((t_len, n_assets), np.nan)
    refits = refit_positions(dates, burn_in_years)
    for j, d in enumerate(refits):
        end = refits[j + 1] if j + 1 < len(refits) else t_len
        x_tr = features[:d].reshape(-1, n_feat)
        y_tr = y_all[:d].reshape(-1)
        ok = np.isfinite(x_tr).all(axis=1) & np.isfinite(y_tr)
        x_tr, y_tr = x_tr[ok], y_tr[ok]
        if y_tr.size <= n_feat:
            continue
        mx = x_tr.mean(axis=0)
        my = y_tr.mean()
        xc = x_tr - mx
        w = np.linalg.solve(xc.T @ xc + alpha * np.eye(n_feat), xc.T @ (y_tr - my))
        pred[d:end] = (features[d:end] - mx) @ w + my
    return pd.DataFrame(pred, index=dates, columns=columns)


def ridge_time_series_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """H3a: ridge on vol-scaled next-day returns, sized by its prediction relative to its trailing std."""
    features = ridge_features(returns)
    target = vol_scaled_next_return(returns)
    pred = walk_forward_ridge(features, target, returns.index)
    scale = pred.expanding(min_periods=ZSCORE_MIN_PERIODS).std(ddof=1)
    signal = (pred / scale).clip(-MAX_EXPOSURE, MAX_EXPOSURE)
    vol = ex_ante_vol(returns)
    exposure = (signal * TARGET_VOL / vol).fillna(0.0)
    return exposure / returns.shape[1]


def ridge_cross_section_weights(returns: pd.DataFrame, *,
                                min_assets: int = MIN_ASSETS) -> pd.DataFrame:
    """H3b: ridge on next-day cross-sectional rank, long the top quintile and short the bottom."""
    features = ridge_features(returns)
    pred = walk_forward_ridge(features, next_day_rank(returns), returns.index)
    return quintile_weights(pred, min_assets=min_assets)


def portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame, cost_bp: pd.Series, *,
                      delay: int = 0) -> pd.Series:
    """Daily net returns. Day t earns the weights of day t-1-delay, less costs on each trade.

    A change in weights at close t costs ``|change| * cost_bp / 1e4`` per asset, summed, and is
    charged to day t+1. ``delay`` holds each position for that many extra days before it earns.
    Missing returns earn nothing.
    """
    w = weights.shift(delay).fillna(0.0)
    r = returns.fillna(0.0)
    prev = w.shift(1).fillna(0.0)
    gross = (prev * r).sum(axis=1)
    cost = ((w - prev).abs() * cost_bp.reindex(w.columns).to_numpy() / 1e4).sum(axis=1)
    net = gross - cost.shift(1).fillna(0.0)
    return net.rename("net")


def newey_west_variance_of_mean(x, lags: int = NW_LAGS) -> float:
    """Newey-West (Bartlett) variance of the sample mean of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    xc = x - x.mean()
    lrv = float(xc @ xc) / n
    for j in range(1, lags + 1):
        gamma = float(xc[j:] @ xc[:-j]) / n
        lrv += 2.0 * (1.0 - j / (lags + 1.0)) * gamma
    return lrv / n


def newey_west_t(x, lags: int = NW_LAGS) -> float:
    """Mean over its Newey-West standard error. NaN without dispersion."""
    x = np.asarray(x, dtype=np.float64)
    var = newey_west_variance_of_mean(x, lags)
    if not np.isfinite(var) or var <= 0.0:
        return float("nan")
    return float(x.mean() / np.sqrt(var))


def min_detectable_sharpe(n_days: int, *, bars_per_year: float = TRADING_DAYS,
                          alpha: float = 0.05, power: float = 0.8) -> float:
    """Annualised Sharpe whose two-sided test at ``alpha`` has ``power`` over ``n_days``."""
    years = n_days / bars_per_year
    return float((stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)) / np.sqrt(years))


def deflated_sharpe(returns, n_trials: int = N_TRIALS) -> float:
    """Deflated Sharpe ratio from ``softedge.eval.stats`` with the null cross-trial variance."""
    r = np.asarray(returns, dtype=np.float64)
    return float(deflated_sharpe_ratio(r, n_trials, null_sr_variance(r.size)))


def evaluate(weights: pd.DataFrame, returns: pd.DataFrame, cost_bp: pd.Series, *,
             delay: int = 0, n_trials: int = N_TRIALS) -> dict:
    """Net-of-cost statistics over the sample from the first day a position earns.

    Keys: ``sharpe`` (annualised), ``nw_t``, ``dsr``, ``sharpe_first_half``,
    ``sharpe_second_half``, ``n_days``, ``min_detectable_sharpe``, ``start`` and ``passes``.
    ``passes`` is True when NW t > 2.5, DSR > 0.95 and the Sharpe is positive in both halves.
    """
    net = portfolio_returns(weights, returns, cost_bp, delay=delay)
    held = weights.shift(delay).fillna(0.0).shift(1).fillna(0.0)
    earning = (held != 0.0).any(axis=1)
    if not earning.any():
        return {"sharpe": float("nan"), "nw_t": float("nan"), "dsr": float("nan"),
                "sharpe_first_half": float("nan"), "sharpe_second_half": float("nan"),
                "n_days": 0, "min_detectable_sharpe": float("nan"), "start": None,
                "passes": False}
    start = earning.idxmax()
    r = net.loc[start:].to_numpy(dtype=np.float64)
    half = r.size // 2
    sr_all = float(sharpe(r, TRADING_DAYS))
    sr1 = float(sharpe(r[:half], TRADING_DAYS))
    sr2 = float(sharpe(r[half:], TRADING_DAYS))
    nw = newey_west_t(r)
    dsr = float(deflated_sharpe_ratio(r, n_trials, null_sr_variance(r.size)))
    passes = bool(nw > PASS_NW_T and dsr > PASS_DSR and sr1 > 0.0 and sr2 > 0.0)
    return {
        "sharpe": sr_all, "nw_t": nw, "dsr": dsr,
        "sharpe_first_half": sr1, "sharpe_second_half": sr2,
        "n_days": int(r.size), "min_detectable_sharpe": min_detectable_sharpe(r.size),
        "start": str(start.date()), "passes": passes,
    }


def null_random_walk_panels(returns: pd.DataFrame, n_panels: int = 5,
                            seed: int = 0) -> list[pd.DataFrame]:
    """Zero-mean Gaussian return panels with each asset's own vol, on the same dates and gaps."""
    rng = np.random.default_rng(seed)
    vol = returns.std(ddof=1).to_numpy(dtype=np.float64)
    panels = []
    for _ in range(n_panels):
        sim = rng.standard_normal(returns.shape) * vol
        panels.append(pd.DataFrame(sim, index=returns.index, columns=returns.columns)
                      .where(returns.notna()))
    return panels


HYPOTHESES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "H1a": lambda r: ts_momentum_weights(r, 21),
    "H1b": lambda r: ts_momentum_weights(r, 63),
    "H1c": lambda r: ts_momentum_weights(r, 252),
    "H1d": lambda r: combine_equal_weight(
        ts_momentum_weights(r, 21), ts_momentum_weights(r, 63), ts_momentum_weights(r, 252)),
    "H2a": lambda r: quintile_weights(-r, min_assets=MIN_ASSETS),
    "H2b": lambda r: quintile_weights(trailing_return(r, 7), min_assets=MIN_ASSETS),
    "H2c": lambda r: quintile_weights(trailing_return(r, 28), min_assets=MIN_ASSETS),
    "H3a": ridge_time_series_weights,
    "H3b": ridge_cross_section_weights,
}
