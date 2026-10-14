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
Z_95 = float(stats.norm.ppf(0.975))


def ex_ante_vol(returns: pd.DataFrame, window: int = VOL_WINDOW,
                bars_per_year: float = TRADING_DAYS) -> pd.DataFrame:
    """Trailing standard deviation over ``window`` days, annualised. NaN until a full window."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(bars_per_year)


def trailing_return(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Compound return over the ``lookback`` days ending at t, NaN until the asset has that many returns.

    A missing day adds no return, so the compound return covers the observed days only. A day
    with no print has no signal, so an asset that stops printing leaves the cross-section on
    its first missing day.
    """
    growth = (1.0 + returns.fillna(0.0)).cumprod()
    past = growth / growth.shift(lookback) - 1.0
    observed = returns.notna().cumsum() > lookback
    return past.where(observed & returns.notna())


def ts_momentum_weights(returns: pd.DataFrame, lookback: int, *,
                        vol_window: int = VOL_WINDOW,
                        bars_per_year: float = TRADING_DAYS) -> pd.DataFrame:
    """Time-series momentum: sign of the trailing return times a 10% vol target, capped at 2x.

    Each asset's exposure is ``sign * TARGET_VOL / ex-ante vol``. The weight is that exposure
    divided by the number of assets, so the panel holds equal risk per asset.
    """
    past = trailing_return(returns, lookback)
    vol = ex_ante_vol(returns, vol_window, bars_per_year)
    exposure = (np.sign(past) * TARGET_VOL / vol).clip(-MAX_EXPOSURE, MAX_EXPOSURE)
    return exposure.fillna(0.0) / returns.shape[1]


def combine_equal_weight(*weights: pd.DataFrame) -> pd.DataFrame:
    """Mean of several weights frames on a common index and columns."""
    stacked = sum(weights[1:], weights[0].copy())
    return stacked / len(weights)


def hold_on_missing(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Keep each asset's previous weight on a day it has no return print.

    A calendar gap, such as a holiday in one market, leaves that asset's position unchanged
    and takes no new signal that day. Before the first print the weight is zero.
    """
    return weights.where(returns.notna()).ffill().fillna(0.0)


def quintile_weights(signal: pd.DataFrame, *, min_assets: int = MIN_ASSETS) -> pd.DataFrame:
    """Dollar-neutral long-short on the top and bottom fifth of each day's cross-section.

    The top ``n // 5`` assets by ``signal`` get weight ``+1/k`` and the bottom ``k`` get
    ``-1/k``, so each leg sums to one. Assets with a missing signal are not ranked. A day with
    fewer than ``min_assets`` valid signals holds no position. Ties resolve by column order.
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

    A refit on day ``d`` trains on rows ``s`` whose target was realised before ``d``. The target
    at row ``s`` is the return of day ``s + 1``, so the training rows are ``s <= d - 2``, which
    is the slice ``[:d - 1]``. The intercept and feature means come from the training rows and
    are not penalised. Rows before the first refit are NaN.
    """
    columns = target.columns if isinstance(target, pd.DataFrame) else None
    y_all = target.to_numpy(dtype=np.float64) if isinstance(target, pd.DataFrame) else target
    t_len, n_assets, n_feat = features.shape
    pred = np.full((t_len, n_assets), np.nan)
    refits = refit_positions(dates, burn_in_years)
    for j, d in enumerate(refits):
        end = refits[j + 1] if j + 1 < len(refits) else t_len
        x_tr = features[: d - 1].reshape(-1, n_feat)
        y_tr = y_all[: d - 1].reshape(-1)
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


def ridge_time_series_weights(returns: pd.DataFrame, *,
                              bars_per_year: float = TRADING_DAYS) -> pd.DataFrame:
    """H3a: ridge on vol-scaled next-day returns, sized by its prediction relative to its trailing std."""
    features = ridge_features(returns)
    target = vol_scaled_next_return(returns)
    pred = walk_forward_ridge(features, target, returns.index)
    scale = pred.expanding(min_periods=ZSCORE_MIN_PERIODS).std(ddof=1)
    signal = (pred / scale).clip(-MAX_EXPOSURE, MAX_EXPOSURE)
    vol = ex_ante_vol(returns, bars_per_year=bars_per_year)
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


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion. Used for the share of null panels that pass."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    lo = 0.0 if successes == 0 else max(0.0, float(centre - half))
    hi = 1.0 if successes == trials else min(1.0, float(centre + half))
    return lo, hi


def observed_sr_variance(annual_sharpes, bars_per_year: float = TRADING_DAYS) -> float:
    """Variance of per-bar Sharpe ratios across trials, from their annualised values."""
    s = np.asarray(annual_sharpes, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return float("nan")
    return float(np.var(s, ddof=1) / bars_per_year)


def deflated_sharpe(returns, n_trials: int = N_TRIALS, sr_variance: float | None = None) -> float:
    """Deflated Sharpe ratio from ``softedge.eval.stats``.

    ``sr_variance`` is the cross-trial variance of per-bar Sharpe ratios. It is floored at the
    null variance, as in the repo's ``dsr_summary``. ``None`` uses the floor alone.
    """
    r = np.asarray(returns, dtype=np.float64)
    floor = null_sr_variance(r.size)
    var = floor if sr_variance is None else max(float(sr_variance), floor)
    return float(deflated_sharpe_ratio(r, n_trials, var))


def pass_rule(*, nw_t: float, dsr: float, sharpe_first_half: float,
              sharpe_second_half: float) -> bool:
    """Registered pass rule: NW t above 2.5, DSR above 0.95, and Sharpe positive in both halves."""
    return bool(nw_t > PASS_NW_T and dsr > PASS_DSR
                and sharpe_first_half > 0.0 and sharpe_second_half > 0.0)


def trial_returns(weights: pd.DataFrame, returns: pd.DataFrame, cost_bp: pd.Series, *,
                  delay: int = 0) -> pd.Series:
    """Net daily returns from the first day a position earns, to the end of the panel."""
    net = portfolio_returns(weights, returns, cost_bp, delay=delay)
    held = weights.shift(delay).fillna(0.0).shift(1).fillna(0.0)
    earning = (held != 0.0).any(axis=1)
    if not earning.any():
        return net.iloc[0:0]
    return net.loc[earning.idxmax():]


def summarize(net: pd.Series, *, n_trials: int = N_TRIALS,
              bars_per_year: float = TRADING_DAYS) -> dict:
    """Statistics for one net return series.

    Keys: ``sharpe`` (annualised), ``nw_t``, ``dsr`` (registered: repo function at ``n_trials``
    with the null variance), ``sharpe_first_half``, ``sharpe_second_half``, ``n_days``,
    ``min_detectable_sharpe``, ``start`` and ``passes``.
    """
    r = net.to_numpy(dtype=np.float64)
    if r.size == 0:
        nan = float("nan")
        return {"sharpe": nan, "nw_t": nan, "dsr": nan, "sharpe_first_half": nan,
                "sharpe_second_half": nan, "n_days": 0, "min_detectable_sharpe": nan,
                "start": None, "passes": False}
    half = r.size // 2
    sr_all = float(sharpe(r, bars_per_year))
    sr1 = float(sharpe(r[:half], bars_per_year))
    sr2 = float(sharpe(r[half:], bars_per_year))
    nw = newey_west_t(r)
    dsr = deflated_sharpe(r, n_trials)
    return {
        "sharpe": sr_all, "nw_t": nw, "dsr": dsr,
        "sharpe_first_half": sr1, "sharpe_second_half": sr2,
        "n_days": int(r.size),
        "min_detectable_sharpe": min_detectable_sharpe(r.size, bars_per_year=bars_per_year),
        "start": str(net.index[0].date()),
        "passes": pass_rule(nw_t=nw, dsr=dsr, sharpe_first_half=sr1, sharpe_second_half=sr2),
    }


def evaluate(weights: pd.DataFrame, returns: pd.DataFrame, cost_bp: pd.Series, *,
             delay: int = 0, n_trials: int = N_TRIALS,
             bars_per_year: float = TRADING_DAYS) -> dict:
    """``summarize`` of the net returns that ``weights`` earn on ``returns``."""
    net = trial_returns(weights, returns, cost_bp, delay=delay)
    return summarize(net, n_trials=n_trials, bars_per_year=bars_per_year)


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


HYPOTHESES: dict[str, Callable[..., pd.DataFrame]] = {
    "H1a": lambda r, bars_per_year=TRADING_DAYS: ts_momentum_weights(
        r, 21, bars_per_year=bars_per_year),
    "H1b": lambda r, bars_per_year=TRADING_DAYS: ts_momentum_weights(
        r, 63, bars_per_year=bars_per_year),
    "H1c": lambda r, bars_per_year=TRADING_DAYS: ts_momentum_weights(
        r, 252, bars_per_year=bars_per_year),
    "H1d": lambda r, bars_per_year=TRADING_DAYS: combine_equal_weight(
        ts_momentum_weights(r, 21, bars_per_year=bars_per_year),
        ts_momentum_weights(r, 63, bars_per_year=bars_per_year),
        ts_momentum_weights(r, 252, bars_per_year=bars_per_year)),
    "H2a": lambda r, bars_per_year=TRADING_DAYS: quintile_weights(-r, min_assets=MIN_ASSETS),
    "H2b": lambda r, bars_per_year=TRADING_DAYS: quintile_weights(
        trailing_return(r, 7), min_assets=MIN_ASSETS),
    "H2c": lambda r, bars_per_year=TRADING_DAYS: quintile_weights(
        trailing_return(r, 28), min_assets=MIN_ASSETS),
    "H3a": lambda r, bars_per_year=TRADING_DAYS: ridge_time_series_weights(
        r, bars_per_year=bars_per_year),
    "H3b": lambda r, bars_per_year=TRADING_DAYS: ridge_cross_section_weights(r),
}
