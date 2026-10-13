"""Seeded synthetic market generators.

Each public generator returns a bar frame built with ``sactrader.data.schema.make_frame``.
Its output is a pure function of the arguments, so the same ``seed`` gives the same frame.

A bar is split into ``substeps`` log-price steps. The steps sum to the bar's log return and
form a Gaussian bridge around it, so high and low are read off the path. The open of bar
``t`` is the close of bar ``t - 1``. Linear prices start at 100. Binary prices are observed
on the same step grid.

Linear generators take ``kind`` in ``("spot", "fx")``. ``with_funding`` turns such a frame
into a perpetual. ``binary_market`` returns a YES token price that settles at 0 or 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import expit

from sactrader.data.schema import make_frame, validate_frame

_START_PRICE = 100.0
LINEAR_KINDS = ("spot", "fx")


def _check(n: int, substeps: int, kind: str | None = None) -> None:
    if n < 2:
        raise ValueError("n must be at least 2")
    if substeps < 1:
        raise ValueError("substeps must be at least 1")
    if kind is not None and kind not in LINEAR_KINDS:
        raise ValueError(f"kind must be one of {LINEAR_KINDS}, got {kind!r}")


def _index(n: int, start: str, bar: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=bar, tz="UTC")


def _bridge_ticks(rng: np.random.Generator, totals: np.ndarray, vol: float | np.ndarray,
                  substeps: int) -> np.ndarray:
    """Cumulative log-price ticks from 0, of length ``n * substeps + 1``.

    Bar ``t`` owns ticks ``t * substeps`` through ``(t + 1) * substeps``. Its steps sum to
    ``totals[t]``, and around that sum they are a bridge with std ``vol`` per bar. For
    ``gbm`` the totals and steps then have the joint law of i.i.d. Gaussian steps.
    """
    z = rng.standard_normal((totals.size, substeps))
    z -= z.mean(axis=1, keepdims=True)
    scale = np.reshape(np.asarray(vol, dtype=np.float64), (-1, 1)) / np.sqrt(substeps)
    steps = totals[:, None] / substeps + scale * z
    return np.concatenate(([0.0], np.cumsum(steps, axis=None)))


def _ohlc(prices: np.ndarray, n: int, substeps: int) -> tuple[np.ndarray, ...]:
    """Open, high, low and close per bar from tick prices of length ``n * substeps + 1``."""
    idx = np.arange(n)[:, None] * substeps + np.arange(substeps + 1)
    win = prices[idx]
    return win[:, 0], win.max(axis=1), win.min(axis=1), win[:, -1]


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    """AR(1) series of length ``n`` with stationary std ``sd``, started from its stationary law."""
    x0 = sd * rng.standard_normal()
    eps = sd * np.sqrt(1.0 - phi**2) * rng.standard_normal(n)
    # The initial condition is the previous value, so the first output is already stationary.
    y, _ = lfilter([1.0], [1.0, -phi], eps, zi=[phi * x0])
    return y


def _linear(rng: np.random.Generator, totals: np.ndarray, vol: float | np.ndarray, *,
            kind: str, start: str, bar: str, symbol: str, substeps: int) -> pd.DataFrame:
    n = totals.size
    prices = _START_PRICE * np.exp(_bridge_ticks(rng, totals, vol, substeps))
    o, h, lo, c = _ohlc(prices, n, substeps)
    return make_frame(_index(n, start, bar), o, h, lo, c, symbol=symbol, kind=kind, bar=bar)


def gbm(n: int, mu: float = 0.0, sigma: float = 0.01, *, kind: str = "spot", seed: int = 0,
        start: str = "2020-01-01", bar: str = "1h", symbol: str, substeps: int = 8,
        ) -> pd.DataFrame:
    """Geometric Brownian motion. Bar log returns are i.i.d. N(mu, sigma**2)."""
    _check(n, substeps, kind)
    rng = np.random.default_rng(seed)
    totals = mu + sigma * rng.standard_normal(n)
    return _linear(rng, totals, sigma, kind=kind, start=start, bar=bar, symbol=symbol,
                   substeps=substeps)


def ar_returns(n: int, phi: float = 0.1, sigma: float = 0.01, *, kind: str = "spot",
               seed: int = 0, start: str = "2020-01-01", bar: str = "1h", symbol: str,
               substeps: int = 8) -> pd.DataFrame:
    """Bar log returns follow AR(1) with coefficient ``phi`` and stationary std ``sigma``."""
    _check(n, substeps, kind)
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must lie in (-1, 1)")
    rng = np.random.default_rng(seed)
    totals = _ar1(rng, n, phi, sigma)
    return _linear(rng, totals, sigma, kind=kind, start=start, bar=bar, symbol=symbol,
                   substeps=substeps)


def planted_signal(n: int, strength: float = 0.1, sigma: float = 0.01, rho: float = 0.9, *,
                   kind: str = "spot", seed: int = 0, start: str = "2020-01-01",
                   bar: str = "1h", symbol: str, substeps: int = 8) -> pd.DataFrame:
    """Bar log return ``t`` is ``strength * sigma * signal[t - 1]`` plus independent noise.

    ``signal`` is AR(1) with coefficient ``rho`` and unit stationary variance. It is stored in
    column ``signal``. ``signal[t]`` is known at the close of bar ``t`` and first affects bar
    ``t + 1``. The driver of bar 0 is drawn from the stationary law before the frame starts.
    The per-bar std of the log return is ``sigma``.
    """
    _check(n, substeps, kind)
    if not -1.0 < strength < 1.0:
        raise ValueError("strength must lie in (-1, 1)")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie in (-1, 1)")
    rng = np.random.default_rng(seed)
    driver = _ar1(rng, n + 1, rho, 1.0)
    noise = sigma * np.sqrt(1.0 - strength**2) * rng.standard_normal(n)
    totals = strength * sigma * driver[:-1] + noise
    frame = _linear(rng, totals, sigma, kind=kind, start=start, bar=bar, symbol=symbol,
                    substeps=substeps)
    frame["signal"] = driver[1:]
    return frame


def regime_switching(n: int, *, drifts: tuple[float, float, float] = (5e-4, -5e-4, 0.0),
                     vols: tuple[float, float, float] = (0.01, 0.012, 0.006),
                     p_stay: float = 0.995, kind: str = "spot", seed: int = 0,
                     start: str = "2020-01-01", bar: str = "1h", symbol: str,
                     substeps: int = 8) -> pd.DataFrame:
    """Markov regimes with per-state log drift and vol.

    States are 0 (up-trend), 1 (down-trend) and 2 (range). A state persists with probability
    ``p_stay``. Otherwise the chain moves to one of the other two states with equal
    probability. The state of each bar is stored in the int column ``regime``.
    """
    _check(n, substeps, kind)
    if len(drifts) != 3 or len(vols) != 3:
        raise ValueError("drifts and vols need one entry per state")
    if not 0.0 < p_stay < 1.0:
        raise ValueError("p_stay must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    # Mod-3 addition of the jumps gives the stay-or-move-to-another-state chain without a loop.
    stay = rng.random(n) < p_stay
    to_next = rng.random(n) < 0.5
    jump = np.where(stay, 0, np.where(to_next, 1, 2))
    jump[0] = 0
    regime = (rng.integers(3) + np.cumsum(jump)) % 3
    vol = np.asarray(vols, dtype=np.float64)[regime]
    totals = np.asarray(drifts, dtype=np.float64)[regime] + vol * rng.standard_normal(n)
    frame = _linear(rng, totals, vol, kind=kind, start=start, bar=bar, symbol=symbol,
                    substeps=substeps)
    frame["regime"] = regime.astype(np.int64)
    return frame


def ou_mean_reverting(n: int, theta: float = 0.02, sigma: float = 0.01, *, kind: str = "spot",
                      seed: int = 0, start: str = "2020-01-01", bar: str = "1h", symbol: str,
                      substeps: int = 8) -> pd.DataFrame:
    """Log price reverts to its start level. The log deviation ``d`` from the start shrinks
    by ``theta`` per bar, and each bar adds a shock with std ``sigma``.
    """
    _check(n, substeps, kind)
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    shocks = sigma * rng.standard_normal(n)
    # dev[t] is the deviation at the close of bar t, starting from 0 at the first open.
    dev = lfilter([1.0], [1.0, -(1.0 - theta)], shocks)
    open_dev = np.concatenate(([0.0], dev[:-1]))
    totals = -theta * open_dev + shocks
    return _linear(rng, totals, sigma, kind=kind, start=start, bar=bar, symbol=symbol,
                   substeps=substeps)


def with_funding(frame: pd.DataFrame, mean: float = 1e-4, std: float = 1e-4,
                 interval_hours: int = 8, *, seed: int = 0) -> pd.DataFrame:
    """Perpetual copy of a spot or fx frame with funding on bars that open at a funding time.

    A funding time is a whole UTC hour divisible by ``interval_hours``. A bar that opens
    exactly there carries a rate drawn from N(mean, std). Every other bar carries 0.0, so
    sub-hourly bars are charged only when they open on the hour.
    """
    kind = frame.attrs.get("kind")
    if kind not in LINEAR_KINDS:
        raise ValueError(f"with_funding needs a spot or fx frame, got kind {kind!r}")
    if interval_hours < 1 or 24 % interval_hours:
        raise ValueError("interval_hours must divide 24")
    rng = np.random.default_rng(seed)
    rates = rng.normal(mean, std, size=len(frame))
    idx = frame.index
    at_funding = np.asarray((idx.hour % interval_hours == 0) & (idx.minute == 0)
                            & (idx.second == 0))
    out = frame.copy()
    out["funding_rate"] = np.where(at_funding, rates, 0.0)
    out.attrs = {**frame.attrs, "kind": "perp"}
    validate_frame(out)
    return out


def binary_market(n: int, *, info_rate: float = 0.05, noise: float = 0.02,
                  longshot_bias: float = 0.0, seed: int = 0, start: str = "2020-01-01",
                  bar: str = "1h", symbol: str, substeps: int = 8) -> pd.DataFrame:
    """YES token price of a binary market that settles at 0 or 1.

    The latent log-odds starts at 0 and is a random walk with variance ``info_rate`` per bar.
    The true probability at the end is ``sigmoid(latent)``, and the resolution is Bernoulli
    with that probability. The observed log-odds is ``(1 - longshot_bias) * latent`` plus
    Gaussian noise with std ``noise``. A positive ``longshot_bias`` shrinks log-odds toward 0,
    which lifts longshot prices and lowers favourite prices. Prices are clipped to
    [0.01, 0.99]. The frame carries ``attrs["resolution"]`` and ``attrs["end_time"]``.
    """
    _check(n, substeps)
    if info_rate < 0.0 or noise < 0.0:
        raise ValueError("info_rate and noise must be non-negative")
    if not 0.0 <= longshot_bias < 1.0:
        raise ValueError("longshot_bias must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    info_sd = np.sqrt(info_rate)
    latent = _bridge_ticks(rng, rng.normal(0.0, info_sd, size=n), info_sd, substeps)
    obs = (1.0 - longshot_bias) * latent + noise * rng.standard_normal(latent.size)
    prices = np.clip(expit(obs), 0.01, 0.99)
    o, h, lo, c = _ohlc(prices, n, substeps)
    resolution = float(rng.random() < expit(latent[-1]))
    frame = make_frame(_index(n, start, bar), o, h, lo, c, symbol=symbol, kind="binary",
                       bar=bar, resolution=resolution)
    frame.attrs["end_time"] = (frame.index[-1] + pd.Timedelta(bar)).isoformat()
    return frame
