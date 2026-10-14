"""Benchmark markets with fresh draws and the failure modes of real markets.

Each generator is a pure function of ``(n, seed, symbol, start)`` and its keyword parameters.
A bench market has the usual OHLC bars and these extra columns:

- ``mu_next``: the true expected log return of the next bar, given the path up to the current
  close. Oracle scoring only, never a feature.
- ``regime``: an int state for diagnostics only, never a feature. Its meaning is set per family.
- ``signal``: an observable AR(1) driver, in the families that have one.
- ``prior_hint``: a prior in units of ``PRIOR_SCALE``, a nominal bar volatility of 1%. It is
  ``mu_next`` plus heavy-tailed noise plus a slow hidden bias. The bias is the part a policy can
  learn from realized returns. Every bench market carries it, so every family has the same
  feature columns and ``family="all"`` mixes tapes that differ only in their dynamics.

Families: ``break`` (the edge flips or vanishes at a random bar of the second half), ``jump``
(GARCH volatility with compound-Poisson crashes), ``threshold`` (the signal predicts only in
high-volatility bars) and ``prior`` (a constant-volatility linear market).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from softedge.data.synthetic import _ar1, _check, _linear

FAMILIES = ("break", "jump", "threshold", "prior")
PRIOR_SCALE = 0.01
_GARCH = (0.08, 0.9)


def _attach_prior(frame: pd.DataFrame, seed: int, *, noise_sd: float = 1.0,
                  noise_df: float = 3.0, bias_phi: float = 0.999,
                  bias_sd: float = 0.3) -> pd.DataFrame:
    # A stream of its own keeps the prior from shifting the market's draws.
    rng = np.random.default_rng([seed, 1])
    n = len(frame)
    noise = noise_sd * np.sqrt((noise_df - 2.0) / noise_df) * rng.standard_t(noise_df, n)
    bias = _ar1(rng, n, bias_phi, bias_sd)
    frame["prior_hint"] = frame["mu_next"].to_numpy() / PRIOR_SCALE + noise + bias
    return frame


def _nonlinear_path(rng: np.random.Generator, n: int, snr_path: np.ndarray, sigma: float,
                    df: float, garch: tuple[float, float], p_stay: float, rho: float):
    """Bar returns, bar vols, expected returns (n + 1 entries), regime and driver.

    The loop mirrors ``synthetic.nonlinear_regime_market`` except that the edge size is read
    per bar from ``snr_path`` (length n + 1), which is what a break needs.
    """
    a, b = garch
    flips = np.where(rng.random(n + 1) < p_stay, 1.0, -1.0)
    flips[0] = 1.0
    regime = (2.0 * rng.integers(2) - 1.0) * np.cumprod(flips)
    driver = _ar1(rng, n + 1, rho, 1.0)
    shocks = rng.standard_t(df, size=n) * np.sqrt((df - 2.0) / df)

    w = sigma**2 * (1.0 - a - b)
    z, x, eps, snr = regime.tolist(), driver.tolist(), shocks.tolist(), snr_path.tolist()
    h = sigma**2
    mom = 0.0
    last8 = [0.0] * 8
    rets: list[float] = []
    vols: list[float] = []
    mus: list[float] = []
    for t in range(n):
        s = h**0.5
        mu = snr[t] * s * (z[t] * np.tanh(2.0 * mom) + 0.5 * np.tanh(2.0 * x[t]))
        shock = s * eps[t]
        rets.append(mu + shock)
        vols.append(s)
        mus.append(mu)
        last8[t % 8] = mu + shock
        mom = sum(last8) / (s * 8.0**0.5) if t >= 7 else 0.0
        h = w + a * shock * shock + b * h
    s = h**0.5
    mus.append(snr[n] * s * (z[n] * np.tanh(2.0 * mom) + 0.5 * np.tanh(2.0 * x[n])))
    return np.asarray(rets), np.asarray(vols), np.asarray(mus), regime, driver


def break_market(n: int, *, snr: float = 0.08, sigma: float = 0.01, df: float = 4.0,
                 garch: tuple[float, float] = _GARCH, p_stay: float = 0.99, rho: float = 0.95,
                 seed: int = 0, symbol: str, start: str = "2020-01-01", bar: str = "1h",
                 substeps: int = 8) -> pd.DataFrame:
    """Hidden-regime market of ``nonlinear_regime_market`` whose edge changes at one bar.

    The break bar is uniform over the second half of the sample. From it on the edge is
    multiplied by ``break_mult``, which is -1 (sign flip) or 0 (edge gone), each with
    probability one half. ``attrs`` holds ``break_bar`` and ``break_mult``. ``regime`` is the
    hidden state in {-1, +1}.
    """
    _check(n, substeps, "spot")
    rng = np.random.default_rng(seed)
    brk = int(rng.integers(n // 2, n))
    mult = -1.0 if rng.random() < 0.5 else 0.0
    snr_path = np.where(np.arange(n + 1) < brk, snr, mult * snr)
    rets, vols, mus, regime, driver = _nonlinear_path(rng, n, snr_path, sigma, df, garch,
                                                      p_stay, rho)
    frame = _linear(rng, rets, vols, kind="spot", start=start, bar=bar, symbol=symbol,
                    substeps=substeps)
    frame["signal"] = driver[1:]
    frame["mu_next"] = mus[1:]
    frame["regime"] = regime[1:].astype(np.int64)
    frame.attrs.update(family="break", break_bar=brk, break_mult=mult)
    return _attach_prior(frame, seed)


def jump_garch(n: int, *, snr: float = 0.1, sigma: float = 0.01,
               garch: tuple[float, float] = _GARCH, jump_rate: float = 0.002,
               jump_mean: float = -0.03, jump_sd: float = 0.02, rho: float = 0.95,
               seed: int = 0, symbol: str, start: str = "2020-01-01", bar: str = "1h",
               substeps: int = 8) -> pd.DataFrame:
    """GARCH(1, 1) volatility with compound-Poisson jumps and a weak signal drift.

    Bar ``t`` has ``N_t`` jumps, Poisson with mean ``jump_rate``. Their total is normal with
    mean ``N_t * jump_mean`` and variance ``N_t * jump_sd**2``. Only the diffusive shock feeds
    the GARCH variance. The expected return of bar ``t`` is
    ``snr * sigma_t * tanh(signal[t - 1]) + jump_rate * jump_mean``. The second term is the
    jump compensator, so ``mu_next`` is the true conditional mean. ``regime`` is 1 where the
    next bar's variance is above its median. ``jumps`` counts each bar's jumps (diagnostic).
    """
    _check(n, substeps, "spot")
    a, b = garch
    rng = np.random.default_rng(seed)
    x = _ar1(rng, n, rho, 1.0)
    eps = rng.standard_normal(n)
    counts = rng.poisson(jump_rate, n)
    jump_ret = jump_mean * counts + jump_sd * np.sqrt(counts) * rng.standard_normal(n)

    w = sigma**2 * (1.0 - a - b)
    e = eps.tolist()
    h = [sigma**2]
    for t in range(n):
        h.append(w + (a * e[t] * e[t] + b) * h[t])
    var = np.asarray(h)
    vol = np.sqrt(var)
    drift = snr * vol * np.tanh(np.concatenate(([0.0], x)))
    rets = drift[:n] + vol[:n] * eps + jump_ret
    mu = drift + jump_rate * jump_mean
    next_var = var[1:]
    frame = _linear(rng, rets, vol[:n], kind="spot", start=start, bar=bar, symbol=symbol,
                    substeps=substeps)
    frame["signal"] = x
    frame["mu_next"] = mu[1:]
    frame["regime"] = (next_var > np.median(next_var)).astype(np.int64)
    frame["jumps"] = counts.astype(np.int64)
    frame.attrs["family"] = "jump"
    return _attach_prior(frame, seed)


def threshold_signal(n: int, *, snr: float = 0.1, sigma: float = 0.01, rho: float = 0.95,
                     vol_rho: float = 0.98, vol_sd: float = 0.5, seed: int = 0, symbol: str,
                     start: str = "2020-01-01", bar: str = "1h", substeps: int = 8,
                     ) -> pd.DataFrame:
    """The signal predicts returns only in bars whose volatility is above the path median.

    Log volatility is an AR(1) with coefficient ``vol_rho`` and std ``vol_sd``, drawn apart
    from returns, so the gate is known in advance. Bar ``t + 1`` drifts by
    ``snr * vol_t * signal_t`` when ``vol_t`` is above the median of the path, and by zero
    otherwise. ``regime`` is that high-volatility flag at each close. The median is a constant
    of the generated path, which a policy never observes.
    """
    _check(n, substeps, "spot")
    rng = np.random.default_rng(seed)
    x = _ar1(rng, n, rho, 1.0)
    vol = sigma * np.exp(_ar1(rng, n, vol_rho, vol_sd))
    eps = rng.standard_normal(n)
    high = vol > np.median(vol)
    drift = snr * np.concatenate(([0.0], vol * x * high))
    frame = _linear(rng, drift[:n] + vol * eps, vol, kind="spot", start=start, bar=bar,
                    symbol=symbol, substeps=substeps)
    frame["signal"] = x
    frame["mu_next"] = drift[1:]
    frame["regime"] = high.astype(np.int64)
    frame.attrs["family"] = "threshold"
    return _attach_prior(frame, seed)


def noisy_prior(n: int, *, snr: float = 0.3, sigma: float = 0.01, rho: float = 0.95,
                seed: int = 0, symbol: str, start: str = "2020-01-01", bar: str = "1h",
                substeps: int = 8) -> pd.DataFrame:
    """Constant-volatility linear market: bar ``t + 1`` drifts by ``snr * sigma * signal_t``.

    Its failure mode is the prior: ``prior_hint`` carries heavy noise and a slow bias, so a
    policy must learn the residual on it. ``regime`` is zero throughout.
    """
    _check(n, substeps, "spot")
    rng = np.random.default_rng(seed)
    x = _ar1(rng, n, rho, 1.0)
    drift = snr * sigma * np.concatenate(([0.0], x))
    rets = drift[:n] + sigma * rng.standard_normal(n)
    frame = _linear(rng, rets, sigma, kind="spot", start=start, bar=bar, symbol=symbol,
                    substeps=substeps)
    frame["signal"] = x
    frame["mu_next"] = drift[1:]
    frame["regime"] = np.zeros(n, dtype=np.int64)
    frame.attrs["family"] = "prior"
    return _attach_prior(frame, seed)


GENERATORS: dict[str, Callable[..., pd.DataFrame]] = {
    "break": break_market,
    "jump": jump_garch,
    "threshold": threshold_signal,
    "prior": noisy_prior,
}


def bench_frames(names: Sequence[str], n: int, *, family: str = "all", seed_offset: int = 0,
                 start: str = "2020-01-01", bar: str = "1h") -> list[pd.DataFrame]:
    """One market per name, seeded ``seed_offset + i``. ``family="all"`` cycles through
    ``FAMILIES`` in name order, so every family appears once the names reach four."""
    if family != "all" and family not in GENERATORS:
        raise ValueError(f"family must be 'all' or one of {FAMILIES}, got {family!r}")
    kinds = FAMILIES if family == "all" else (family,)
    return [GENERATORS[kinds[i % len(kinds)]](n, seed=seed_offset + i, symbol=name, start=start,
                                              bar=bar)
            for i, name in enumerate(names)]
