"""Significance and overfitting tests for strategy Sharpe ratios.

Inputs are per-bar returns. Sharpe values here are per bar unless the argument
is named ``bars_per_year``. Formulas follow Bailey and Lopez de Prado (2012) for
the probabilistic Sharpe ratio, Bailey and Lopez de Prado (2014) for the deflated
Sharpe ratio, Politis and Romano (1994) for the stationary bootstrap, and Bailey,
Borwein, Lopez de Prado and Zhu (2017) for combinatorially symmetric
cross-validation (CSCV).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

from sactrader.eval.metrics import dispersion_ok, sharpe

EULER_GAMMA = 0.5772156649015329
# Bootstrap rows are generated in chunks of about this many index entries.
_CHUNK_ENTRIES = 2_000_000


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """Probability that the true per-bar Sharpe exceeds ``sr_benchmark``.

    The denominator corrects for skew and kurtosis of the sample. Returns nan when
    the series has no dispersion or the correction term is not positive.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1 or r.size < 3:
        raise ValueError("PSR needs a 1-D series of at least 3 returns")
    sd = r.std(ddof=1)
    if not dispersion_ok(r, np.asarray(sd)):
        return float("nan")
    sr = r.mean() / sd
    skew = stats.skew(r)
    excess_kurt = stats.kurtosis(r)
    # Raw kurtosis is excess + 3, so (raw - 1) / 4 equals (excess + 2) / 4.
    var_term = 1.0 - skew * sr + (excess_kurt + 2.0) / 4.0 * sr * sr
    if var_term <= 0.0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(r.size - 1) / np.sqrt(var_term)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-bar Sharpe of ``n_trials`` zero-mean trials.

    ``sr_variance`` is the variance of the trial Sharpe ratios across the search.
    The Euler-Mascheroni approximation needs two or more trials. One trial has
    expected maximum zero, which is the value returned here.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if sr_variance < 0.0:
        raise ValueError("sr_variance must be non-negative")
    if n_trials == 1:
        return 0.0
    z_hi = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z_lo = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z_hi + EULER_GAMMA * z_lo))


def deflated_sharpe_ratio(returns, n_trials: int, sr_variance: float) -> float:
    """PSR against the expected maximum Sharpe of the search that produced ``returns``."""
    benchmark = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=benchmark)


def stationary_bootstrap_indices(
    n: int, mean_block: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano resamples of ``range(n)``, shape ``(n_boot, n)``.

    Each position starts a new block with probability ``1 / mean_block``, otherwise
    it continues the previous block. Indices wrap around at ``n``, so every
    resample is a full-length series.
    """
    if mean_block < 1.0:
        raise ValueError("mean_block must be at least 1")
    pos = np.arange(n)
    new_block = rng.random((n_boot, n)) < 1.0 / mean_block
    new_block[:, 0] = True
    anchor = np.maximum.accumulate(np.where(new_block, pos, 0), axis=1)
    starts = rng.integers(0, n, size=(n_boot, n))
    block_start = np.take_along_axis(starts, anchor, axis=1)
    return (block_start + (pos - anchor)) % n


def sharpe_diff_test(
    r_a,
    r_b,
    bars_per_year: float,
    n_boot: int = 2000,
    mean_block: float | None = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired stationary-bootstrap test of annualized Sharpe(a) - Sharpe(b).

    Both series are resampled with the same indices, which keeps their pairing.
    ``lo`` and ``hi`` are the percentile interval at level ``1 - alpha``.
    ``p_value`` is the one-sided share of bootstrap differences at or below zero.
    """
    a = np.asarray(r_a, dtype=np.float64)
    b = np.asarray(r_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.size < 3:
        raise ValueError("r_a and r_b must be 1-D series of equal length of at least 3")
    n = a.size
    block = n ** (1.0 / 3.0) if mean_block is None else float(mean_block)
    rng = np.random.default_rng(seed)
    chunk = max(1, min(n_boot, _CHUNK_ENTRIES // n))

    diffs = []
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = stationary_bootstrap_indices(n, block, k, rng)
        diffs.append(sharpe(a[idx], bars_per_year) - sharpe(b[idx], bars_per_year))
        done += k
    d = np.concatenate(diffs)
    lo, hi = np.quantile(d, [alpha / 2.0, 1.0 - alpha / 2.0])
    point = float(sharpe(a, bars_per_year) - sharpe(b, bars_per_year))
    return {
        "diff": point,
        "lo": float(lo),
        "hi": float(hi),
        "p_value": float(np.mean(d <= 0.0)),
    }


def pbo_cscv(perf: np.ndarray, n_splits: int = 16) -> float:
    """Probability of backtest overfitting by CSCV.

    ``perf`` is a T x N matrix of per-bar returns of N configurations. Rows are cut
    into ``n_splits`` contiguous blocks. For every choice of half the blocks as the
    in-sample set, the configuration with the best in-sample Sharpe is picked. Its
    out-of-sample rank among the N configurations gives a relative rank w. The
    logit of w is nonpositive when the pick falls at or below the median, and the
    PBO is the share of such splits.
    """
    x = np.asarray(perf, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("perf must be a T x N matrix with N >= 2")
    if n_splits < 2 or n_splits % 2:
        raise ValueError("n_splits must be even and at least 2")
    t_rows, n_cfg = x.shape
    if t_rows < n_splits:
        raise ValueError("need at least one row per split")

    blocks = np.array_split(np.arange(t_rows), n_splits)
    half = n_splits // 2
    all_blocks = set(range(n_splits))
    logits = np.empty(math.comb(n_splits, half))
    for k, combo in enumerate(itertools.combinations(range(n_splits), half)):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in sorted(all_blocks - set(combo))])
        is_sr = sharpe(x[is_rows].T, 1.0, axis=-1)
        oos_sr = sharpe(x[oos_rows].T, 1.0, axis=-1)
        best = int(np.argmax(is_sr))
        # Average rank of the in-sample pick among the N out-of-sample values, 1 to N.
        below = np.sum(oos_sr < oos_sr[best])
        equal = np.sum(oos_sr == oos_sr[best])
        rank = below + (equal + 1) / 2.0
        w = rank / (n_cfg + 1.0)
        logits[k] = np.log(w / (1.0 - w))
    return float(np.mean(logits <= 0.0))
