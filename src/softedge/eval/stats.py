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

import numpy as np
from scipy import stats

from softedge.eval.metrics import DEGENERATE_REL, dispersion_ok, sharpe

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


def sharpe_moments(returns) -> dict[str, float]:
    """Per-bar Sharpe ratio, skew, raw kurtosis and length of a return series."""
    r = np.asarray(returns, dtype=np.float64)
    sd = r.std(ddof=1)
    sr = float(r.mean() / sd) if dispersion_ok(r, np.asarray(sd)) else 0.0
    if sr == 0.0:
        return {"n": float(r.size), "sr": 0.0, "skew": 0.0, "kurt": 3.0}
    return {"n": float(r.size), "sr": sr, "skew": float(stats.skew(r)),
            "kurt": float(stats.kurtosis(r, fisher=False))}


def null_sr_variance(n: int) -> float:
    """Variance of the per-bar Sharpe estimate of a zero-skill strategy over ``n`` bars.

    With N such trials the estimates scatter with this variance, so it is the smallest honest
    value for the cross-trial variance in the deflated Sharpe ratio.
    """
    return 1.0 / max(n - 1, 1)


def min_track_record_length(returns, sr_benchmark: float = 0.0, prob: float = 0.95) -> float:
    """Bars needed for PSR(sr_benchmark) to reach ``prob`` given the series' Sharpe and moments
    (Bailey and Lopez de Prado, 2012). Infinite when the Sharpe does not exceed the benchmark."""
    m = sharpe_moments(returns)
    if m["sr"] <= sr_benchmark:
        return float("inf")
    var_term = 1.0 - m["skew"] * m["sr"] + (m["kurt"] - 1.0) / 4.0 * m["sr"] ** 2
    z = stats.norm.ppf(prob)
    return float(1.0 + max(var_term, 0.0) * (z / (m["sr"] - sr_benchmark)) ** 2)


def dsr_summary(returns, bars_per_year: float, n_trials: int,
                sr_variance: float | None = None) -> dict[str, float]:
    """Deflated Sharpe ratio report for one series.

    ``sr_variance`` is the variance of per-bar Sharpe ratios across the ``n_trials``
    configurations tried on the same data. It is floored at ``null_sr_variance(n)``; ``None``
    uses that floor alone. Keys: annualized Sharpe, per-bar moments, PSR against zero, the
    expected maximum Sharpe of ``n_trials`` zero-skill trials (``sr0``, per bar and annualized),
    DSR = PSR against ``sr0``, and the minimum track record (bars and years) for PSR(0) >= 0.95.
    """
    r = np.asarray(returns, dtype=np.float64)
    m = sharpe_moments(r)
    floor = null_sr_variance(r.size)
    var = floor if sr_variance is None else max(float(sr_variance), floor)
    sr0 = expected_max_sharpe(max(int(n_trials), 1), var)
    ann = float(np.sqrt(bars_per_year))
    min_trl = min_track_record_length(r)
    return {
        "sharpe": m["sr"] * ann, "sr_bar": m["sr"], "skew": m["skew"], "kurt": m["kurt"],
        "n_bars": m["n"], "n_trials": float(n_trials), "sr_variance": var,
        "psr": probabilistic_sharpe_ratio(r) if r.size >= 3 else float("nan"),
        "sr0_bar": sr0, "sr0": sr0 * ann,
        "dsr": probabilistic_sharpe_ratio(r, sr_benchmark=sr0) if r.size >= 3 else float("nan"),
        "min_trl_bars": min_trl, "min_trl_years": min_trl / bars_per_year,
    }


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
    if t_rows < 2 * n_splits:
        raise ValueError("need at least two rows per split")

    blocks = np.array_split(np.arange(t_rows), n_splits)
    counts = np.array([b.size for b in blocks], dtype=np.float64)
    means = np.stack([x[b].mean(axis=0) for b in blocks])
    m2 = np.stack([((x[b] - x[b].mean(axis=0)) ** 2).sum(axis=0) for b in blocks])
    scale = np.abs(x).max()

    half = n_splits // 2
    combo_sets = [set(c) for c in itertools.combinations(range(n_splits), half)]
    combos = np.array([sorted(s) for s in combo_sets])
    comps = np.array([[i for i in range(n_splits) if i not in s] for s in combo_sets])
    is_sr = _subset_sharpe(combos, counts, means, m2, scale)
    oos_sr = _subset_sharpe(comps, counts, means, m2, scale)

    rows = np.arange(combos.shape[0])
    best = np.argmax(is_sr, axis=1)
    oos_best = oos_sr[rows, best][:, None]
    # Average rank of the in-sample pick among the N out-of-sample values, 1 to N.
    below = np.sum(oos_sr < oos_best, axis=1)
    equal = np.sum(oos_sr == oos_best, axis=1)
    rank = below + (equal + 1) / 2.0
    w = rank / (n_cfg + 1.0)
    logits = np.log(w / (1.0 - w))
    return float(np.mean(logits <= 0.0))


def _subset_sharpe(
    index: np.ndarray,
    counts: np.ndarray,
    means: np.ndarray,
    m2: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Per-bar Sharpe of each subset of blocks, shape (K, N).

    ``index`` lists the block numbers of each subset. Subset moments come from the
    block means and sums of squared deviations. Two subsets are merged with the
    pairwise update of Chan et al., which stays accurate where a raw sum of squares
    would cancel.
    """
    n = counts[index[:, 0]]
    mean = means[index[:, 0]]
    acc = m2[index[:, 0]]
    for j in range(1, index.shape[1]):
        nb = counts[index[:, j]]
        delta = means[index[:, j]] - mean
        n_new = n + nb
        mean = mean + delta * (nb / n_new)[:, None]
        acc = acc + m2[index[:, j]] + delta**2 * (n * nb / n_new)[:, None]
        n = n_new
    sd = np.sqrt(acc / (n - 1.0)[:, None])
    ok = sd > DEGENERATE_REL * scale
    out = np.zeros_like(mean)
    np.divide(mean, sd, out=out, where=ok)
    return out
