import numpy as np
import pandas as pd
import pytest

from softedge.data.bench import (
    FAMILIES,
    break_market,
    jump_garch,
    noisy_prior,
    threshold_signal,
)
from softedge.features.causal import compute_features
from softedge.pipeline import load_frames

GENERATORS = {"break": break_market, "jump": jump_garch, "threshold": threshold_signal,
              "prior": noisy_prior}
START = "2021-01-01"


def _log_return(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame["close"].to_numpy() / frame["open"].to_numpy())


@pytest.mark.parametrize("family", FAMILIES)
def test_same_seed_gives_identical_frames(family):
    gen = GENERATORS[family]
    a = gen(3000, seed=7, symbol="X", start=START)
    b = gen(3000, seed=7, symbol="X", start=START)
    c = gen(3000, seed=8, symbol="X", start=START)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["close"], c["close"])


@pytest.mark.parametrize("family", FAMILIES)
def test_mu_next_is_the_expected_next_return(family):
    # Regress the realized return of bar t+1 on mu_next[t]. A slope near one means mu_next is
    # the conditional mean. Signal autocorrelation is zero here, so a drift one bar off would
    # give a slope near zero instead.
    fr = GENERATORS[family](100_000, seed=3, symbol="X", start=START, snr=0.5, rho=0.0)
    r = _log_return(fr)
    mu = fr["mu_next"].to_numpy()
    slope = np.cov(r[1:], mu[:-1])[0, 1] / np.var(mu[:-1], ddof=1)
    assert 0.92 < slope < 1.08


@pytest.mark.parametrize("seed", range(8))
def test_break_happens_in_second_half(seed):
    n = 2000
    fr = break_market(n, seed=seed, symbol="X", start=START)
    brk = fr.attrs["break_bar"]
    assert n // 2 <= brk < n
    mu = fr["mu_next"].to_numpy()
    assert np.any(mu[: brk - 1] != 0.0)
    if fr.attrs["break_mult"] == 0.0:
        # The edge vanishes from bar brk on, so mu_next of the bar before it is exactly zero.
        assert np.all(mu[brk - 1:] == 0.0)


def test_jump_frequency_matches_the_rate():
    fr = jump_garch(200_000, seed=5, symbol="X", start=START, jump_rate=0.002)
    freq = (fr["jumps"].to_numpy() > 0).mean()
    # Expected 400 bars with a jump, Poisson sd 20. The band is about five sd wide.
    assert 0.0015 < freq < 0.0025


def test_threshold_signal_predicts_only_in_high_vol_bars():
    fr = threshold_signal(200_000, snr=0.5, seed=2, symbol="X", start=START)
    r = _log_return(fr)
    x = fr["signal"].to_numpy()[:-1]
    gate = fr["regime"].to_numpy()[:-1] == 1
    rn = r[1:]
    low = np.corrcoef(x[~gate], rn[~gate])[0, 1]
    high = np.corrcoef(x[gate], rn[gate])[0, 1]
    assert abs(low) < 0.02
    assert high > 0.1


def test_prior_hint_carries_a_slow_hidden_bias():
    n = 100_000
    fr = noisy_prior(n, seed=4, symbol="X", start=START)
    resid = fr["prior_hint"].to_numpy() - fr["mu_next"].to_numpy() / 0.01
    # Iid noise leaves 500-bar block means with variance var/500. The slow bias adds far more.
    blocks = resid[: n // 500 * 500].reshape(-1, 500).mean(axis=1)
    assert blocks.var() > 5 * resid.var() / 500
    assert np.corrcoef(fr["prior_hint"], fr["mu_next"])[0, 1] > 0.1


@pytest.mark.parametrize("family", FAMILIES)
def test_features_at_t_ignore_bars_after_t(family):
    fr = GENERATORS[family](3000, seed=1, symbol="X", start=START)
    alt = GENERATORS[family](3000, seed=2, symbol="X", start=START)
    t = 1500
    later = fr.index > fr.index[t]
    pert = fr.copy()
    for col in ("open", "high", "low", "close", "volume", "signal", "prior_hint"):
        pert.loc[later, col] = alt.loc[later, col]
    base = compute_features(fr).to_numpy()
    moved = compute_features(pert).to_numpy()
    np.testing.assert_array_equal(base[: t + 1], moved[: t + 1])
    assert not np.array_equal(base[t + 1:], moved[t + 1:])


@pytest.mark.parametrize("family", FAMILIES)
def test_diagnostics_are_not_features(family):
    cols = compute_features(GENERATORS[family](2000, seed=0, symbol="X", start=START)).columns
    assert not {"regime", "mu_next", "jumps"} & set(cols)


def test_every_family_has_the_same_feature_columns():
    frames = load_frames("synthetic_bench", "2021-01", "2021-03",
                         symbols=[f"B{i}" for i in range(4)])
    assert {tuple(compute_features(fr).columns) for fr in frames} == {
        tuple(compute_features(frames[0]).columns)}
    assert {fr.attrs["family"] for fr in frames} == set(FAMILIES)


def test_seed_offset_changes_the_draw_and_repeats_for_the_same_offset():
    names = ["B0", "B1", "B2", "B3"]
    first = load_frames("synthetic_bench", "2021-01", "2021-03", symbols=names, seed_offset=0)
    again = load_frames("synthetic_bench", "2021-01", "2021-03", symbols=names, seed_offset=0)
    other = load_frames("synthetic_bench", "2021-01", "2021-03", symbols=names,
                        seed_offset=1000)
    for a, b, c in zip(first, again, other, strict=True):
        np.testing.assert_array_equal(a["close"].to_numpy(), b["close"].to_numpy())
        assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_family_filter_keeps_one_family_and_unknown_names_fail():
    frames = load_frames("synthetic_bench", "2021-01", "2021-02", symbols=["B0", "B1"],
                         family="threshold")
    assert {fr.attrs["family"] for fr in frames} == {"threshold"}
    with pytest.raises(ValueError):
        load_frames("synthetic_bench", "2021-01", "2021-02", symbols=["B0"], family="nope")
