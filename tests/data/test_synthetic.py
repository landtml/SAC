import numpy as np
import pandas as pd
import pytest

from sactrader.data.schema import validate_frame
from sactrader.data.synthetic import (
    ar_returns,
    binary_market,
    gbm,
    ou_mean_reverting,
    planted_signal,
    regime_switching,
    with_funding,
)

N = 500


def _log_returns(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame["close"] / frame["open"]).to_numpy()


GENERATORS = {
    "gbm": lambda seed: gbm(N, 1e-4, 0.01, seed=seed, symbol="X"),
    "gbm_fx": lambda seed: gbm(N, sigma=0.002, kind="fx", seed=seed, symbol="X"),
    "ar_returns": lambda seed: ar_returns(N, seed=seed, symbol="X"),
    "planted_signal": lambda seed: planted_signal(N, seed=seed, symbol="X"),
    "regime_switching": lambda seed: regime_switching(N, seed=seed, symbol="X"),
    "ou_mean_reverting": lambda seed: ou_mean_reverting(N, seed=seed, symbol="X"),
    "with_funding": lambda seed: with_funding(gbm(N, seed=seed, symbol="X"), seed=seed),
    "binary_market": lambda seed: binary_market(N, seed=seed, symbol="X"),
}


@pytest.mark.parametrize("name", list(GENERATORS))
def test_every_generator_passes_validate_frame(name):
    frame = GENERATORS[name](seed=1)
    validate_frame(frame)
    assert len(frame) == N


@pytest.mark.parametrize("name", list(GENERATORS))
def test_same_seed_gives_identical_frame_and_other_seed_differs(name):
    a = GENERATORS[name](seed=3)
    b = GENERATORS[name](seed=3)
    c = GENERATORS[name](seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs == b.attrs
    assert not a["close"].equals(c["close"])


def test_gbm_log_return_mean_and_std_within_three_standard_errors():
    mu, sigma, n = 2e-4, 0.01, 20000
    r = _log_returns(gbm(n, mu, sigma, symbol="X", seed=0))
    assert abs(r.mean() - mu) < 3 * sigma / np.sqrt(n)
    assert abs(r.std(ddof=1) - sigma) < 3 * sigma / np.sqrt(2 * n)


def test_ar_returns_lag1_autocorrelation_matches_phi():
    phi = 0.3
    r = _log_returns(ar_returns(20000, phi=phi, symbol="X", seed=0))
    assert abs(np.corrcoef(r[:-1], r[1:])[0, 1] - phi) < 0.03


def test_planted_signal_regression_slope_matches_strength_times_sigma():
    strength, sigma = 0.1, 0.01
    frame = planted_signal(50000, strength=strength, sigma=sigma, rho=0.9, symbol="X", seed=0)
    r = _log_returns(frame)
    signal = frame["signal"].to_numpy()
    slope = np.polyfit(signal[:-1], r[1:], 1)[0]
    assert abs(slope - strength * sigma) < 0.2 * strength * sigma


def test_ou_return_regression_slope_on_log_deviation_is_minus_theta():
    theta, sigma, n = 0.02, 0.01, 50000
    frame = ou_mean_reverting(n, theta=theta, sigma=sigma, symbol="X", seed=0)
    dev = np.log(frame["open"] / frame["open"].iloc[0]).to_numpy()
    slope = np.polyfit(dev, _log_returns(frame), 1)[0]
    assert abs(slope + theta) < 0.2 * theta


def test_regime_states_and_switch_rate():
    n, p_stay = 200000, 0.995
    frame = regime_switching(n, p_stay=p_stay, symbol="X", seed=0)
    regime = frame["regime"].to_numpy()
    assert regime.dtype == np.int64
    assert set(np.unique(regime)) == {0, 1, 2}
    assert abs(np.mean(regime[1:] != regime[:-1]) - (1 - p_stay)) < 0.001


def test_funding_nonzero_only_at_interval_hours():
    base = gbm(24 * 30, symbol="X", seed=0)
    frame = with_funding(base, interval_hours=8, seed=0)
    assert frame.attrs["kind"] == "perp"
    pd.testing.assert_frame_equal(frame[base.columns], base)
    charged = frame.index[frame["funding_rate"].to_numpy() != 0.0]
    assert (charged.hour % 8 == 0).all()
    assert (charged.minute == 0).all()
    assert len(charged) == 30 * 3


def test_binary_calibration_at_mid_life_without_longshot_bias():
    n, markets = 24, 400
    mid_price = np.empty(markets)
    resolution = np.empty(markets)
    for seed in range(markets):
        frame = binary_market(n, seed=seed, symbol="YES")
        mid_price[seed] = frame["close"].iloc[n // 2]
        resolution[seed] = frame.attrs["resolution"]
    bucket = (mid_price >= 0.6) & (mid_price < 0.8)
    assert bucket.sum() >= 50
    assert abs(resolution[bucket].mean() - mid_price[bucket].mean()) < 0.1


@pytest.mark.parametrize("info_rate, longshot_bias", [(0.05, 0.0), (0.5, 0.0), (0.05, 0.3)])
def test_binary_prices_lie_strictly_inside_unit_interval(info_rate, longshot_bias):
    for seed in range(20):
        frame = binary_market(200, info_rate=info_rate, longshot_bias=longshot_bias,
                              seed=seed, symbol="YES")
        prices = frame[["open", "high", "low", "close"]].to_numpy()
        assert ((prices > 0.0) & (prices < 1.0)).all()


def test_binary_attrs_hold_resolution_and_end_time():
    frame = binary_market(24, symbol="YES", seed=2)
    assert frame.attrs["kind"] == "binary"
    assert frame.attrs["resolution"] in (0.0, 1.0)
    assert frame.attrs["end_time"] == (frame.index[-1] + pd.Timedelta("1h")).isoformat()


def test_longshot_bias_scales_observed_log_odds_and_keeps_resolution():
    base = binary_market(50, noise=0.0, seed=7, symbol="YES")
    biased = binary_market(50, noise=0.0, longshot_bias=0.3, seed=7, symbol="YES")
    assert biased.attrs["resolution"] == base.attrs["resolution"]

    def logit(p):
        return np.log(p / (1.0 - p))

    p0 = base["close"].to_numpy()
    p1 = biased["close"].to_numpy()
    interior = (p0 > 0.02) & (p0 < 0.98) & (p1 > 0.02) & (p1 < 0.98)
    assert interior.sum() > 10
    np.testing.assert_allclose(logit(p1[interior]), 0.7 * logit(p0[interior]), atol=1e-9)


@pytest.mark.parametrize("bad", [
    lambda: gbm(10, kind="perp", symbol="X"),
    lambda: with_funding(binary_market(10, symbol="X", seed=0)),
    lambda: ar_returns(10, phi=1.0, symbol="X"),
])
def test_invalid_arguments_raise(bad):
    with pytest.raises(ValueError):
        bad()
