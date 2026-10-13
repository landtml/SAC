import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis

from softedge.data.schema import validate_frame
from softedge.data.synthetic import nonlinear_regime_market

N_LONG = 50000
SNR = 0.08  # default snr of nonlinear_regime_market
P_STAY = 0.99  # default p_stay of nonlinear_regime_market


def _log_returns(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame["close"] / frame["open"]).to_numpy()


def _sharpe(pnl: np.ndarray) -> float:
    return pnl.mean() / pnl.std(ddof=1)


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    return nonlinear_regime_market(N_LONG, symbol="X", seed=0)


def test_frame_is_schema_valid_with_diagnostic_columns():
    frame = nonlinear_regime_market(500, symbol="X", seed=1)
    validate_frame(frame)
    assert frame.attrs["kind"] == "spot"
    assert frame["regime"].dtype == np.int64
    assert set(np.unique(frame["regime"])) <= {-1, 1}
    assert np.isfinite(frame[["signal", "mu_next"]].to_numpy()).all()


def test_same_seed_gives_identical_frame_and_other_seed_differs():
    a = nonlinear_regime_market(500, symbol="X", seed=3)
    b = nonlinear_regime_market(500, symbol="X", seed=3)
    c = nonlinear_regime_market(500, symbol="X", seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs == b.attrs
    assert not a["close"].equals(c["close"])


def test_bar_returns_have_excess_kurtosis_above_one_for_df4(market):
    assert kurtosis(_log_returns(market)) > 1.0


def test_absolute_returns_show_volatility_clustering(market):
    absr = np.abs(_log_returns(market))
    assert np.corrcoef(absr[:-1], absr[1:])[0, 1] > 0.1


def test_regime_switch_rate_is_within_30_percent_of_one_minus_p_stay(market):
    regime = market["regime"].to_numpy()
    rate = np.mean(regime[1:] != regime[:-1])
    assert abs(rate - (1 - P_STAY)) < 0.3 * (1 - P_STAY)


def test_expected_return_predicts_the_next_bar_return(market):
    r = _log_returns(market)
    mu = market["mu_next"].to_numpy()
    assert np.corrcoef(mu[:-1], r[1:])[0, 1] > 0.03


def test_oracle_sign_of_expected_return_earns_positive_sharpe_near_snr_eg(market):
    r = _log_returns(market)
    mu = market["mu_next"].to_numpy()[:-1]
    pnl = np.sign(mu) * r[1:]
    assert pnl.mean() > 0
    # sigma_t is not stored, so the scale of g comes from the realized return std.
    mean_abs_g = np.mean(np.abs(mu)) / (SNR * r.std(ddof=1))
    assert 0.5 * SNR * mean_abs_g < _sharpe(pnl) < 2 * SNR * mean_abs_g


def test_regime_blind_momentum_has_less_than_half_the_oracle_sharpe(market):
    r = _log_returns(market)
    mu = market["mu_next"].to_numpy()[:-1]
    oracle = _sharpe(np.sign(mu) * r[1:])
    # Row j of past8 sums r[j:j + 8], so it is known at the close of row j + 7.
    past8 = np.convolve(r, np.ones(8), mode="valid")
    momentum = _sharpe(np.sign(past8[:-1]) * r[8:])
    assert momentum < 0.5 * oracle


@pytest.mark.parametrize("kwargs", [
    {"kind": "perp"},
    {"sigma": 0.0},
    {"df": 2.0},
    {"garch": (0.5, 0.5)},
    {"p_stay": 1.0},
    {"rho": 1.0},
])
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        nonlinear_regime_market(50, symbol="X", **kwargs)
