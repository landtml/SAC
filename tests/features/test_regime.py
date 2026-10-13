import numpy as np
import pandas as pd
import pytest

from sactrader.data.schema import make_frame
from sactrader.features.regime import (
    _GAUSS_LOG_ABS_STD,
    RegimeConfig,
    regime_features,
    regime_warmup_bars,
)

N = 1200
WARM = 64  # longest default window
COLUMNS = [f"{name}_{n}" for n in (16, 64) for name in ("vr", "er", "ac1", "volvol", "skew")]


def _frame(close, kind="spot"):
    close = np.asarray(close, dtype=np.float64)
    idx = pd.date_range("2024-01-01", periods=close.size, freq="1h", tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    kw = {"resolution": 1.0} if kind == "binary" else {}
    return make_frame(idx, open_, np.maximum(open_, close), np.minimum(open_, close), close,
                      symbol="TST", kind=kind, bar="1h", **kw)


def _prices(log_returns):
    return 100.0 * np.exp(np.cumsum(log_returns))


def _ar1(phi, n, seed, sigma=0.01):
    """Stationary AR(1) as a truncated moving average, so no loop over rows is needed."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n + 60)
    return np.convolve(eps, phi ** np.arange(60), mode="valid")[:n]


def _mean(out, col):
    return float(out[col].iloc[WARM:].mean())


def test_contract_columns_index_dtype_and_warmup():
    rng = np.random.default_rng(1)
    frame = _frame(_prices(rng.normal(0.0, 0.01, N)))
    out = regime_features(frame)
    assert list(out.columns) == COLUMNS
    assert out.index.equals(frame.index)
    assert set(out.dtypes) == {np.dtype("float32")}
    assert regime_warmup_bars() == WARM
    assert out.iloc[:WARM]["vr_64"].isna().all()
    assert not out.iloc[WARM:].isna().any().any()
    assert out.iloc[WARM:].abs().max().max() <= 5.0


def test_rejects_windows_shorter_than_eight_bars():
    frame = _frame(_prices(np.random.default_rng(2).normal(0.0, 0.01, N)))
    with pytest.raises(ValueError):
        regime_features(frame, RegimeConfig(windows=(4, 16)))


def test_causal_rows_before_t_ignore_later_prices():
    rng = np.random.default_rng(3)
    close = _prices(rng.normal(0.0, 0.01, N))
    t = 700
    shocked = close.copy()
    shocked[t:] *= np.exp(rng.normal(0.0, 0.2, N - t))
    base = regime_features(_frame(close))
    after = regime_features(_frame(shocked))
    np.testing.assert_array_equal(base.iloc[:t].to_numpy(), after.iloc[:t].to_numpy())
    assert not np.array_equal(base.iloc[t:].to_numpy(), after.iloc[t:].to_numpy())


def test_scale_free_under_price_multiplier():
    close = _prices(np.random.default_rng(4).normal(0.0, 0.01, N))
    a = regime_features(_frame(close)).to_numpy()
    b = regime_features(_frame(7.0 * close)).to_numpy()
    np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-5)


def test_ar1_signs_for_trend_and_autocorrelation():
    pos = regime_features(_frame(_prices(_ar1(0.5, N, seed=5))))
    neg = regime_features(_frame(_prices(_ar1(-0.5, N, seed=6))))
    assert _mean(pos, "ac1_64") > 1.0
    assert _mean(pos, "vr_64") > 0.2
    assert _mean(neg, "ac1_64") < -1.0
    assert _mean(neg, "vr_64") < -0.2


def test_straight_line_drift_with_tiny_noise_is_efficient():
    rng = np.random.default_rng(7)
    log_price = np.arange(N) * 0.001 + rng.normal(0.0, 1e-5, N)
    out = regime_features(_frame(100.0 * np.exp(log_price)))
    assert _mean(out, "er_16") > 0.99
    assert _mean(out, "er_64") > 0.99


def test_gaussian_log_abs_std_matches_closed_form():
    assert _GAUSS_LOG_ABS_STD == pytest.approx(np.pi / (2.0 * np.sqrt(2.0)), abs=1e-8)


def test_volvol_near_zero_on_gaussian_and_positive_with_jumps():
    # A 64-bar window std is noisy row by row, so the mean needs a long series to settle.
    n = 20000
    rng = np.random.default_rng(8)
    gauss = regime_features(_frame(_prices(rng.normal(0.0, 0.01, n))))
    assert abs(_mean(gauss, "volvol_64")) < 0.1
    scale = np.where(rng.uniform(size=n) < 0.05, 0.2, 0.01)
    jumpy = regime_features(_frame(_prices(rng.normal(0.0, 1.0, n) * scale)))
    assert _mean(jumpy, "volvol_64") > 0.1


def test_flat_prices_give_zeros_not_nan_after_warmup():
    out = regime_features(_frame(np.full(N, 50.0)))
    assert not out.iloc[WARM:].isna().any().any()
    assert (out.iloc[WARM:].to_numpy() == 0.0).all()


def test_binary_frame_uses_logit_increments():
    logit = _ar1(0.5, N, seed=9, sigma=0.05)
    prob = 1.0 / (1.0 + np.exp(-logit))
    out = regime_features(_frame(prob, kind="binary"))
    assert list(out.columns) == COLUMNS
    assert not out.iloc[WARM:].isna().any().any()
    d = np.diff(logit)[-64:]
    expected_er = 2.0 * (abs(d.sum()) / abs(d).sum() - 0.5)
    assert out["er_64"].iloc[-1] == pytest.approx(expected_er, abs=1e-4)


def test_clip_bounds_every_column():
    rng = np.random.default_rng(10)
    jumps = np.where(rng.uniform(size=N) < 0.05, 30.0, 1.0)
    out = regime_features(_frame(_prices(rng.normal(0.0, 0.01, N) * jumps)),
                          RegimeConfig(clip=0.5))
    assert (out.iloc[WARM:].abs() <= 0.5).all().all()
