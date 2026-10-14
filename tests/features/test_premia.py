import numpy as np
import pandas as pd
import pytest

from softedge.data.schema import make_frame
from softedge.data.synthetic import binary_market, gbm
from softedge.features import premia
from softedge.features.premia import (
    CARRY_WINDOW,
    TSMOM_LOOKBACK,
    TSMOM_SKIP,
    TSMOM_TARGET,
    premia_features,
    slope_on_index,
)
from softedge.pipeline import to_tapes

N = 900
K = 500  # changes after this bar must not move any earlier row


def _daily(close, symbol="TST", kind="spot", rate_diff=None):
    close = np.asarray(close, dtype=np.float64)
    idx = pd.date_range("2020-01-01", periods=close.size, freq="1D", tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    frame = make_frame(idx, open_, np.maximum(open_, close), np.minimum(open_, close), close,
                       symbol=symbol, kind=kind, bar="1D")
    if rate_diff is not None:
        frame["rate_diff"] = np.asarray(rate_diff, dtype=np.float64)
    return frame


def _path(log_returns):
    return 100.0 * np.exp(np.cumsum(log_returns))


def _ewm_var_ref(x, span):
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = (1.0 - alpha) * out[i - 1] + alpha * x[i]
    return out


def test_tsmom_skips_the_last_month():
    # Up 0.1% a bar for the whole year, then a 5% drop in each of the last 21 bars. The
    # 12-month return that skips the last month is still up, so the signal stays long.
    steps = np.r_[np.full(N - TSMOM_SKIP, 0.001), np.full(TSMOM_SKIP, -0.05)]
    feats = premia_features(_daily(_path(steps)))
    assert feats["tsmom_12m"].iloc[-1] > 0.0
    full_window = steps[-TSMOM_LOOKBACK:].sum()
    assert full_window < 0.0  # the rule would have flipped had it kept the last month


def test_tsmom_is_sign_times_inverse_vol_after_warmup():
    steps = np.random.default_rng(1).normal(-0.0005, 0.01, N)
    close = _path(steps)
    feats = premia_features(_daily(close))["tsmom_12m"].to_numpy(np.float64)
    r = np.diff(np.log(close))
    # sigma[t] for t >= 1 starts its EWMA at the first return, as the feature does.
    sigma = np.r_[np.nan, np.sqrt(_ewm_var_ref(r * r, 32))]
    t = np.arange(TSMOM_LOOKBACK, N)
    lookback = np.log(close[t - TSMOM_SKIP]) - np.log(close[t - TSMOM_LOOKBACK])
    expected = np.sign(lookback) * TSMOM_TARGET / np.maximum(sigma[t], 1e-8)
    np.testing.assert_allclose(feats[t], np.clip(expected, -5.0, 5.0), rtol=1e-4, atol=1e-6)


def test_tsmom_is_zero_before_a_full_lookback():
    feats = premia_features(_daily(_path(np.random.default_rng(2).normal(0, 0.01, N))))
    assert np.all(feats["tsmom_12m"].iloc[:TSMOM_LOOKBACK] == 0.0)


def test_carry_trend_keeps_carry_only_when_the_trend_agrees():
    n = 200
    up = _path(np.full(n, 0.002))
    down = _path(np.full(n, -0.002))
    carry = np.full(n, 0.05)  # a 5% differential maps to 1.0 under the carry scaling
    agree = premia_features(_daily(up, kind="fx", rate_diff=carry))["carry_trend"]
    assert np.allclose(agree.iloc[CARRY_WINDOW + 1:], 1.0)
    dropped = premia_features(_daily(down, kind="fx", rate_diff=carry))["carry_trend"]
    assert np.all(dropped.iloc[CARRY_WINDOW + 1:] == 0.0)
    short_carry = premia_features(_daily(down, kind="fx", rate_diff=-carry))["carry_trend"]
    assert np.allclose(short_carry.iloc[CARRY_WINDOW + 1:], -1.0)


def test_carry_trend_absent_without_a_rate_differential():
    feats = premia_features(_daily(_path(np.full(N, 0.001))))
    assert "carry_trend" not in feats.columns


@pytest.mark.parametrize("kind", ("spot", "fx"))
def test_future_rows_do_not_change_past_premia(kind):
    rng = np.random.default_rng(5)
    base_steps = rng.normal(0.0, 0.01, N)
    alt_steps = rng.normal(0.0, 0.01, N)
    rate = rng.uniform(-0.02, 0.05, N)
    base = _daily(_path(base_steps), kind=kind, rate_diff=rate)
    alt_close = np.r_[_path(base_steps)[: K + 1], _path(alt_steps)[K + 1:]]
    alt_rate = np.r_[rate[: K + 1], rng.uniform(-0.02, 0.05, N - K - 1)]
    changed = _daily(alt_close, kind=kind, rate_diff=alt_rate)
    full = premia_features(base)
    perturbed = premia_features(changed)
    np.testing.assert_allclose(perturbed.iloc[: K + 1].to_numpy(), full.iloc[: K + 1].to_numpy(),
                               atol=1e-6)
    assert not np.allclose(perturbed.iloc[K + 1:].to_numpy(), full.iloc[K + 1:].to_numpy())


def test_tsmom_is_scale_free():
    close = _path(np.random.default_rng(7).normal(0.0, 0.01, N))
    a = premia_features(_daily(close))["tsmom_12m"].to_numpy()
    b = premia_features(_daily(7.0 * close))["tsmom_12m"].to_numpy()
    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-6)


def test_slope_uses_the_last_value_stamped_at_or_before_each_bar():
    idx = pd.date_range("2020-01-01", periods=6, freq="1D", tz="UTC")
    ten = pd.Series([2.0, 2.5, 3.0], index=idx[[0, 2, 4]])
    two = pd.Series([1.0, 1.0, 1.5], index=idx[[0, 3, 4]])
    got = slope_on_index(idx, ten, two)
    # Each day reads the last stamped value of each tenor: day 2 takes the new ten-year (2.5)
    # against the old two-year (1.0), and day 4 takes both new values (3.0 - 1.5).
    np.testing.assert_allclose(got, [1.0, 1.0, 1.5, 1.5, 1.5, 1.5])


def test_slope_is_zero_before_both_yields_exist():
    idx = pd.date_range("2019-12-30", periods=4, freq="1D", tz="UTC")
    ten = pd.Series([2.0], index=idx[[2]])
    two = pd.Series([1.0], index=idx[[1]])
    got = slope_on_index(idx, ten, two)
    np.testing.assert_allclose(got, [np.nan, np.nan, 1.0, 1.0], equal_nan=True)


def test_curve_slope_reads_treasury_yields_only_for_treasury_tapes(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, freq="1D", tz="UTC")
    yields = {
        "DGS10": pd.Series([2.0, 2.2, 2.4, 2.6, 2.8], index=idx),
        "DGS2": pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx),
    }
    monkeypatch.setattr(premia, "_fred_yield", lambda series_id: yields[series_id])
    ust = _daily(_path(np.full(5, 0.001)), symbol="UST10")
    ust_feats = premia_features(ust)
    np.testing.assert_allclose(ust_feats["curve_slope"].to_numpy(), [1.0, 1.2, 1.4, 1.6, 1.8],
                               rtol=1e-5)
    spx = premia_features(_daily(_path(np.full(5, 0.001)), symbol="SPX"))
    assert np.all(spx["curve_slope"].to_numpy() == 0.0)


def test_binary_markets_get_no_premia_columns():
    frame = binary_market(200, seed=3, symbol="B", start="2021-01-01")
    assert premia_features(frame).shape[1] == 0


def test_premia_features_are_finite_and_clipped():
    frame = gbm(600, sigma=0.01, seed=0, symbol="G", start="2021-01-01")
    feats = premia_features(frame)
    assert np.isfinite(feats.to_numpy()).all()
    assert feats.abs().to_numpy().max() <= 5.0


def test_to_tapes_adds_premia_only_on_request():
    frames = [gbm(600, sigma=0.01, seed=i, symbol=f"S{i}") for i in range(2)]
    plain = to_tapes(frames)
    with_premia = to_tapes(frames, premia=True)
    assert "tsmom_12m" in with_premia[0].feature_names
    assert "tsmom_12m" not in plain[0].feature_names
    assert len(with_premia[0]) <= len(plain[0])
