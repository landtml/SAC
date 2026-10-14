"""Squeeze gate: fixed thresholds, causality, and the gated funding prior feature."""

import numpy as np
import pandas as pd
import pytest

from softedge.data.schema import make_frame
from softedge.features import squeeze
from softedge.features.causal import compute_features


def daily_oi(values, start="2022-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(np.asarray(values, dtype=float), index=idx, name="oi_value")


def perp_daily(oi, seed=1):
    n = len(oi)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    frame = make_frame(idx, close, close * 1.01, close * 0.99, close, np.full(n, 10.0),
                       symbol="BTCUSDT-BASIS", kind="perp", bar="1D",
                       funding_rate=rng.normal(1e-4, 2e-4, n))
    frame["oi_value"] = np.asarray(oi, dtype=float)
    return frame


def test_thresholds_are_the_preregistered_values():
    assert (squeeze.WINDOW_DAYS, squeeze.HISTORY_DAYS, squeeze.QUANTILE) == (7, 180, 0.8)


def test_gate_closes_while_growth_exceeds_trailing_80th_percentile():
    oi = np.full(300, 100.0)
    oi[250:] = 150.0  # one jump: seven-day growth is positive on days 250 to 256 only
    gate = squeeze.squeeze_gate(daily_oi(oi), "1D")
    assert (gate.iloc[250:257] == 0.0).all()
    assert (gate.iloc[257:] == 1.0).all()


def test_gate_is_closed_until_the_history_is_complete():
    gate = squeeze.squeeze_gate(daily_oi(np.full(300, 100.0)), "1D")
    # Growth exists from day 7, so the first 180-day distribution ends on day 186.
    assert (gate.iloc[:186] == 0.0).all()
    assert gate.iloc[186] == 1.0


def test_leading_days_without_a_sample_are_closed():
    oi = np.full(260, np.nan)
    oi[50:] = 100.0
    gate = squeeze.squeeze_gate(daily_oi(oi), "1D")
    # Growth exists from day 57 and the distribution from day 236.
    assert (gate.iloc[:236] == 0.0).all()
    assert gate.iloc[236] == 1.0


def test_missing_day_holds_the_last_sample():
    oi = np.linspace(100.0, 160.0, 260)
    gapped = oi.copy()
    gapped[120] = np.nan
    held = oi.copy()
    held[120] = oi[119]
    pd.testing.assert_series_equal(squeeze.squeeze_gate(daily_oi(gapped), "1D"),
                                   squeeze.squeeze_gate(daily_oi(held), "1D"))


def test_gate_ignores_oi_after_bar_t():
    rng = np.random.default_rng(0)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.05, 400)))
    t = 300
    alt = base.copy()
    alt[t + 1:] = base[t + 1:] * np.exp(rng.normal(0.0, 1.0, base.size - t - 1))
    g0 = squeeze.squeeze_gate(daily_oi(base), "1D")
    g1 = squeeze.squeeze_gate(daily_oi(alt), "1D")
    assert (g0.iloc[: t + 1] == g1.iloc[: t + 1]).all()
    assert (g0.iloc[t + 1:] != g1.iloc[t + 1:]).any()


def test_gate_needs_daily_bars():
    with pytest.raises(ValueError, match="daily"):
        squeeze.squeeze_gate(daily_oi(np.full(300, 100.0)), "1h")


def test_gated_prior_is_funding_ewm_times_gate():
    oi = 100.0 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.0, 0.03, 400)))
    frame = perp_daily(oi)
    table = compute_features(frame)
    gate = squeeze.squeeze_gate(frame["oi_value"], "1D").to_numpy()
    np.testing.assert_allclose(table["funding_ewm_gated"].to_numpy(),
                               table["funding_ewm"].to_numpy() * gate, rtol=1e-5, atol=1e-6)


def test_gated_prior_ignores_oi_after_bar_t():
    rng = np.random.default_rng(3)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, 400)))
    t = 300
    alt = base.copy()
    alt[t + 1:] = base[t + 1:] * np.exp(rng.normal(0.0, 1.0, base.size - t - 1))
    a = compute_features(perp_daily(base))["funding_ewm_gated"].to_numpy()
    b = compute_features(perp_daily(alt))["funding_ewm_gated"].to_numpy()
    assert np.array_equal(a[: t + 1], b[: t + 1])
    assert not np.array_equal(a[t + 1:], b[t + 1:])


def test_gated_prior_needs_an_open_interest_column():
    frame = perp_daily(np.full(300, 100.0))
    del frame["oi_value"]
    assert "funding_ewm_gated" not in compute_features(frame).columns
