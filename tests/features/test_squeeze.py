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


T0 = 200  # decision day of the designed series below


def step_oi(first_step_day: int = 220, n: int = 300) -> np.ndarray:
    """Flat open interest with one step up from ``first_step_day``: growth is positive there."""
    oi = np.full(n, 100.0)
    oi[first_step_day:] = 150.0
    return oi


def oi_from_growth(g: np.ndarray) -> pd.Series:
    """Daily USD open interest whose seven-day log growth is exactly ``g[t]`` for ``t >= 7``."""
    log_oi = np.zeros(len(g))
    for t in range(7, len(g)):
        log_oi[t] = log_oi[t - 7] + g[t]
    idx = pd.date_range("2022-01-01", periods=len(g), freq="D", tz="UTC")
    return pd.Series(100.0 * np.exp(log_oi), index=idx, name="oi_value")


def designed_growth(seed: int) -> np.ndarray:
    """Growth in which bar ``T0`` is closed by the rule, and is open if the rule read bar T0+1.

    The window ending T0 holds 180 values; the oldest is the minimum, and the growth at T0 is
    the 145th smallest. The 80th percentile then lies just below it, so growth at T0 exceeds the
    limit. Dropping the minimum and adding a value at T0+1 lifts the percentile to at least the
    growth at T0, so a rule that read the next day's growth would open the gate at T0.
    """
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0, 0.02, T0 + 2)
    first = T0 - 179
    g[first] = -1.0
    below = np.sort(g[first + 1:T0])
    g[T0] = (below[142] + below[143]) / 2.0
    return g


def test_thresholds_are_fixed_at_the_framing_values():
    assert (squeeze.WINDOW_DAYS, squeeze.HISTORY_DAYS, squeeze.QUANTILE) == (7, 180, 0.8)


def test_gate_closes_while_growth_exceeds_trailing_80th_percentile():
    oi = np.full(300, 100.0)
    oi[250:] = 150.0  # one jump: seven-day growth is positive on days 250 to 256 only
    gate = squeeze.squeeze_gate(daily_oi(oi), "1D")
    assert (gate.iloc[250:257] == 0.0).all()
    assert (gate.iloc[257:] == 1.0).all()


def test_gate_is_open_until_the_history_is_complete():
    # A flat series has no growth to close on, and no gate is computable before day 186.
    gate = squeeze.squeeze_gate(daily_oi(np.full(300, 100.0)), "1D")
    assert (gate == 1.0).all()


def test_leading_days_without_a_sample_are_open():
    oi = np.full(300, np.nan)
    oi[50:] = step_oi(first_step_day=200, n=250)  # the step is day 250 of the series
    gate = squeeze.squeeze_gate(daily_oi(oi), "1D")
    assert (gate.iloc[:50] == 1.0).all()
    assert (gate.iloc[250:257] == 0.0).all()


def test_missing_day_is_open_and_is_not_filled():
    # Without the gap, the step closes the gate on days 220 to 226. A missing day 120 makes every
    # growth and limit whose window holds it missing, so the gate is open there. Filling day 120
    # from day 121 would compute those windows and close the gate again on days 220 to 226.
    full = daily_oi(step_oi())
    gapped = full.copy()
    gapped.iloc[120] = np.nan
    assert (squeeze.squeeze_gate(full, "1D").iloc[220:227] == 0.0).all()
    gate = squeeze.squeeze_gate(gapped, "1D")
    assert (gate.iloc[120:307] == 1.0).all()
    pd.testing.assert_series_equal(gate.iloc[307:], squeeze.squeeze_gate(full, "1D").iloc[307:])


def test_dropped_day_counts_as_missing_on_the_calendar():
    # A day with no row is the same as a day with a missing sample, so windows count calendar
    # days, not rows.
    full = daily_oi(step_oi())
    with_gap = full.copy()
    with_gap.iloc[120] = np.nan
    dropped = full.drop(full.index[120])
    expected = squeeze.squeeze_gate(with_gap, "1D").drop(full.index[120])
    pd.testing.assert_series_equal(squeeze.squeeze_gate(dropped, "1D"), expected)


def test_gate_ignores_oi_after_bar_t():
    # Bar T0 is closed by the rule. A shock to day T0+1 must leave bar T0 closed. Read as
    # the next day, a positive shock lifts the limit and a negative one lowers the growth.
    g = designed_growth(seed=0)
    base = squeeze.squeeze_gate(oi_from_growth(g), "1D")
    assert base.iloc[T0] == 0.0
    for shock in (2.0, -2.0):
        alt_g = g.copy()
        alt_g[T0 + 1] = shock
        alt = squeeze.squeeze_gate(oi_from_growth(alt_g), "1D")
        assert (alt.iloc[: T0 + 1] == base.iloc[: T0 + 1]).all(), shock


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
    # A level shift from bar t+1 on: a leak through the next day's open interest would move
    # the growth at bar t above its limit and close the gate there.
    alt = base.copy()
    alt[t + 1:] = base[t + 1:] * 1.5
    a = compute_features(perp_daily(base))["funding_ewm_gated"].to_numpy()
    b = compute_features(perp_daily(alt))["funding_ewm_gated"].to_numpy()
    assert np.array_equal(a[: t + 1], b[: t + 1])
    assert not np.array_equal(a[t + 1:], b[t + 1:])


def test_gated_prior_needs_an_open_interest_column():
    frame = perp_daily(np.full(300, 100.0))
    del frame["oi_value"]
    assert "funding_ewm_gated" not in compute_features(frame).columns
