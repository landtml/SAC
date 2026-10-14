"""Variance risk premium frames: daily payoff, causal observations, and the ``vrp`` feature."""

import numpy as np
import pandas as pd
import pytest

from softedge.data.variance import variance_frame, vrp_frames
from softedge.features.causal import compute_features
from softedge.pipeline import to_tapes


def _days(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")


def _market(n: int = 120, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Daily opens of an underlying and hourly DVOL candles that end on the hour."""
    rng = np.random.default_rng(seed)
    days = _days(n)
    opens = pd.Series(30_000 * np.exp(np.cumsum(0.03 * rng.standard_normal(n))), index=days)
    ends = pd.date_range(days[0], days[-1] + pd.Timedelta("2D"), freq="h")
    dvol = pd.Series(0.45 + 0.1 * np.sin(np.arange(len(ends)) / 500.0), index=ends)
    return opens, dvol


def test_close_moves_by_implied_minus_realized_variance_each_day():
    opens, dvol = _market()
    fr = variance_frame(opens, dvol, "BTC")
    implied = dvol.reindex(opens.index).to_numpy()
    r = np.diff(np.log(opens.to_numpy()))
    payoff = implied[:-1] ** 2 / 365.0 - r**2
    np.testing.assert_allclose(fr["close"].pct_change().to_numpy()[1:], payoff, rtol=1e-9)


def test_dvol_is_the_last_candle_ending_at_or_before_the_open():
    days = _days(3)
    opens = pd.Series([1.0, 1.1, 1.2], index=days)
    ends = pd.DatetimeIndex([
        days[0],                      # ends at day 0 open: used there
        days[0] + pd.Timedelta("1h"),  # ends after day 0 open: not used there
        days[1],                      # ends at day 1 open: used there
        days[2] + pd.Timedelta("1h"),  # ends after day 2 open: not used there
    ], tz="UTC")
    dvol = pd.Series([0.40, 0.90, 0.50, 0.99], index=ends)
    fr = variance_frame(opens, dvol, "BTC")
    np.testing.assert_array_equal(fr["dvol_open"].to_numpy(), [0.40, 0.50, 0.50])


def test_rv30_is_root_of_mean_squared_daily_returns_over_the_30_prior_days():
    opens, dvol = _market()
    fr = variance_frame(opens, dvol, "BTC")
    r = np.diff(np.log(opens.to_numpy()))
    t = 45
    expected = np.sqrt(365.0 * np.mean(r[t - 30:t] ** 2))
    assert fr["rv30"].iloc[t] == pytest.approx(expected, rel=1e-9)
    assert np.isnan(fr["rv30"].iloc[29])


def test_rv30_excludes_the_day_it_describes():
    opens, dvol = _market()
    spiked = opens.copy()
    spiked.iloc[46:] *= 1.5  # changes only the return of day 45, open to next open
    base = variance_frame(opens, dvol, "BTC")
    late = variance_frame(spiked, dvol, "BTC")
    assert base["rv30"].iloc[45] == late["rv30"].iloc[45]
    assert late["rv30"].iloc[46] > base["rv30"].iloc[46]


@pytest.mark.parametrize("t", [40, 70])
def test_observations_at_open_t_ignore_data_after_it(t):
    opens, dvol = _market(n=100)
    later_opens = opens.copy()
    later_opens.iloc[t + 1:] *= 1.7
    later_dvol = dvol.copy()
    later_dvol[later_dvol.index > opens.index[t]] *= 2.0
    base = variance_frame(opens, dvol, "BTC")
    late = variance_frame(later_opens, later_dvol, "BTC")
    cols = ["close", "dvol_open", "rv30"]
    np.testing.assert_array_equal(base[cols].iloc[:t + 1].to_numpy(),
                                  late[cols].iloc[:t + 1].to_numpy())
    feats_base = compute_features(base).iloc[:t + 1]
    feats_late = compute_features(late).iloc[:t + 1]
    np.testing.assert_array_equal(feats_base.to_numpy(), feats_late.to_numpy())


def test_vrp_feature_is_implied_minus_trailing_realized_variance():
    opens, dvol = _market()
    fr = variance_frame(opens, dvol, "ETH")
    feats = compute_features(fr)
    t = 60
    expected = fr["dvol_open"].iloc[t] ** 2 - fr["rv30"].iloc[t] ** 2
    assert feats["vrp"].iloc[t] == pytest.approx(expected, abs=1e-6)


def test_tape_next_bar_return_is_the_daily_payoff():
    opens, dvol = _market(n=400)
    fr = variance_frame(opens, dvol, "BTC")
    (tp,) = to_tapes([fr])
    assert "vrp" in tp.feature_names
    start = len(fr) - len(tp)
    days = range(start, len(fr) - 1)
    implied = fr["dvol_open"].to_numpy()
    r = np.diff(np.log(opens.to_numpy()))
    payoff = [implied[t] ** 2 / 365.0 - r[t] ** 2 for t in days]
    np.testing.assert_allclose(tp.long_ret[:-1], payoff, rtol=1e-9)


def test_vrp_frames_join_hourly_spot_opens_with_dvol(monkeypatch):
    from softedge.data import binance, deribit

    opens, dvol = _market(n=60)
    hours = pd.date_range(opens.index[0], opens.index[-1] + pd.Timedelta("23h"), freq="h")
    hourly = pd.DataFrame({"open": opens.reindex(hours.floor("D")).to_numpy()}, index=hours)
    calls = []

    def fake_spot(symbol, interval, start, end):
        calls.append((symbol, interval))
        return hourly

    monkeypatch.setattr(binance, "load_spot", fake_spot)
    monkeypatch.setattr(deribit, "load_dvol", lambda cur, start, end, res: dvol)
    (fr,) = vrp_frames("2022-01", "2022-03", ("BTC",))
    assert calls == [("BTCUSDT", "1h")]
    assert fr.attrs["symbol"] == "BTC-VRP" and fr.attrs["kind"] == "spot"
    np.testing.assert_allclose(fr["open"].iloc[1:].to_numpy(), fr["close"].iloc[:-1].to_numpy())
    assert len(fr) == len(opens)
