import numpy as np
import pandas as pd
import pytest

from softedge.data import fred_markets, long_daily


def _csv(dates, values) -> bytes:
    lines = ["observation_date,VALUE"] + [f"{d},{v}" for d, v in zip(dates, values, strict=True)]
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture
def fred(monkeypatch):
    """Synthetic FRED bodies by series id. A series not set reads as an empty header."""
    store: dict[str, bytes] = {}
    monkeypatch.setattr(fred_markets, "_fred_csv", lambda sid: store.get(sid, _csv([], [])))
    return store


def test_gbp_return_is_the_quoted_level_change(fred):
    fred["DEXUSUK"] = _csv(["2020-01-01", "2020-01-02", "2020-01-03"], [1.25, 1.375, 1.1])
    r = long_daily.asset_returns("GBP")
    assert r.index[0] == pd.Timestamp("2020-01-02", tz="UTC")
    assert r.iloc[0] == pytest.approx(0.10)
    assert r.iloc[1] == pytest.approx(1.1 / 1.375 - 1.0)


def test_jpy_is_inverted_to_usd_terms(fred):
    # DEXJPUS is yen per dollar, so the yen's dollar return is the inverse level's change.
    fred["DEXJPUS"] = _csv(["2020-01-01", "2020-01-02"], [100.0, 110.0])
    r = long_daily.asset_returns("JPY")
    assert r.iloc[0] == pytest.approx(100.0 / 110.0 - 1.0)


def test_bond_proxy_follows_the_stated_formula(fred):
    # r_t = y_{t-1} / 252 - 8 * (y_t - y_{t-1}) with yields in decimals.
    fred["DGS10"] = _csv(["2020-01-01", "2020-01-02", "2020-01-03"], [4.0, 5.0, 4.5])
    r = long_daily.asset_returns("UST10")
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(0.04 / 252 - 8.0 * (0.05 - 0.04))
    assert r.iloc[1] == pytest.approx(0.05 / 252 - 8.0 * (0.045 - 0.05))


def test_oil_drops_non_positive_prices_before_returns(fred):
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    fred["DCOILWTICO"] = _csv(dates, [50.0, 0.0, 55.0, -1.0, 60.0])
    r = long_daily.asset_returns("WTI")
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(55.0 / 50.0 - 1.0)
    assert r.iloc[1] == pytest.approx(60.0 / 55.0 - 1.0)


def test_costs_follow_asset_class():
    cost = pd.Series(long_daily.COST_BP)
    fx = ["GBP", "JPY", "CHF", "CAD", "AUD", "EUR"]
    assert (cost[fx] == 2.0).all()
    assert cost["UST10"] == 2.0
    assert (cost[["NASDAQ", "NIKKEI", "WTI"]] == 5.0).all()


def test_panel_keeps_each_asset_history_without_filling(fred):
    fred["DEXUSUK"] = _csv(["2020-01-01", "2020-01-02", "2020-01-03"], [1.25, 1.375, 1.1])
    fred["NASDAQCOM"] = _csv(["2020-01-02", "2020-01-03", "2020-01-06"], [100.0, 101.0, 99.0])
    panel, costs = long_daily.load_daily_panel()
    assert list(panel.columns) == list(long_daily.FRED_SERIES)
    assert panel.index.is_monotonic_increasing
    assert np.isnan(panel.loc["2020-01-06", "GBP"])
    assert np.isnan(panel.loc["2020-01-02", "NASDAQ"])
    assert panel.loc["2020-01-03", "NASDAQ"] == pytest.approx(101.0 / 100.0 - 1.0)
    assert costs["NASDAQ"] == 5.0


def test_chf_and_cad_are_inverted_to_usd_terms(fred):
    fred["DEXSZUS"] = _csv(["2020-01-01", "2020-01-02"], [0.9, 0.95])
    fred["DEXCAUS"] = _csv(["2020-01-01", "2020-01-02"], [1.25, 1.2])
    assert long_daily.asset_returns("CHF").iloc[0] == pytest.approx(0.9 / 0.95 - 1.0)
    assert long_daily.asset_returns("CAD").iloc[0] == pytest.approx(1.25 / 1.2 - 1.0)
