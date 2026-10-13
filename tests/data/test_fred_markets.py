import numpy as np
import pandas as pd
import pytest

from sactrader.data import fred_markets
from sactrader.data.schema import validate_frame


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content


@pytest.fixture(autouse=True)
def isolated(request, tmp_path, monkeypatch):
    # Network tests keep the caller's cache directory.
    if request.node.get_closest_marker("network"):
        return
    monkeypatch.setenv("SACTRADER_DATA", str(tmp_path))
    monkeypatch.setattr(fred_markets, "REQUEST_INTERVAL_S", 0.0)


def serve(monkeypatch, csv_by_id: dict[str, str]) -> list[dict]:
    calls: list[dict] = []

    def fake_get(url, *, params=None, **kwargs):
        assert url == fred_markets.FRED_CSV_URL
        calls.append(dict(params))
        return FakeResponse(csv_by_id[params["id"]].encode())

    monkeypatch.setattr(fred_markets, "http_get", fake_get)
    return calls


def utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_price_bars_are_built_from_closes(monkeypatch):
    csv = "observation_date,SP500\n2024-01-01,100\n2024-01-02,102\n2024-01-03,101\n"
    serve(monkeypatch, {"SP500": csv})
    df = fred_markets.load_fred_market("SPX")
    validate_frame(df)
    assert df.attrs == {"symbol": "SPX", "kind": "spot", "bar": "1D"}
    np.testing.assert_allclose(df["open"], [100, 100, 102])
    np.testing.assert_allclose(df["high"], [100, 102, 102])
    np.testing.assert_allclose(df["low"], [100, 100, 101])
    np.testing.assert_allclose(df["close"], [100, 102, 101])
    assert df["volume"].isna().all()


def test_missing_and_non_positive_prices_are_dropped(monkeypatch):
    csv = ("observation_date,DCOILWTICO\n2024-01-02,70\n2024-01-03,.\n"
           "2024-01-04,-5\n2024-01-05,72\n")
    serve(monkeypatch, {"DCOILWTICO": csv})
    df = fred_markets.load_fred_market("WTI")
    validate_frame(df)
    assert list(df.index) == [utc("2024-01-02"), utc("2024-01-05")]
    np.testing.assert_allclose(df["close"], [70, 72])
    # The second bar opens at the last kept price, not at a dropped row.
    np.testing.assert_allclose(df["open"], [70, 70])


def test_bond_index_matches_hand_computation(monkeypatch):
    # Fri 2024-01-05 at 4.00%, Mon 2024-01-08 at 5.00%, Tue 2024-01-09 at 4.00%. UST10 has
    # D = 8 and C = 80. The Friday-to-Monday step spans 3 calendar days.
    # Step 1: ret = 0.04 * 3/365 - 8 * 0.01 + 0.5 * 80 * 0.01**2 = -0.0756712328767
    # Step 2: ret = 0.05 * 1/365 + 8 * 0.01 + 0.5 * 80 * 0.01**2 = +0.0841369863014
    # Closes are 100, 100 * (1 + ret1), then the product of both factors. The convexity term
    # is 0.004 per step.
    csv = "observation_date,DGS10\n2024-01-05,4.00\n2024-01-08,5.00\n2024-01-09,4.00\n"
    serve(monkeypatch, {"DGS10": csv})
    df = fred_markets.load_fred_market("UST10")
    validate_frame(df)
    close = [100.0, 92.43287671232876, 100.20990039407019]
    np.testing.assert_allclose(df["close"], close, rtol=1e-10)
    np.testing.assert_allclose(df["open"], [100.0, 100.0, close[1]], rtol=1e-10)
    np.testing.assert_allclose(df["high"], [100.0, 100.0, close[2]], rtol=1e-10)
    np.testing.assert_allclose(df["low"], [100.0, close[1], close[1]], rtol=1e-10)


def test_month_window_covers_whole_month_and_opens_from_history(monkeypatch):
    csv = ("observation_date,SP500\n2023-12-29,100\n2024-01-02,102\n"
           "2024-01-31,110\n2024-02-01,120\n")
    serve(monkeypatch, {"SP500": csv})
    df = fred_markets.load_fred_market("SPX", start="2024-01", end="2024-01")
    assert list(df.index) == [utc("2024-01-02"), utc("2024-01-31")]
    # The first bar opens at the close of 2023-12-29, which lies before the window.
    np.testing.assert_allclose(df["open"], [100, 102])
    np.testing.assert_allclose(df["close"], [102, 110])


def test_fresh_cache_skips_download(monkeypatch):
    calls = serve(monkeypatch, {"DGS2": "observation_date,DGS2\n2024-01-02,4.0\n2024-01-03,4.1\n"})
    fred_markets.load_fred_market("UST2")
    fred_markets.load_fred_market("UST2")
    assert len(calls) == 1


def test_unknown_market_raises(monkeypatch):
    serve(monkeypatch, {})
    with pytest.raises(ValueError, match="unknown market"):
        fred_markets.load_fred_market("DAX")


@pytest.mark.network
def test_spx_and_ust10_live_2015_2024():
    spx = fred_markets.load_fred_market("SPX", start="2015-01", end="2024-12")
    ust = fred_markets.load_fred_market("UST10", start="2015-01", end="2024-12")
    validate_frame(spx)
    validate_frame(ust)
    assert spx.attrs["symbol"] == "SPX" and ust.attrs["symbol"] == "UST10"
    # FRED's SP500 starts on 2016-10-13, so the 2015 part of this window is empty. The UST10
    # window holds about 2500 daily yields.
    assert len(spx) > 2000 and len(ust) > 2400
    assert spx["close"].between(1000, 7000).all()
    assert (ust["close"] > 0).all()
