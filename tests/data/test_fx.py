import os
import time

import numpy as np
import pandas as pd
import pytest

from softedge.data import fx
from softedge.data.schema import validate_frame


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content


@pytest.fixture(autouse=True)
def isolated(request, tmp_path, monkeypatch):
    # Network tests keep the caller's cache directory.
    if request.node.get_closest_marker("network"):
        return
    monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path))
    monkeypatch.setattr(fx, "REQUEST_INTERVAL_S", 0.0)


def serve(monkeypatch, csv_by_id: dict[str, str]) -> list[dict]:
    calls: list[dict] = []

    def fake_get(url, *, params=None, **kwargs):
        assert url == fx.FRED_CSV_URL
        calls.append(dict(params))
        return FakeResponse(csv_by_id[params["id"]].encode())

    monkeypatch.setattr(fx, "http_get", fake_get)
    return calls


def test_missing_values_are_dropped(monkeypatch):
    csv = "observation_date,DEXUSEU\n2024-01-02,1.10\n2024-01-03,.\n2024-01-04,\n2024-01-05,1.12\n"
    serve(monkeypatch, {"DEXUSEU": csv})
    df = fx.load_fx_daily("EURUSD")
    assert list(df.index) == [pd.Timestamp("2024-01-02", tz="UTC"),
                              pd.Timestamp("2024-01-05", tz="UTC")]
    np.testing.assert_allclose(df["close"], [1.10, 1.12])
    # The second bar opens at the previous available close, not at a dropped row.
    np.testing.assert_allclose(df["open"], [1.10, 1.10])
    assert not df[["open", "high", "low", "close"]].isna().any().any()
    assert df["volume"].isna().all()


def test_ohlc_built_from_closes(monkeypatch):
    csv = "observation_date,DEXUSEU\n2024-01-01,1.10\n2024-01-02,1.12\n2024-01-03,1.11\n"
    serve(monkeypatch, {"DEXUSEU": csv})
    df = fx.load_fx_daily("EURUSD")
    validate_frame(df)
    assert df.attrs == {"symbol": "EURUSD", "kind": "fx", "bar": "1D"}
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert (df.index.hour == 0).all()
    np.testing.assert_allclose(df["open"], [1.10, 1.10, 1.12])
    np.testing.assert_allclose(df["high"], [1.10, 1.12, 1.12])
    np.testing.assert_allclose(df["low"], [1.10, 1.10, 1.11])
    np.testing.assert_allclose(df["close"], [1.10, 1.12, 1.11])
    assert df["volume"].isna().all()


def test_window_bounds_are_inclusive_and_open_uses_history(monkeypatch):
    csv = ("observation_date,DEXUSEU\n2024-01-01,1\n2024-01-02,2\n"
           "2024-01-03,3\n2024-01-04,4\n2024-01-05,5\n")
    serve(monkeypatch, {"DEXUSEU": csv})
    df = fx.load_fx_daily("EURUSD", start="2024-01-03", end="2024-01-04")
    assert len(df) == 2
    # The first bar opens at the close of 2024-01-02, which lies before the window.
    np.testing.assert_allclose(df["open"], [2.0, 3.0])
    np.testing.assert_allclose(df["close"], [3.0, 4.0])


def test_invert_takes_reciprocal(monkeypatch):
    monkeypatch.setitem(fx.FRED_FX_SERIES, "EURUSD", ("DEXUSEU", True))
    serve(monkeypatch, {"DEXUSEU": "observation_date,DEXUSEU\n2024-01-01,2\n2024-01-02,4\n"})
    df = fx.load_fx_daily("EURUSD")
    np.testing.assert_allclose(df["close"], [0.5, 0.25])


def test_fresh_cache_skips_download_and_stale_cache_refetches(monkeypatch):
    csv = "observation_date,DEXUSEU\n2024-01-01,1\n2024-01-02,2\n"
    calls = serve(monkeypatch, {"DEXUSEU": csv})
    fx.load_fx_daily("EURUSD")
    fx.load_fx_daily("EURUSD")
    assert len(calls) == 1

    path = fx.cache_dir("fred") / "DEXUSEU.csv"
    old = time.time() - 2 * 86400
    os.utime(path, (old, old))
    fx.load_fx_daily("EURUSD")
    assert len(calls) == 2


def test_unknown_pair_raises(monkeypatch):
    serve(monkeypatch, {})
    with pytest.raises(ValueError, match="unknown pair"):
        fx.load_fx_daily("XAUUSD")


@pytest.mark.parametrize("pair", fx.EM_FX_PAIRS)
def test_em_pairs_parse_like_g10_pairs(monkeypatch, pair):
    series_id, invert = fx.FRED_FX_SERIES[pair]
    assert invert is False
    csv = f"observation_date,{series_id}\n2024-01-02,2.0\n2024-01-03,.\n2024-01-04,2.5\n"
    serve(monkeypatch, {series_id: csv})
    df = fx.load_fx_daily(pair)
    validate_frame(df)
    assert df.attrs == {"symbol": pair, "kind": "fx", "bar": "1D"}
    np.testing.assert_allclose(df["close"], [2.0, 2.5])
    np.testing.assert_allclose(df["open"], [2.0, 2.0])


@pytest.mark.network
def test_eurusd_since_2024_live():
    df = fx.load_fx_daily("EURUSD", start="2024-01-01")
    validate_frame(df)
    assert df.attrs["symbol"] == "EURUSD" and df.attrs["bar"] == "1D"
    assert df.index[0] >= pd.Timestamp("2024-01-01", tz="UTC")
    assert len(df) > 400
    assert df["close"].between(0.5, 2.0).all()
