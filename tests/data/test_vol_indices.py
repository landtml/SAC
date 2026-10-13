import os
import time

import numpy as np
import pandas as pd
import pytest

from sactrader.data import vol_indices
from sactrader.data.schema import make_frame


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content


@pytest.fixture(autouse=True)
def isolated(request, tmp_path, monkeypatch):
    # Network tests keep the caller's cache directory.
    if request.node.get_closest_marker("network"):
        return
    monkeypatch.setenv("SACTRADER_DATA", str(tmp_path))
    monkeypatch.setattr(vol_indices, "REQUEST_INTERVAL_S", 0.0)


def serve(monkeypatch, csv_by_id: dict[str, str]) -> list[str]:
    calls: list[str] = []

    def fake_get(url, *, params=None, **kwargs):
        assert url == vol_indices.FRED_CSV_URL
        calls.append(params["id"])
        return FakeResponse(csv_by_id[params["id"]].encode())

    monkeypatch.setattr(vol_indices, "http_get", fake_get)
    return calls


def fred_csv(series_id: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(f"{date},{value}\n" for date, value in rows)
    return f"observation_date,{series_id}\n{body}"


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def frame(symbol: str, kind: str, stamps: list[str], bar: str) -> pd.DataFrame:
    close = np.linspace(100.0, 101.0, len(stamps))
    return make_frame(
        pd.DatetimeIndex([utc(s) for s in stamps]), close, close * 1.01, close * 0.99, close,
        symbol=symbol, kind=kind, bar=bar,
    )


def test_values_are_decimal_and_known_at_next_midnight(monkeypatch):
    serve(monkeypatch, {"VIXCLS": fred_csv("VIXCLS", [
        ("2024-03-04", "20"), ("2024-03-05", "."), ("2024-03-06", "15.5")])})
    s = vol_indices.load_vol_index("VIXCLS")
    # The missing 5 March value is dropped; 4 March is known from 5 March, 00:00 UTC.
    assert list(s.index) == [utc("2024-03-05"), utc("2024-03-07")]
    np.testing.assert_allclose(s, [0.20, 0.155])
    assert s.index.tz is not None and str(s.index.tz) == "UTC"


def test_bar_closing_before_next_midnight_does_not_see_the_day_value(monkeypatch):
    serve(monkeypatch, {"VIXCLS": fred_csv("VIXCLS", [
        ("2024-03-04", "20"), ("2024-03-05", "30"), ("2024-03-06", "40")])})
    stamps = ["2024-03-05 22:00", "2024-03-05 23:00", "2024-03-06 00:00"]
    spot = frame("SPX", "spot", stamps, "1h")
    out = vol_indices.attach_fred_implied_vol(spot)
    # The 5 March value is known at 6 March 00:00. The 22:00 bar closes at 23:00 and cannot see
    # it. The 23:00 bar closes exactly at 6 March 00:00 and sees it.
    np.testing.assert_allclose(out["implied_vol"], [0.20, 0.30, 0.30])


def test_daily_bar_sees_its_own_close_at_the_boundary(monkeypatch):
    serve(monkeypatch, {"VIXCLS": fred_csv("VIXCLS", [
        ("2024-03-05", "30"), ("2024-03-06", "40"), ("2024-03-07", "50")])})
    daily = frame("SPX", "spot", ["2024-03-05", "2024-03-06", "2024-03-07"], "1D")
    out = vol_indices.attach_fred_implied_vol(daily)
    # Each daily bar closes at midnight after its own label, when that day's value is known.
    np.testing.assert_allclose(out["implied_vol"], [0.30, 0.40, 0.50])


def test_nan_before_the_first_value_and_attrs_kept(monkeypatch):
    serve(monkeypatch, {"VIXCLS": fred_csv("VIXCLS", [
        ("2024-03-05", "15.5"), ("2024-03-06", "16.0")])})
    daily = frame("SPX", "spot", ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"], "1D")
    out = vol_indices.attach_fred_implied_vol(daily)
    assert np.isnan(out["implied_vol"].iloc[0])
    np.testing.assert_allclose(out["implied_vol"].iloc[1:], [0.155, 0.16, 0.16])
    assert out["implied_vol"].dtype == np.float64
    assert out.attrs == daily.attrs
    assert "implied_vol" not in daily.columns


@pytest.mark.parametrize("symbol, series_id", [
    ("SPX", "VIXCLS"), ("NDX", "VXNCLS"), ("WTI", "OVXCLS"), ("BRENT", "OVXCLS"),
])
def test_mapped_symbol_reads_its_series(monkeypatch, symbol, series_id):
    calls = serve(monkeypatch, {series_id: fred_csv(series_id, [("2024-03-05", "20")])})
    out = vol_indices.attach_fred_implied_vol(
        frame(symbol, "spot", ["2024-03-06", "2024-03-07"], "1D"))
    assert calls == [series_id]
    np.testing.assert_allclose(out["implied_vol"], [0.20, 0.20])


def test_unknown_symbol_returned_unchanged_without_a_fetch(monkeypatch):
    calls = serve(monkeypatch, {})
    fx_bars = frame("EURUSD", "fx", ["2024-03-05", "2024-03-06"], "1D")
    out = vol_indices.attach_fred_implied_vol(fx_bars)
    assert calls == []
    assert "implied_vol" not in out.columns
    assert out.equals(fx_bars)
    assert out.attrs == fx_bars.attrs


def test_fresh_cache_skips_download(monkeypatch):
    calls = serve(monkeypatch, {"VIXCLS": fred_csv("VIXCLS", [("2024-03-05", "20")])})
    vol_indices.load_vol_index("VIXCLS")
    vol_indices.load_vol_index("VIXCLS")
    assert len(calls) == 1

    path = vol_indices.cache_dir("fred") / "VIXCLS.csv"
    old = time.time() - 2 * 86400
    os.utime(path, (old, old))
    vol_indices.load_vol_index("VIXCLS")
    assert len(calls) == 2


@pytest.mark.network
def test_vix_2020_peak_above_half_live():
    s = vol_indices.load_vol_index("VIXCLS")
    in_2020 = s[(s.index >= utc("2020-01-01")) & (s.index < utc("2021-01-01"))]
    assert in_2020.max() > 0.5
