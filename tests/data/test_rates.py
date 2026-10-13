import os
import time

import numpy as np
import pandas as pd
import pytest

from sactrader.data import rates
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
    monkeypatch.setattr(rates, "REQUEST_INTERVAL_S", 0.0)


def fred_csv(series_id: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(f"{date},{value}\n" for date, value in rows)
    return f"observation_date,{series_id}\n{body}"


def serve(monkeypatch, csv_by_id: dict[str, str]) -> list[str]:
    # An extension series is empty unless a test supplies rows for it.
    empty = {series: fred_csv(series, []) for series in rates.SHORT_RATE_EXTENSION.values()}
    csv_by_id = {**empty, **csv_by_id}
    calls: list[str] = []

    def fake_get(url, *, params=None, **kwargs):
        assert url == rates.FRED_CSV_URL
        calls.append(params["id"])
        return FakeResponse(csv_by_id[params["id"]].encode())

    monkeypatch.setattr(rates, "http_get", fake_get)
    return calls


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def test_percent_becomes_decimal_and_takes_effect_next_month(monkeypatch):
    csv = fred_csv("IR3TIB01USM156N", [("2024-01-01", "5.25"), ("2024-02-01", "."),
                                       ("2024-03-01", "5.0")])
    serve(monkeypatch, {"IR3TIB01USM156N": csv})
    s = rates.load_short_rate("USD")
    # The January average is known from 1 February. The missing February value leaves it in
    # effect until the March average is known from 1 April.
    assert s.index[0] == utc("2024-02-01")
    assert s[utc("2024-02-01")] == pytest.approx(0.0525)
    assert s[utc("2024-02-29")] == pytest.approx(0.0525)
    assert s[utc("2024-03-01")] == pytest.approx(0.0525)
    assert s[utc("2024-04-01")] == pytest.approx(0.05)


def test_month_value_is_not_visible_before_month_end(monkeypatch):
    serve(monkeypatch, {
        "IR3TIB01USM156N": fred_csv("IR3TIB01USM156N", [
            ("2024-01-01", "5.0"), ("2024-02-01", "5.0"), ("2024-03-01", "4.5")]),
        "IR3TIB01EZM156N": fred_csv("IR3TIB01EZM156N", [
            ("2024-01-01", "3.0"), ("2024-02-01", "3.0"), ("2024-03-01", "2.0")]),
    })
    idx = pd.DatetimeIndex([utc("2024-03-15"), utc("2024-03-31T23:59"), utc("2024-04-01")])
    diff = rates.rate_diff("EURUSD", idx)
    # The March averages are published at the start of April, not in mid-March.
    np.testing.assert_allclose(diff, [0.03 - 0.05, 0.03 - 0.05, 0.02 - 0.045])


def test_forward_fill_gives_one_value_per_day(monkeypatch):
    serve(monkeypatch, {"IR3TIB01USM156N": fred_csv("IR3TIB01USM156N", [
        ("2024-01-01", "5.0"), ("2024-02-01", "4.0")])})
    s = rates.load_short_rate("USD")
    steps = s.index.to_series().diff().dropna()
    assert (steps == pd.Timedelta("1D")).all()
    assert s[utc("2024-02-10")] == pytest.approx(0.05)
    assert s.index.tz is not None and str(s.index.tz) == "UTC"


def test_base_minus_quote_sign(monkeypatch):
    serve(monkeypatch, {
        "IR3TIB01EZM156N": fred_csv("IR3TIB01EZM156N", [("2024-01-01", "2.0")]),
        "IR3TIB01USM156N": fred_csv("IR3TIB01USM156N", [("2024-01-01", "5.0")]),
        "IR3TIB01JPM156N": fred_csv("IR3TIB01JPM156N", [("2024-01-01", "0.1")]),
    })
    idx = pd.DatetimeIndex([utc("2024-03-15")])
    assert rates.rate_diff("EURUSD", idx)[0] == pytest.approx(0.02 - 0.05)
    assert rates.rate_diff("USDJPY", idx)[0] == pytest.approx(0.05 - 0.001)


def test_nan_until_both_series_start(monkeypatch):
    serve(monkeypatch, {
        "IR3TIB01USM156N": fred_csv("IR3TIB01USM156N", [("1999-12-01", "6.0")]),
        "IR3TIB01EZM156N": fred_csv("IR3TIB01EZM156N", [("2000-01-01", "4.0")]),
    })
    # USD is known from 2000-01-01, EUR only from 2000-02-01 (the January average).
    idx = pd.DatetimeIndex([utc("1999-12-31"), utc("2000-01-15"), utc("2000-02-01")])
    diff = rates.rate_diff("EURUSD", idx)
    assert np.isnan(diff[0]) and np.isnan(diff[1])
    assert diff[2] == pytest.approx(0.04 - 0.06)


def test_daily_extension_takes_over_after_monthly_series_stops(monkeypatch):
    serve(monkeypatch, {
        "IR3TIB01EZM156N": fred_csv("IR3TIB01EZM156N", [("2024-01-01", "3.0")]),
        # Wed 31 Jan is known on Thu 1 Feb, the same instant as the last monthly value.
        "ECBESTRVOLWGTTRMDMNRT": fred_csv("ECBESTRVOLWGTTRMDMNRT", [
            ("2024-01-31", "3.5"), ("2024-02-01", "3.6"), ("2024-02-02", "3.7")]),
    })
    s = rates.load_short_rate("EUR")
    assert s[utc("2024-02-01")] == pytest.approx(0.03)
    assert s[utc("2024-02-02")] == pytest.approx(0.036)
    # Saturday and Sunday hold the Friday value.
    assert s[utc("2024-02-04")] == pytest.approx(0.036)
    assert s[utc("2024-02-05")] == pytest.approx(0.037)


def test_unknown_currency_or_pair_raises(monkeypatch):
    serve(monkeypatch, {})
    idx = pd.DatetimeIndex([utc("2024-03-15")])
    with pytest.raises(ValueError, match="no short rate"):
        rates.load_short_rate("XXX")
    with pytest.raises(ValueError, match="no short rate"):
        rates.rate_diff("EURXXX", idx)
    with pytest.raises(ValueError, match="six letters"):
        rates.rate_diff("EUR", idx)


def test_fresh_cache_skips_download_and_stale_cache_refetches(monkeypatch):
    calls = serve(monkeypatch, {"IR3TIB01USM156N": fred_csv("IR3TIB01USM156N", [
        ("2024-01-01", "5.0")])})
    rates.load_short_rate("USD")
    rates.load_short_rate("USD")
    assert len(calls) == 1

    path = rates.cache_dir("fred") / "IR3TIB01USM156N.csv"
    old = time.time() - 2 * 86400
    os.utime(path, (old, old))
    rates.load_short_rate("USD")
    assert len(calls) == 2


def fx_frame(symbol: str, days: list[str], kind: str = "fx") -> pd.DataFrame:
    n = len(days)
    close = np.linspace(1.1, 1.2, n)
    return make_frame(
        pd.DatetimeIndex([utc(d) for d in days]), close, close * 1.01, close * 0.99, close,
        symbol=symbol, kind=kind, bar="1D",
    )


def test_attach_adds_column_and_keeps_attrs(monkeypatch):
    serve(monkeypatch, {
        "IR3TIB01EZM156N": fred_csv("IR3TIB01EZM156N", [("2024-01-01", "3.0")]),
        "IR3TIB01USM156N": fred_csv("IR3TIB01USM156N", [("2024-01-01", "5.0")]),
    })
    frame = fx_frame("EURUSD", ["2024-03-14", "2024-03-15"])
    out = rates.attach_rate_diff(frame)
    assert "rate_diff" not in frame.columns
    assert out.attrs == frame.attrs
    assert out["rate_diff"].dtype == np.float64
    np.testing.assert_allclose(out["rate_diff"], [0.03 - 0.05, 0.03 - 0.05])


def test_attach_fills_zero_where_rate_is_undefined(monkeypatch):
    serve(monkeypatch, {
        "IR3TIB01EZM156N": fred_csv("IR3TIB01EZM156N", [("2024-01-01", "3.0")]),
        "IR3TIB01USM156N": fred_csv("IR3TIB01USM156N", [("2024-01-01", "5.0")]),
    })
    frame = fx_frame("EURUSD", ["2023-12-31", "2024-03-15"])
    out = rates.attach_rate_diff(frame)
    assert out["rate_diff"].iloc[0] == 0.0
    assert out["rate_diff"].iloc[1] == pytest.approx(-0.02)


def test_attach_rejects_non_fx_frame(monkeypatch):
    serve(monkeypatch, {})
    spot = fx_frame("EURUSD", ["2024-03-14", "2024-03-15"], kind="spot")
    with pytest.raises(ValueError, match="fx frame"):
        rates.attach_rate_diff(spot)


@pytest.mark.network
def test_eurusd_diff_over_2010_2024_live():
    index = pd.date_range("2010-01-01", "2024-12-31", freq="D", tz="UTC")
    diff = rates.rate_diff("EURUSD", index)
    assert np.isfinite(diff).all()
    assert np.abs(diff).max() <= 0.1
