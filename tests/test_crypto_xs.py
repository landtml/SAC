"""Crypto cross-section tests. Offline: the archive and the S3 listing are stubbed.

Fixtures are synthetic 1d kline rows in the archive's headerless CSV layout, served through a
stub of ``binance.fetch_bytes``, so the loader reads them exactly as it reads real zips.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from softedge.data import binance, crypto_xs

BASE = "https://data.binance.vision/data/spot/monthly/klines"


def kline_zip(symbol: str, month: str, closes: dict[str, float],
              volumes: dict[str, float] | None = None) -> bytes:
    """One monthly 1d zip; ``closes`` maps a UTC day to its close, ``volumes`` overrides volume."""
    rows = []
    for day, close in closes.items():
        open_ms = int(pd.Timestamp(day, tz="UTC").value // 1_000_000)
        volume = (volumes or {}).get(day, 1.0)
        rows.append(f"{open_ms},{close},{close},{close},{close},{volume},{open_ms + 86_399_999},"
                    f"1,1,1,1,0")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{symbol}-1d-{month}.csv", "\n".join(rows) + "\n")
    return buf.getvalue()


def days_of(month: str, closes: list[float]) -> dict[str, float]:
    """Consecutive days from the first of ``month`` with the given closes."""
    first = pd.Timestamp(f"{month}-01")
    return {str((first + pd.Timedelta(days=i)).date()): c for i, c in enumerate(closes)}


class Archive:
    """Serves zips by URL and records every request, so tests can see what was asked for."""

    def __init__(self, zips: dict[str, bytes]) -> None:
        self.zips = zips
        self.requests: list[str] = []

    def __call__(self, url: str, dest, **_: object) -> bytes:
        self.requests.append(url)
        if url not in self.zips:
            raise FileNotFoundError(url)
        return self.zips[url]


def serve(monkeypatch, tmp_path, zips: dict[str, bytes]) -> Archive:
    monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path / "data"))
    archive = Archive(zips)
    monkeypatch.setattr(binance, "fetch_bytes", archive)
    return archive


def url_of(symbol: str, month: str) -> str:
    return f"{BASE}/{symbol}/1d/{symbol}-1d-{month}.zip"


# Exclusion rules -------------------------------------------------------------------------------

def test_universe_keeps_only_usdt_pairs():
    symbols = ["BTCUSDT", "ETHBTC", "BNBBUSD", "ETHUSDT"]
    assert crypto_xs.universe(symbols) == ["BTCUSDT", "ETHUSDT"]


def test_universe_drops_stablecoin_pairs():
    symbols = ["USDCUSDT", "BUSDUSDT", "FDUSDUSDT", "TUSDUSDT", "DAIUSDT", "USDEUSDT", "SOLUSDT"]
    assert crypto_xs.universe(symbols) == ["SOLUSDT"]


def test_universe_drops_leveraged_tokens_and_keeps_coins_with_suffix_letters():
    symbols = ["BTCUPUSDT", "ETHDOWNUSDT", "XRPBULLUSDT", "BNBBEARUSDT", "JUPUSDT", "SYRUPUSDT"]
    assert crypto_xs.universe(symbols) == ["JUPUSDT", "SYRUPUSDT"]


# Returns ---------------------------------------------------------------------------------------

def test_daily_returns_are_close_to_close_simple_returns():
    close = pd.Series([100.0, 110.0, 99.0],
                      index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
    ret = crypto_xs.daily_returns(close, "TESTUSDT")
    assert np.isnan(ret.iloc[0])
    assert ret.iloc[1] == pytest.approx(0.10)
    assert ret.iloc[2] == pytest.approx(-0.10)


def test_return_across_a_missing_day_is_nan_and_warned_not_bridged():
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04"], tz="UTC")
    close = pd.Series([100.0, 110.0, 121.0], index=index)
    with pytest.warns(UserWarning, match="gap"):
        ret = crypto_xs.daily_returns(close, "TESTUSDT")
    assert ret.iloc[1] == pytest.approx(0.10)
    assert np.isnan(ret.iloc[2])


# Loading ---------------------------------------------------------------------------------------

def test_returns_are_nan_before_listing_and_after_delisting(monkeypatch, tmp_path):
    # AAAUSDT lists on Feb 1 and its file stops after Feb 5; BBBUSDT trades in January only.
    zips = {
        url_of("AAAUSDT", "2019-02"): kline_zip(
            "AAAUSDT", "2019-02", days_of("2019-02", [10.0, 11.0, 12.1, 12.1, 13.31])),
        url_of("BBBUSDT", "2019-01"): kline_zip(
            "BBBUSDT", "2019-01", days_of("2019-01", [5.0, 5.5, 4.95, 5.0])),
    }
    serve(monkeypatch, tmp_path, zips)
    archive = {"AAAUSDT": ["2019-02"], "BBBUSDT": ["2019-01"]}
    returns, _ = crypto_xs.load_crypto_xs("2019-01-01", "2019-02-28", archive=archive)

    assert np.isnan(returns.loc["2019-01-15", "AAAUSDT"])  # before listing
    assert returns.loc["2019-02-02", "AAAUSDT"] == pytest.approx(0.10)
    assert returns.loc["2019-02-03", "AAAUSDT"] == pytest.approx(0.10)
    assert returns.loc["2019-02-04", "AAAUSDT"] == pytest.approx(0.0)
    assert np.isnan(returns.loc["2019-02-10", "AAAUSDT"])  # after delisting
    assert returns.loc["2019-01-02", "BBBUSDT"] == pytest.approx(0.10)
    assert returns.loc["2019-01-03", "BBBUSDT"] == pytest.approx(-0.10)
    assert np.isnan(returns.loc["2019-02-02", "BBBUSDT"])  # after delisting


def test_requests_cover_only_listed_months(monkeypatch, tmp_path):
    zips = {url_of("AAAUSDT", "2019-02"): kline_zip("AAAUSDT", "2019-02",
                                                    days_of("2019-02", [10.0, 11.0]))}
    archive = serve(monkeypatch, tmp_path, zips)
    crypto_xs.load_crypto_xs("2019-01-01", "2019-12-31",
                             archive={"AAAUSDT": ["2019-02", "2019-03"]})
    assert archive.requests == [url_of("AAAUSDT", "2019-02"), url_of("AAAUSDT", "2019-03")]


# Mask ------------------------------------------------------------------------------------------

def test_day_is_used_only_with_at_least_thirty_coins(monkeypatch, tmp_path):
    names = [f"C{i:02d}USDT" for i in range(30)]
    flat = [1.0, 1.1, 1.21]
    zips, archive = {}, {}
    for name in names:
        closes = days_of("2019-01", flat)
        if name == names[0]:  # this coin is delisted after 2019-01-02
            closes = {d: c for d, c in closes.items() if d <= "2019-01-02"}
        zips[url_of(name, "2019-01")] = kline_zip(name, "2019-01", closes)
        archive[name] = ["2019-01"]
    serve(monkeypatch, tmp_path, zips)
    returns, mask = crypto_xs.load_crypto_xs("2019-01-01", "2019-01-31", archive=archive)

    assert returns.notna().sum(axis=1).loc["2019-01-02"] == 30
    assert mask.loc["2019-01-02"]
    assert returns.notna().sum(axis=1).loc["2019-01-03"] == 29
    assert not mask.loc["2019-01-03"]


def test_empty_archive_gives_empty_outputs(monkeypatch, tmp_path):
    serve(monkeypatch, tmp_path, {})
    returns, mask = crypto_xs.load_crypto_xs("2019-01-01", "2019-01-31", archive={})
    assert returns.empty
    assert mask.empty


# Listing ---------------------------------------------------------------------------------------

class Page:
    def __init__(self, body: str) -> None:
        self.content = body.encode()


def test_listing_follows_pagination(monkeypatch):
    head = ('<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            '<IsTruncated>true</IsTruncated>')
    first = head + ('<CommonPrefixes><Prefix>data/spot/monthly/klines/AAAUSDT/</Prefix>'
                    '</CommonPrefixes></ListBucketResult>')
    last = ('<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            '<IsTruncated>false</IsTruncated><CommonPrefixes>'
            '<Prefix>data/spot/monthly/klines/BBBUSDT/</Prefix></CommonPrefixes>'
            '</ListBucketResult>')
    calls = []

    def fake_get(url, *, params=None, **_):
        calls.append(dict(params))
        return Page(first if "marker" not in params else last)

    monkeypatch.setattr(crypto_xs, "http_get", fake_get)
    keys, prefixes = crypto_xs._s3_list("data/spot/monthly/klines/", delimiter="/")
    assert prefixes == ["data/spot/monthly/klines/AAAUSDT/", "data/spot/monthly/klines/BBBUSDT/"]
    assert len(calls) == 2
    assert calls[1]["marker"] == "data/spot/monthly/klines/AAAUSDT/"


def test_symbol_with_fewer_than_two_bars_is_skipped_with_a_warning(monkeypatch, tmp_path):
    zips = {
        url_of("ONEUSDT", "2019-01"): kline_zip("ONEUSDT", "2019-01", days_of("2019-01", [7.0])),
        url_of("BBBUSDT", "2019-01"): kline_zip("BBBUSDT", "2019-01", days_of("2019-01", [5.0, 5.5])),
    }
    serve(monkeypatch, tmp_path, zips)
    archive = {"ONEUSDT": ["2019-01"], "BBBUSDT": ["2019-01"]}
    with pytest.warns(UserWarning, match="ONEUSDT"):
        returns, _ = crypto_xs.load_crypto_xs("2019-01-01", "2019-01-31", archive=archive)
    assert list(returns.columns) == ["BBBUSDT"]
    assert returns.loc["2019-01-02", "BBBUSDT"] == pytest.approx(0.10)


def test_reused_ticker_splits_into_two_coins_with_no_return_across_the_junction(
        monkeypatch, tmp_path):
    # LUNAUSDT's old coin trades to May 3 here; the reused ticker's new coin starts May 31.
    zips = {
        url_of("LUNAUSDT", "2022-05"): kline_zip(
            "LUNAUSDT", "2022-05",
            {"2022-05-01": 10.0, "2022-05-02": 11.0, "2022-05-03": 12.1, "2022-05-31": 1.0}),
        url_of("LUNAUSDT", "2022-06"): kline_zip(
            "LUNAUSDT", "2022-06", {"2022-06-01": 1.5, "2022-06-02": 1.2}),
    }
    serve(monkeypatch, tmp_path, zips)
    archive = {"LUNAUSDT": ["2022-05", "2022-06"]}
    with pytest.warns(UserWarning):  # the archive gap between the two coins
        returns, _ = crypto_xs.load_crypto_xs("2022-05-01", "2022-06-30", archive=archive)

    assert returns.loc["2022-05-02", "LUNAUSDT"] == pytest.approx(0.10)
    assert returns.loc["2022-05-31":, "LUNAUSDT"].isna().all()  # old coin has no later bars
    assert np.isnan(returns.loc["2022-05-31", "LUNA2USDT"])  # new coin's first bar
    assert returns.loc["2022-06-01", "LUNA2USDT"] == pytest.approx(0.5)
    assert returns.loc["2022-05-04":"2022-05-30"].isna().all().all()  # nothing spans the junction


def test_negative_volume_rows_keep_their_closes_and_warn_naming_the_day(monkeypatch, tmp_path):
    zips = {
        url_of("NEGUSDT", "2019-01"): kline_zip(
            "NEGUSDT", "2019-01", days_of("2019-01", [5.0, 5.5, 4.95]),
            volumes={"2019-01-02": -3.0}),
    }
    serve(monkeypatch, tmp_path, zips)
    with pytest.warns(UserWarning, match="2019-01-02"):
        returns, _ = crypto_xs.load_crypto_xs("2019-01-01", "2019-01-31",
                                              archive={"NEGUSDT": ["2019-01"]})
    assert returns.loc["2019-01-02", "NEGUSDT"] == pytest.approx(0.10)
    assert returns.loc["2019-01-03", "NEGUSDT"] == pytest.approx(-0.10)
