"""Binance loader tests. Offline except the ``network`` test, which reads the live archive.

Fixtures are synthetic rows in the archive's CSV layout. ``fetch_bytes`` is replaced by an
archive stub that serves zips by URL and records each request.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from softedge.data import binance
from softedge.data.schema import validate_frame

FIXTURES = Path(__file__).parent / "fixtures" / "binance"
SPOT_1H = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h"
PERP_1H = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h"
FUNDING = "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT"


class Archive:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[str] = []

    def __call__(self, url: str, dest: Path, **_: object) -> bytes:
        self.requests.append(url)
        if url not in self.files:
            raise FileNotFoundError(url)
        return self.files[url]


def zip_bytes(csv: bytes, name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, csv)
    return buf.getvalue()


def fixture_zip(name: str) -> bytes:
    return zip_bytes((FIXTURES / name).read_bytes(), name)


def serve(monkeypatch, tmp_path: Path, files: dict[str, bytes]) -> Archive:
    monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path / "data"))
    archive = Archive(files)
    monkeypatch.setattr(binance, "fetch_bytes", archive)
    return archive


def test_spot_reads_headerless_millisecond_rows(monkeypatch, tmp_path):
    serve(monkeypatch, tmp_path,
          {f"{SPOT_1H}/BTCUSDT-1h-2024-01.zip": fixture_zip("spot_klines_2024-01.csv")})
    df = binance.load_spot("btcusdt", "1h", "2024-01", "2024-01")
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2024-01-01T00:00", tz="UTC")
    assert df["open"].tolist() == [42000.0, 42010.0, 42020.0]
    assert df["trades"].tolist() == [500.0, 501.0, 502.0]
    assert df.attrs == {"symbol": "BTCUSDT", "kind": "spot", "bar": "1h"}


def test_perp_skips_header_row_and_builds_valid_frame(monkeypatch, tmp_path):
    serve(monkeypatch, tmp_path, {
        f"{PERP_1H}/BTCUSDT-1h-2024-01.zip": fixture_zip("perp_klines_2024-01.csv"),
        f"{FUNDING}/BTCUSDT-fundingRate-2024-01.zip": fixture_zip("perp_funding_2024-01.csv"),
    })
    df = binance.load_perp("BTCUSDT", "1h", "2024-01", "2024-01")
    validate_frame(df)
    assert len(df) == 24
    assert list(df.columns) == [
        "open", "high", "low", "close", "volume", "funding_rate",
        "quote_volume", "trades", "taker_buy_base",
    ]
    assert df.attrs == {"symbol": "BTCUSDT", "kind": "perp", "bar": "1h"}
    assert df.index.is_monotonic_increasing and str(df.index.tz) == "UTC"


def test_microsecond_rows_load_alongside_millisecond_rows(monkeypatch, tmp_path):
    archive = serve(monkeypatch, tmp_path, {
        f"{SPOT_1H}/BTCUSDT-1h-2024-01.zip": fixture_zip("spot_klines_2024-01.csv"),
        f"{SPOT_1H}/BTCUSDT-1h-2025-01.zip": fixture_zip("spot_klines_2025-01.csv"),
    })
    with pytest.warns(UserWarning, match="gap"):  # the year between the two fixtures is missing
        df = binance.load_spot("BTCUSDT", "1h", "2024-01", "2025-01")
    assert list(df.index) == [
        pd.Timestamp(t, tz="UTC") for t in (
            "2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00",
            "2025-01-01T00:00", "2025-01-01T01:00",
        )
    ]
    assert len(archive.requests) == 13  # every month from 2024-01 to 2025-01 was asked for


def test_month_answered_404_is_skipped(monkeypatch, tmp_path):
    archive = serve(monkeypatch, tmp_path, {
        f"{SPOT_1H}/BTCUSDT-1h-2024-01.zip": fixture_zip("spot_klines_2024-01.csv"),
    })
    df = binance.load_spot("BTCUSDT", "1h", "2024-01", "2024-02")
    assert len(df) == 3
    assert archive.requests == [
        f"{SPOT_1H}/BTCUSDT-1h-2024-01.zip",
        f"{SPOT_1H}/BTCUSDT-1h-2024-02.zip",
    ]


def test_no_listed_month_raises(monkeypatch, tmp_path):
    serve(monkeypatch, tmp_path, {})
    with pytest.raises(ValueError):
        binance.load_spot("BTCUSDT", "1h", "2024-01", "2024-02")


def test_funding_lands_in_the_bar_that_opens_at_or_before_it(monkeypatch, tmp_path):
    serve(monkeypatch, tmp_path, {
        f"{PERP_1H}/BTCUSDT-1h-2024-01.zip": fixture_zip("perp_klines_2024-01.csv"),
        f"{FUNDING}/BTCUSDT-fundingRate-2024-01.zip": fixture_zip("perp_funding_2024-01.csv"),
    })
    rate = binance.load_perp("BTCUSDT", "1h", "2024-01", "2024-01")["funding_rate"]
    assert rate[pd.Timestamp("2024-01-01T00:00", tz="UTC")] == pytest.approx(0.0001)
    assert rate[pd.Timestamp("2024-01-01T08:00", tz="UTC")] == pytest.approx(-0.00005)
    assert rate[pd.Timestamp("2024-01-01T16:00", tz="UTC")] == pytest.approx(0.00015)
    assert int((rate != 0).sum()) == 3
    # The 2024-01-02 00:00 event falls after the last bar and is ignored.
    assert rate.sum() == pytest.approx(0.0002)


def test_second_call_reads_the_frame_cache(monkeypatch, tmp_path):
    archive = serve(monkeypatch, tmp_path, {
        f"{PERP_1H}/BTCUSDT-1h-2024-01.zip": fixture_zip("perp_klines_2024-01.csv"),
        f"{FUNDING}/BTCUSDT-fundingRate-2024-01.zip": fixture_zip("perp_funding_2024-01.csv"),
    })
    first = binance.load_perp("BTCUSDT", "1h", "2024-01", "2024-01")
    requests_after_first = len(archive.requests)
    second = binance.load_perp("BTCUSDT", "1h", "2024-01", "2024-01")
    assert len(archive.requests) == requests_after_first
    pd.testing.assert_frame_equal(second, first, check_freq=False)
    assert second.attrs == first.attrs


def test_bad_rows_are_dropped_and_high_low_widened(monkeypatch, tmp_path):
    csv = b"\n".join([
        b"1704067200000,100,101,99,100.5,1,1704070799999,100,1,0.5,50,0",
        # Same open time as the row above: the last row wins.
        b"1704067200000,100,101,99,100.7,1,1704070799999,100,1,0.5,50,0",
        # High 99 sits below open 100 and close 102, so high widens to 102.
        b"1704070800000,100,99,98,102,1,1704074399999,100,1,0.5,50,0",
        # Zero open price: dropped.
        b"1704074400000,0,1,0,1,1,1704077999999,100,1,0.5,50,0",
    ]) + b"\n"
    serve(monkeypatch, tmp_path, {f"{SPOT_1H}/BTCUSDT-1h-2024-01.zip": zip_bytes(csv, "x.csv")})
    df = binance.load_spot("BTCUSDT", "1h", "2024-01", "2024-01")
    validate_frame(df)
    assert len(df) == 2
    assert df["close"].tolist() == [100.7, 102.0]
    assert df["high"].tolist() == [101.0, 102.0]


@pytest.mark.network
def test_live_btcusdt_perp_january_2024():
    df = binance.load_perp("BTCUSDT", "1h", "2024-01", "2024-01")
    validate_frame(df)
    assert len(df) == 744
    funded = df[df["funding_rate"] != 0]
    per_day = funded.groupby(funded.index.date).size()
    assert len(per_day) == 31
    assert (per_day == 3).all()


def _hourly_month(start: str, hours: int, seed: float) -> bytes:
    """A spot kline zip of ``hours`` consecutive bars from ``start`` (UTC, ``YYYY-MM-DDTHH``)."""
    rows = []
    open_ms = int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    for i in range(hours):
        t = open_ms + i * 3_600_000
        price = seed + i * 0.01
        rows.append(f"{t},{price},{price + 1},{price - 1},{price + 0.5},1,{t + 3_599_999},"
                    f"100,1,0.5,50,0")
    return zip_bytes(("\n".join(rows) + "\n").encode(), "x.csv")


def test_gap_of_missing_months_warns_and_names_the_gap(monkeypatch, tmp_path):
    # January and March are in the archive, February is not: the bars bridge a 1,400-bar gap.
    serve(monkeypatch, tmp_path, {
        f"{SPOT_1H}/BTCUSDT-1h-2024-01.zip": _hourly_month("2024-01-01T00:00", 3, 100.0),
        f"{SPOT_1H}/BTCUSDT-1h-2024-03.zip": _hourly_month("2024-03-01T00:00", 3, 200.0),
    })
    with pytest.warns(UserWarning, match="gap") as record:
        df = binance.load_spot("BTCUSDT", "1h", "2024-01", "2024-03")
    message = str(record[0].message)
    assert "2024-01-01 02:00" in message and "2024-03-01 00:00" in message
    assert len(df) == 6  # the bridged frame is still returned, as the pegs loader expects


def test_single_missing_bar_does_not_warn(monkeypatch, tmp_path):
    import warnings

    rows = [
        "1704067200000,100,101,99,100.5,1,1704070799999,100,1,0.5,50,0",
        # The 01:00 bar is missing: a 2-bar step, which is within tolerance.
        "1704074400000,100,101,99,100.5,1,1704077999999,100,1,0.5,50,0",
    ]
    serve(monkeypatch, tmp_path, {
        f"{SPOT_1H}/BTCUSDT-1h-2024-01.zip": zip_bytes(("\n".join(rows) + "\n").encode(), "x.csv"),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = binance.load_spot("BTCUSDT", "1h", "2024-01", "2024-01")
    assert len(df) == 2


def test_contiguous_months_do_not_warn(monkeypatch, tmp_path):
    import warnings

    serve(monkeypatch, tmp_path, {
        f"{SPOT_1H}/BTCUSDT-1h-2024-01.zip": _hourly_month("2024-01-31T22:00", 2, 100.0),
        f"{SPOT_1H}/BTCUSDT-1h-2024-02.zip": _hourly_month("2024-02-01T00:00", 2, 100.0),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = binance.load_spot("BTCUSDT", "1h", "2024-01", "2024-02")
    assert len(df) == 4
