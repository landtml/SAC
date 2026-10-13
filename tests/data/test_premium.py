"""Premium loader and alignment tests. Offline except the ``network`` test.

Archive rows are canned in the kline CSV layout. ``fetch_bytes`` is replaced by a stub that
serves zips by URL and records each request. The cache root is a temporary directory.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sactrader.data import premium
from sactrader.data.schema import make_frame

PREMIUM_1H = "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/BTCUSDT/1h"
HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)
HOUR_MS = 3_600_000


class Archive:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[str] = []

    def __call__(self, url: str, dest: Path, **_: object) -> bytes:
        self.requests.append(url)
        if url not in self.files:
            raise FileNotFoundError(url)
        return self.files[url]


def serve(monkeypatch, tmp_path: Path, files: dict[str, bytes]) -> Archive:
    monkeypatch.setenv("SACTRADER_DATA", str(tmp_path / "data"))
    archive = Archive(files)
    monkeypatch.setattr(premium, "fetch_bytes", archive)
    return archive


def ms(text: str) -> int:
    return pd.Timestamp(text, tz="UTC").value // 1_000_000


def kline_row(open_ms: int, close: float, scale: int = 1) -> str:
    """One premium kline in the archive layout. ``scale`` 1000 writes microseconds."""
    close_ms = open_ms + HOUR_MS - 1
    return (f"{open_ms * scale},{close},{close},{close},{close},0,"
            f"{close_ms * scale},0,60,0,0,0")


def month_zip(rows: list[str], header: bool = False) -> bytes:
    lines = [HEADER] if header else []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1h.csv", "\n".join(lines + rows) + "\n")
    return buf.getvalue()


def perp_frame(start: str, periods: int, bar: str):
    index = pd.date_range(start, periods=periods, freq=bar, tz="UTC")
    flat = np.full(periods, 100.0)
    return make_frame(index, flat, flat, flat, flat, np.zeros(periods), symbol="BTCUSDT",
                      kind="perp", bar=bar, funding_rate=np.zeros(periods))


def test_millisecond_and_microsecond_rows_load_in_utc(monkeypatch, tmp_path):
    archive = serve(monkeypatch, tmp_path, {
        f"{PREMIUM_1H}/BTCUSDT-1h-2024-01.zip": month_zip([
            kline_row(ms("2024-01-01T00:00"), 0.0001),
            kline_row(ms("2024-01-01T01:00"), -0.0002),
        ]),
        f"{PREMIUM_1H}/BTCUSDT-1h-2025-01.zip": month_zip([
            kline_row(ms("2025-01-01T00:00"), 0.0003, scale=1000),
        ]),
    })
    series = premium.load_premium("btcusdt", "1h", "2024-01", "2025-01")
    assert list(series.index) == [
        pd.Timestamp("2024-01-01T00:00", tz="UTC"),
        pd.Timestamp("2024-01-01T01:00", tz="UTC"),
        pd.Timestamp("2025-01-01T00:00", tz="UTC"),
    ]
    np.testing.assert_allclose(series.to_numpy(), [0.0001, -0.0002, 0.0003])
    assert series.name == "premium"
    assert series.dtype == np.float64
    assert str(series.index.tz) == "UTC"
    assert series.index.is_monotonic_increasing
    assert len(archive.requests) == 13  # every month from 2024-01 to 2025-01 was asked for


def test_header_row_is_skipped(monkeypatch, tmp_path):
    serve(monkeypatch, tmp_path, {
        f"{PREMIUM_1H}/BTCUSDT-1h-2024-01.zip": month_zip(
            [kline_row(ms("2024-01-01T00:00"), 0.0001)], header=True),
    })
    series = premium.load_premium("BTCUSDT", "1h", "2024-01", "2024-01")
    np.testing.assert_allclose(series.to_numpy(), [0.0001])


def test_month_the_archive_lacks_is_skipped(monkeypatch, tmp_path):
    archive = serve(monkeypatch, tmp_path, {
        f"{PREMIUM_1H}/BTCUSDT-1h-2024-01.zip": month_zip([kline_row(ms("2024-01-01"), 0.0001)]),
        f"{PREMIUM_1H}/BTCUSDT-1h-2024-03.zip": month_zip([kline_row(ms("2024-03-01"), 0.0002)]),
    })
    series = premium.load_premium("BTCUSDT", "1h", "2024-01", "2024-03")
    np.testing.assert_allclose(series.to_numpy(), [0.0001, 0.0002])
    assert archive.requests == [
        f"{PREMIUM_1H}/BTCUSDT-1h-{month}.zip" for month in ("2024-01", "2024-02", "2024-03")
    ]


def test_no_listed_month_raises(monkeypatch, tmp_path):
    serve(monkeypatch, tmp_path, {})
    with pytest.raises(ValueError):
        premium.load_premium("BTCUSDT", "1h", "2024-01", "2024-02")


def test_unsupported_interval_is_rejected_without_a_request(monkeypatch, tmp_path):
    archive = serve(monkeypatch, tmp_path, {})
    with pytest.raises(ValueError):
        premium.load_premium("BTCUSDT", "7m", "2024-01", "2024-01")
    assert archive.requests == []


def test_hourly_frame_takes_the_premium_with_the_same_open_time(monkeypatch, tmp_path):
    archive = serve(monkeypatch, tmp_path, {
        f"{PREMIUM_1H}/BTCUSDT-1h-2024-01.zip": month_zip([
            kline_row(ms("2024-01-01T00:00"), 0.0001),
            kline_row(ms("2024-01-01T01:00"), 0.0002),
            kline_row(ms("2024-01-01T02:00"), 0.0003),
            kline_row(ms("2024-01-01T03:00"), 0.0004),
        ]),
    })
    frame = perp_frame("2024-01-01T00:00", periods=5, bar="1h")
    out = premium.attach_premium(frame)
    np.testing.assert_allclose(
        out["premium"].to_numpy(), [0.0001, 0.0002, 0.0003, 0.0004, np.nan], equal_nan=True)
    assert out["premium"].dtype == np.float64
    assert "premium" not in frame.columns
    assert out.attrs == frame.attrs
    assert archive.requests == [f"{PREMIUM_1H}/BTCUSDT-1h-2024-01.zip"]


def test_daily_bar_takes_the_hourly_kline_closing_at_its_close(monkeypatch, tmp_path):
    rows = [kline_row(ms("2024-01-01") + hour * HOUR_MS, 1e-4 * (hour + 1)) for hour in range(24)]
    rows.append(kline_row(ms("2024-01-02"), 9e-3))
    archive = serve(monkeypatch, tmp_path, {
        f"{PREMIUM_1H}/BTCUSDT-1h-2024-01.zip": month_zip(rows),
    })
    out = premium.attach_premium(perp_frame("2024-01-01", periods=2, bar="1D"))
    # Jan 1 closes at Jan 2 00:00, so its 23:00 kline counts. The Jan 2 00:00 kline closes at
    # 01:00 and is excluded. Jan 2 closes at Jan 3 00:00, so the last kline before that is 00:00.
    np.testing.assert_allclose(out["premium"].to_numpy(), [24e-4, 9e-3])
    assert archive.requests == [f"{PREMIUM_1H}/BTCUSDT-1h-2024-01.zip"]


def test_bars_before_the_first_listed_month_are_nan(monkeypatch, tmp_path):
    serve(monkeypatch, tmp_path, {
        f"{PREMIUM_1H}/BTCUSDT-1h-2024-01.zip": month_zip([
            kline_row(ms("2024-01-01T00:00"), 0.0001),
            kline_row(ms("2024-01-01T01:00"), 0.0002),
        ]),
    })
    out = premium.attach_premium(perp_frame("2023-12-31T22:00", periods=4, bar="1h"))
    np.testing.assert_allclose(
        out["premium"].to_numpy(), [np.nan, np.nan, 0.0001, 0.0002], equal_nan=True)


def test_spot_frame_is_rejected_without_a_request(monkeypatch, tmp_path):
    archive = serve(monkeypatch, tmp_path, {})
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    flat = np.full(3, 100.0)
    spot = make_frame(index, flat, flat, flat, flat, np.zeros(3), symbol="BTCUSDT", kind="spot",
                      bar="1h")
    with pytest.raises(ValueError):
        premium.attach_premium(spot)
    assert archive.requests == []


def test_parsed_series_is_cached_once_the_end_month_has_passed(monkeypatch, tmp_path):
    archive = serve(monkeypatch, tmp_path, {
        f"{PREMIUM_1H}/BTCUSDT-1h-2024-01.zip": month_zip([kline_row(ms("2024-01-01"), 0.0001)]),
    })
    first = premium.load_premium("BTCUSDT", "1h", "2024-01", "2024-01")
    requests_after_first = len(archive.requests)
    second = premium.load_premium("BTCUSDT", "1h", "2024-01", "2024-01")
    assert len(archive.requests) == requests_after_first
    pd.testing.assert_series_equal(second, first, check_freq=False)
    cached = list((tmp_path / "data" / "binance" / "premium").glob("*.parquet"))
    assert len(cached) == 1


def test_series_is_not_cached_while_the_end_month_is_open(monkeypatch, tmp_path):
    serve(monkeypatch, tmp_path, {
        f"{PREMIUM_1H}/BTCUSDT-1h-2024-01.zip": month_zip([kline_row(ms("2024-01-01"), 0.0001)]),
    })
    premium.load_premium("BTCUSDT", "1h", "2024-01", "2030-01")
    assert list((tmp_path / "data" / "binance" / "premium").glob("*.parquet")) == []


@pytest.mark.network
def test_live_btcusdt_premium_january_2024():
    series = premium.load_premium("BTCUSDT", "1h", "2024-01", "2024-01")
    assert len(series) == 744
    assert series.index.is_monotonic_increasing
    assert (series.abs() < 0.01).mean() >= 0.99
