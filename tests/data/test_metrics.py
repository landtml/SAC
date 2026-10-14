"""Open-interest loader and alignment tests. Offline: ``fetch_bytes`` is replaced by a stub that
serves canned daily zips by URL, and the cache root is a temporary directory."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from softedge.data import metrics
from softedge.data.schema import make_frame

BASE = "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT"
HEADER = ("create_time,symbol,sum_open_interest,sum_open_interest_value,"
          "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
          "count_long_short_ratio,sum_taker_long_short_vol_ratio")


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
    monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path / "data"))
    archive = Archive(files)
    monkeypatch.setattr(metrics, "fetch_bytes", archive)
    return archive


def day_zip(day: str, values: dict[str, float]) -> bytes:
    """One daily metrics file: rows keyed by ``HH:MM`` snapshot time, value in USD."""
    rows = [f"{day} {hhmm}:00,BTCUSDT,1,{usd},1,1,1,1" for hhmm, usd in values.items()]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(f"BTCUSDT-metrics-{day}.csv", "\n".join([HEADER, *rows]) + "\n")
    return buf.getvalue()


def hourly_frame(start: str, n: int):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    price = np.linspace(100.0, 101.0, n)
    return make_frame(idx, price, price, price, price, np.ones(n), symbol="BTCUSDT-BASIS",
                      kind="spot", bar="1h")


def test_parse_reads_usd_open_interest_by_snapshot_time():
    series = metrics._parse_metrics(day_zip("2024-01-01", {"00:00": 5.0, "23:55": 7.0}))
    assert series.name == "oi_value"
    assert str(series.index.tz) == "UTC"
    assert series.index[-1] == pd.Timestamp("2024-01-01 23:55", tz="UTC")
    assert series.tolist() == [5.0, 7.0]


def test_loader_fetches_each_day_from_2022_and_warns_on_missing_days(monkeypatch, tmp_path):
    files = {f"{BASE}/BTCUSDT-metrics-2022-01-01.zip": day_zip("2022-01-01", {"00:00": 1.0}),
             f"{BASE}/BTCUSDT-metrics-2022-01-03.zip": day_zip("2022-01-03", {"00:00": 3.0})}
    archive = serve(monkeypatch, tmp_path, files)
    with pytest.warns(UserWarning, match="missing day"):
        series = metrics.load_open_interest("BTCUSDT", "2021-12", "2022-01")
    assert archive.requests[0] == f"{BASE}/BTCUSDT-metrics-2022-01-01.zip"
    assert len(archive.requests) == 31  # every day of January 2022, none of 2021
    assert series.tolist() == [1.0, 3.0]


def test_attach_takes_last_snapshot_strictly_before_each_bar_close(monkeypatch):
    snaps = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(["2024-01-01 23:55", "2024-01-02 00:00", "2024-01-02 00:05"],
                               tz="UTC"),
        name="oi_value")
    monkeypatch.setattr(metrics, "load_open_interest", lambda sym, start, end: snaps)
    out = metrics.attach_open_interest(hourly_frame("2024-01-01 22:00", 3), "BTCUSDT")
    # 22:00 closes at 23:00 with no snapshot before it. 23:00 closes at 00:00, so the 00:00
    # snapshot is not used. 00:00 closes at 01:00 and takes the 00:05 snapshot.
    np.testing.assert_allclose(out["oi_value"].to_numpy(), [np.nan, 1.0, 3.0])


def test_attach_ignores_snapshots_at_or_after_bar_close(monkeypatch):
    before = pd.Series([1.0], index=pd.DatetimeIndex(["2024-01-01 23:55"], tz="UTC"),
                       name="oi_value")
    after = pd.Series([1.0, 9.0, 9.0],
                      index=pd.DatetimeIndex(["2024-01-01 23:55", "2024-01-02 00:00",
                                              "2024-01-02 00:30"], tz="UTC"),
                      name="oi_value")
    frame = hourly_frame("2024-01-01 22:00", 2)  # bars 22:00 and 23:00
    monkeypatch.setattr(metrics, "load_open_interest", lambda sym, start, end: before)
    a = metrics.attach_open_interest(frame, "BTCUSDT")["oi_value"].to_numpy()
    monkeypatch.setattr(metrics, "load_open_interest", lambda sym, start, end: after)
    b = metrics.attach_open_interest(frame, "BTCUSDT")["oi_value"].to_numpy()
    np.testing.assert_array_equal(a, b)
