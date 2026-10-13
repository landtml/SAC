"""Tests for the Deribit DVOL loader. Every test runs offline except the marked network test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from softedge.data import deribit
from softedge.data.schema import make_frame

HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def ms(text: str) -> int:
    return round(pd.Timestamp(text, tz="UTC").timestamp() * 1000)


T0 = ms("2024-01-01")


class _Response:
    def __init__(self, body: dict) -> None:
        self._body = body

    def json(self) -> dict:
        return self._body


class FakeDeribit:
    """Serves the DVOL endpoint from ``closes`` (open ms -> close, percent), newest page first."""

    def __init__(self, closes: dict[int, float], page_size: int = 1000) -> None:
        self.closes = closes
        self.page_size = page_size
        self.calls: list[dict] = []

    def __call__(self, url: str, *, params: dict | None = None, **_: object) -> _Response:
        params = dict(params or {})
        self.calls.append(params)
        step = HOUR_MS if params["resolution"] == "3600" else DAY_MS
        lo, hi = params["start_timestamp"], params["end_timestamp"]
        opened = sorted(t for t in self.closes if lo <= t <= hi)
        page = opened[-self.page_size:]
        data = []
        for t in page:
            c = self.closes[t]
            data.append([t, c - 1, c + 1, c - 2, c])
        cont = page[0] - step if len(opened) > len(page) else None
        return _Response({"jsonrpc": "2.0", "result": {"data": data, "continuation": cont}})


class _Clock:
    """Stands in for the ``time`` module so pacing is checked without real sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _no_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("an offline test reached the network")


def _bars(start: str, periods: int, bar: str):
    idx = pd.date_range(start, periods=periods, freq=bar, tz="UTC")
    flat = np.full(periods, 100.0)
    return make_frame(idx, flat, flat, flat, flat, symbol="BTCUSDT", kind="spot", bar=bar)


@pytest.fixture(autouse=True)
def _offline(request, tmp_path, monkeypatch):
    monkeypatch.setattr(deribit, "MIN_INTERVAL_S", 0.0)
    if request.node.get_closest_marker("network") is None:
        monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path))
        monkeypatch.setattr(deribit, "http_get", _no_network)


def test_pages_follow_continuation_back_to_window_start(monkeypatch):
    fake = FakeDeribit({T0 + k * HOUR_MS: 40.0 + k for k in range(7)}, page_size=3)
    monkeypatch.setattr(deribit, "http_get", fake)

    s = deribit.load_dvol("BTC", "2024-01", "2024-01", "1h")

    np.testing.assert_allclose(s.to_numpy(), (40.0 + np.arange(7)) / 100)
    assert {c["start_timestamp"] for c in fake.calls} == {T0}
    # Each page is the newest three candles; continuation is the candle before that page.
    assert [c["end_timestamp"] for c in fake.calls] == [ms("2024-02-01") - 1, T0 + 3 * HOUR_MS, T0]


def test_value_is_the_close_in_decimal_indexed_by_candle_end(monkeypatch):
    fake = FakeDeribit({T0 + 10 * HOUR_MS: 55.0})
    monkeypatch.setattr(deribit, "http_get", fake)

    s = deribit.load_dvol("BTC", "2024-01", "2024-01", "1h")

    # The fake's open, high and low are 54, 56 and 53, so this checks that the close is read.
    assert s.iloc[0] == pytest.approx(0.55)
    assert s.index[0] == pd.Timestamp("2024-01-01 11:00", tz="UTC")


def test_hourly_bars_see_the_candle_that_ends_at_their_close(monkeypatch):
    fake = FakeDeribit({T0 + k * HOUR_MS: 40.0 + k for k in range(24)})
    monkeypatch.setattr(deribit, "http_get", fake)
    frame = _bars("2024-01-01", 24, "1h")

    out = deribit.attach_implied_vol(frame, "BTC")

    # Bar k closes at hour k + 1, the instant candle k ends, so the bar sees candle k.
    np.testing.assert_allclose(out["implied_vol"], (40.0 + np.arange(24)) / 100)
    assert out["implied_vol"].dtype == np.float64
    assert out.attrs == frame.attrs
    assert "implied_vol" not in frame.columns


def test_bar_closing_before_a_candle_end_does_not_see_it(monkeypatch):
    fake = FakeDeribit({T0 + 9 * HOUR_MS: 50.0, T0 + 10 * HOUR_MS: 60.0})
    monkeypatch.setattr(deribit, "http_get", fake)
    frame = _bars("2024-01-01 10:00", 2, "30min")  # bars close at 10:30 and at 11:00

    out = deribit.attach_implied_vol(frame, "BTC")

    # The 10:00 candle ends at 11:00: hidden from the 10:30 close, visible at the 11:00 close.
    np.testing.assert_allclose(out["implied_vol"], [0.50, 0.60])


def test_daily_bars_see_the_last_hourly_candle_before_close(monkeypatch):
    fake = FakeDeribit({T0 + k * HOUR_MS: 40.0 + k for k in range(72)})
    monkeypatch.setattr(deribit, "http_get", fake)
    frame = _bars("2024-01-01", 2, "1D")  # bars close at the midnights of Jan 2 and Jan 3

    out = deribit.attach_implied_vol(frame, "BTC")

    # Candles 23 and 47 end at those midnights, so they are the last ones known at each close.
    np.testing.assert_allclose(out["implied_vol"], [0.63, 0.87])


def test_nan_before_the_first_candle(monkeypatch):
    fake = FakeDeribit({T0 + 2 * HOUR_MS: 40.0, T0 + 3 * HOUR_MS: 41.0})
    monkeypatch.setattr(deribit, "http_get", fake)
    frame = _bars("2024-01-01", 3, "1h")  # bars close at 01:00, 02:00 and 03:00

    out = deribit.attach_implied_vol(frame, "BTC")

    assert np.isnan(out["implied_vol"].iloc[:2]).all()
    assert out["implied_vol"].iloc[2] == pytest.approx(0.40)


def test_frame_before_the_series_is_all_nan(monkeypatch):
    fake = FakeDeribit({})
    monkeypatch.setattr(deribit, "http_get", fake)

    out = deribit.attach_implied_vol(_bars("2019-06-01", 3, "1D"), "BTC")

    assert out["implied_vol"].isna().all()


def test_fx_and_binary_frames_are_rejected(monkeypatch):
    fake = FakeDeribit({T0: 40.0})
    monkeypatch.setattr(deribit, "http_get", fake)
    idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    fx = make_frame(idx, [1.1] * 3, [1.2] * 3, [1.0] * 3, [1.1] * 3,
                    symbol="EURUSD", kind="fx", bar="1D")
    binary = make_frame(idx, [0.4] * 3, [0.45] * 3, [0.35] * 3, [0.4] * 3,
                        symbol="YES", kind="binary", bar="1D", resolution=1.0)

    for frame in (fx, binary):
        with pytest.raises(ValueError):
            deribit.attach_implied_vol(frame, "BTC")
    assert fake.calls == []


def test_completed_month_is_cached_and_not_fetched_again(monkeypatch, tmp_path):
    fake = FakeDeribit({T0: 40.0})
    monkeypatch.setattr(deribit, "http_get", fake)

    first = deribit.load_dvol("BTC", "2024-01", "2024-01", "1h")
    second = deribit.load_dvol("BTC", "2024-01", "2024-01", "1h")

    pd.testing.assert_series_equal(first, second)
    assert len(fake.calls) == 1
    cached = tmp_path / "deribit" / "dvol" / "BTC" / "1h" / "2024-01.json"
    assert cached.exists()
    assert list(cached.parent.glob("*.part")) == []


def test_month_in_progress_is_fetched_and_not_cached(monkeypatch, tmp_path):
    this_month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
    fake = FakeDeribit({})
    monkeypatch.setattr(deribit, "http_get", fake)

    deribit.load_dvol("BTC", this_month, this_month, "1h")
    deribit.load_dvol("BTC", this_month, this_month, "1h")

    assert len(fake.calls) == 2
    assert not (tmp_path / "deribit" / "dvol" / "BTC" / "1h" / f"{this_month}.json").exists()


def test_requests_keep_half_a_second_apart(monkeypatch):
    clock = _Clock()
    fake = FakeDeribit({})
    stamps: list[float] = []

    def timed(url, *, params=None, **kwargs):
        stamps.append(clock.monotonic())
        return fake(url, params=params, **kwargs)

    monkeypatch.setattr(deribit, "time", clock)
    monkeypatch.setattr(deribit, "_last_request", float("-inf"))
    monkeypatch.setattr(deribit, "MIN_INTERVAL_S", 0.5)
    monkeypatch.setattr(deribit, "http_get", timed)

    deribit.load_dvol("BTC", "2024-01", "2024-03", "1h")  # three months, one page each

    assert len(stamps) == 3
    assert np.diff(stamps).min() >= 0.5


def test_error_reply_raises(monkeypatch):
    def reply(url, *, params=None, **_):
        return _Response({"jsonrpc": "2.0", "error": {"code": 10001, "message": "bad request"}})

    monkeypatch.setattr(deribit, "http_get", reply)
    with pytest.raises(RuntimeError, match="bad request"):
        deribit.load_dvol("BTC", "2024-01", "2024-01", "1h")


def test_unknown_currency_or_resolution_is_rejected():
    with pytest.raises(ValueError):
        deribit.load_dvol("DOGE")
    with pytest.raises(ValueError):
        deribit.load_dvol("BTC", resolution="1m")


@pytest.mark.network
def test_live_btc_dvol_2024_daily_values_are_plausible():
    s = deribit.load_dvol("BTC", "2024-01", "2024-12", "1D")

    assert len(s) > 0
    assert s.between(0.2, 2.0).all()
