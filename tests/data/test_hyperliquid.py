"""Hyperliquid loader tests. Offline except the ``network`` test, which reads the live API.

``_post_json`` is replaced by a fake info endpoint that answers from in-memory rows, filtering by
the request window and capping replies the way the API does. Rows use the live field names.
"""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from softedge.data import hyperliquid
from softedge.data.schema import validate_frame

HOUR = 3_600_000
CANDLE_CAP = 5000
FUNDING_PAGE = 500
OHLC = (100.0, 110.0, 90.0, 105.0)


def ms(text: str) -> int:
    return pd.Timestamp(text, tz="UTC").value // 1_000_000


def candle(open_ms: int, ohlc: tuple[float, float, float, float], volume: str = "10.0",
           trades: int = 5, width: int = HOUR) -> dict:
    o, h, lo, c = ohlc
    return {"t": open_ms, "T": open_ms + width - 1, "s": "BTC", "i": "1h",
            "o": str(o), "h": str(h), "l": str(lo), "c": str(c), "v": volume, "n": trades}


def event(time_ms: int, rate: str) -> dict:
    return {"coin": "BTC", "fundingRate": rate, "premium": "0.0", "time": time_ms}


class InfoServer:
    """Fake info endpoint: filters rows by the request window and caps each reply."""

    def __init__(self, *, candles=(), funding=(), meta=None) -> None:
        self.candles = sorted(candles, key=lambda r: r["t"])
        self.funding = sorted(funding, key=lambda r: r["time"])
        self.meta = meta
        self.bodies: list[dict] = []

    def __call__(self, body: dict) -> object:
        self.bodies.append(body)
        if body["type"] == "meta":
            return self.meta
        if body["type"] == "candleSnapshot":
            req = body["req"]
            rows = [r for r in self.candles if req["startTime"] <= r["t"] <= req["endTime"]]
            return rows[-CANDLE_CAP:]
        if body["type"] == "fundingHistory":
            rows = [r for r in self.funding if body["startTime"] <= r["time"] <= body["endTime"]]
            return rows[:FUNDING_PAGE]
        raise AssertionError(f"unexpected request {body['type']}")

    def requests_of(self, kind: str) -> list[dict]:
        return [b for b in self.bodies if b["type"] == kind]


class FakeReply:
    def __init__(self, status: int, payload: object = None) -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}
        self._payload = payload

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


@pytest.fixture(autouse=True)
def _cache_root(request, monkeypatch, tmp_path):
    # The network test writes to whatever SOFTEDGE_DATA the shell sets, so the shared cache fills.
    if request.node.get_closest_marker("network") is None:
        monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path / "data"))


def test_list_perps_keeps_name_leverage_and_size_decimals(monkeypatch):
    meta = {"universe": [
        {"name": "BTC", "szDecimals": 5, "maxLeverage": 40, "marginTableId": 56},
        {"name": "MATIC", "szDecimals": 0, "maxLeverage": 10, "marginTableId": 7,
         "isDelisted": True},
    ]}
    monkeypatch.setattr(hyperliquid, "_post_json", InfoServer(meta=meta))
    table = hyperliquid.list_perps()
    assert list(table.columns) == ["coin", "max_leverage", "sz_decimals"]
    assert table["coin"].tolist() == ["BTC", "MATIC"]
    assert table["max_leverage"].tolist() == [40, 10]
    assert table["sz_decimals"].tolist() == [5, 0]


def test_candle_strings_become_floats_and_open_times_become_utc():
    rows = [candle(ms("2024-01-01T00:00"), (42000.5, 42100.0, 41900.0, 42050.0),
                   volume="12.25", trades=7)]
    frame = hyperliquid._parse_candles(rows, now_ms=ms("2024-02-01"))
    assert frame.index.tolist() == [pd.Timestamp("2024-01-01T00:00", tz="UTC")]
    assert frame["open"].iloc[0] == 42000.5
    assert frame["volume"].iloc[0] == 12.25
    assert frame["trades"].iloc[0] == 7.0
    assert frame.dtypes.eq("float64").all()


def test_candle_still_open_at_request_time_is_dropped():
    done = candle(ms("2024-01-01T00:00"), OHLC)
    open_now = candle(ms("2024-01-01T01:00"), OHLC)
    frame = hyperliquid._parse_candles([done, open_now], now_ms=ms("2024-01-01T01:30"))
    assert frame.index.tolist() == [pd.Timestamp("2024-01-01T00:00", tz="UTC")]


def test_funding_pages_forward_from_last_event_plus_one(monkeypatch):
    times = [ms("2024-01-01T00:00") + 151 + i * HOUR for i in range(600)]
    server = InfoServer(funding=[event(t, str(i)) for i, t in enumerate(times)])
    monkeypatch.setattr(hyperliquid, "_post_json", server)
    rate = hyperliquid.load_funding("BTC", "2024-01", "2024-01")
    assert len(rate) == 600
    assert rate.name == "funding_rate"
    assert rate.iloc[599] == 599.0
    assert rate.index.tolist()[0] == pd.Timestamp("2024-01-01T00:00:00.151", tz="UTC")
    starts = [b["startTime"] for b in server.requests_of("fundingHistory")]
    assert starts == [ms("2024-01-01T00:00"), times[499] + 1, times[599] + 1]


def test_perp_frame_sums_funding_into_bars_and_validates(monkeypatch):
    candles = [
        candle(ms("2024-01-01T00:00"), (100.0, 110.0, 90.0, 105.0)),
        candle(ms("2024-01-01T01:00"), (105.0, 108.0, 101.0, 107.0)),
        # The high sits below open and close, so the bar must be widened to pass the schema.
        candle(ms("2024-01-01T02:00"), (107.0, 106.0, 104.0, 112.0)),
    ]
    funding = [
        event(ms("2024-01-01T00:00:00.151"), "0.5"),
        event(ms("2024-01-01T00:59:59.900"), "0.25"),  # same bar as the one above
        event(ms("2024-01-01T02:00:00.000"), "0.125"),  # opens exactly on the boundary
        event(ms("2024-01-01T03:00:00.000"), "0.0625"),  # after the last bar closes
    ]
    monkeypatch.setattr(hyperliquid, "_post_json",
                        InfoServer(candles=candles, funding=funding))
    frame = hyperliquid.load_perp("BTC", "1h", "2024-01", "2024-01")
    validate_frame(frame)
    assert frame.attrs == {"symbol": "BTC", "kind": "perp", "bar": "1h"}
    assert list(frame.columns) == [
        "open", "high", "low", "close", "volume", "funding_rate", "trades",
    ]
    assert frame["funding_rate"].tolist() == [0.75, 0.0, 0.125]
    assert frame["high"].iloc[2] == 112.0


@pytest.mark.parametrize(("interval", "alias"), [
    ("15m", "15min"), ("1h", "1h"), ("4h", "4h"), ("1d", "1D"),
])
def test_interval_maps_to_pandas_alias(interval, alias):
    assert hyperliquid._bar_alias(interval) == alias


@pytest.mark.parametrize("interval", ["1m", "5m", "1w", "1M", "h", "2x"])
def test_unsupported_interval_is_refused_before_any_request(monkeypatch, interval):
    def no_requests(body):
        raise AssertionError(f"request sent for interval {interval!r}")

    monkeypatch.setattr(hyperliquid, "_post_json", no_requests)
    with pytest.raises(ValueError):
        hyperliquid.load_perp("BTC", interval, "2024-01", "2024-01")


def test_history_the_api_does_not_serve_is_left_out(monkeypatch):
    # Only March 2026 is served, as measured for 1h candles. February returns nothing.
    served = [candle(ms("2026-03-19T09:00") + i * HOUR, OHLC) for i in range(48)]
    monkeypatch.setattr(hyperliquid, "_post_json", InfoServer(candles=served))
    frame = hyperliquid.load_perp("BTC", "1h", "2026-02", "2026-03")
    assert frame.index[0] == pd.Timestamp("2026-03-19T09:00", tz="UTC")
    assert len(frame) == 48


def test_ended_month_is_read_back_without_requests(monkeypatch, tmp_path):
    server = InfoServer(
        candles=[candle(ms("2024-01-01T00:00") + i * HOUR, OHLC) for i in range(5)],
        funding=[event(ms("2024-01-01T00:00:00.151") + i * HOUR, "0.0001") for i in range(5)],
    )
    monkeypatch.setattr(hyperliquid, "_post_json", server)
    first = hyperliquid.load_perp("BTC", "1h", "2024-01", "2024-01")
    sent = len(server.bodies)
    again = hyperliquid.load_perp("BTC", "1h", "2024-01", "2024-01")
    assert len(server.bodies) == sent
    pd.testing.assert_frame_equal(first, again)
    assert again.attrs == first.attrs
    raw = tmp_path / "data" / "hyperliquid" / "raw" / "BTC"
    assert (raw / "candles-1h" / "2024-01.json").exists()
    assert (raw / "funding" / "2024-01.json").exists()
    assert not (raw / "funding" / "2024-01.json.part").exists()


def test_month_in_progress_is_fetched_each_time_and_not_cached(monkeypatch, tmp_path):
    server = InfoServer(funding=[event(ms("2024-01-02T00:00:00.151"), "0.0001")])
    monkeypatch.setattr(hyperliquid, "_post_json", server)
    monkeypatch.setattr(hyperliquid, "_now_ms", lambda: ms("2024-01-15T00:00"))
    hyperliquid.load_funding("BTC", "2024-01", "2024-01")
    hyperliquid.load_funding("BTC", "2024-01", "2024-01")
    # One request returns the event and the next returns nothing, so each call makes two.
    assert len(server.requests_of("fundingHistory")) == 4
    assert not (tmp_path / "data" / "hyperliquid" / "raw" / "BTC" / "funding"
                / "2024-01.json").exists()


def test_post_json_retries_429_and_503_then_returns(monkeypatch):
    monkeypatch.setattr(hyperliquid, "MIN_GAP_S", 0.0)
    monkeypatch.setattr(hyperliquid, "BACKOFF_S", 0.0)
    replies = [FakeReply(429), FakeReply(503), FakeReply(200, {"universe": []})]
    sent = _fake_post(monkeypatch, replies)
    assert hyperliquid._post_json({"type": "meta"}) == {"universe": []}
    assert len(sent) == 3


def test_post_json_gives_up_after_five_tries(monkeypatch):
    monkeypatch.setattr(hyperliquid, "MIN_GAP_S", 0.0)
    monkeypatch.setattr(hyperliquid, "BACKOFF_S", 0.0)
    sent = _fake_post(monkeypatch, [FakeReply(500) for _ in range(5)])
    with pytest.raises(RuntimeError):
        hyperliquid._post_json({"type": "meta"})
    assert len(sent) == 5


def test_post_json_raises_client_errors_without_retry(monkeypatch):
    sent = _fake_post(monkeypatch, [FakeReply(422)])
    with pytest.raises(requests.HTTPError):
        hyperliquid._post_json({"type": "meta"})
    assert len(sent) == 1


def _fake_post(monkeypatch, replies: list[FakeReply]) -> list[str]:
    queue = list(replies)
    sent: list[str] = []

    def post(url, **_kwargs):
        sent.append(url)
        return queue.pop(0)

    monkeypatch.setattr(hyperliquid.requests, "post", post)
    return sent


@pytest.mark.network
def test_live_btc_daily_bars_over_2024():
    frame = hyperliquid.load_perp("BTC", "1d", "2024-01", "2024-12")
    validate_frame(frame)
    assert frame.attrs == {"symbol": "BTC", "kind": "perp", "bar": "1D"}
    assert frame.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert len(frame) == 366
    assert (frame["funding_rate"] != 0).sum() > 300
