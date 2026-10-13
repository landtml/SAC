import json

import numpy as np
import pandas as pd
import pytest

from sactrader.data import polymarket
from sactrader.data.schema import validate_frame

TOKEN = "12345678901234567890" * 4
T0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp())
HOUR = 3600
DAY = 86400


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def isolated(request, tmp_path, monkeypatch):
    # Network tests keep the caller's cache directory.
    if request.node.get_closest_marker("network"):
        return
    monkeypatch.setenv("SACTRADER_DATA", str(tmp_path))
    monkeypatch.setattr(polymarket, "REQUEST_INTERVAL_S", 0.0)


def ts(seconds: int) -> pd.Timestamp:
    return pd.to_datetime(seconds, unit="s", utc=True)


def market(**overrides) -> dict:
    row = {
        "id": "1",
        "slug": "will-it-rain",
        "question": "Will it rain?",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["1", "0"]),
        "clobTokenIds": json.dumps([TOKEN, "222"]),
        "startDate": "2024-01-01T00:00:00Z",
        "endDate": "2024-01-03T12:00:00Z",
        "closedTime": "2024-01-03 12:00:00+00",
        "volumeNum": 500000.0,
    }
    row.update(overrides)
    return row


def serve_markets(monkeypatch, pages: dict[int, list[dict]]) -> list[dict]:
    calls: list[dict] = []

    def fake_get(url, *, params=None, **kwargs):
        assert url == polymarket.GAMMA_URL
        calls.append(dict(params))
        return FakeResponse(pages.get(params["offset"], []))

    monkeypatch.setattr(polymarket, "http_get", fake_get)
    return calls


def serve_clob(monkeypatch, points: list[tuple[int, float]]) -> list[dict]:
    calls: list[dict] = []

    def fake_get(url, *, params=None, **kwargs):
        assert url == polymarket.CLOB_PRICES_URL
        calls.append(dict(params))
        history = [{"t": t, "p": p} for t, p in points
                   if params["startTs"] <= t <= params["endTs"]]
        return FakeResponse({"history": history})

    monkeypatch.setattr(polymarket, "http_get", fake_get)
    return calls


def hourly_points(hours, price_of=lambda h: 0.5) -> list[tuple[int, float]]:
    return [(T0 + h * HOUR + 1800, price_of(h)) for h in hours]


def binary_market(hours: int = 60, **overrides) -> dict:
    row = {
        "slug": "will-it-rain",
        "yes_token": TOKEN,
        "start": ts(T0),
        "end": ts(T0 + hours * HOUR),
        "scheduled_end": ts(T0 + hours * HOUR),
        "resolution": 1.0,
    }
    row.update(overrides)
    return row


def test_only_settled_yes_no_markets_are_kept(monkeypatch):
    rows = [
        market(id="1"),
        market(id="2", outcomePrices=json.dumps(["0", "1"]), slug="no-wins"),
        market(id="3", outcomes=json.dumps(["Up", "Down"])),
        market(id="4", outcomePrices=json.dumps(["0.55", "0.45"])),
    ]
    serve_markets(monkeypatch, {0: rows})
    df = polymarket.list_resolved_markets()
    assert list(df["id"]) == ["1", "2"]
    assert list(df["resolution"]) == [1.0, 0.0]


def test_scan_stops_at_first_market_below_min_volume(monkeypatch):
    full = [market(id=str(i), volumeNum=1e6 - i) for i in range(100)]
    second = ([market(id=str(100 + i), volumeNum=5e5) for i in range(3)]
              + [market(id="low", volumeNum=1e4)]
              + [market(id=str(200 + i), volumeNum=5e5) for i in range(96)])
    calls = serve_markets(monkeypatch, {0: full, 100: second})
    df = polymarket.list_resolved_markets(min_volume=1e5, max_markets=500)
    assert len(df) == 103
    assert "low" not in set(df["id"])
    assert [c["offset"] for c in calls] == [0, 100]
    assert calls[0] == {"closed": "true", "limit": 100, "offset": 0,
                        "order": "volumeNum", "ascending": "false"}


def test_max_markets_truncates_and_stops_paging(monkeypatch):
    calls = serve_markets(monkeypatch, {
        0: [market(id=str(i)) for i in range(100)],
        100: [market(id="unused")],
    })
    df = polymarket.list_resolved_markets(max_markets=5)
    assert list(df["id"]) == ["0", "1", "2", "3", "4"]
    assert [c["offset"] for c in calls] == [0]


def test_end_is_later_of_end_date_and_closed_time(monkeypatch):
    rows = [
        market(id="a", endDate="2024-01-03T12:00:00Z", closedTime="2024-01-04 08:00:00+00"),
        market(id="b", endDate="2024-01-05T00:00:00Z", closedTime="2024-01-04 08:00:00+00"),
        market(id="c", endDate=None, closedTime="2024-01-06 00:00:00+00"),
    ]
    serve_markets(monkeypatch, {0: rows})
    df = polymarket.list_resolved_markets().set_index("id")
    assert df.loc["a", "end"] == pd.Timestamp("2024-01-04 08:00", tz="UTC")
    assert df.loc["b", "end"] == pd.Timestamp("2024-01-05", tz="UTC")
    assert df.loc["c", "end"] == pd.Timestamp("2024-01-06", tz="UTC")


def test_yes_token_is_kept_as_exact_string(monkeypatch):
    serve_markets(monkeypatch, {0: [market()]})
    df = polymarket.list_resolved_markets()
    assert df.loc[0, "yes_token"] == TOKEN
    assert df.loc[0, "volume"] == 500000.0


def test_markets_cached_per_argument_pair(monkeypatch):
    calls = serve_markets(monkeypatch, {0: [market()]})
    first = polymarket.list_resolved_markets()
    second = polymarket.list_resolved_markets()
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_price_history_requests_consecutive_14_day_windows(monkeypatch):
    calls = serve_clob(monkeypatch, [])
    polymarket.load_price_history(TOKEN, 0, 40 * DAY)
    windows = [(c["startTs"], c["endTs"]) for c in calls]
    assert windows == [(0, 14 * DAY), (14 * DAY, 28 * DAY), (28 * DAY, 40 * DAY)]
    assert all(c["market"] == TOKEN and c["fidelity"] == 60 for c in calls)


def test_price_history_is_sorted_and_deduplicated(monkeypatch):
    # The point at 14 days falls inside both windows and must appear once.
    points = [(300, 0.3), (100, 0.1), (14 * DAY, 0.7), (14 * DAY + 50, 0.75)]
    serve_clob(monkeypatch, points)
    series = polymarket.load_price_history(TOKEN, 0, 20 * DAY)
    assert series.index.is_monotonic_increasing
    assert not series.index.has_duplicates
    assert list(series.index) == [ts(100), ts(300), ts(14 * DAY), ts(14 * DAY + 50)]
    np.testing.assert_allclose(series.to_numpy(), [0.1, 0.3, 0.7, 0.75])


def test_cached_span_serves_inner_requests_without_network(monkeypatch):
    calls = serve_clob(monkeypatch, [(t, 0.5) for t in range(0, 30 * DAY, HOUR)])
    polymarket.load_price_history(TOKEN, 0, 20 * DAY)
    fetched = len(calls)
    inner = polymarket.load_price_history(TOKEN, 5 * DAY, 10 * DAY)
    assert len(calls) == fetched
    assert inner.index.min() >= ts(5 * DAY) and inner.index.max() <= ts(10 * DAY)

    # Extending the span merges it, so a later request over the whole span is cached.
    polymarket.load_price_history(TOKEN, 10 * DAY, 30 * DAY)
    fetched = len(calls)
    whole = polymarket.load_price_history(TOKEN, 0, 30 * DAY)
    assert len(calls) == fetched
    assert whole.index.min() == ts(0) and whole.index.max() == ts(30 * DAY - HOUR)


def test_resample_repeats_previous_close_over_empty_bars(monkeypatch):
    points = [
        (T0 + 600, 0.40),
        (T0 + 3000, 0.45),
        (T0 + 2 * HOUR + 1200, 0.30),
        (T0 + 5 * HOUR + 300, 0.60),
    ] + hourly_points(range(6, 60))
    serve_clob(monkeypatch, points)
    df = polymarket.load_market(binary_market())
    assert len(df) == 60
    head = df.iloc[:6]
    np.testing.assert_allclose(head["open"], [0.40, 0.45, 0.30, 0.30, 0.30, 0.60])
    np.testing.assert_allclose(head["high"], [0.45, 0.45, 0.30, 0.30, 0.30, 0.60])
    np.testing.assert_allclose(head["low"], [0.40, 0.45, 0.30, 0.30, 0.30, 0.60])
    np.testing.assert_allclose(head["close"], [0.45, 0.45, 0.30, 0.30, 0.30, 0.60])


def test_bars_before_first_point_are_dropped(monkeypatch):
    serve_clob(monkeypatch, hourly_points(range(2, 60)))
    df = polymarket.load_market(binary_market())
    assert df.index[0] == ts(T0 + 2 * HOUR)
    assert len(df) == 58


def test_prices_are_clipped_to_open_interval(monkeypatch):
    serve_clob(monkeypatch, hourly_points(range(60), lambda h: 0.0 if h == 1 else 0.9999))
    df = polymarket.load_market(binary_market())
    validate_frame(df)
    assert df["close"].iloc[0] == pytest.approx(0.999)
    assert df["close"].iloc[1] == pytest.approx(0.001)


def test_fewer_than_48_bars_raises(monkeypatch):
    serve_clob(monkeypatch, hourly_points(range(47)))
    with pytest.raises(ValueError, match="need 48"):
        polymarket.load_market(binary_market(hours=60))


def test_exactly_48_bars_is_accepted(monkeypatch):
    serve_clob(monkeypatch, hourly_points(range(48)))
    df = polymarket.load_market(binary_market(hours=60))
    assert len(df) == 48


def test_frame_matches_binary_contract(monkeypatch):
    serve_clob(monkeypatch, hourly_points(range(60)))
    row = pd.Series(binary_market(slug="s" * 80, resolution=0.0))
    df = polymarket.load_market(row)
    validate_frame(df)
    assert df.attrs["kind"] == "binary"
    assert df.attrs["bar"] == "1h"
    assert df.attrs["symbol"] == "s" * 60
    assert df.attrs["resolution"] == 0.0
    assert df.attrs["end_time"] == "2024-01-03T12:00:00+00:00"


def test_end_time_is_the_scheduled_end_not_the_close(monkeypatch):
    serve_clob(monkeypatch, hourly_points(range(60)))
    df = polymarket.load_market(binary_market(hours=60, scheduled_end=ts(T0 + 100 * HOUR)))
    assert pd.Timestamp(df.attrs["end_time"]) == pd.Timestamp(ts(T0 + 100 * HOUR))
    df = polymarket.load_market(binary_market(hours=60, scheduled_end=None))
    assert "end_time" not in df.attrs


@pytest.mark.network
def test_top_resolved_market_loads_live():
    top = polymarket.list_resolved_markets(max_markets=1)
    assert len(top) == 1
    df = polymarket.load_market(top.iloc[0])
    validate_frame(df)
    assert df.attrs["kind"] == "binary"
    assert df.attrs["resolution"] in (0.0, 1.0)
    assert len(df) >= 48
