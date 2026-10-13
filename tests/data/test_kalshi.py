import pandas as pd
import pytest

from sactrader.data import kalshi
from sactrader.data.schema import validate_frame

T0 = pd.Timestamp("2026-10-01T00:00:00Z")
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
SETTLED = T0 + 3 * DAY + pd.Timedelta(minutes=10)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def api_time(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def market(**overrides) -> dict:
    row = {
        "ticker": "KXTEST-26OCT01-T5",
        "event_ticker": "KXTEST-26OCT01",
        "title": "Will it rain?",
        "market_type": "binary",
        "status": "finalized",
        "result": "yes",
        "open_time": api_time(T0),
        "close_time": api_time(T0 + 3 * DAY),
        "expiration_time": api_time(T0 + 9 * DAY),
        "latest_expiration_time": api_time(T0 + 10 * DAY),
        "settlement_ts": api_time(SETTLED),
        "volume_fp": "250000.00",
    }
    row.update(overrides)
    return row


def candle(open_time: pd.Timestamp, bid=None, ask=None, last=None) -> dict:
    # Candles are labelled by their end time, one hour after the open.
    return {
        "end_period_ts": int((open_time + HOUR).timestamp()),
        "yes_bid": {"close_dollars": bid},
        "yes_ask": {"close_dollars": ask},
        "price": {"close_dollars": last},
        "volume_fp": "0.00",
        "open_interest_fp": "0.00",
    }


class FakeKalshi:
    """Stands in for ``http_get``: pages are keyed by (series, cursor), candles are shared."""

    def __init__(self, pages=None, candles=None):
        self.pages = pages or {}
        self.candles = candles or []
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, *, params=None, **kwargs):
        params = dict(params or {})
        self.calls.append((url, params))
        if url == f"{kalshi.API_URL}/markets":
            key = (params["series_ticker"], params.get("cursor"))
            return FakeResponse(self.pages.get(key, {"markets": [], "cursor": ""}))
        if url.endswith("/candlesticks"):
            lo, hi = params["start_ts"], params["end_ts"]
            # Both window ends are inclusive, so a boundary candle can come back twice.
            inside = [c for c in self.candles
                      if lo <= c["end_period_ts"] - HOUR.total_seconds() <= hi]
            return FakeResponse({"candlesticks": inside, "ticker": url.split("/")[-2]})
        raise AssertionError(f"unexpected request {url}")


@pytest.fixture(autouse=True)
def isolated(request, tmp_path, monkeypatch):
    # Network tests keep the caller's cache directory.
    if request.node.get_closest_marker("network"):
        return
    monkeypatch.setenv("SACTRADER_DATA", str(tmp_path))
    monkeypatch.setattr(kalshi, "REQUEST_INTERVAL_S", 0.0)


def serve(monkeypatch, fake: FakeKalshi) -> FakeKalshi:
    monkeypatch.setattr(kalshi, "http_get", fake)
    return fake


def serve_series(monkeypatch, by_series: dict[str, list[dict]], candles=None) -> FakeKalshi:
    pages = {(s, None): {"markets": ms, "cursor": ""} for s, ms in by_series.items()}
    return serve(monkeypatch, FakeKalshi(pages=pages, candles=candles))


def listing_row(**overrides) -> dict:
    row = {
        "ticker": "KXTEST-26OCT01-T5",
        "series_ticker": "KXTEST",
        "event_ticker": "KXTEST-26OCT01",
        "title": "Will it rain?",
        "start": T0,
        "end": T0 + 3 * DAY,
        "scheduled_end": T0 + 3 * DAY,
        "resolution": 1.0,
        "volume": 250000.0,
    }
    row.update(overrides)
    return row


def hourly_candles(hours, last_of=lambda h: 0.5) -> list[dict]:
    return [candle(T0 + h * HOUR, last=f"{last_of(h):.4f}") for h in range(hours)]


def market_calls(fake: FakeKalshi) -> list[dict]:
    return [params for url, params in fake.calls if url.endswith("/markets")]


def test_listing_queries_each_series_and_follows_each_cursor(monkeypatch):
    pages = {
        ("KXA", None): {"markets": [market(ticker="A-1", volume_fp="300000")], "cursor": "c1"},
        ("KXA", "c1"): {"markets": [market(ticker="A-2", volume_fp="200000")], "cursor": "c2"},
        ("KXA", "c2"): {"markets": [market(ticker="A-3", volume_fp="900000")], "cursor": ""},
        ("KXB", None): {"markets": [market(ticker="B-1", volume_fp="400000")], "cursor": ""},
    }
    fake = serve(monkeypatch, FakeKalshi(pages=pages))

    rows = kalshi.list_settled_markets(series=("KXA", "KXB"), min_volume=0, max_markets=10)

    queried = [(p["series_ticker"], p.get("cursor")) for p in market_calls(fake)]
    assert queried == [("KXA", None), ("KXA", "c1"), ("KXA", "c2"), ("KXB", None)]
    assert all(p["status"] == "settled" and p["limit"] == kalshi.PAGE_SIZE
               for p in market_calls(fake))
    assert list(rows["ticker"]) == ["A-3", "B-1", "A-1", "A-2"]
    assert rows.set_index("ticker")["series_ticker"].to_dict() == {
        "A-1": "KXA", "A-2": "KXA", "A-3": "KXA", "B-1": "KXB"}


def test_listing_keeps_only_yes_or_no_binary_markets(monkeypatch):
    markets = [
        market(ticker="KEEP-YES", result="yes"),
        market(ticker="KEEP-NO", result="no"),
        market(ticker="DROP-VOID", result="void"),
        market(ticker="DROP-EMPTY", result=""),
        market(ticker="DROP-SCALAR", market_type="scalar"),
    ]
    serve_series(monkeypatch, {"KXA": markets})

    rows = kalshi.list_settled_markets(series=("KXA",), min_volume=0, max_markets=10)

    assert sorted(rows["ticker"]) == ["KEEP-NO", "KEEP-YES"]
    assert rows.set_index("ticker")["resolution"].to_dict() == {"KEEP-YES": 1.0, "KEEP-NO": 0.0}


def test_listing_applies_volume_floor_and_cap_after_sorting_across_series(monkeypatch):
    serve_series(monkeypatch, {
        "KXA": [market(ticker="A-5E5", volume_fp="500000"),
                market(ticker="A-2E5", volume_fp="200000"),
                market(ticker="A-BELOW", volume_fp="10000")],
        "KXB": [market(ticker="B-3E5", volume_fp="300000"),
                market(ticker="B-9E5", volume_fp="900000")],
    })

    rows = kalshi.list_settled_markets(series=("KXA", "KXB"), min_volume=1e5, max_markets=3)

    assert list(rows["ticker"]) == ["B-9E5", "A-5E5", "B-3E5"]
    assert list(rows["volume"]) == [900000.0, 500000.0, 300000.0]


def test_listing_maps_fields_and_takes_scheduled_end_from_latest_expiration(monkeypatch):
    serve_series(monkeypatch, {"KXTEST": [market()]})

    rows = kalshi.list_settled_markets(series=("KXTEST",), min_volume=0, max_markets=5)

    row = rows.iloc[0]
    assert row["ticker"] == "KXTEST-26OCT01-T5"
    assert row["series_ticker"] == "KXTEST"
    assert row["event_ticker"] == "KXTEST-26OCT01"
    assert row["start"] == T0
    assert row["scheduled_end"] == T0 + 10 * DAY
    # The later of close and settlement bounds the candles, so settlement wins here.
    assert row["end"] == SETTLED
    assert row["resolution"] == 1.0
    assert row["volume"] == 250000.0


def test_scheduled_end_falls_back_to_expiration_time(monkeypatch):
    market_without_latest = market()
    del market_without_latest["latest_expiration_time"]
    serve_series(monkeypatch, {"KXTEST": [market_without_latest]})

    rows = kalshi.list_settled_markets(series=("KXTEST",), min_volume=0, max_markets=5)

    assert rows.iloc[0]["scheduled_end"] == T0 + 9 * DAY


def test_end_time_attribute_is_latest_expiration_not_actual_close(monkeypatch):
    # An early close moves close_time to the actual close, so close_time is not the schedule.
    early_close = market(close_time=api_time(T0 + 2 * DAY))
    serve_series(monkeypatch, {"KXTEST": [early_close]}, candles=hourly_candles(50))
    rows = kalshi.list_settled_markets(series=("KXTEST",), min_volume=0, max_markets=5)

    frame = kalshi.load_candles(rows.iloc[0])

    assert frame.attrs["end_time"] == (T0 + 10 * DAY).isoformat()
    assert frame.attrs["end_time"] != (T0 + 2 * DAY).isoformat()


def test_listing_repeat_call_reads_the_cache(monkeypatch):
    fake = serve_series(monkeypatch, {"KXTEST": [market()]})
    first = kalshi.list_settled_markets(series=("KXTEST",), min_volume=0, max_markets=5)
    calls_after_first = len(fake.calls)

    second = kalshi.list_settled_markets(series=("KXTEST",), min_volume=0, max_markets=5)

    assert len(fake.calls) == calls_after_first
    pd.testing.assert_frame_equal(first, second)


def test_cache_key_includes_the_series_tuple(monkeypatch):
    fake = serve_series(monkeypatch, {"KXA": [market(ticker="A-1")],
                                      "KXB": [market(ticker="B-1")]})
    kalshi.list_settled_markets(series=("KXA",), min_volume=0, max_markets=5)
    only_a = kalshi.list_settled_markets(series=("KXA", "KXB"), min_volume=0, max_markets=5)

    assert sorted(only_a["ticker"]) == ["A-1", "B-1"]
    assert {p["series_ticker"] for p in market_calls(fake)} == {"KXA", "KXB"}


def test_dollar_strings_become_prices_without_scaling(monkeypatch):
    candles = [candle(T0, last="0.4200"), candle(T0 + HOUR, last="0.6000")]
    candles += hourly_candles(48, last_of=lambda h: 0.5)[2:]
    serve(monkeypatch, FakeKalshi(candles=candles))

    frame = kalshi.load_candles(listing_row())

    assert frame["close"].iloc[0] == pytest.approx(0.42)
    assert frame["close"].iloc[1] == pytest.approx(0.60)


def test_bid_ask_mid_is_preferred_to_the_last_trade(monkeypatch):
    candles = [
        candle(T0, bid="0.3000", ask="0.5000", last="0.9000"),
        candle(T0 + HOUR, bid="0.3000", ask=None, last="0.6000"),
    ]
    candles += hourly_candles(50)[2:]
    serve(monkeypatch, FakeKalshi(candles=candles))

    frame = kalshi.load_candles(listing_row())

    assert frame["close"].iloc[0] == pytest.approx(0.40)
    # A one-sided quote is not a price, so the last trade is used.
    assert frame["close"].iloc[1] == pytest.approx(0.60)


def test_prices_clip_to_the_open_unit_interval(monkeypatch):
    candles = [candle(T0 + h * HOUR, last="0.0000" if h % 2 == 0 else "1.0000")
               for h in range(50)]
    serve(monkeypatch, FakeKalshi(candles=candles))

    frame = kalshi.load_candles(listing_row())

    assert frame["close"].min() == pytest.approx(kalshi.PRICE_FLOOR)
    assert frame["close"].max() == pytest.approx(kalshi.PRICE_CEIL)


def test_leading_empty_candles_drop_and_gaps_forward_fill(monkeypatch):
    candles = [candle(T0 + h * HOUR) for h in range(2)]
    # Hour 10 is absent from the response altogether.
    candles += [candle(T0 + h * HOUR, last=f"{0.2 + 0.001 * h:.4f}")
                for h in range(2, 60) if h != 10]
    serve(monkeypatch, FakeKalshi(candles=candles))

    frame = kalshi.load_candles(listing_row(end=T0 + 60 * HOUR, scheduled_end=T0 + 60 * HOUR))

    assert frame.index[0] == T0 + 2 * HOUR
    assert (frame.index.to_series().diff().dropna() == HOUR).all()
    assert frame.loc[T0 + 10 * HOUR, "close"] == pytest.approx(frame.loc[T0 + 9 * HOUR, "close"])


def test_frame_satisfies_the_binary_contract(monkeypatch):
    serve(monkeypatch, FakeKalshi(candles=hourly_candles(50)))

    frame = kalshi.load_candles(listing_row())

    validate_frame(frame)
    assert frame.attrs["kind"] == "binary"
    assert frame.attrs["symbol"] == "KXTEST-26OCT01-T5"
    assert frame.attrs["resolution"] == 1.0
    assert frame.attrs["bar"] == "60min"
    assert str(frame.index.tz) == "UTC"


def test_fewer_than_48_bars_raises(monkeypatch):
    serve(monkeypatch, FakeKalshi(candles=hourly_candles(30)))

    with pytest.raises(ValueError, match="bars"):
        kalshi.load_candles(listing_row(end=T0 + 30 * HOUR, scheduled_end=T0 + 30 * HOUR))


def test_load_market_resamples_hourly_candles_to_daily_bars(monkeypatch):
    # Hour h of day d trades at 0.2 + 0.01 * (h % 24), so each day has open 0.2 and close 0.43.
    hours = 48 * 24
    candles = [candle(T0 + h * HOUR, last=f"{0.2 + 0.01 * (h % 24):.4f}") for h in range(hours)]
    fake = serve(monkeypatch, FakeKalshi(candles=candles))
    row = listing_row(end=T0 + hours * HOUR, scheduled_end=T0 + hours * HOUR)

    frame = kalshi.load_market(row, bar="1D")

    validate_frame(frame)
    assert len(frame) == 48
    assert frame.attrs["bar"] == "1D"
    assert frame.iloc[0]["open"] == pytest.approx(0.20)
    assert frame.iloc[0]["close"] == pytest.approx(0.43)
    assert frame.iloc[0]["high"] == pytest.approx(0.43)
    assert frame.iloc[0]["low"] == pytest.approx(0.20)
    # A 48-day span needs two candlestick windows of at most 30 days each.
    candle_calls = [params for url, params in fake.calls if url.endswith("/candlesticks")]
    assert len(candle_calls) == 2
    assert all(p["end_ts"] - p["start_ts"] <= kalshi.MAX_WINDOW_S for p in candle_calls)


def test_load_market_rejects_bars_finer_than_a_candle(monkeypatch):
    serve(monkeypatch, FakeKalshi(candles=hourly_candles(50)))

    with pytest.raises(ValueError, match="at least 60 minutes"):
        kalshi.load_market(listing_row(), bar="30min")


@pytest.mark.network
def test_top_settled_market_loads_from_the_live_api():
    listing = kalshi.list_settled_markets(max_markets=1)
    assert len(listing) == 1
    frame = kalshi.load_market(listing.iloc[0])
    validate_frame(frame)
    assert frame.attrs["kind"] == "binary"
    assert frame.attrs["resolution"] in (0.0, 1.0)
