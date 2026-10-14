import numpy as np
import pandas as pd
import pytest

from softedge.data.delivery import (
    DEFAULT_ROLL_COST,
    DEFAULT_ROLL_DAYS,
    delivery_basis_frame,
    expiry_of,
    front_chain,
    quarterly_expiries,
)
from softedge.data.schema import make_frame
from softedge.features.causal import compute_features

DAY = pd.Timedelta("1D")


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def days(first: str, last: str) -> pd.DatetimeIndex:
    return pd.date_range(first, last, freq="1D", tz="UTC")


def bars(index, close) -> pd.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    return pd.DataFrame({"close": close, "volume": np.ones(len(close))}, index=index)


def spot_frame(index, close) -> pd.DataFrame:
    c = np.asarray(close, dtype=np.float64)
    return make_frame(index, c, c, c, c, np.ones(len(c)), symbol="BTCUSDT", kind="spot", bar="1D")


def test_expiry_is_the_last_friday_of_the_quarter_at_0800_utc():
    assert expiry_of("BTCUSDT_250627") == utc("2025-06-27 08:00")
    assert quarterly_expiries("2025-01", "2025-12") == [
        utc("2025-03-28 08:00"), utc("2025-06-27 08:00"),
        utc("2025-09-26 08:00"), utc("2025-12-26 08:00"),
    ]


def test_spot_symbol_has_no_expiry():
    with pytest.raises(ValueError, match="expiry suffix"):
        expiry_of("BTCUSDT")


def test_front_rolls_seven_days_before_expiry_without_a_price_jump():
    index = days("2024-01-01", "2024-01-08")
    closes = {"A": bars(index, np.full(8, 100.0)), "B": bars(index, np.full(8, 105.0))}
    exp = {"A": utc("2024-01-09 08:00"), "B": utc("2024-02-09 08:00")}
    chain = front_chain(closes, exp, roll_days=7, roll_cost=0.001)
    # Close of 2024-01-01 is 7.3 days before expiry (still A); close of 2024-01-02 is 6.3 days.
    assert list(chain["held"].iloc[:3]) == ["A", "B", "B"]
    # The 5% level gap between A and B is not a return: the chain does not jump at the roll.
    assert chain["close"].iloc[1] / chain["close"].iloc[0] == pytest.approx(1.0)
    # The roll is traded at the close of 2024-01-02, so its cost lands on the next step.
    assert chain["close"].iloc[2] / chain["close"].iloc[1] == pytest.approx(1.001)


def test_expiring_contract_is_held_until_its_successor_is_listed():
    index = days("2024-01-01", "2024-01-08")
    a = bars(index, np.full(8, 100.0))
    b = bars(index[4:], np.full(4, 105.0))  # listed on 2024-01-05, after the roll window opens
    exp = {"A": utc("2024-01-09 08:00"), "B": utc("2024-02-09 08:00")}
    chain = front_chain({"A": a, "B": b}, exp, roll_days=7, roll_cost=0.001)
    assert list(chain["held"].iloc[:5]) == ["A", "A", "A", "A", "B"]
    assert chain["close"].loc[utc("2024-01-06")] / chain["close"].loc[utc("2024-01-05")] \
        == pytest.approx(1.001)


def test_missing_bar_of_the_held_contract_raises():
    index = days("2024-01-01", "2024-01-08")
    b = bars(index, np.full(8, 105.0)).drop(utc("2024-01-05"))
    closes = {"A": bars(index, np.full(8, 100.0)), "B": b}
    exp = {"A": utc("2024-01-10 08:00"), "B": utc("2024-02-09 08:00")}
    with pytest.raises(ValueError, match="no bar"):
        front_chain(closes, exp, roll_days=7, roll_cost=0.001)


def test_long_spot_short_future_earns_the_basis_as_it_converges():
    index = days("2024-01-01", "2024-01-15")
    expiry = utc("2024-01-31 08:00")
    left = np.asarray((expiry - (index + DAY)) / DAY, dtype=np.float64)
    fut = 100.0 * (1.0 + 0.10 * left / 365.0)
    frame = delivery_basis_frame(spot_frame(index, np.full(15, 100.0)),
                                 {"A": bars(index, fut)}, {"A": expiry}, symbol="BTC-DELIV")
    # A 10% annual basis decays as the future falls from its close-to-expiry value. The short
    # future gains exactly the ratio of the two future prices (about 0.10 * 14 / 365).
    f0, f1 = 1.0 + 0.10 * left[0] / 365.0, 1.0 + 0.10 * left[-1] / 365.0
    gain = frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0
    assert gain == pytest.approx(f0 / f1 - 1.0, rel=2e-3)


def test_basis_ann_is_the_annualized_basis_known_at_the_close():
    index = days("2024-01-01", "2024-01-03")
    expiry = utc("2024-01-11 08:00")
    frame = delivery_basis_frame(spot_frame(index, [100.0, 100.0, 100.0]),
                                 {"A": bars(index, [101.0, 101.0, 101.0])}, {"A": expiry},
                                 symbol="BTC-DELIV")
    # Close of 2024-01-01 is 9 days and 8 hours before expiry.
    want = (101.0 / 100.0 - 1.0) * 365.0 / (9 + 8 / 24)
    assert frame["basis_ann"].iloc[0] == pytest.approx(want, rel=1e-9)


def _two_contract_market(seed: int):
    index = days("2024-01-01", "2024-02-20")
    rng = np.random.default_rng(seed)
    spot = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    a = spot * (1.0 + 0.02 * rng.normal(size=len(index)).cumsum() / 10)
    b = spot * (1.0 + 0.03 + 0.02 * rng.normal(size=len(index)).cumsum() / 10)
    exp = {"A": utc("2024-01-31 08:00"), "B": utc("2024-03-29 08:00")}
    return index, spot_frame(index, spot), {"A": bars(index, a), "B": bars(index, b)}, exp


def test_basis_ann_at_t_ignores_data_after_t():
    index, spot, contracts, exp = _two_contract_market(seed=0)
    base = delivery_basis_frame(spot, contracts, exp, symbol="BTC-DELIV")
    cut = index[20]
    rng = np.random.default_rng(1)
    later = index > cut
    spot2 = spot.copy()
    spot2.loc[later, ["open", "high", "low", "close"]] *= rng.uniform(0.98, 1.02, (later.sum(), 1))
    contracts2 = {}
    for sym, b in contracts.items():
        b2 = b.copy()
        b2.loc[later, "close"] *= rng.uniform(0.98, 1.02, later.sum())
        contracts2[sym] = b2
    other = delivery_basis_frame(spot2, contracts2, exp, symbol="BTC-DELIV")
    np.testing.assert_array_equal(base.loc[:cut, "basis_ann"].to_numpy(),
                                  other.loc[:cut, "basis_ann"].to_numpy())


def test_basis_ann_reaches_the_feature_table_clipped_to_five():
    index, spot, contracts, exp = _two_contract_market(seed=2)
    frame = delivery_basis_frame(spot, contracts, exp, symbol="BTC-DELIV")
    table = compute_features(frame)
    assert "basis_ann" in table.columns
    np.testing.assert_allclose(table["basis_ann"].to_numpy(),
                               frame["basis_ann"].clip(-5.0, 5.0).to_numpy(), rtol=1e-6,
                               atol=1e-7)


def test_default_roll_is_seven_days_and_two_normal_perp_trades():
    from softedge.tape import DEFAULT_COSTS

    perp = DEFAULT_COSTS["perp"]
    assert DEFAULT_ROLL_DAYS == 7
    assert DEFAULT_ROLL_COST == pytest.approx(2 * (perp.fee_bps + perp.half_spread_bps) * 1e-4)


def test_load_frames_routes_the_delivery_universe(monkeypatch):
    import softedge.data.delivery as delivery
    from softedge.pipeline import load_frames

    seen = {}

    def fake(symbols, start, end, **kwargs):
        seen.update(symbols=tuple(symbols), start=start, end=end)
        return ["sentinel"]

    monkeypatch.setattr(delivery, "load_delivery_basis", fake)
    assert load_frames("delivery_basis", "2024-01", "2024-03", symbols=["BTCUSDT"]) == ["sentinel"]
    assert seen == {"symbols": ("BTCUSDT",), "start": "2024-01", "end": "2024-03"}


def test_loader_builds_one_frame_per_coin_from_contracts_and_spot(monkeypatch):
    import softedge.data.binance as binance
    import softedge.data.delivery as delivery

    index = days("2024-01-01", "2024-02-20")
    exp = {"BTCUSDT_240531": utc("2024-05-31 08:00"), "BTCUSDT_240329": utc("2024-03-29 08:00")}
    listed = [delivery.Contract(s, e, ("2024-01",)) for s, e in exp.items()]
    prices = {s: bars(index, np.full(len(index), 100.0 + i)) for i, s in enumerate(exp)}
    monkeypatch.setattr(delivery, "list_contracts", lambda base, start, end: listed)
    monkeypatch.setattr(delivery, "load_contract", lambda c: prices[c.symbol])
    spot_1h = make_frame(pd.date_range("2024-01-01", periods=24 * len(index), freq="1h", tz="UTC"),
                         *(np.full(24 * len(index), 100.0),) * 4, np.ones(24 * len(index)),
                         symbol="BTCUSDT", kind="spot", bar="1h")
    monkeypatch.setattr(binance, "load_spot", lambda *a, **k: spot_1h)
    (frame,) = delivery.load_delivery_basis(["BTCUSDT"], "2024-01", "2024-02")
    assert frame.attrs["symbol"] == "BTC-DELIV" and frame.attrs["kind"] == "perp"
    assert {"basis_ann", "funding_rate"} <= set(frame.columns)
    assert len(frame) > 40


@pytest.mark.network
def test_archive_lists_the_june_2025_quarterly_contract():
    from softedge.data.delivery import list_contracts

    got = {c.symbol: c for c in list_contracts("BTCUSDT", "2025-01", "2025-06")}
    assert "BTCUSDT_250627" in got and "2025-03" in got["BTCUSDT_250627"].months
