import numpy as np
import pandas as pd
import pytest

from softedge.data.basis import basis_frame
from softedge.data.schema import make_frame, validate_frame

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def at(*hours: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([T0 + pd.Timedelta(hours=h) for h in hours])


def leg(kind, symbol, index, close, volume, funding=None, bar="1h"):
    close = np.asarray(close, dtype=np.float64)
    return make_frame(index, close, close, close, close, volume, symbol=symbol, kind=kind,
                      bar=bar, funding_rate=funding)


def test_closes_compound_the_leg_spread_and_schema_holds():
    idx = at(0, 1, 2, 3)
    spot = leg("spot", "BTCUSDT", idx, [100, 110, 110, 121], [1, 2, 3, 4])
    perp = leg("perp", "BTCUSDT", idx, [100, 100, 110, 99], [5, 1, 2, 6],
               funding=[0.0001, -0.0002, 0.0, 0.0003])
    out = basis_frame(spot, perp)
    validate_frame(out)
    assert out.attrs == {"symbol": "BTCUSDT-BASIS", "kind": "perp", "bar": "1h", "short_borrow": True}
    # Spot returns are 0.1, 0, 0.1 and perp returns are 0, 0.1, -0.1. The spreads 0.1, -0.1
    # and 0.2 move the close by factors 1.1, 0.9 and 1.2.
    np.testing.assert_allclose(out["close"], [100, 110, 99, 118.8])
    np.testing.assert_allclose(out["open"], [100, 100, 110, 99])
    np.testing.assert_allclose(out["high"], [100, 110, 110, 118.8])
    np.testing.assert_allclose(out["low"], [100, 100, 99, 99])


def test_volume_is_the_smaller_leg_and_funding_flips_sign():
    idx = at(0, 1, 2, 3)
    spot = leg("spot", "BTCUSDT", idx, [100, 110, 110, 121], [1, 2, 3, 4])
    perp = leg("perp", "BTCUSDT", idx, [100, 100, 110, 99], [5, 1, 2, 6],
               funding=[0.0001, -0.0002, 0.0, 0.0003])
    out = basis_frame(spot, perp)
    np.testing.assert_allclose(out["volume"], [1, 1, 2, 4])
    np.testing.assert_allclose(out["funding_rate"], [-0.0001, 0.0002, 0.0, -0.0003])


def test_identical_legs_give_a_flat_close():
    idx = at(0, 1, 2)
    prices = [100, 105, 98]
    spot = leg("spot", "BTCUSDT", idx, prices, [1, 1, 1])
    perp = leg("perp", "BTCUSDT", idx, prices, [1, 1, 1], funding=[0.0, 0.0, 0.0])
    out = basis_frame(spot, perp)
    np.testing.assert_allclose(out["close"], [100, 100, 100])


def test_only_common_bars_enter_the_spread():
    spot = leg("spot", "BTCUSDT", at(0, 1, 2, 3), [100, 110, 110, 121], [1, 1, 1, 1])
    perp = leg("perp", "BTCUSDT", at(0, 2, 3, 4), [100, 100, 110, 50], [1, 1, 1, 1],
               funding=[0.0, 0.0, 0.0, 0.0])
    out = basis_frame(spot, perp)
    assert list(out.index) == list(at(0, 2, 3))
    # Hour 4 exists only in perp and must not move the close. The return from hour 0 to
    # hour 2 spans the spot bar at hour 1, which is missing from perp.
    np.testing.assert_allclose(out["close"], [100, 110, 110])


def test_fewer_than_two_common_bars_raises():
    spot = leg("spot", "BTCUSDT", at(0, 1), [100, 101], [1, 1])
    perp = leg("perp", "BTCUSDT", at(2, 3), [100, 101], [1, 1], funding=[0.0, 0.0])
    with pytest.raises(ValueError, match="at least 2"):
        basis_frame(spot, perp)


def test_mismatched_bars_or_symbols_and_swapped_legs_raise():
    spot = leg("spot", "BTCUSDT", at(0, 1), [100, 101], [1, 1])
    perp_4h = leg("perp", "BTCUSDT", at(0, 1), [100, 101], [1, 1], funding=[0.0, 0.0],
                  bar="4h")
    perp_eth = leg("perp", "ETHUSDT", at(0, 1), [100, 101], [1, 1], funding=[0.0, 0.0])
    perp = leg("perp", "BTCUSDT", at(0, 1), [100, 101], [1, 1], funding=[0.0, 0.0])
    with pytest.raises(ValueError, match="legs differ"):
        basis_frame(spot, perp_4h)
    with pytest.raises(ValueError, match="legs differ"):
        basis_frame(spot, perp_eth)
    with pytest.raises(ValueError, match="spot frame, then a perp frame"):
        basis_frame(perp, spot)


def test_spread_of_two_perps_nets_their_funding():
    from softedge.data.basis import spread_frame
    from softedge.data.schema import make_frame

    idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")

    def perp(closes, funding, symbol):
        c = np.asarray(closes, dtype=float)
        return make_frame(idx, c, c, c, c, np.ones(4), symbol=symbol, kind="perp", bar="1D",
                          funding_rate=np.asarray(funding, dtype=float))

    a = perp([100, 110, 99, 99], [0.001, 0.0, 0.002, 0.0], "BTCUSDT")
    b = perp([50, 54, 54, 49.5], [0.003, 0.001, 0.0, 0.0], "BTC")
    out = spread_frame(a, b, "BTC-XV")
    want = 100 * np.cumprod([1.0, 1 + 0.10 - 0.08, 1 - 0.10 - 0.0, 1 + 0.0 + 1 / 12])
    np.testing.assert_allclose(out["close"].to_numpy(), want)
    np.testing.assert_allclose(out["funding_rate"].to_numpy(), [-0.002, -0.001, 0.002, 0.0])
    assert out.attrs["symbol"] == "BTC-XV" and out.attrs["kind"] == "perp"


def test_short_basis_pays_spot_borrow_on_the_tape():
    from softedge.features.causal import FeatureConfig, compute_features
    from softedge.tape import CostSpec, build_tape

    idx = pd.date_range("2024-01-01", periods=80, freq="1D", tz="UTC")
    c = np.linspace(100, 120, 80)

    def leg(kind):
        extra = {"funding_rate": np.full(80, 1e-4)} if kind == "perp" else {}
        return make_frame(idx, c, c, c, c, np.ones(80), symbol="BTCUSDT", kind=kind, bar="1D", **extra)

    frame = basis_frame(leg("spot"), leg("perp"))
    assert frame.attrs["short_borrow"] is True
    cfg = FeatureConfig(horizons=(1, 4), ema_spans=(4,), vol_long_span=8)
    with_borrow = build_tape(frame, compute_features(frame, cfg), CostSpec(borrow_apr=0.1))
    without = build_tape(frame, compute_features(frame, cfg), CostSpec(borrow_apr=0.0))
    extra = with_borrow.carry_short[:-1] - without.carry_short[:-1]
    np.testing.assert_allclose(extra, 0.1 / with_borrow.bars_per_year)
    np.testing.assert_array_equal(with_borrow.carry_long, without.carry_long)


def test_spread_nets_the_premium_with_fundings_sign():
    idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    c = np.array([100.0, 101.0, 102.0, 103.0])
    spot = make_frame(idx, c, c, c, c, np.ones(4), symbol="BTCUSDT", kind="spot", bar="1D")
    perp = make_frame(idx, c, c, c, c, np.ones(4), symbol="BTCUSDT", kind="perp", bar="1D",
                      funding_rate=np.full(4, 1e-4))
    perp["premium"] = [2e-4, -1e-4, 0.0, 3e-4]
    out = basis_frame(spot, perp)
    np.testing.assert_allclose(out["premium"].to_numpy(), [-2e-4, 1e-4, 0.0, -3e-4])
    assert "premium" not in basis_frame(spot, perp.drop(columns="premium")).columns
