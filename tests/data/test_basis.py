import numpy as np
import pandas as pd
import pytest

from sactrader.data.basis import basis_frame
from sactrader.data.schema import make_frame, validate_frame

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
    assert out.attrs == {"symbol": "BTCUSDT-BASIS", "kind": "perp", "bar": "1h"}
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
