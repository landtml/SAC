import numpy as np
import pandas as pd
import pytest

from sactrader.data.schema import FrameError, make_frame
from sactrader.tape import CostSpec, build_tape


def _frame(close, kind="spot", resolution=None, funding=None):
    close = np.asarray(close, dtype=np.float64)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="1h", tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 if kind == "binary" else 1.01)
    low = np.minimum(open_, close) * 0.99
    if kind == "binary":
        high = np.minimum(np.maximum(open_, close) + 0.01, 0.999)
    return make_frame(idx, open_, high, low, close, np.ones(len(close)), symbol="X",
                      kind=kind, bar="1h", funding_rate=funding, resolution=resolution)


def _features(frame, warmup=2):
    f = pd.DataFrame({"a": np.arange(len(frame), dtype=float)}, index=frame.index)
    f.iloc[:warmup] = np.nan
    return f


def test_linear_tape_returns_and_warmup():
    fr = _frame([100, 110, 99, 99, 120])
    tape = build_tape(fr, _features(fr), CostSpec(fee_bps=5, half_spread_bps=1), bars_per_year=8760)
    assert len(tape) == 3
    np.testing.assert_allclose(tape.long_ret, [0.0, 120 / 99 - 1, 0.0])
    np.testing.assert_allclose(tape.short_ret, -tape.long_ret)
    np.testing.assert_allclose(tape.cost_long, 6e-4)
    assert tape.features[0, 0] == 2.0
    assert not tape.terminal_at_end
    assert (tape.long_worst <= tape.long_ret).all() and (tape.short_worst <= tape.short_ret).all()


def test_perp_funding_charged_in_next_bar():
    fund = np.array([0.0, 0.0, 0.0001, 0.0, -0.0002])
    fr = _frame([100, 100, 100, 100, 100], kind="perp", funding=fund)
    tape = build_tape(fr, _features(fr, warmup=0), bars_per_year=8760)
    np.testing.assert_allclose(tape.carry_long, [0.0, 0.0001, 0.0, -0.0002, 0.0])
    np.testing.assert_allclose(tape.carry_short, -tape.carry_long)


def test_binary_settlement_on_last_row():
    fr = _frame([0.4, 0.5, 0.8], kind="binary", resolution=1.0)
    tape = build_tape(fr, _features(fr, warmup=0), CostSpec(fee_bps=0.0, half_spread_abs=0.01), bars_per_year=8760)
    assert tape.terminal_at_end
    assert tape.long_ret[-1] == pytest.approx(1 / 0.8 - 1)
    assert tape.short_ret[-1] == pytest.approx(-1.0)
    assert tape.short_ret[0] == pytest.approx(0.5 / 0.6 - 1)
    assert tape.cost_short[2] == pytest.approx(0.01 / 0.2)


def test_slice_cut_row_has_no_returns():
    fr = _frame([0.4, 0.5, 0.6, 0.7, 0.8], kind="binary", resolution=0.0)
    tape = build_tape(fr, _features(fr, warmup=0), bars_per_year=8760)
    head = tape.slice(0, 3)
    assert not head.terminal_at_end
    assert head.long_ret[-1] == 0.0 and head.short_growth[-1] == 1.0
    tail = tape.time_slice("2024-01-01 02:00", None)
    assert tail.terminal_at_end and len(tail) == 3


def test_nan_after_warmup_rejected():
    fr = _frame([100, 101, 102, 103])
    f = _features(fr, warmup=1)
    f.iloc[2] = np.nan
    with pytest.raises(ValueError, match="after warm-up"):
        build_tape(fr, f, bars_per_year=8760)


def test_schema_rejects_bad_frames():
    with pytest.raises(FrameError):
        _frame([0.5, 1.2], kind="binary", resolution=1.0)
    with pytest.raises(FrameError):
        _frame([100, 101], kind="perp")
