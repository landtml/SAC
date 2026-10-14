import numpy as np
import pandas as pd
import pytest

from softedge.data import binance, pegs
from softedge.data.schema import make_frame, validate_frame
from softedge.features.causal import compute_features
from softedge.pipeline import load_frames


def _spot(close, symbol, peg, start="2024-01-01"):
    close = np.asarray(close, dtype=np.float64)
    idx = pd.date_range(start, periods=len(close), freq="1h", tz="UTC")
    frame = make_frame(idx, close, close, close, close, np.ones(len(close)),
                       symbol=symbol, kind="spot", bar="1h")
    if peg is not None:
        frame.attrs["peg"] = peg
    return frame


def _walk(n, seed, scale=0.001, centre=1.0):
    rng = np.random.default_rng(seed)
    return centre * np.exp(np.cumsum(rng.normal(0.0, scale, n)))


def test_fixed_peg_deviation_is_log_price_over_one():
    frame = _spot([1.001, 0.999, 1.0], "USDCUSDT", peg=1.0)
    dev = pegs.peg_deviation(frame)
    np.testing.assert_allclose(dev.to_numpy(), np.log([1.001, 0.999, 1.0]))


def test_trailing_median_deviation_matches_a_direct_reference():
    # 60 days of hourly bars; the window is the trailing 30 days, the current bar included.
    close = _walk(24 * 60, seed=1, scale=0.002, centre=0.03)
    frame = _spot(close, "WBETHETH", peg=pegs.TRAILING)
    dev = pegs.peg_deviation(frame).to_numpy()
    times = frame.index
    for t in (200, 700, 800, len(close) - 1):
        window = close[(times > times[t] - pd.Timedelta("30D")) & (times <= times[t])]
        assert dev[t] == pytest.approx(np.log(close[t] / np.median(window)))


def test_trailing_median_waits_for_a_week_of_history():
    # Before a week of bars the median of a few bars is too noisy to call a peg: NaN there,
    # so the feature is missing for the first week, inside the feature warm-up.
    close = _walk(24 * 20, seed=8, scale=0.002)
    frame = _spot(close, "BNSOLSOL", peg=pegs.TRAILING)
    dev = pegs.peg_deviation(frame).to_numpy()
    assert np.isnan(dev[: 24 * 7]).all()
    assert np.isfinite(dev[24 * 7 :]).all()
    fixed = pegs.peg_deviation(_spot(close, "USDCUSDT", peg=1.0)).to_numpy()
    assert np.isfinite(fixed).all()


def test_trailing_median_never_reads_bars_after_t():
    close = _walk(24 * 40, seed=2, scale=0.002)
    frame = _spot(close, "BNSOLSOL", peg=pegs.TRAILING)
    full = pegs.peg_deviation(frame).to_numpy()
    head = pegs.peg_deviation(frame.iloc[:500]).to_numpy()
    np.testing.assert_allclose(head, full[:500])


@pytest.mark.parametrize("peg", [1.0, pegs.TRAILING])
def test_changing_bars_after_t_leaves_peg_dev_at_t_unchanged(peg):
    # Look-ahead test for the prior feature: rewrite every bar after t, then the peg_dev
    # rows up to t must not move. The rewritten rows must move, so the test can fail.
    close = _walk(24 * 50, seed=3, scale=0.002)
    t = 900
    altered = close.copy()
    altered[t + 1:] = altered[t + 1:] * 1.05
    before = compute_features(_spot(close, "X", peg=peg))["peg_dev"].to_numpy()
    after = compute_features(_spot(altered, "X", peg=peg))["peg_dev"].to_numpy()
    np.testing.assert_array_equal(before[: t + 1], after[: t + 1])
    assert not np.allclose(before[t + 1:], after[t + 1:])


def test_peg_dev_column_only_for_pegged_frames():
    plain = _spot(_walk(300, seed=4), "PLAIN", peg=None)
    assert "peg_dev" not in compute_features(plain).columns
    assert "peg_dev" in compute_features(_spot(_walk(300, seed=4), "P", peg=1.0)).columns


def test_attach_peg_marks_stablecoins_fixed_and_staking_tokens_trailing():
    raw = _spot(_walk(50, seed=5), "USDCUSDT", peg=None)
    assert pegs.attach_peg(raw).attrs["peg"] == 1.0
    stake = _spot(_walk(50, seed=6), "WBETHETH", peg=None)
    assert pegs.attach_peg(stake).attrs["peg"] == pegs.TRAILING
    validate_frame(pegs.attach_peg(stake))


def _runs(symbol, runs, peg=None):
    """Spot frame from runs of ``(start, closes)``. Each run starts where the archive resumes,
    so the hours between two runs are a gap."""
    parts = [_spot(closes, symbol, peg=peg, start=start) for start, closes in runs]
    frame = pd.concat(parts)
    frame.attrs = dict(parts[0].attrs)
    return frame


def test_series_splits_at_gaps_longer_than_two_bars_and_drops_short_pieces(monkeypatch):
    # Run one has a single missing bar (a two-hour step, not a gap). Runs one and two are
    # separated by a four-hour gap; runs two and three by a week. Run three lasts five days.
    one = _walk(24 * 31, seed=9, scale=0.001)
    two = _walk(24 * 31, seed=10, scale=0.001)
    three = _walk(24 * 5, seed=11, scale=0.001)
    frame = _runs("USDCUSDT", [("2020-01-01", one), ("2020-02-01 03:00", two),
                               ("2020-03-10", three)])
    frame = frame.drop(frame.index[100])
    monkeypatch.setattr(binance, "load_spot", lambda *a, **k: frame)
    out = pegs.load_peg_frames(symbols=["USDCUSDT"])
    assert [f.attrs["symbol"] for f in out] == ["USDCUSDT-1", "USDCUSDT-2"]
    assert len(out[0]) == 24 * 31 - 1
    assert out[1].index[0] == pd.Timestamp("2020-02-01 03:00", tz="UTC")
    assert all(f.attrs["peg"] == 1.0 and f.attrs["kind"] == "spot" for f in out)
    for f in out:
        validate_frame(f)


def test_trailing_median_restarts_in_each_piece(monkeypatch):
    # The second piece trades about 60% above the first. Its peg must come from its own bars,
    # and its first week has no peg, even though it starts more than 30 days into the series.
    first = _walk(24 * 40, seed=1, scale=0.002, centre=0.03)
    second = _walk(24 * 40, seed=2, scale=0.002, centre=0.05)
    frame = _runs("WBETHETH", [("2024-01-01", first), ("2024-02-12", second)])
    monkeypatch.setattr(binance, "load_spot", lambda *a, **k: frame)
    out = pegs.load_peg_frames(symbols=["WBETHETH"])
    assert [f.attrs["symbol"] for f in out] == ["WBETHETH-1", "WBETHETH-2"]
    assert out[1].attrs["peg"] == pegs.TRAILING
    dev = pegs.peg_deviation(out[1]).to_numpy()
    assert np.isnan(dev[: 24 * 7]).all() and np.isfinite(dev[24 * 7 :]).all()
    times = out[1].index
    for t in (24 * 10, 24 * 30, len(second) - 1):
        window = second[(times > times[t] - pd.Timedelta("30D")) & (times <= times[t])]
        assert dev[t] == pytest.approx(np.log(second[t] / np.median(window)))


def test_unsplit_series_is_returned_whole_under_its_own_symbol(monkeypatch):
    frame = _runs("BNSOLSOL", [("2024-01-01", _walk(24 * 10, seed=12))])
    monkeypatch.setattr(binance, "load_spot", lambda *a, **k: frame)
    (out,) = pegs.load_peg_frames(symbols=["BNSOLSOL"])
    assert out.attrs["symbol"] == "BNSOLSOL" and len(out) == 24 * 10
    validate_frame(out)


def test_pegs_universe_loads_hourly_spot_pairs_with_their_peg(monkeypatch):
    calls = []

    def fake_load_spot(symbol, interval="1h", start="2020-01", end="2026-09"):
        calls.append((symbol, interval, start, end))
        return _spot(_walk(48, seed=7), symbol, peg=None)

    monkeypatch.setattr(binance, "load_spot", fake_load_spot)
    frames = load_frames("pegs", "2024-01", "2024-06", symbols=["USDCUSDT", "WBETHETH"])
    assert calls == [("USDCUSDT", "1h", "2024-01", "2024-06"),
                     ("WBETHETH", "1h", "2024-01", "2024-06")]
    assert [f.attrs["peg"] for f in frames] == [1.0, pegs.TRAILING]
    assert all(f.attrs["kind"] == "spot" for f in frames)
