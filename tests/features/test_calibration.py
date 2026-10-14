import numpy as np
import pandas as pd
import pytest

from softedge.data.schema import make_frame
from softedge.features.calibration import CalibConfig, calib_edge
from softedge.pipeline import load_frames, to_tapes

CFG = CalibConfig(min_markets=10, bins=10, shrink=5.0)


def _binary(prices, resolution, start, symbol):
    """Binary frame of hourly bars with the given YES prices, resolved one bar after its last bar."""
    idx = pd.date_range(start=start, periods=len(prices), freq="1h", tz="UTC")
    p = np.asarray(prices, dtype=np.float64)
    frame = make_frame(idx, p, p, p, p, symbol=symbol, kind="binary", bar="1h",
                       resolution=resolution)
    frame.attrs["resolved_at"] = (idx[-1] + pd.Timedelta("1h")).isoformat()
    return frame


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _planted(n_markets, n_bars=24, seed=0, bias=0.6):
    """Markets priced in log-odds shrunk by ``bias`` toward 0.5 (longshots too high, favourites
    too low). Outcomes follow the unshrunk log-odds, so the planted mispricing has a known sign.
    Start times are spread over 500 hours, so resolutions are staggered."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 1.5, n_markets)
    outcome = (rng.random(n_markets) < _sigmoid(latent)).astype(np.float64)
    starts = rng.integers(0, 500, n_markets)
    frames = []
    for m in range(n_markets):
        path = latent[m] + 0.05 * rng.standard_normal(n_bars)
        start = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=int(starts[m]))
        frames.append(_binary(_sigmoid(bias * path), outcome[m], start, f"M{m}"))
    return frames


def _flip_outcome(frame):
    out = frame.copy()
    out.attrs = dict(frame.attrs)
    out.attrs["resolution"] = 1.0 - frame.attrs["resolution"]
    return out


def test_bin_outcome_rate_is_shrunk_toward_the_price():
    # Two markets sit in the 0.2-0.3 bin at price 0.25, one resolving YES and one NO. Each has
    # weight 1, so the bin holds 2 markets with outcome sum 1. Shrinking with 2 pseudo-markets:
    # f = (1 + 2 * 0.25) / (2 + 2) = 0.375, so the edge is 0.125 at every bar of the third market.
    a = _binary([0.25] * 4, 1.0, "2020-01-01 00:00", "A")
    b = _binary([0.25] * 4, 0.0, "2020-01-01 00:00", "B")
    c = _binary([0.25] * 4, 0.0, "2020-01-01 05:00", "C")
    edges = calib_edge([a, b, c], CalibConfig(min_markets=2, bins=10, shrink=2.0))
    np.testing.assert_allclose(edges[2].to_numpy(), 0.125, atol=1e-6)
    assert (edges[0] == 0.0).all() and (edges[1] == 0.0).all()


def test_zero_shrink_uses_the_bin_outcome_rate_as_is():
    # Same bin with no shrinkage: the edge is the raw outcome rate minus the price.
    a =_binary([0.25] * 4, 1.0, "2020-01-01 00:00", "A")
    b = _binary([0.25] * 4, 0.0, "2020-01-01 00:00", "B")
    c = _binary([0.25] * 4, 0.0, "2020-01-01 05:00", "C")
    edges = calib_edge([a, b, c], CalibConfig(min_markets=2, bins=10, shrink=0.0))
    # S_w = 2 markets in the bin, S_y = 1 YES, so f = 0.5 and the edge is 0.5 - 0.25 = 0.25.
    np.testing.assert_allclose(edges[2].to_numpy(), 0.25, atol=1e-6)


def test_empty_bin_has_zero_edge():
    # Only the 0.2-0.3 bin has resolved markets; a bar priced at 0.85 has no evidence.
    a = _binary([0.25] * 4, 1.0, "2020-01-01 00:00", "A")
    b = _binary([0.25] * 4, 0.0, "2020-01-01 00:00", "B")
    d = _binary([0.85] * 4, 1.0, "2020-01-01 05:00", "D")
    edges = calib_edge([a, b, d], CalibConfig(min_markets=2, bins=10, shrink=0.0))
    assert (edges[2] == 0.0).all()


def test_zero_below_min_markets_and_nonzero_once_reached():
    a = _binary([0.25] * 4, 1.0, "2020-01-01 00:00", "A")
    b = _binary([0.25] * 4, 0.0, "2020-01-01 00:00", "B")
    c = _binary([0.25] * 4, 0.0, "2020-01-01 05:00", "C")
    assert (calib_edge([a, b, c], CalibConfig(min_markets=3, shrink=2.0))[2] == 0.0).all()
    assert (calib_edge([a, b, c], CalibConfig(min_markets=2, shrink=2.0))[2] != 0.0).all()


def test_market_resolving_at_a_bar_open_is_not_yet_known_at_that_bar():
    # A resolves at 04:00, the open of D's first bar, so A must not count there. It counts from
    # 05:00 on.
    a = _binary([0.25] * 4, 1.0, "2020-01-01 00:00", "A")
    d = _binary([0.25] * 4, 0.0, "2020-01-01 04:00", "D")
    edges = calib_edge([a, d], CalibConfig(min_markets=1, shrink=1.0))
    assert edges[1].iloc[0] == 0.0
    assert edges[1].iloc[1] != 0.0


def test_planted_longshot_bias_recovers_its_sign():
    # 1000 markets at bias 0.4: the favourite band holds few extreme markets, and at 400 its sign
    # was noise. Seeds 1 to 3 all give the planted signs with wide margins.
    frames = _planted(1000, seed=1, bias=0.4)
    edges = calib_edge(frames, CFG)
    price = np.concatenate([f["close"].to_numpy() for f in frames])
    edge = np.concatenate([e.to_numpy() for e in edges])
    longshot = (price > 0.15) & (price < 0.30)
    favourite = (price > 0.70) & (price < 0.85)
    assert edge[longshot].mean() < 0.0
    assert edge[favourite].mean() > 0.0


def _resolved_ts(frame):
    return pd.Timestamp(frame.attrs["resolved_at"])


def test_outcomes_resolving_at_or_after_t_do_not_change_the_feature_at_t():
    frames = _planted(120, seed=2)
    base = calib_edge(frames, CFG)
    # Cut at resolution instants themselves, where the boundary case (resolving exactly at t)
    # happens.
    cuts = sorted({_resolved_ts(f) for f in frames})[::31]
    checked_nonzero = 0
    for t in cuts:
        after = [_flip_outcome(f) if _resolved_ts(f) >= t else f for f in frames]
        new = calib_edge(after, CFG)
        for b, n in zip(base, new, strict=True):
            mask = b.index == t
            np.testing.assert_array_equal(b[mask].to_numpy(), n[mask].to_numpy())
            checked_nonzero += int(np.count_nonzero(b[mask].to_numpy()))
    assert checked_nonzero > 0


def test_outcomes_resolved_before_t_do_change_the_feature_at_t():
    # The negative control: the look-ahead test would pass trivially if the outcomes never
    # entered the feature at all.
    frames = _planted(120, seed=2)
    base = calib_edge(frames, CFG)
    t = max(f.index[-1] for f in frames)
    before = [_flip_outcome(f) if _resolved_ts(f) < t else f for f in frames]
    new = calib_edge(before, CFG)
    changed = any(not np.array_equal(b[b.index == t].to_numpy(), n[n.index == t].to_numpy())
                  for b, n in zip(base, new, strict=True))
    assert changed


def test_rejects_linear_frames_and_missing_resolution_time():
    from softedge.data.synthetic import gbm

    with pytest.raises(ValueError, match="binary"):
        calib_edge([gbm(100, seed=0, symbol="L")], CFG)
    frame = _binary([0.4] * 10, 1.0, "2020-01-01", "A")
    del frame.attrs["resolved_at"]
    with pytest.raises(ValueError, match="resolved_at"):
        calib_edge([frame], CFG)


def test_to_tapes_adds_calib_edge_column_matching_the_function():
    frames = _planted(60, n_bars=100, seed=3)
    tapes = to_tapes(frames, calib=CFG)
    ref = calib_edge(frames, CFG)
    for tp, edge in zip(tapes, ref, strict=True):
        idx = tp.feature_names.index("calib_edge")
        rows = edge.to_numpy()[len(edge) - len(tp):]
        np.testing.assert_allclose(tp.features[:, idx], rows, atol=1e-6)
    assert "calib_edge" not in to_tapes(frames)[0].feature_names


@pytest.mark.parametrize("universe", ["polymarket", "kalshi"])
def test_loaders_record_the_actual_close_as_resolution_time(monkeypatch, universe):
    from softedge.data import kalshi, polymarket

    end = pd.Timestamp("2025-03-01T12:00:00Z")
    listing = pd.DataFrame({"end": [end], "slug": ["m"], "ticker": ["m"]})
    frame = _binary([0.4] * 10, 1.0, "2025-02-01", "m")
    if universe == "polymarket":
        monkeypatch.setattr(polymarket, "list_resolved_markets", lambda **kw: listing)
        monkeypatch.setattr(polymarket, "load_market", lambda row: frame)
    else:
        monkeypatch.setattr(kalshi, "list_settled_markets", lambda **kw: listing)
        monkeypatch.setattr(kalshi, "load_market", lambda row: frame)
    (out,) = load_frames(universe)
    assert out.attrs["resolved_at"] == end.isoformat()
