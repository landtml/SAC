import numpy as np
import pandas as pd
import pytest

from softedge import experiment
from softedge.data.schema import make_frame
from softedge.features.crosssec import xs_funding_features
from softedge.pipeline import to_tapes

START = pd.Timestamp("2024-01-01", tz="UTC")


def carry_frame(symbol, income, start=START, seed=0):
    """Daily carry frame: ``funding_rate`` is minus the income, as in ``spread_frame``."""
    income = np.asarray(income, dtype=float)
    n = len(income)
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    return make_frame(idx, close, close * 1.01, close * 0.99, close, np.ones(n),
                      symbol=symbol, kind="perp", bar="1D", funding_rate=-income)


def test_xs_funding_is_robust_z_of_trailing_income_and_topk_picks_the_richest():
    frames = [carry_frame(s, [v] * 5) for s, v in zip("ABCD", [1.0, 2.0, 3.0, 10.0], strict=True)]
    out = xs_funding_features(frames, window=3, topk=2)
    # Days 0 and 1 have no full 3-day window yet.
    for o in out:
        assert (o.iloc[:2] == 0.0).all().all()
    # Trailing means 1, 2, 3, 10: median 2.5, MAD 1.0. D's 7.5 is clipped at 5.
    day2 = np.array([o["funding_xs"].iloc[2] for o in out])
    assert np.allclose(day2, [-1.5, -0.5, 0.5, 5.0])
    assert [o["funding_topk"].iloc[2] for o in out] == [0.0, 0.0, 1.0, 1.0]
    assert list(out[0].columns) == ["funding_xs", "funding_topk"]


def test_a_coin_joins_the_cross_section_only_with_a_full_window():
    base = [carry_frame(s, [v] * 6) for s, v in zip("ABCD", [1.0, 2.0, 3.0, 10.0], strict=True)]
    late = carry_frame("E", [100.0] * 4, start=START + pd.Timedelta(days=2), seed=4)
    ref = xs_funding_features(base, window=3, topk=2)
    out = xs_funding_features([*base, late], window=3, topk=2)
    # E's window is incomplete on days 2 and 3 (days 0 and 1 are missing), so the others are unchanged.
    for k in range(4):
        assert np.array_equal(out[k].iloc[:4].to_numpy(), ref[k].iloc[:4].to_numpy())
    assert (out[4].loc[:START + pd.Timedelta(days=3)] == 0.0).all().all()
    # On day 4 E's window is 2..4 and full: median of 1, 2, 3, 10, 100 is 3, MAD 2, so E is clipped at 5.
    assert out[4].loc[START + pd.Timedelta(days=4), "funding_xs"] == pytest.approx(5.0)


def test_xs_funding_reads_no_bar_after_t():
    rng = np.random.default_rng(0)
    days = 12
    base = {s: rng.normal(1.0, 1.0, days) for s in "ABCDE"}
    ref = xs_funding_features([carry_frame(s, base[s]) for s in "ABCDE"], window=3, topk=2)
    for t in range(1, days - 2):
        frames = []
        for s in "ABCDE":
            v = base[s].copy()
            v[t + 1:] = rng.normal(50.0, 10.0, days - t - 1)
            frames.append(carry_frame(s, v))
        # A coin listed after t, with a large income, must not move the cross-section at t.
        frames.append(carry_frame("F", rng.normal(50.0, 1.0, days - t - 1),
                                  start=START + pd.Timedelta(days=t + 1), seed=9))
        out = xs_funding_features(frames, window=3, topk=2)
        for k in range(5):
            assert np.array_equal(out[k].iloc[:t + 1].to_numpy(), ref[k].iloc[:t + 1].to_numpy()), t


def test_xs_funding_needs_at_least_min_coins_and_daily_bars():
    two = xs_funding_features([carry_frame("A", [1.0] * 5), carry_frame("B", [2.0] * 5)], window=3)
    assert all((o == 0.0).all().all() for o in two)
    hourly = carry_frame("A", [1.0] * 5)
    hourly.attrs["bar"] = "1h"
    with pytest.raises(ValueError, match="daily"):
        xs_funding_features([hourly, carry_frame("B", [1.0] * 5), carry_frame("C", [1.0] * 5)])


def test_to_tapes_adds_the_funding_rank_only_on_request():
    frames = [carry_frame(s, np.linspace(0.5, 2.0, 320) * (i + 1), seed=i)
              for i, s in enumerate("ABCD")]
    plain = to_tapes(frames)
    tapes = to_tapes(frames, xs_funding={"window": 7, "topk": 2})
    for tp, ref in zip(tapes, plain, strict=True):
        assert tp.feature_names[:-2] == ref.feature_names
        assert tp.feature_names[-2:] == ("funding_xs", "funding_topk")
        assert np.isfinite(tp.features).all()
    assert all("funding_xs" not in tp.feature_names for tp in plain)


def test_build_all_tapes_passes_the_xs_funding_block(monkeypatch):
    frames = [carry_frame(s, np.linspace(0.5, 2.0, 320), seed=i) for i, s in enumerate("ABCD")]
    monkeypatch.setattr(experiment, "load_frames", lambda *a, **k: frames)
    data = {"start": "2024-01", "end": "2024-12"}
    plain = experiment.build_all_tapes({"universe": "crypto_basis", "data": data})
    tapes = experiment.build_all_tapes(
        {"universe": "crypto_basis", "data": {**data, "xs_funding": {"window": 7, "topk": 2}}})
    assert all("funding_xs" in tp.feature_names for tp in tapes)
    assert all("funding_xs" not in tp.feature_names for tp in plain)
