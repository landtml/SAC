import numpy as np
import pandas as pd
import pytest

from softedge.data.synthetic import gbm
from softedge.features.causal import compute_features
from softedge.features.cross import CROSS_HORIZONS, cross_features

T0 = pd.Timestamp("2024-01-01", tz="UTC")
PRICE = ["open", "high", "low", "close"]
ROOT3 = np.sqrt(3.0)


def _one_bar(values):
    """A one-row table at T0 holding ``ret_h{h}`` for each horizon in ``values``."""
    return pd.DataFrame({f"ret_h{h}": [v] for h, v in values.items()},
                        index=pd.DatetimeIndex([T0]))


def _row(out, sym):
    return out[sym].iloc[0].to_dict()


def test_hand_computed_three_assets():
    tables = {
        "A": _one_bar({1: 1.0, 4: 0.5, 16: 0.4, 64: 3.0}),
        "B": _one_bar({1: -2.0, 4: 0.5, 16: -0.2, 64: -1.0}),
        "C": _one_bar({1: 0.5, 4: 0.5, 16: 0.3, 64: -1.0}),
    }
    out = cross_features(tables)
    # Mean times sqrt(3) is the sum divided by sqrt(3).
    market = {
        "mkt_ret_h1": (-0.5 / 3) * ROOT3,
        "mkt_ret_h4": (1.5 / 3) * ROOT3,
        "mkt_ret_h16": (0.5 / 3) * ROOT3,
        "mkt_ret_h64": (1.0 / 3) * ROOT3,
        # Population std (ddof 0) of (1.0, -2.0, 0.5): mean -1/6, sum of squares 31/6.
        "mkt_dispersion": np.sqrt((31 / 6) / 3),
        # Two of three 16-bar returns are positive: (2/3 - 1/2) * 2.
        "mkt_breadth": 1 / 3,
    }
    # Each asset's 16-bar return minus the mean of the other two.
    rel = {"A": 0.4 - (-0.2 + 0.3) / 2, "B": -0.2 - (0.4 + 0.3) / 2, "C": 0.3 - (0.4 - 0.2) / 2}
    for sym in "ABC":
        got = _row(out, sym)
        for name, want in market.items():
            np.testing.assert_allclose(got[name], want, rtol=1e-6, err_msg=f"{sym} {name}")
        np.testing.assert_allclose(got["rel_ret_h16"], rel[sym], rtol=1e-6, err_msg=sym)


def test_market_return_is_clipped():
    tables = {s: _one_bar({1: 0.0, 4: 4.0, 16: 0.0, 64: 0.0}) for s in "ABC"}
    # 4 * sqrt(3) = 6.93 is above the default clip.
    assert _row(cross_features(tables), "A")["mkt_ret_h4"] == pytest.approx(5.0)
    assert _row(cross_features(tables, clip=10.0), "A")["mkt_ret_h4"] == pytest.approx(
        4.0 * ROOT3, rel=1e-6)
    negative = {s: _one_bar({1: 0.0, 4: -4.0, 16: 0.0, 64: 0.0}) for s in "ABC"}
    assert _row(cross_features(negative), "B")["mkt_ret_h4"] == pytest.approx(-5.0)


def test_rows_below_min_assets_are_zero():
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    full = pd.DataFrame({f"ret_h{h}": [0.5, -0.5] for h in CROSS_HORIZONS}, index=index)
    partial = full.copy()
    partial.iloc[1] = np.nan  # C has no finite return at the second timestamp
    tables = {"A": full, "B": full.copy(), "C": partial}

    out = cross_features(tables)
    assert (out["A"].iloc[1].to_numpy() == 0.0).all()
    assert (out["C"].iloc[1].to_numpy() == 0.0).all()
    assert out["A"].iloc[0]["mkt_ret_h1"] == pytest.approx(1.5 / ROOT3, rel=1e-6)

    # With two assets allowed, the second timestamp uses A and B alone: sum / sqrt(2).
    loose = cross_features(tables, min_assets=2)
    assert loose["A"].iloc[1]["mkt_ret_h1"] == pytest.approx(-1.0 / np.sqrt(2.0), rel=1e-6)
    assert loose["A"].iloc[1]["mkt_dispersion"] == pytest.approx(0.0, abs=1e-7)


def test_nan_return_drops_out_of_its_own_horizon_only():
    values = {1: 1.0, 4: 0.0, 16: 0.0, 64: 1.0}
    tables = {s: _one_bar(values) for s in "ABC"}
    tables["D"] = _one_bar({**values, 64: np.nan})
    got = _row(cross_features(tables), "A")
    # Four finite one-bar returns, so the sum is over four.
    assert got["mkt_ret_h1"] == pytest.approx(4.0 / 2.0, rel=1e-6)
    # Three finite 64-bar returns, so the sum is over three.
    assert got["mkt_ret_h64"] == pytest.approx(3.0 / ROOT3, rel=1e-6)


def _misaligned_tables(seed=0):
    rng = np.random.default_rng(seed)
    base = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    positions = {
        "A": np.arange(60),
        "B": np.arange(5, 60),  # starts late
        "C": np.delete(np.arange(60), [8, 9, 30]),  # gaps
        "D": np.arange(40),  # ends early, with warm-up NaNs in its 64-bar return
    }
    tables = {}
    for sym, pos in positions.items():
        data = {f"ret_h{h}": rng.normal(size=len(pos)) for h in CROSS_HORIZONS}
        if sym == "D":
            data["ret_h64"][:10] = np.nan
        tables[sym] = pd.DataFrame(data, index=base[pos]).astype(np.float32)
    return tables


def _reference(tables, horizons, min_assets, clip):
    """Row-by-row version of the definitions, with no vectorization, to check the fast path."""
    needed = sorted({*horizons, 1, 16})
    result = {}
    for sym, table in tables.items():
        rows = []
        for t in table.index:
            ret = {h: {} for h in needed}
            for other, other_table in tables.items():
                if t not in other_table.index:
                    continue
                for h in needed:
                    x = float(other_table.at[t, f"ret_h{h}"])
                    if np.isfinite(x):
                        ret[h][other] = x
            if any(len(ret[h]) < min_assets for h in needed):
                rows.append([0.0] * (len(horizons) + 3))
                continue
            row = []
            for h in horizons:
                vals = list(ret[h].values())
                mean = sum(vals) / len(vals)
                row.append(float(np.clip(mean * len(vals) ** 0.5, -clip, clip)))
            others = [v for s, v in ret[16].items() if s != sym]
            own = ret[16].get(sym)
            row.append(own - sum(others) / len(others) if own is not None and others else 0.0)
            v1 = list(ret[1].values())
            m1 = sum(v1) / len(v1)
            row.append(float(np.sqrt(sum((v - m1) ** 2 for v in v1) / len(v1))))
            v16 = list(ret[16].values())
            row.append((sum(v > 0 for v in v16) / len(v16) - 0.5) * 2.0)
            rows.append(row)
        result[sym] = np.array(rows)
    return result


def test_matches_row_by_row_reference_on_misaligned_tables():
    tables = _misaligned_tables()
    out = cross_features(tables)
    want = _reference(tables, CROSS_HORIZONS, min_assets=3, clip=5.0)
    names = [*(f"mkt_ret_h{h}" for h in CROSS_HORIZONS), "rel_ret_h16", "mkt_dispersion",
             "mkt_breadth"]
    some_zero = some_live = False
    for sym, table in tables.items():
        assert out[sym].index.equals(table.index)
        assert list(out[sym].columns) == names
        assert all(out[sym].dtypes == np.float32)
        assert np.isfinite(out[sym].to_numpy()).all()
        np.testing.assert_allclose(out[sym].to_numpy(), want[sym], atol=1e-5, rtol=1e-5,
                                   err_msg=sym)
        some_zero |= bool((out[sym].to_numpy() == 0.0).all(axis=1).any())
        some_live |= bool((out[sym].to_numpy() != 0.0).any())
    assert some_zero and some_live  # both branches of the gate are exercised


def test_causal_under_future_shock():
    frames = [gbm(600, sigma=0.01, seed=i, symbol=f"S{i}", start="2024-01-01") for i in range(3)]
    tables = {f.attrs["symbol"]: compute_features(f) for f in frames}
    base = cross_features(tables)

    cut = 300
    shocked_frame = frames[0].copy()
    rows = shocked_frame.index[cut + 1:]  # every price after row ``cut`` moves by 50%
    shocked_frame.loc[rows, PRICE] = shocked_frame.loc[rows, PRICE] * 1.5
    shocked = dict(tables)
    shocked["S0"] = compute_features(shocked_frame)
    after = cross_features(shocked)

    for sym in tables:
        np.testing.assert_array_equal(base[sym].to_numpy()[:cut + 1],
                                      after[sym].to_numpy()[:cut + 1], err_msg=sym)
    changed = [not np.array_equal(base[s].to_numpy()[cut + 1:], after[s].to_numpy()[cut + 1:])
               for s in tables]
    assert any(changed)


def test_rejects_too_few_tables_and_missing_columns():
    good = _one_bar({h: 0.1 for h in CROSS_HORIZONS})
    with pytest.raises(ValueError):
        cross_features({"A": good})
    with pytest.raises(ValueError):
        cross_features({"A": good, "B": good, "C": good.drop(columns=["ret_h16"])})
