import pandas as pd

from sactrader.data.synthetic import binary_market, gbm
from sactrader.pipeline import BINARY_FEATURES, Period, SplitSpec, split_tapes, to_tapes


def test_split_cuts_linear_tapes_and_keeps_binary_markets_whole():
    lin = gbm(2000, seed=0, symbol="L", start="2021-01-01")
    b1 = binary_market(200, seed=1, symbol="B1", start="2021-01-01")
    b2 = binary_market(200, seed=2, symbol="B2", start="2021-02-20")
    tapes = to_tapes([lin, b1, b2])
    assert [tp.kind for tp in tapes] == ["spot", "binary", "binary"]
    assert tapes[1].feature_names == to_tapes([b1], feature_cfg=BINARY_FEATURES)[0].feature_names
    split = SplitSpec(Period(None, "2021-02-01"), Period("2021-02-01", "2021-02-15"),
                      Period("2021-02-15", None))
    parts = split_tapes(tapes, split)
    train_syms = [tp.symbol for tp in parts["train"]]
    test_syms = [tp.symbol for tp in parts["test"]]
    assert train_syms == ["L", "B1"] and "B2" in test_syms and "B1" not in test_syms
    lin_train = parts["train"][0]
    assert pd.Timestamp(int(lin_train.times[-1]), tz="UTC") < pd.Timestamp("2021-02-01", tz="UTC")
    assert not lin_train.terminal_at_end and parts["test"][0].symbol == "L"


def test_resample_hourly_perp_to_daily():
    import numpy as np

    from sactrader.data.schema import validate_frame
    from sactrader.data.synthetic import with_funding
    from sactrader.pipeline import resample_frame

    fr = with_funding(gbm(24 * 10, seed=3, symbol="P"), seed=1)
    fr["taker_buy_base"] = 1.0
    day = resample_frame(fr, "1D")
    validate_frame(day)
    assert len(day) == 10 and day.attrs["bar"] == "1D" and day.attrs["kind"] == "perp"
    first = fr.iloc[:24]
    assert day["open"].iloc[0] == first["open"].iloc[0]
    assert day["close"].iloc[0] == first["close"].iloc[-1]
    assert day["high"].iloc[0] == first["high"].max() and day["low"].iloc[0] == first["low"].min()
    assert np.isclose(day["funding_rate"].iloc[0], first["funding_rate"].sum())
    assert day["taker_buy_base"].iloc[0] == 24.0


def test_cross_features_join_linear_tapes_only():
    import numpy as np

    linear = [gbm(600, sigma=0.01, seed=i, symbol=f"S{i}") for i in range(3)]
    binary = binary_market(200, seed=9, symbol="B")
    plain = to_tapes(linear + [binary])
    tapes = to_tapes(linear + [binary], cross=True)
    for tp, ref in zip(tapes[:3], plain[:3], strict=True):
        assert "mkt_ret_h16" in tp.feature_names and "rel_ret_h16" in tp.feature_names
        assert tp.feature_names[:len(ref.feature_names)] == ref.feature_names
        assert len(tp) == len(ref) and np.abs(tp.features).max() <= 5.0
    assert "mkt_ret_h16" not in tapes[3].feature_names
    # Two linear frames are too few for a cross-section, so nothing is added.
    assert to_tapes(linear[:2], cross=True)[0].feature_names == plain[0].feature_names
