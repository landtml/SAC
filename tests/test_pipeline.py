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
