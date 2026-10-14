import pandas as pd

from softedge.data.synthetic import binary_market, gbm
from softedge.pipeline import BINARY_FEATURES, Period, SplitSpec, split_tapes, to_tapes


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

    from softedge.data.schema import validate_frame
    from softedge.data.synthetic import with_funding
    from softedge.pipeline import resample_frame

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


def test_regime_features_are_added_on_request():
    frames = [gbm(600, sigma=0.01, seed=i, symbol=f"S{i}") for i in range(2)]
    plain, with_regime = to_tapes(frames), to_tapes(frames, regime=True)
    assert "vr_16" in with_regime[0].feature_names and "vr_16" not in plain[0].feature_names
    assert len(with_regime[0]) <= len(plain[0])


def _write_legacy_cache(frame, path) -> None:
    """Write a bar frame the way releases before the rename did: attrs under the old key."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"sactrader": json.dumps(frame.attrs).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_resampling_a_perp_cached_under_the_old_project_key(tmp_path, monkeypatch):
    from softedge.data.binance import _frame_path
    from softedge.data.synthetic import gbm, with_funding
    from softedge.pipeline import load_frames, resample_frame

    monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path))
    perp = with_funding(gbm(24 * 40, seed=1, symbol="BTCUSDT", start="2024-01-01"), seed=1)
    _write_legacy_cache(perp, _frame_path("perp", "BTCUSDT", "1h", "2024-01", "2024-02"))
    (frame,) = load_frames("crypto_perp", "2024-01", "2024-02", symbols=["BTCUSDT"])
    assert resample_frame(frame, "1D").attrs["kind"] == "perp"


def test_basis_of_legs_cached_under_the_old_project_key(tmp_path, monkeypatch):
    from softedge.data.binance import _frame_path
    from softedge.data.synthetic import gbm, with_funding
    from softedge.pipeline import load_frames, resample_frame

    monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path))
    spot = gbm(24 * 40, seed=0, symbol="BTCUSDT", start="2024-01-01")
    perp = with_funding(gbm(24 * 40, seed=1, symbol="BTCUSDT", start="2024-01-01"), seed=1)
    _write_legacy_cache(spot, _frame_path("spot", "BTCUSDT", "1h", "2024-01", "2024-02"))
    _write_legacy_cache(perp, _frame_path("perp", "BTCUSDT", "1h", "2024-01", "2024-02"))
    (basis,) = load_frames("crypto_basis", "2024-01", "2024-02", symbols=["BTCUSDT"])
    assert basis.attrs["symbol"] == "BTCUSDT-BASIS"
    assert resample_frame(basis, "1D").attrs["bar"] == "1D"


def test_cli_download_accepts_vrp(monkeypatch):
    from softedge import pipeline
    from softedge.cli import main

    seen = []
    monkeypatch.setattr(pipeline, "load_frames", lambda universe, *a, **k: seen.append(universe) or [])
    assert main(["download", "vrp"]) == 0
    assert seen == ["vrp"]
