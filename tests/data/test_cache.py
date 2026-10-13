import numpy as np
import pandas as pd

from softedge.data.cache import cache_dir, load_frame, save_frame
from softedge.data.schema import make_frame, validate_frame


def test_frame_roundtrip_keeps_attrs(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path))
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    c = np.array([1.0, 2.0, 3.0])
    df = make_frame(idx, c, c, c, c, None, symbol="X", kind="perp", bar="1h",
                    funding_rate=np.zeros(3))
    path = cache_dir("t") / "x.parquet"
    save_frame(df, path)
    back = load_frame(path)
    pd.testing.assert_frame_equal(back, df, check_freq=False)
    assert back.attrs == df.attrs
    validate_frame(back)


def test_load_frame_reads_attrs_cached_under_the_old_project_key(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from softedge.data.cache import load_frame
    from softedge.data.synthetic import gbm

    frame = gbm(10, sigma=0.01, seed=0, symbol="OLD")
    table = pa.Table.from_pandas(frame)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"sactrader": json.dumps(frame.attrs).encode()})
    path = tmp_path / "old.parquet"
    pq.write_table(table, path)
    assert load_frame(path).attrs["symbol"] == "OLD"
