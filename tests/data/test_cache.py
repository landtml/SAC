import numpy as np
import pandas as pd

from sactrader.data.cache import cache_dir, load_frame, save_frame
from sactrader.data.schema import make_frame, validate_frame


def test_frame_roundtrip_keeps_attrs(tmp_path, monkeypatch):
    monkeypatch.setenv("SACTRADER_DATA", str(tmp_path))
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
