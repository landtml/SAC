"""With ``open_interest`` off, crypto_basis tapes are identical to those before the squeeze hook.

The digest was computed on the code before the open-interest hook, from synthetic markets, so
the test fails if any tape array or field changes while the flag is off.
"""

import dataclasses
import hashlib
from unittest import mock

import numpy as np
import pandas as pd

import softedge.pipeline as pipeline
from softedge.data import binance
from softedge.data.schema import make_frame
from softedge.tape import MarketTape

PINNED_DIGEST = "7e4813b94085e98d46b666257118cbe0de1dbec9de27c4cd5dc2ccb7f88a96b8"


def hourly(kind: str, sym: str, n: int, seed: int, **extra):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    return make_frame(idx, close, close * 1.001, close * 0.999, close, np.full(n, 5.0),
                      symbol=sym, kind=kind, bar="1h", **extra)


def spot(s, i, start, end):
    return hourly("spot", s, 24 * 500, sum(map(ord, s)) % 1000)


def perp(s, i, start, end):
    n = 24 * 500
    rng = np.random.default_rng(7)
    return hourly("perp", s, n, sum(map(ord, s)) % 1000,
                  funding_rate=rng.normal(1e-4, 2e-4, n))


def crypto_basis_tapes():
    with mock.patch.object(binance, "load_spot", spot), \
            mock.patch.object(binance, "load_perp", perp):
        frames = pipeline.load_frames("crypto_basis", "2023-01", "2024-06",
                                      symbols=["BTCUSDT", "ETHUSDT"])
    frames = [pipeline.resample_frame(f, "1D") for f in frames]
    return frames, pipeline.to_tapes(frames)


def switch_off_digest() -> str:
    _, tapes = crypto_basis_tapes()
    h = hashlib.sha256()
    for tp in tapes:
        for field in dataclasses.fields(MarketTape):
            value = getattr(tp, field.name)
            if isinstance(value, np.ndarray):
                h.update(field.name.encode())
                h.update(np.ascontiguousarray(value).tobytes())
            else:
                h.update(repr(value).encode())
    return h.hexdigest()


def test_flag_off_tapes_match_the_pinned_digest():
    assert switch_off_digest() == PINNED_DIGEST


def test_flag_off_adds_no_open_interest_column_or_gated_feature():
    frames, tapes = crypto_basis_tapes()
    assert all("oi_value" not in f.columns for f in frames)
    assert all("funding_ewm_gated" not in tp.feature_names for tp in tapes)
