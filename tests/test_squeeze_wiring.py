"""Wiring of the open-interest gate: the loader flag, the pipeline hook, and the config pair."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import softedge.experiment as experiment
import softedge.pipeline as pipeline
from softedge.data import binance, metrics
from softedge.data.schema import make_frame

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "carry"


def hourly(kind: str, n: int = 48, **extra):
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    price = np.linspace(100.0, 101.0, n)
    return make_frame(idx, price, price, price, price, np.ones(n), symbol="BTCUSDT",
                      kind=kind, bar="1h", **extra)


def test_crypto_basis_attaches_open_interest_only_when_asked(monkeypatch):
    monkeypatch.setattr(binance, "load_spot", lambda s, i, start, end: hourly("spot"))
    monkeypatch.setattr(binance, "load_perp",
                        lambda s, i, start, end: hourly("perp", funding_rate=np.zeros(48)))
    oi = pd.Series([1.0], index=pd.DatetimeIndex(["2024-01-01 00:00"], tz="UTC"))
    monkeypatch.setattr(metrics, "load_open_interest", lambda sym, start, end: oi)

    plain = pipeline.load_frames("crypto_basis", symbols=["BTCUSDT"])
    gated = pipeline.load_frames("crypto_basis", symbols=["BTCUSDT"], open_interest=True)
    assert "oi_value" not in plain[0].columns
    assert "oi_value" in gated[0].columns


def test_build_all_tapes_forwards_the_open_interest_flag(monkeypatch):
    seen = {}

    def fake_load_frames(universe, start, end, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(experiment, "load_frames", fake_load_frames)
    experiment.build_all_tapes({"universe": "crypto_basis", "data": {"open_interest": True}})
    assert seen["open_interest"] is True
    experiment.build_all_tapes({"universe": "crypto_basis", "data": {}})
    assert seen["open_interest"] is False


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())


def test_gated_config_differs_from_the_funding_prior_only_in_the_gate():
    gated = _load("crypto_basis_1d_squeeze_gate.yaml")
    base = _load("crypto_basis_1d_squeeze_base.yaml")
    prior = _load("crypto_basis_1d_prior_v2.yaml")

    assert gated["data"]["open_interest"] is True
    assert gated["env"]["prior_feature"] == "funding_ewm_gated"
    assert "open_interest" not in base["data"]
    assert base["env"]["prior_feature"] == "funding_ewm"

    def without(cfg: dict) -> dict:
        cfg = {k: v for k, v in cfg.items() if k != "name"}
        cfg["data"] = {k: v for k, v in cfg["data"].items() if k != "open_interest"}
        cfg["env"] = {k: v for k, v in cfg["env"].items() if k != "prior_feature"}
        return cfg

    assert without(gated) == without(base)
    assert without(base) == without(prior)


def test_daily_resample_keeps_the_23_55_snapshot_of_each_day(monkeypatch):
    from softedge.pipeline import resample_frame

    snaps = pd.Series(
        [10.0, 20.0, 30.0],
        index=pd.DatetimeIndex(["2024-01-01 23:55", "2024-01-02 00:00", "2024-01-02 23:55"],
                               tz="UTC"),
        name="oi_value")
    monkeypatch.setattr(metrics, "load_open_interest", lambda sym, start, end: snaps)
    frame = metrics.attach_open_interest(hourly("spot", n=48), "BTCUSDT")
    daily = resample_frame(frame, "1D")
    # Day 1 closes at 2024-01-02 00:00, so its last hourly bar takes the 23:55 snapshot.
    # Day 2 closes at 2024-01-03 00:00 and takes the 23:55 snapshot of its own day.
    assert daily["oi_value"].tolist() == [10.0, 30.0]
