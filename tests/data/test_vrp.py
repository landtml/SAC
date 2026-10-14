from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from softedge.data import fred_markets, vol_indices, vrp
from softedge.env.vec_env import ActionMap, TapeSet
from softedge.features.causal import FeatureConfig, compute_features
from softedge.pipeline import load_frames
from softedge.tape import build_tape

PRIOR_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "vrp" / "vrp_spx_1d_prior.yaml"
SHORT_FEATURES = FeatureConfig(horizons=(1, 4, 16), ema_spans=(16,), vol_long_span=64)


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content


@pytest.fixture(autouse=True)
def no_pacing(monkeypatch):
    monkeypatch.setattr(fred_markets, "REQUEST_INTERVAL_S", 0.0)
    monkeypatch.setattr(vol_indices, "REQUEST_INTERVAL_S", 0.0)


def series_csv(series_id: str, dates: list[str], values: np.ndarray) -> str:
    body = "".join(f"{d},{v}\n" for d, v in zip(dates, values, strict=True))
    return f"observation_date,{series_id}\n{body}"


def sample(seed: int = 0, n: int = 90):
    rng = np.random.default_rng(seed)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2019-01-02", periods=n)]
    spx = np.round(2500.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))), 2)
    vix = np.round(rng.uniform(12.0, 35.0, n), 2)
    return dates, spx, vix


def build(monkeypatch, cache, dates, spx, vix, *, prior: bool = True):
    series = {
        "SP500": series_csv("SP500", dates, spx),
        "VIXCLS": series_csv("VIXCLS", dates, vix),
    }
    monkeypatch.setenv("SOFTEDGE_DATA", str(cache))

    def fake_get(url, *, params=None, **kwargs):
        return FakeResponse(series[params["id"]].encode())

    monkeypatch.setattr(fred_markets, "http_get", fake_get)
    monkeypatch.setattr(vol_indices, "http_get", fake_get)
    return vrp.load_vrp_frame("2000-01", "2100-01", prior=prior)


def reference_rv(close: np.ndarray, t: int) -> float:
    logs = np.log(close)
    sq = [(logs[j] - logs[j - 1]) ** 2 for j in range(t - 20, t + 1)]
    return 252.0 * sum(sq) / 21.0


def test_realized_variance_is_252_times_mean_of_21_squared_returns():
    _, spx, _ = sample()
    rv = vrp.realized_variance(spx)
    # The first full window of 21 returns ends at row 21, since row 0 has no return.
    assert np.isnan(rv[:21]).all() and np.isfinite(rv[21:]).all()
    for t in (21, 40, 89):
        assert rv[t] == pytest.approx(reference_rv(spx, t), rel=1e-12)


def test_index_moves_by_implied_now_minus_realized_next():
    _, spx, vix = sample()
    iv = (vix / 100.0) ** 2
    rv = vrp.realized_variance(spx)
    index = vrp.short_variance_closes(iv, rv)
    for t in range(22, 60):
        expected = (iv[t - 1] - rv[t]) / 252.0
        assert index[t] / index[t - 1] - 1.0 == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("which", ["iv", "rv"])
def test_missing_input_after_warm_up_raises_instead_of_zeroing(which):
    _, spx, vix = sample()
    iv = (vix / 100.0) ** 2
    rv = vrp.realized_variance(spx)
    if which == "iv":
        iv[40] = np.nan
    else:
        rv[60] = np.nan
    with pytest.raises(ValueError, match="after warm-up"):
        vrp.short_variance_closes(iv, rv)


def test_prior_is_implied_minus_realized_and_zero_while_gated(monkeypatch, tmp_path):
    dates, spx, vix = sample()
    frame = build(monkeypatch, tmp_path / "a", dates, spx, vix)
    close = frame["close"].to_numpy()
    rolling_high = pd.Series(close).rolling(63, min_periods=1).max().to_numpy()
    prior = frame["vrp_prior"].to_numpy()
    iv = (vix / 100.0) ** 2
    open_rows = closed_rows = 0
    for t in range(21, len(frame)):
        raw = iv[t] - reference_rv(spx, t)
        if close[t] >= rolling_high[t]:
            open_rows += 1
            assert prior[t] == pytest.approx(raw, rel=1e-9)
        else:
            closed_rows += 1
            assert prior[t] == 0.0
    # Guards against a vacuous pass: both gate states must occur in the sample.
    assert open_rows > 0 and closed_rows > 0
    assert (prior[:21] == 0.0).all()


@pytest.mark.parametrize("series", ["SP500", "VIXCLS"])
def test_changing_prices_or_vix_after_bar_t_leaves_bar_t_unchanged(monkeypatch, tmp_path, series):
    dates, spx, vix = sample()
    k = 50
    base = build(monkeypatch, tmp_path / "base", dates, spx, vix)
    spx2, vix2 = spx.copy(), vix.copy()
    if series == "SP500":
        spx2[k + 1:] = np.round(spx[k + 1:] * 1.3, 2)
    else:
        vix2[k + 1:] = np.round(vix[k + 1:] + 15.0, 2)
    alt = build(monkeypatch, tmp_path / "alt", dates, spx2, vix2)
    np.testing.assert_allclose(alt["vrp_prior"].iloc[: k + 1], base["vrp_prior"].iloc[: k + 1],
                               rtol=0, atol=0)
    np.testing.assert_allclose(alt["close"].iloc[: k + 1], base["close"].iloc[: k + 1],
                               rtol=0, atol=0)
    # Guards against a vacuous pass: the rows after k must actually change.
    assert not np.allclose(alt["vrp_prior"].iloc[k + 1:], base["vrp_prior"].iloc[k + 1:])


def test_universe_adds_prior_column_only_when_implied_vol_is_requested(monkeypatch, tmp_path):
    dates, spx, vix = sample()
    build(monkeypatch, tmp_path / "a", dates, spx, vix, prior=False)
    plain = load_frames("vrp_spx", "2000-01", "2100-01")[0]
    with_prior = load_frames("vrp_spx", "2000-01", "2100-01", implied_vol=True)[0]
    assert "vrp_prior" not in plain.columns
    assert "vrp_prior" in with_prior.columns
    assert plain.attrs["symbol"] == vrp.SYMBOL and plain.attrs["kind"] == "spot"
    assert plain.attrs["bar"] == "1D"


def test_shipped_prior_config_holds_short_variance_when_implied_exceeds_realized(monkeypatch, tmp_path):
    # VIX 20 is 0.04 implied variance. Daily moves of 0.5% give realized variance near 0.006,
    # so the gap is positive on nearly every bar. Exposure earns the index, so it must be long.
    rng = np.random.default_rng(1)
    n = 200
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2019-01-02", periods=n)]
    spx = np.round(2500.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, n))), 2)
    vix = np.full(n, 20.0)
    frame = build(monkeypatch, tmp_path / "a", dates, spx, vix)
    tape = build_tape(frame, compute_features(frame, SHORT_FEATURES))
    env = yaml.safe_load(PRIOR_CONFIG.read_text())["env"]
    rows = np.arange(len(tape) - 1)
    amap = ActionMap(prior_feature=env["prior_feature"], prior_gain=env["prior_gain"], residual=0.0)
    target = TapeSet([tape], window=4, action_map=amap).targets(
        np.zeros(len(rows), dtype=np.int64), rows, np.zeros(len(rows)))
    payoff = target * tape.long_ret[rows]
    assert target.mean() > 0
    assert payoff.sum() > 0


def test_prior_feature_copies_the_prior_without_gaps(monkeypatch, tmp_path):
    dates, spx, vix = sample()
    frame = build(monkeypatch, tmp_path / "a", dates, spx, vix)
    feats = compute_features(frame)
    assert np.isfinite(feats["vrp_prior"].to_numpy()).all()
    np.testing.assert_allclose(feats["vrp_prior"].to_numpy(), frame["vrp_prior"].to_numpy(),
                               rtol=0, atol=1e-6)
