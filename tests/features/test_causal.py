import numpy as np
import pandas as pd
import pytest

from sactrader.data.schema import make_frame
from sactrader.features.causal import FeatureConfig, compute_features, warmup_bars
from sactrader.tape import build_tape

KINDS = ("spot", "perp", "fx", "binary")
N = 600  # past the 256-bar horizon and the 512-bar long-vol span
PRICE_COLS = (
    *(f"ret_h{h}" for h in (1, 4, 16, 64, 256)),
    *(f"ema_gap_{s}" for s in (16, 64, 256)),
    "vol_ratio",
    "range_vol_ratio",
    "body",
)
TIME_COLS = ("hour_sin", "hour_cos", "dow_sin", "dow_cos")
BINARY_COLS = ("logit", "dlogit_1", "dlogit_4", "dlogit_24", "log_hours_to_end")
BINARY_END = "2024-02-01 00:00:00+00:00"


def _bars(kind, n, seed):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, n)
    if kind == "binary":
        close = 1.0 / (1.0 + np.exp(-np.cumsum(steps)))
    else:
        close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, n))
    if kind == "binary":
        high = np.minimum(high, 0.999)
        low = np.maximum(low, 0.001)
    volume = rng.gamma(2.0, 100.0, n)
    taker = volume * rng.uniform(0.0, 1.0, n)
    funding = np.where(np.arange(n) % 8 == 0, rng.normal(0.0, 2e-4, n), 0.0)
    signal = rng.normal(0.0, 3.0, n)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume,
            "taker": taker, "funding": funding, "signal": signal}


def _build(kind, b, *, bar="1h", volume=True, extras=True):
    idx = pd.date_range("2024-01-01", periods=len(b["close"]), freq=bar, tz="UTC")
    kw = {}
    if kind == "perp":
        kw["funding_rate"] = b["funding"]
    if kind == "binary":
        kw["resolution"] = 1.0
    frame = make_frame(idx, b["open"], b["high"], b["low"], b["close"],
                       b["volume"] if volume else None,
                       symbol="TST", kind=kind, bar=bar, **kw)
    if extras:
        frame["taker_buy_base"] = b["taker"]
        frame["signal"] = b["signal"]
    if kind == "binary":
        frame.attrs["end_time"] = BINARY_END
    return frame


def _expected_columns(kind):
    cols = [*PRICE_COLS, "volume_z", "taker_imbalance"]
    if kind == "perp":
        cols += ["funding_last", "funding_ewm"]
    cols += list(TIME_COLS)  # every frame here has 1h bars
    if kind == "binary":
        cols += list(BINARY_COLS)
    return cols + ["signal"]


def _ewm_ref(x, span):
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = (1.0 - alpha) * out[i - 1] + alpha * x[i]
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_columns_follow_kind_in_fixed_order(kind):
    out = compute_features(_build(kind, _bars(kind, N, 1)))
    assert list(out.columns) == _expected_columns(kind)
    again = compute_features(_build(kind, _bars(kind, N, 1)))
    pd.testing.assert_frame_equal(out, again)


@pytest.mark.parametrize("kind", KINDS)
def test_nan_only_in_warmup_and_output_is_float32(kind):
    frame = _build(kind, _bars(kind, N, 2))
    out = compute_features(frame)
    w = warmup_bars()
    assert out.index.equals(frame.index)
    assert (out.dtypes == np.float32).all()
    assert out.iloc[w:].notna().all().all()
    assert np.isfinite(out.iloc[w:].to_numpy()).all()
    # The bound is tight: the 256-bar horizon is still undefined on the last warm-up row.
    assert np.isnan(out["ret_h256"].iloc[w - 1])


@pytest.mark.parametrize("kind", KINDS)
def test_values_clipped_to_bound(kind):
    frame = _build(kind, _bars(kind, N, 3))
    out = compute_features(frame).to_numpy()
    assert np.nanmax(np.abs(out)) <= 5.0
    tight = compute_features(frame, FeatureConfig(clip=0.5)).to_numpy()
    assert np.nanmax(np.abs(tight)) == 0.5


@pytest.mark.parametrize("kind", KINDS)
def test_prefix_matches_full_frame(kind):
    frame = _build(kind, _bars(kind, N, 4))
    k = 300
    head = compute_features(frame.iloc[:k])
    full = compute_features(frame)
    np.testing.assert_allclose(head.to_numpy(), full.iloc[:k].to_numpy(),
                               atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("kind", KINDS)
def test_future_rows_do_not_change_past(kind):
    base = _bars(kind, N, 5)
    alt = _bars(kind, N, 6)
    k = 400
    changed = {key: np.concatenate([base[key][: k + 1], alt[key][k + 1:]]) for key in base}
    full = compute_features(_build(kind, base))
    perturbed = compute_features(_build(kind, changed))
    np.testing.assert_allclose(perturbed.iloc[: k + 1].to_numpy(),
                               full.iloc[: k + 1].to_numpy(), atol=1e-6, equal_nan=True)
    # Guards against a vacuous pass: the rows after k must actually differ.
    assert not np.allclose(perturbed.iloc[k + 1:].to_numpy(),
                           full.iloc[k + 1:].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("kind", ("spot", "perp", "fx"))
@pytest.mark.parametrize("factor", (1000.0, 1e-3))
def test_price_scale_leaves_price_features_unchanged(kind, factor):
    base = _bars(kind, N, 7)
    scaled = {key: (val * factor if key in ("open", "high", "low", "close") else val)
              for key, val in base.items()}
    a = compute_features(_build(kind, base))[list(PRICE_COLS)]
    b = compute_features(_build(kind, scaled))[list(PRICE_COLS)]
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), atol=1e-5, equal_nan=True)


def test_price_features_match_scalar_reference():
    b = _bars("spot", N, 8)
    out = compute_features(_build("spot", b))
    logc = np.log(b["close"])
    ret = np.diff(logc)  # ret[i] is the return into row i + 1
    sq = ret**2
    sigma = np.sqrt(_ewm_ref(sq, 32))  # sigma[i] belongs to row i + 1
    sigma_long = np.sqrt(_ewm_ref(sq, 512))
    expected = {
        "ret_h1": ret / sigma,
        "ema_gap_16": (logc[1:] - _ewm_ref(logc, 16)[1:]) / (sigma * 4.0),
        "vol_ratio": np.log(sigma / sigma_long),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(out[name].to_numpy()[1:], np.clip(values, -5.0, 5.0),
                                   atol=1e-5)
    body = (b["close"] - b["open"]) / (b["high"] - b["low"])
    np.testing.assert_allclose(out["body"].to_numpy(), np.clip(body, -5.0, 5.0), atol=1e-5)


def test_binary_logit_and_lags_match_formula():
    b = _bars("binary", N, 9)
    out = compute_features(_build("binary", b))
    raw = np.log(b["close"] / (1.0 - b["close"])) / 4.0
    np.testing.assert_allclose(out["logit"].to_numpy(), raw, atol=1e-5)
    for k in (1, 4, 24):
        expected = np.full(N, np.nan)
        expected[k:] = raw[k:] - raw[:-k]
        np.testing.assert_allclose(out[f"dlogit_{k}"].to_numpy(), expected,
                                   atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("end", [pd.Timestamp("2024-01-01 10:00", tz="UTC"),
                                 "2024-01-01 10:00:00+00:00"])
def test_hours_to_end_uses_bar_close_and_clamps(end):
    frame = _build("binary", _bars("binary", 20, 10))
    frame.attrs["end_time"] = end
    out = compute_features(frame)["log_hours_to_end"].to_numpy()
    hours = np.maximum(9.0 - np.arange(20), 0.0)  # bar t closes at t + 1 hours
    np.testing.assert_allclose(out, np.log1p(hours) / 5.0, atol=1e-6)


def test_intraday_time_codes_come_from_open_time():
    out = compute_features(_build("spot", _bars("spot", N, 11)))
    assert out["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert out["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert out["hour_sin"].iloc[6] == pytest.approx(1.0, abs=1e-6)  # 06:00 is a quarter turn
    assert out["dow_sin"].iloc[0] == pytest.approx(0.0, abs=1e-6)  # 2024-01-01 is a Monday
    assert out["dow_cos"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_optional_columns_follow_frame_contents():
    b = _bars("spot", N, 12)
    daily = compute_features(_build("spot", b, bar="1D"))
    assert not set(TIME_COLS) & set(daily.columns)
    bare = compute_features(_build("spot", b, volume=False, extras=False))
    assert not {"volume_z", "taker_imbalance", "signal"} & set(bare.columns)
    frame = _build("binary", _bars("binary", N, 13))
    del frame.attrs["end_time"]
    assert "log_hours_to_end" not in compute_features(frame).columns


def test_funding_carries_last_nonzero_rate_in_bps():
    rate = np.array([0.0, 0.0, 1e-4, 0.0, 0.0, -2e-4, 0.0, 0.0])
    idx = pd.date_range("2024-01-01", periods=8, freq="8h", tz="UTC")
    c = np.full(8, 100.0)
    frame = make_frame(idx, c, c * 1.001, c * 0.999, c, np.ones(8), symbol="TST",
                       kind="perp", bar="8h", funding_rate=rate)
    out = compute_features(frame)
    carried = np.array([0.0, 0.0, 1.0, 1.0, 1.0, -2.0, -2.0, -2.0])
    np.testing.assert_allclose(out["funding_last"].to_numpy(), carried, atol=1e-6)
    np.testing.assert_allclose(out["funding_ewm"].to_numpy(), _ewm_ref(carried, 64), atol=1e-6)


def test_missing_and_flat_bars_stay_finite_and_neutral():
    b = _bars("spot", N, 14)
    b["volume"][50] = np.nan
    b["volume"][60] = 0.0
    b["taker"][60] = 0.0
    b["open"][70] = b["close"][70]
    b["high"][70] = b["close"][70]
    b["low"][70] = b["close"][70]
    out = compute_features(_build("spot", b))
    assert out["volume_z"].iloc[50] == 0.0
    assert out["body"].iloc[70] == 0.0
    assert np.isfinite(out.iloc[warmup_bars():].to_numpy()).all()


def test_signal_copied_and_clipped():
    frame = _build("spot", _bars("spot", N, 15))
    assert np.abs(frame["signal"]).max() > 5.0  # the clip has work to do
    out = compute_features(frame)
    expected = np.clip(frame["signal"].to_numpy(), -5.0, 5.0).astype(np.float32)
    np.testing.assert_array_equal(out["signal"].to_numpy(), expected)


@pytest.mark.parametrize("kind", KINDS)
def test_tape_builds_for_every_kind(kind):
    frame = _build(kind, _bars(kind, N, 16))
    feats = compute_features(frame)
    tape = build_tape(frame, feats)
    assert len(tape) == N - warmup_bars()
    assert tape.feature_names == tuple(feats.columns)
    assert tape.features.shape == (len(tape), feats.shape[1])
    assert np.isfinite(tape.features).all()


def test_warmup_follows_config():
    assert warmup_bars() == 256
    cfg = FeatureConfig(horizons=(1, 8), ema_spans=(4,), clip=0.5)
    assert warmup_bars(cfg) == 24  # the binary lag of 24 bars is the longest here
    for kind in KINDS:
        out = compute_features(_build(kind, _bars(kind, N, 17)), cfg)
        assert "ret_h8" in out.columns and "ema_gap_4" in out.columns
        assert np.isfinite(out.iloc[warmup_bars(cfg):].to_numpy()).all()
        assert np.nanmax(np.abs(out.to_numpy())) <= 0.5
