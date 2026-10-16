import numpy as np
import pandas as pd
import pytest

import softedge.eval.active as active_mod
from softedge.cli import main
from softedge.data.synthetic import gbm
from softedge.env.vec_env import ActionMap, TapeSet, rollout
from softedge.eval.active import active_table
from softedge.eval.tuned_rule import (
    THETAS,
    choose_settings,
    deadzone_targets,
    pick_best,
    raw_prior,
    rollout_targets,
    tuned_rule_returns,
    zero_residual_targets,
)
from softedge.experiment import NO_TRADE_BANDS
from softedge.features.causal import FeatureConfig, compute_features
from softedge.pipeline import Period, SplitSpec
from softedge.tape import build_tape

CFG = FeatureConfig(horizons=(1, 4, 16), ema_spans=(16,), vol_long_span=64)
WINDOW = 4
BARS = 40


def _tape(seed, symbol="S", n=300, drift=0.0):
    frame = gbm(n, sigma=0.01, seed=seed, symbol=symbol)
    if drift:
        frame = frame.copy()
        for col in ("open", "high", "low", "close"):
            frame[col] = frame[col] * np.exp(drift * np.arange(len(frame)))
    return build_tape(frame, compute_features(frame, CFG))


def _tapes():
    return [_tape(1, "A"), _tape(2, "B")]


def _amap(**kw):
    return ActionMap(**{"prior_feature": "ret_h4", "prior_gain": 4.0, "residual": 0.5, **kw})


def _sr(x) -> float:
    return float(np.mean(x) / np.std(x, ddof=1))


# Theta 0 and band 0 are the zero-residual policy: the same rows, the same loop, the same bits.
@pytest.mark.parametrize("amap", [_amap(), _amap(vol_target=0.2), _amap(residual=0.0)])
def test_theta_zero_band_zero_reproduces_zero_residual_rollout(amap):
    ts = TapeSet(_tapes(), WINDOW, None, amap)
    want = rollout(ts, lambda obs: np.zeros(len(obs)))
    got = rollout_targets(ts, deadzone_targets(ts, 0.0), 0.0)
    for w, g in zip(want, got, strict=True):
        assert set(g) == set(w)
        for key in w:
            np.testing.assert_array_equal(g[key], w[key])


@pytest.mark.parametrize("band", NO_TRADE_BANDS)
def test_theta_zero_matches_rollout_with_band(band):
    ts = TapeSet(_tapes(), WINDOW, None, _amap())
    want = rollout(ts, lambda obs: np.zeros(len(obs)), band=band)
    got = rollout_targets(ts, deadzone_targets(ts, 0.0), band)
    for w, g in zip(want, got, strict=True):
        np.testing.assert_array_equal(g["ret"], w["ret"])
        np.testing.assert_array_equal(g["target"], w["target"])


def test_deadzone_zeroes_targets_below_theta():
    ts = TapeSet(_tapes(), WINDOW, None, _amap(prior_gain=50.0))
    raw = raw_prior(ts)
    np.testing.assert_allclose(np.clip(raw, -1.0, 1.0), ts.prior)
    zero = zero_residual_targets(ts)
    theta = float(np.median(np.abs(raw)))
    rows = deadzone_targets(ts, theta)

    inside = np.abs(raw) < theta
    assert inside.any() and (~inside).any()
    assert np.all(rows[inside] == 0.0)
    np.testing.assert_array_equal(rows[~inside], zero[~inside])
    assert np.any(zero[inside] != 0.0)  # the deadzone changes something


def test_theta_is_applied_to_raw_prior_not_clipped_prior():
    # Gain 50 pushes |raw| past 1, so the clip changes the prior. A deadzone of 2 must read the
    # raw value: rows with |raw| >= 2 keep their target although the clipped prior is at most 1.
    ts = TapeSet(_tapes(), WINDOW, None, _amap(prior_gain=50.0))
    raw = raw_prior(ts)
    assert np.any(np.abs(raw) > 1.0)
    rows = deadzone_targets(ts, 2.0)
    keep = np.abs(raw) >= 2.0
    np.testing.assert_array_equal(rows[keep], zero_residual_targets(ts)[keep])
    assert np.all(rows[~keep] == 0.0)


def test_pick_best_breaks_ties_to_smaller_theta_then_smaller_band():
    scores = {(0.5, 0.1): 1.0, (0.25, 0.2): 1.0, (0.25, 0.05): 1.0, (2.0, 0.0): 0.9}
    assert pick_best(scores) == (0.25, 0.05)
    assert pick_best({(0.0, 0.2): 1.0, (2.0, 0.0): 1.1}) == (2.0, 0.0)
    assert pick_best({(th, b): 0.0 for th in THETAS for b in NO_TRADE_BANDS}) == (0.0, 0.0)


def test_constant_zero_prior_scores_every_pair_equally_and_picks_the_first():
    # Gain zero leaves the prior at zero everywhere, so every pair trades nothing: all scores tie.
    theta, band, scores = choose_settings(_tapes(), WINDOW, _amap(prior_gain=0.0))
    assert len(set(scores.values())) == 1
    assert (theta, band) == (0.0, 0.0)


def _split():
    return SplitSpec(Period(None, None), Period(None, None), Period("2024-06-01", None))


def test_selection_never_reads_test_tapes(monkeypatch):
    import softedge.eval.tuned_rule as tuned_mod

    val = [_tape(3, "V1"), _tape(4, "V2")]
    train = [_tape(5, "T")]
    test_a = [_tape(6, "X1"), _tape(7, "X2")]
    test_b = [_tape(8, "Y1", drift=0.002), _tape(9, "Y2", drift=-0.002)]
    cfg = {"env": {"window": WINDOW, "prior_feature": "ret_h4", "prior_gain": 4.0,
                   "residual": 0.5}}

    def run_with(test):
        parts = {"train": train, "val": val, "test": test}
        monkeypatch.setattr(tuned_mod, "build_all_tapes", lambda cfg: [])
        monkeypatch.setattr(tuned_mod, "make_splits", lambda cfg: [_split()])
        monkeypatch.setattr(tuned_mod, "split_tapes", lambda tapes, split: parts)
        return tuned_rule_returns(None, cfg)

    series_a, settings_a = run_with(test_a)
    series_b, settings_b = run_with(test_b)

    assert settings_a == settings_b
    assert len(settings_a) == 1
    assert not series_a.equals(series_b)  # the test tapes do change the returns, not the choice


def test_choice_on_validation_only_matches_split_settings():
    val = [_tape(3, "V1"), _tape(4, "V2")]
    theta, band, scores = choose_settings(val, WINDOW, _amap())
    assert (theta, band) == pick_best(scores)
    assert set(scores) == {(th, b) for th in THETAS for b in NO_TRADE_BANDS}


# Tuned rule in the active table, on a toy run directory.

BAR_DAYS = pd.date_range("2024-01-01", periods=BARS, freq="D", tz="UTC", name="time")


def _write_carry_run(root, name, *, prior_feature=True, columns):
    run = root / name
    run.mkdir(parents=True)
    text = "universe: toy\n"
    if prior_feature:
        text += "env:\n  prior_feature: ret_h4\n"
    (run / "config.yaml").write_text(text)
    pd.DataFrame(columns, index=BAR_DAYS).to_csv(run / "pooled_returns.csv")
    return run


def _alt(n=BARS):
    return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def test_tuned_mode_subtracts_tuned_rule_not_prior_only(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    prior_only = 0.0005 * np.sin(np.arange(BARS))
    tuned = 0.0002 * np.cos(np.arange(BARS))
    active = 0.002 + 0.01 * _alt()
    _write_carry_run(root, "carry", columns={
        "sac_banded": tuned + active, "prior_only": prior_only})
    calls = []

    def fake_tuned(run, cfg):
        calls.append(run.name)
        return pd.Series(tuned, index=BAR_DAYS), []

    monkeypatch.setattr(active_mod, "tuned_rule_returns", fake_tuned)
    row = active_table(root, baseline="tuned").set_index("run").loc["carry"]

    assert calls == ["carry"]
    assert row["sharpe"] == pytest.approx(_sr(active) * np.sqrt(365.25), rel=1e-12)
    assert row["baseline_sharpe"] == pytest.approx(_sr(tuned) * np.sqrt(365.25), rel=1e-12)


def test_rule_mode_keeps_prior_only_and_reports_its_sharpe(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    prior_only = 0.0005 * np.sin(np.arange(BARS))
    _write_carry_run(root, "carry", columns={
        "sac_banded": prior_only + 0.002 + 0.01 * _alt(), "prior_only": prior_only})
    row = active_table(root).set_index("run").loc["carry"]

    assert row["baseline_sharpe"] == pytest.approx(_sr(prior_only) * np.sqrt(365.25), rel=1e-12)
    assert list(active_table(root).columns) == active_mod.COLUMNS


def test_tuned_mode_skips_runs_without_prior_feature(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    prior_only = 0.0005 * np.sin(np.arange(BARS))
    _write_carry_run(root, "no_feature", prior_feature=False, columns={
        "sac_banded": prior_only + 0.01 * _alt(), "prior_only": prior_only})
    monkeypatch.setattr(active_mod, "tuned_rule_returns",
                        lambda run, cfg: pytest.fail("tuned rule built for a run without one"))

    assert active_table(root, baseline="tuned").empty


def test_unknown_baseline_is_refused(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    with pytest.raises(ValueError):
        active_table(root, baseline="oracle")


def test_cli_baseline_tuned_prints_without_out(tmp_path, monkeypatch, capsys):
    root = tmp_path / "runs"
    root.mkdir()
    prior_only = 0.0005 * np.sin(np.arange(BARS))
    _write_carry_run(root, "carry", columns={
        "sac_banded": prior_only + 0.01 * _alt(), "prior_only": prior_only})
    monkeypatch.setattr(active_mod, "tuned_rule_returns",
                        lambda run, cfg: (pd.Series(prior_only, index=BAR_DAYS), []))
    before = sorted(p for p in tmp_path.rglob("*"))

    assert main(["active", "--runs", str(root), "--baseline", "tuned"]) == 0

    out = capsys.readouterr().out
    assert "baseline_sharpe" in out and "carry" in out
    assert "validation-tuned rule" in out
    assert sorted(p for p in tmp_path.rglob("*")) == before  # nothing written without --out
