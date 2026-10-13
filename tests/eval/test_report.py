import numpy as np
import pandas as pd
import pytest

from sactrader.data import synthetic
from sactrader.env.vec_env import TapeSet, rollout
from sactrader.eval.report import (
    per_tape_table,
    pooled_returns,
    rollout_exposures,
    summary_table,
    to_markdown,
)
from sactrader.features.causal import FeatureConfig, compute_features
from sactrader.tape import build_tape

CFG = FeatureConfig(horizons=(1, 4), ema_spans=(8,), vol_long_span=32)
WINDOW = 4
START = WINDOW - 1


def _tape(frame):
    return build_tape(frame, compute_features(frame, CFG))


def _spot(seed, n=200, symbol="AAA", bar="1h"):
    return _tape(synthetic.gbm(n, mu=0.0005, sigma=0.01, seed=seed, symbol=symbol, bar=bar))


def _runs(tapes):
    runs = {}
    for name, level in (("buy_hold", 1.0), ("half", 0.5), ("flat", 0.0)):
        runs[name] = [
            rollout_exposures(t, np.full(len(t), level * t.max_leverage), START) for t in tapes
        ]
    return runs


def test_rollout_exposures_matches_vec_env_rollout_at_full_leverage():
    tapes = [
        _spot(1),
        _spot(2, n=150, symbol="BBB"),
        _tape(synthetic.binary_market(200, seed=3, symbol="YES")),
    ]
    expected = rollout(TapeSet(tapes, window=WINDOW), lambda obs: np.ones(len(obs)))
    for tape, want in zip(tapes, expected, strict=True):
        got = rollout_exposures(tape, np.full(len(tape), tape.max_leverage), START)
        assert got.keys() == want.keys()
        for key in want:
            np.testing.assert_allclose(got[key], want[key], rtol=0, atol=1e-12, err_msg=key)


def test_pooled_return_is_the_row_mean_for_identical_timestamps():
    a, b = _spot(1), _spot(2)
    assert np.array_equal(a.times, b.times)
    run = [
        rollout_exposures(a, np.ones(len(a)), START),
        rollout_exposures(b, np.full(len(b), -0.5), START),
    ]
    pooled = pooled_returns(run, [a, b], START)
    expected = (run[0]["ret"] + run[1]["ret"]) / 2
    np.testing.assert_allclose(pooled.to_numpy(), expected, rtol=0, atol=1e-15)
    stamps = pd.to_datetime(a.times[START:START + expected.size], unit="ns", utc=True)
    assert pooled.index.equals(stamps)
    assert pooled.index.is_monotonic_increasing


def test_pooled_return_averages_only_tapes_with_a_step_at_that_time():
    long_t, short_t = _spot(1, n=200), _spot(2, n=150)
    assert np.array_equal(short_t.times, long_t.times[:len(short_t)])
    run = [
        rollout_exposures(long_t, np.ones(len(long_t)), START),
        rollout_exposures(short_t, np.full(len(short_t), -1.0), START),
    ]
    pooled = pooled_returns(run, [long_t, short_t], START).to_numpy()
    k = len(run[1]["ret"])
    np.testing.assert_allclose(pooled[:k], (run[0]["ret"][:k] + run[1]["ret"]) / 2, atol=1e-15)
    np.testing.assert_allclose(pooled[k:], run[0]["ret"][k:], atol=1e-15)


def test_pooled_return_rejects_mixed_bar_sizes():
    hourly = _spot(1)
    daily = _spot(2, bar="1D")
    run = [
        rollout_exposures(hourly, np.zeros(len(hourly)), START),
        rollout_exposures(daily, np.zeros(len(daily)), START),
    ]
    with pytest.raises(ValueError, match="bar"):
        pooled_returns(run, [hourly, daily], START)


def test_per_tape_table_has_one_row_per_strategy_and_tape():
    tapes = [_spot(1, symbol="AAA"), _spot(2, n=150, symbol="BBB")]
    df = per_tape_table(_runs(tapes), tapes, START)
    assert list(df.columns) == [
        "strategy", "symbol", "kind", "n_bars", "total_return", "cagr", "sharpe", "sortino",
        "max_drawdown", "turnover_per_year", "cost_per_year", "avg_abs_exposure",
    ]
    assert len(df) == 3 * len(tapes)
    buy_hold = df[df["strategy"] == "buy_hold"]
    assert buy_hold["symbol"].tolist() == ["AAA", "BBB"]
    assert buy_hold["n_bars"].tolist() == [len(t) - START - 1 for t in tapes]
    assert (buy_hold["avg_abs_exposure"] == 1.0).all()
    assert (df.loc[df["strategy"] == "flat", "turnover_per_year"] == 0.0).all()


def test_summary_table_fills_diff_columns_for_every_strategy_but_the_reference():
    tapes = [_spot(1), _spot(2, symbol="BBB")]
    df = summary_table(_runs(tapes), tapes, START, reference="buy_hold", n_boot=200)
    assert df["strategy"].tolist() == ["buy_hold", "half", "flat"]
    diff_cols = ["sharpe_diff", "diff_lo", "diff_hi", "p_value"]
    assert df.loc[0, diff_cols].isna().all()
    assert df.loc[1:, diff_cols].notna().all().all()
    assert df["p_value"].dropna().between(0.0, 1.0).all()
    assert (df.loc[1:, "diff_lo"] <= df.loc[1:, "diff_hi"]).all()


def test_to_markdown_renders_header_rule_and_one_line_per_row():
    tapes = [_spot(1), _spot(2, symbol="BBB")]
    df = summary_table(_runs(tapes), tapes, START, reference="buy_hold", n_boot=200)
    lines = to_markdown(df).splitlines()
    assert len(lines) == len(df) + 2
    assert lines[0].startswith("| strategy |")
    assert lines[1] == "| " + " | ".join(["---"] * len(df.columns)) + " |"
    assert "n/a" in lines[2]
