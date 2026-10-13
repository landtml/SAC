import numpy as np
import pytest

from sactrader.eval.backtest import backtest, backtest_exposures
from sactrader.eval.metrics import summarize
from sactrader.portfolio import step_portfolio
from sactrader.tape import CostSpec

from .helpers import make_tape

TARGETS = [1.5, -2.5, 0.3, 0.8, -0.5, 0.0, 1.0]


def _policy(t, w):
    # Trade through a sequence of targets, then hold whatever exposure has drifted to.
    return TARGETS[t] if t < len(TARGETS) else w


def _manual_loop(tape, policy):
    """Scalar step_portfolio calls, one row at a time, independent of backtest()."""
    n = len(tape) if tape.terminal_at_end else len(tape) - 1
    lim = tape.max_leverage
    w = 0.0
    rows = []
    for t in range(n):
        target = min(max(policy(t, w), -lim), lim)
        r = step_portfolio(
            w, target,
            tape.long_ret[t], tape.short_ret[t], tape.short_growth[t],
            tape.long_worst[t], tape.short_worst[t],
            tape.carry_long[t], tape.carry_short[t],
            tape.cost_long[t], tape.cost_short[t],
            tape.impact, tape.maint_margin,
        )
        rows.append((w, target, float(r.ret), float(r.cost), float(r.carry),
                     float(r.turnover), bool(r.liquidated)))
        w = float(r.w_next)
    return np.array(rows, dtype=object)


def _spot_tape():
    costs = CostSpec(fee_bps=5, half_spread_bps=1, borrow_apr=0.2, impact=0.01, max_leverage=2.0)
    return make_tape([100, 105, 98, 110, 97, 101, 104, 99], costs=costs)


def _perp_tape():
    costs = CostSpec(fee_bps=4.5, half_spread_bps=0.5, impact=0.01, max_leverage=2.0)
    funding = np.array([0.0, 0.0001, -0.0002, 0.0003, 0.0, 0.0001, -0.0001, 0.0002])
    return make_tape([100, 105, 98, 110, 97, 101, 104, 99], kind="perp", costs=costs,
                     funding=funding)


def _binary_tape():
    costs = CostSpec(impact=0.01, max_leverage=2.0)
    return make_tape([0.40, 0.45, 0.38, 0.52, 0.47, 0.60, 0.55, 0.66], kind="binary",
                     costs=costs, resolution=1.0)


@pytest.mark.parametrize("make", [_spot_tape, _perp_tape, _binary_tape])
def test_backtest_matches_manual_loop(make):
    tape = make()
    res = backtest(tape, _policy)
    expected = _manual_loop(tape, _policy)
    n = len(expected)
    assert len(res.returns) == n

    np.testing.assert_allclose(res.w_prev, expected[:, 0].astype(float), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(res.exposure, expected[:, 1].astype(float), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(res.returns, expected[:, 2].astype(float), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(res.cost, expected[:, 3].astype(float), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(res.carry, expected[:, 4].astype(float), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(res.turnover, expected[:, 5].astype(float), rtol=1e-12, atol=1e-15)
    assert res.liquidated.tolist() == expected[:, 6].tolist()
    np.testing.assert_allclose(res.equity, np.cumprod(1.0 + res.returns), rtol=1e-14)


def test_step_count_excludes_last_row_unless_settlement():
    assert len(backtest(_spot_tape(), _policy).returns) == 7
    assert len(backtest(_binary_tape(), _policy).returns) == 8


def test_binary_settlement_is_the_final_step():
    # Hold a YES token through a close of 0.80. The settlement pays the full 1.0.
    costs = CostSpec(max_leverage=2.0)
    tape = make_tape([0.40, 0.45, 0.52, 0.80], kind="binary", costs=costs, resolution=1.0)
    assert tape.terminal_at_end
    res = backtest(tape, lambda t, w: 1.0 if t == 0 else w)
    assert len(res.returns) == 4
    # Holding a unit with no trade and no carry earns w_prev times the step return.
    np.testing.assert_allclose(res.returns[1:], res.w_prev[1:] * tape.long_ret[1:4], rtol=1e-12)
    assert res.returns[-1] == pytest.approx(0.25 * res.w_prev[-1])


def test_targets_are_clipped_to_tape_leverage():
    tape = _spot_tape()  # max_leverage 2.0
    res = backtest(tape, lambda t, w: 5.0 if t % 2 == 0 else -5.0)
    assert np.all(np.abs(res.exposure) <= 2.0)
    assert res.exposure[0] == 2.0 and res.exposure[1] == -2.0


def test_liquidation_wipes_out_equity_and_is_counted():
    # The low of 40 on row 1 drops a 2x long by 60%, past the 0.5% maintenance margin.
    costs = CostSpec(max_leverage=2.0)
    tape = make_tape([100.0, 100.0, 100.0, 100.0], costs=costs,
                     low=[99.9, 40.0, 99.9, 99.9], high=[100.1, 100.1, 100.1, 100.1])
    res = backtest(tape, lambda t, w: 2.0)
    assert res.liquidated[0]
    assert res.returns[0] == pytest.approx(-1.0)
    assert res.equity[-1] == 0.0
    assert summarize(res, tape.bars_per_year)["n_liquidations"] == 1.0


def test_backtest_exposures_equals_policy_on_the_same_path():
    tape = _spot_tape()
    path = np.linspace(-1.5, 1.5, len(tape))
    a = backtest_exposures(tape, path)
    b = backtest(tape, lambda t, _w: path[t])
    np.testing.assert_array_equal(a.returns, b.returns)
    np.testing.assert_array_equal(a.exposure, b.exposure)
    with pytest.raises(ValueError):
        backtest_exposures(tape, path[:-1])


def test_flat_path_has_no_returns_costs_or_exposure():
    tape = _spot_tape()
    res = backtest(tape, lambda t, w: 0.0)
    assert np.all(res.returns == 0.0)
    assert np.all(res.cost == 0.0) and np.all(res.turnover == 0.0)
    s = summarize(res, tape.bars_per_year)
    assert s["avg_exposure"] == 0.0
    assert s["cost_per_year"] == 0.0
    assert s["n_liquidations"] == 0.0


def test_summary_includes_backtest_fields():
    tape = _spot_tape()
    res = backtest(tape, _policy)
    s = summarize(res, tape.bars_per_year)
    assert s["avg_exposure"] == pytest.approx(np.mean(res.exposure))
    assert s["avg_abs_exposure"] == pytest.approx(np.mean(np.abs(res.exposure)))
    assert s["turnover_per_year"] == pytest.approx(np.mean(res.turnover) * tape.bars_per_year)
    assert s["cost_per_year"] == pytest.approx(np.mean(res.cost) * tape.bars_per_year)
    assert s["carry_per_year"] == pytest.approx(np.mean(res.carry) * tape.bars_per_year)
    assert s["n_bars"] == len(res.returns)
