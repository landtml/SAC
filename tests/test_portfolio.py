import numpy as np
import pytest

from sactrader.portfolio import log_reward, step_portfolio


def _step(w_prev, target, lr=0.0, sr=None, sg=None, lw=None, sw=None,
          kl=0.0, ks=0.0, cl=0.0, cs=0.0, impact=0.0, maint=0.0):
    sr = -lr if sr is None else sr
    sg = 1.0 + lr if sg is None else sg
    lw = lr if lw is None else lw
    sw = sr if sw is None else sw
    a = lambda x: np.atleast_1d(np.asarray(x, dtype=np.float64))  # noqa: E731
    return step_portfolio(a(w_prev), a(target), a(lr), a(sr), a(sg), a(lw), a(sw),
                          a(kl), a(ks), a(cl), a(cs), impact, maint)


def test_flat_stays_flat():
    r = _step(0.0, 0.0, lr=0.05, cl=0.001, cs=0.001)
    assert r.ret[0] == 0.0 and r.w_next[0] == 0.0 and r.turnover[0] == 0.0


def test_long_pays_entry_cost_and_drifts():
    r = _step(0.0, 1.0, lr=0.10, cl=0.001)
    assert r.ret[0] == pytest.approx(0.10 - 0.001)
    assert r.cost[0] == pytest.approx(0.001)
    assert r.w_next[0] == pytest.approx(1.10 / (1.0 + 0.10 - 0.001))


def test_linear_short_loses_on_rally_and_exposure_grows():
    r = _step(-1.0, -1.0, lr=0.10)
    assert r.ret[0] == pytest.approx(-0.10)
    assert r.w_next[0] == pytest.approx(-1.10 / 0.90)


def test_flip_long_to_short_trades_both_sides():
    r = _step(1.0, -1.0, lr=0.0, cl=0.001, cs=0.002)
    assert r.turnover[0] == pytest.approx(2.0)
    assert r.cost[0] == pytest.approx(0.001 + 0.002)


def test_binary_no_token_holds_constant_exposure():
    # NO token at 0.5 moves to 0.4 when YES goes 0.5 -> 0.6.
    sr = 0.4 / 0.5 - 1.0
    r = _step(-1.0, -1.0, lr=0.6 / 0.5 - 1.0, sr=sr, sg=1.0 + sr)
    assert r.ret[0] == pytest.approx(-0.2)
    assert r.w_next[0] == pytest.approx(-1.0)


def test_carry_charged_on_notional():
    r = _step(2.0, 2.0, lr=0.0, kl=0.0001)
    assert r.ret[0] == pytest.approx(-0.0002)
    r = _step(-2.0, -2.0, lr=0.0, ks=-0.0001)  # short receives funding
    assert r.ret[0] == pytest.approx(0.0002)


def test_impact_is_superlinear():
    small = _step(0.0, 0.5, impact=0.01).cost[0]
    big = _step(0.0, 1.0, impact=0.01).cost[0]
    assert big / small == pytest.approx(2.0**1.5)


def test_leveraged_long_liquidates_on_intrabar_drop():
    r = _step(3.0, 3.0, lr=-0.05, lw=-0.34, maint=0.005)
    assert r.liquidated[0]
    assert r.w_next[0] == 0.0
    assert r.ret[0] == pytest.approx(max(3 * -0.34 - 0.005 * 3, -1.0))


def test_unlevered_position_does_not_liquidate_on_large_drop():
    r = _step(1.0, 1.0, lr=-0.5, maint=0.005)
    assert not r.liquidated[0]
    assert r.ret[0] == pytest.approx(-0.5)


def test_batch_shapes_and_log_reward():
    w = np.array([0.0, 1.0, -1.0])
    z = np.zeros(3)
    r = step_portfolio(w, w, z + 0.01, z - 0.01, z + 1.01, z, z, z, z, z, z, 0.0, 0.0)
    assert r.ret.shape == (3,)
    np.testing.assert_allclose(log_reward(r.ret), np.log1p(r.ret))
    assert np.isfinite(log_reward(np.array([-1.0])))[0]
