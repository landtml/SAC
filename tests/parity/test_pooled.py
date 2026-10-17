"""Tests of the pooled parity network and its panel backtest."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from softedge.parity import backtest as bt  # noqa: E402
from softedge.parity import panel_world as pw  # noqa: E402
from softedge.parity import pooled as pl  # noqa: E402

N_WEEKS = 260


@pytest.fixture(scope="module")
def panel():
    r, c, listed, null = pw.simulate(jax.random.key(3), 3, 6, N_WEEKS)
    return np.asarray(r), np.asarray(c), np.asarray(listed), np.asarray(null)


@pytest.fixture(scope="module")
def model():
    m = pl.PooledPolicy(rngs=nnx.Rngs(1))
    m.out.kernel[...] = m.out.kernel[...] * 300.0
    return m


class Fixed:
    def __init__(self, e):
        self.e = e

    def observe(self, logret, cash=None):
        pass

    def target(self, wealth, peak):
        return np.full(np.shape(wealth), self.e)


def test_accounting_matches_numpy_backtest_with_late_listings(panel, model):
    r, c, listed, _ = panel
    for p in range(r.shape[0]):
        out = pl.runner(model, pl.Config(fixed=0.6, governor="none"))(r[p], c[p], listed[p],
                                                                      0.0005)
        for i in np.flatnonzero(listed[p].any(axis=1)):
            s = int(np.argmax(listed[p, i]))
            ref = bt.simulate(r[p, i:i + 1, s:], Fixed(0.6), cash=c[p, s:], cost=0.0005)
            np.testing.assert_allclose(out["growth"][i, s:], ref["growth"][0], atol=1e-12)
            assert not out["growth"][i, :s].any()


def test_late_listings_and_absent_assets_occur(panel):
    _, _, listed, _ = panel
    starts = [int(np.argmax(row)) for row in listed.reshape(-1, N_WEEKS) if row.any()]
    assert max(starts) > 0
    assert listed[:, 0].all()


def mirrored(r, c, listed):
    return np.where(listed, 2.0 * c[None, :] - r, 0.0)


@pytest.mark.parametrize("cfg", [pl.Config(gate=False, governor="none"),
                                 pl.Config(gate=True, governor="none")])
def test_mirror_maps_exposure_to_one_minus_itself(panel, model, cfg):
    r, c, listed, _ = panel
    a = pl.runner(model, cfg)(r[0], c[0], listed[0], 0.0005)
    b = pl.runner(model, cfg)(mirrored(r[0], c[0], listed[0]), c[0], listed[0], 0.0005)
    clock = np.cumsum(listed[0], axis=1)
    decided = listed[0] & (clock > pl.FIRST_DECISION)
    np.testing.assert_allclose(a["target"][decided] + b["target"][decided], 1.0, atol=1e-12)
    if not cfg.gate:
        assert np.abs(a["target"][decided] - 0.5).max() > 0.05


def test_flipping_any_future_leaves_every_earlier_target_unchanged(panel, model):
    r, c, listed, _ = panel
    run = pl.runner(model, pl.Config(gate=True, governor="crash"))
    base = run(r[1], c[1], listed[1], 0.0005)
    rng = np.random.default_rng(0)
    for cut in (60, 120, 200):
        flips = np.where(rng.random(r[1].shape) < 0.5, -1.0, 1.0)
        other = r[1].copy()
        x = r[1] - c[1][None, :]
        other[:, cut + 1:] = (c[1][None, :] + flips * x)[:, cut + 1:]
        other = np.where(listed[1], other, 0.0)
        out = run(other, c[1], listed[1], 0.0005)
        assert np.array_equal(base["target"][:, :cut + 1], out["target"][:, :cut + 1])


def test_solo_matches_a_panel_of_one(panel, model):
    r, c, listed, _ = panel
    solo = pl.runner(model, pl.Config(), solo=True)(r[2], c[2], listed[2], 0.0005)
    for i in range(r.shape[1]):
        one = pl.runner(model, pl.Config())(r[2, i:i + 1], c[2], listed[2, i:i + 1], 0.0005)
        np.testing.assert_allclose(solo["growth"][i], one["growth"][0], atol=1e-14)


def test_held_exposure_never_exceeds_last_cap(panel, model):
    r, c, listed, _ = panel
    for p in range(r.shape[0]):
        out = pl.runner(model, pl.Config(fixed=1.0, premium=False, direction=False))(
            r[p], c[p], listed[p], 0.0005)
        held_next = out["held"][:, 1:]
        cap = out["cap"][:, :-1]
        both = listed[p][:, 1:] & listed[p][:, :-1]
        assert (held_next[both] <= cap[both] + 1e-12).all()


def test_crash_governor_keeps_wealth_above_floor_when_losses_fit_the_estimate():
    """A weekly loss no larger than the crash estimate cannot reach the floor."""
    rng = np.random.default_rng(5)
    r = np.log1p(np.clip(rng.standard_t(3, (8, 400)) * 0.03, -0.09, 0.5))
    c = np.full(400, 0.0005)
    listed = np.ones_like(r, dtype=bool)
    m = pl.PooledPolicy(rngs=nnx.Rngs(0))
    out = pl.runner(m, pl.Config(fixed=1.0, premium=False, direction=False))(r, c, listed, 0.0)
    wealth = np.exp(np.cumsum(out["growth"], axis=1))
    peak = np.maximum.accumulate(wealth, axis=1)
    assert (wealth >= pl.FLOOR * peak - 1e-12).all()


def test_null_panels_rarely_deploy(model):
    r, c, listed, null = pw.simulate(jax.random.key(11), 24, 8, 520)
    r, c, listed = (np.asarray(a)[np.asarray(null)] for a in (r, c, listed))
    deployed, weeks = 0, 0
    run = pl.runner(model, pl.Config())
    for p in range(r.shape[0]):
        out = run(r[p], c[p], listed[p], 0.0005)
        live = out["scored"]
        deployed += ((out["w_prem"] > 0) | (out["w_dir"] > 0))[live].sum()
        weeks += live.sum()
    assert r.shape[0] >= 5
    assert deployed / weeks <= 0.02


def test_pooled_ledger_is_a_supermartingale_under_the_sign_null():
    """Capital of a pooled ledger on fair-coin settlements rarely reaches 1 / alpha."""
    rng = np.random.default_rng(7)
    n_panels, n_assets, n_weeks = 2000, 10, 300
    led = pl.init_ledger(n_assets)
    settle = jax.jit(jax.vmap(pl.settle, in_axes=(0, 0, None)))
    place = jax.jit(jax.vmap(pl.place, in_axes=(0, 0, None)))
    leds = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_panels,) + x.shape), led)
    log_k = np.zeros(n_panels)
    hit = np.zeros(n_panels, bool)
    valid = jnp.ones(n_assets, bool)
    common = rng.normal(size=(n_weeks, n_panels, 1))
    for t in range(n_weeks):
        size = np.abs(common[t] + rng.standard_t(3, (n_panels, n_assets)))
        z = jnp.asarray(np.where(rng.random((n_panels, 1)) < 0.5, -1.0, 1.0) * size)
        frac, bet, placed = leds.frac, leds.bet, leds.placed
        u = np.where(placed, bet * np.clip(z, -2, 2) / 2, 0.0)
        cnt = placed.sum(axis=1)
        ubar = np.where(cnt > 0, u.sum(axis=1) / np.maximum(cnt, 1), 0.0)
        log_k += np.log1p(np.asarray(frac) * ubar)
        hit |= log_k >= np.log(10.0)
        leds = settle(leds, z, valid)
        forecast = jnp.asarray(rng.normal(size=(n_panels, n_assets)) + 0.5)
        leds = place(leds, forecast, valid)
    assert hit.mean() <= 0.1


def test_premium_logit_is_finite_and_odd_on_one_asset():
    prem = pl.Premium(jnp.array([0.002]), jnp.array([0.5]), jnp.array([0.01]))
    var = jnp.array([0.0004])
    valid = jnp.array([True])
    a = pl.premium_logit(prem, var, valid)
    b = pl.premium_logit(prem._replace(total=-prem.total), var, valid)
    assert np.isfinite(np.asarray(a)).all()
    np.testing.assert_allclose(a, -b)
    assert float(pl.premium_logit(prem, var, jnp.array([False]))[0]) == 0.0


def test_gradients_are_finite_and_reach_every_layer(model):
    r, c, listed, _ = pw.simulate(jax.random.key(4), 2, 4, 160)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss(p):
        return -pl.panel_growth(nnx.merge(graphdef, p, rest), r, c, listed, 0.001,
                                jnp.array([0.0, 1.0]))

    grads = jax.grad(loss)(params)
    for path, g in jax.tree_util.tree_leaves_with_path(grads):
        assert np.isfinite(np.asarray(g)).all(), path
    for name in ("hidden", "out"):
        assert any(np.abs(np.asarray(g)).max() > 0 for p, g in
                   jax.tree_util.tree_leaves_with_path(grads) if name in str(p))


def test_save_and_load_round_trip(tmp_path, panel, model):
    r, c, listed, _ = panel
    path = str(tmp_path / "pooled.npz")
    pl.save(model, path)
    again = pl.load(path)
    a = pl.runner(model)(r[0], c[0], listed[0], 0.0005)
    b = pl.runner(again)(r[0], c[0], listed[0], 0.0005)
    np.testing.assert_array_equal(a["target"], b["target"])


def test_panel_world_is_reproducible():
    a = pw.simulate(jax.random.key(9), 2, 3, 100)
    b = pw.simulate(jax.random.key(9), 2, 3, 100)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
