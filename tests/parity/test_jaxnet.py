import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("chex")

import chex  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from softedge.parity import backtest as bt  # noqa: E402
from softedge.parity import jax_world as jw  # noqa: E402
from softedge.parity import jaxnet as jn  # noqa: E402

jax.config.update("jax_enable_x64", True)


def make_model(seed=0):
    model = jn.ParityPolicy((16, 16), rngs=nnx.Rngs(seed))
    rng = np.random.default_rng(seed)
    model.out.kernel[...] = jnp.asarray(rng.normal(0.0, 0.5, model.out.kernel[...].shape))
    model.set_scaling(rng.standard_normal((500, jn.N_ODD + jn.N_EVEN)) * 0.5)
    return model


def paths(n_paths=6, n_steps=400, seed=1):
    r, c, _ = jw.simulate(jax.random.key(seed), n_paths, n_steps)
    return np.asarray(r, dtype=float), np.asarray(c, dtype=float)


def test_rollout_matches_the_numpy_backtest():
    model = make_model()
    r, c = paths()
    growth, held = jax.vmap(lambda a, b: jn.rollout(model, a, b, 0.001))(r, c)
    ref = bt.simulate(r, jn.JaxPolicy(model, r.shape[0], gate=False), cash=c, cost=0.001)
    np.testing.assert_allclose(np.asarray(held), ref["held"], atol=1e-10)
    np.testing.assert_allclose(np.asarray(growth), ref["growth"], atol=1e-10)


def test_mirroring_the_market_mirrors_the_exposure():
    model = make_model(2)
    r, c = paths(seed=3)
    on = bt.simulate(r, jn.JaxPolicy(model, 6, gate=False, governor=False), cash=c)
    mirror = bt.simulate(2 * c - r, jn.JaxPolicy(model, 6, gate=False, governor=False), cash=c)
    np.testing.assert_allclose(mirror["target"][:, bt.FIRST_DECISION:],
                               1.0 - on["target"][:, bt.FIRST_DECISION:], atol=1e-12)


def test_future_returns_do_not_move_past_decisions():
    model = make_model(4)
    r, c = paths(seed=5)
    scrambled = r.copy()
    scrambled[:, 251:] = 2 * c[:, 251:] - r[:, 251:]
    a = bt.simulate(r, jn.JaxPolicy(model, 6), cash=c)
    b = bt.simulate(scrambled, jn.JaxPolicy(model, 6), cash=c)
    np.testing.assert_array_equal(a["target"][:, :251], b["target"][:, :251])


def test_world_is_reproducible_and_null_markets_are_symmetric():
    a = jw.simulate(jax.random.key(7), 2000, 300)
    b = jw.simulate(jax.random.key(7), 2000, 300)
    chex.assert_trees_all_close(a, b)
    r, c, null = (np.asarray(v) for v in a)
    x = (r - c)[null]
    z = x / x.std(axis=1, keepdims=True)
    assert abs(z.mean()) < 0.01
    assert 0.35 < null.mean() < 0.45


def test_gradients_of_net_growth_are_finite_and_reach_every_layer():
    model = make_model(6)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    r, c = paths(n_paths=8, n_steps=300, seed=8)

    def loss(p):
        return -jn.mean_growth(nnx.merge(graphdef, p, rest), r, c, 0.001)

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    chex.assert_tree_all_finite(grads)
    assert all(float(jnp.abs(g).max()) > 0.0 for g in leaves)


def test_save_and_load_give_the_same_convictions(tmp_path):
    model = make_model(9)
    u = np.random.default_rng(10).standard_normal((20, jn.N_ODD + jn.N_EVEN))
    jn.save(model, str(tmp_path / "w.npz"))
    other = jn.load(str(tmp_path / "w.npz"), hidden=(16, 16))
    np.testing.assert_array_equal(np.asarray(other.conviction(u)), np.asarray(model.conviction(u)))


def test_conviction_is_odd_and_direction_off_holds_one_half():
    model = make_model(11)
    u = np.random.default_rng(12).standard_normal((50, jn.N_ODD + jn.N_EVEN))
    mirror = u.copy()
    mirror[:, : jn.N_ODD] *= -1.0
    np.testing.assert_allclose(np.asarray(model.conviction(mirror)),
                               -np.asarray(model.conviction(u)), atol=1e-12)
    r, c = paths(seed=13)
    res = bt.simulate(r, jn.JaxPolicy(model, 6, direction=False, governor=False), cash=c)
    assert np.all(res["target"][:, bt.FIRST_DECISION:] == 0.5)
