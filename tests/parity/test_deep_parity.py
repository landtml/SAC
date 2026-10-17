import numpy as np

from softedge.parity import backtest as bt
from softedge.parity.deep import MLP, AdamW, OddNet
from softedge.parity.deep_network import (
    N_EVEN,
    N_ODD,
    DeepConfig,
    DeepParityNetwork,
    MarketMemory,
)
from softedge.parity.world import simulate_world


def small_net(seed=0):
    rng = np.random.default_rng(seed)
    net = OddNet(3, 2, (8, 8), rng)
    for p in net.mlp.params:
        p += 0.3 * rng.standard_normal(p.shape)
    return net


def test_mlp_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    mlp = MLP((4, 6, 5, 1), rng, out_scale=1.0)
    x = rng.standard_normal((7, 4))
    y = rng.standard_normal((7, 1))

    def loss():
        out, _ = mlp.forward(x)
        return float(np.mean((out - y) ** 2))

    out, cache = mlp.forward(x)
    grads = mlp.backward(cache, 2.0 * (out - y) / y.size)
    for p, g in zip(mlp.params, grads, strict=True):
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + 1e-6
            up = loss()
            p[idx] = old - 1e-6
            down = loss()
            p[idx] = old
            assert abs((up - down) / 2e-6 - g[idx]) < 1e-6


def test_odd_net_gradients_match_finite_differences():
    net = small_net()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((9, 5))
    y = rng.standard_normal(9)
    _, grads = net.loss_and_grads(x, y)
    for p, g in zip(net.mlp.params, grads, strict=True):
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + 1e-6
            up = net.loss_and_grads(x, y)[0]
            p[idx] = old - 1e-6
            down = net.loss_and_grads(x, y)[0]
            p[idx] = old
            assert abs((up - down) / 2e-6 - g[idx]) < 1e-6


def test_odd_net_is_exactly_odd_in_its_odd_inputs():
    net = small_net()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 5))
    net.fit_scaling(x + rng.standard_normal((50, 5)))
    mirror = x.copy()
    mirror[:, :3] *= -1.0
    np.testing.assert_allclose(net.predict(mirror), -net.predict(x), atol=1e-14)


def test_adamw_fits_a_small_odd_target():
    rng = np.random.default_rng(4)
    net = OddNet(3, 2, (16, 16), rng)
    x = rng.standard_normal((2000, 5))
    y = np.tanh(x[:, 0] * (1.0 + x[:, 3] ** 2)) + 0.5 * x[:, 1]
    net.fit_scaling(x)
    opt = AdamW(net.mlp.params, lr=3e-3)
    first = net.loss_and_grads(x, y)[0]
    for _ in range(600):
        idx = rng.integers(0, 2000, 256)
        opt.step(net.loss_and_grads(x[idx], y[idx])[1])
    assert net.loss_and_grads(x, y)[0] < 0.1 * first


def test_state_round_trip_gives_the_same_predictions(tmp_path):
    net = small_net()
    x = np.random.default_rng(5).standard_normal((20, 5))
    np.savez(tmp_path / "w.npz", **net.state())
    with np.load(tmp_path / "w.npz") as state:
        other = OddNet.from_state(state)
    np.testing.assert_array_equal(other.predict(x), net.predict(x))


def test_memory_inputs_split_by_parity():
    rng = np.random.default_rng(6)
    x = 0.01 * rng.standard_t(4, (16, 300))
    a, b = MarketMemory(16), MarketMemory(16)
    for t in range(300):
        a.observe(x[:, t])
        b.observe(-x[:, t])
    ia, ib = a.inputs(), b.inputs()
    assert ia.shape == (16, N_ODD + N_EVEN)
    np.testing.assert_allclose(ib[:, :N_ODD], -ia[:, :N_ODD], atol=1e-12)
    np.testing.assert_allclose(ib[:, N_ODD:], ia[:, N_ODD:], atol=1e-12)


def test_world_is_reproducible_and_null_paths_are_symmetric():
    a = simulate_world(400, 300, seed=7)
    b = simulate_world(400, 300, seed=7)
    np.testing.assert_array_equal(a.logret, b.logret)
    excess = (a.logret - a.cash)[a.null]
    z = excess / excess.std(axis=1, keepdims=True)
    assert abs(z.mean()) < 0.02
    assert 0.3 < a.null.mean() < 0.5


def test_deep_network_mirrors_and_ignores_the_future():
    rng = np.random.default_rng(8)
    model = OddNet(N_ODD, N_EVEN, (8,), rng)
    for p in model.mlp.params:
        p += 0.2 * rng.standard_normal(p.shape)
    cash = 0.0005
    r = cash + 0.01 * rng.standard_t(4, (8, 400)) + 0.002
    cfg = DeepConfig(governor=False, gate=False)
    on = bt.simulate(r, DeepParityNetwork(8, model, cfg))
    mirror = bt.simulate(2 * cash - r, DeepParityNetwork(8, model, cfg))
    np.testing.assert_allclose(mirror["target"][:, bt.FIRST_DECISION:],
                               1.0 - on["target"][:, bt.FIRST_DECISION:], atol=1e-12)
    scrambled = r.copy()
    scrambled[:, 201:] = 2 * cash - scrambled[:, 201:]
    other = bt.simulate(scrambled, DeepParityNetwork(8, model, DeepConfig()))
    base = bt.simulate(r, DeepParityNetwork(8, model, DeepConfig()))
    np.testing.assert_array_equal(base["target"][:, :201], other["target"][:, :201])


def test_direction_off_holds_one_half():
    model = OddNet(N_ODD, N_EVEN, (8,), np.random.default_rng(9))
    r = 0.0005 + 0.01 * np.random.default_rng(10).standard_normal((4, 200))
    res = bt.simulate(r, DeepParityNetwork(4, model, DeepConfig(direction=False,
                                                                governor=False)))
    assert np.all(res["target"][:, bt.FIRST_DECISION:] == 0.5)
