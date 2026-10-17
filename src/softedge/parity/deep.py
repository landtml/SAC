"""A small deep-learning library written from scratch in numpy.

It holds only what the deep parity network needs: a multilayer perceptron with SiLU activations
and hand-written backpropagation, a wrapper that makes the network odd in chosen inputs, and the
AdamW optimiser. A batch is the first axis of every array.
"""

from __future__ import annotations

import numpy as np


def silu(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SiLU, x * sigmoid(x), and its derivative."""
    s = 0.5 * (1.0 + np.tanh(0.5 * x))
    return x * s, s * (1.0 + x * (1.0 - s))


class MLP:
    """Dense layers with SiLU between them and a linear output."""

    def __init__(self, sizes: tuple[int, ...], rng: np.random.Generator, *,
                 out_scale: float = 0.01):
        self.sizes = tuple(sizes)
        self.params: list[np.ndarray] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            last = i == len(sizes) - 2
            scale = out_scale / np.sqrt(a) if last else np.sqrt(2.0 / a)
            self.params.append(rng.standard_normal((a, b)) * scale)
            self.params.append(np.zeros(b))

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list]:
        cache = []
        h = x
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            w, b = self.params[2 * i], self.params[2 * i + 1]
            pre = h @ w + b
            if i < n_layers - 1:
                out, deriv = silu(pre)
                cache.append((h, deriv))
                h = out
            else:
                cache.append((h, None))
                h = pre
        return h, cache

    def backward(self, cache: list, grad_out: np.ndarray) -> list[np.ndarray]:
        grads: list[np.ndarray] = [np.empty(0)] * len(self.params)
        g = grad_out
        for i in reversed(range(len(cache))):
            h, deriv = cache[i]
            if deriv is not None:
                g = g * deriv
            grads[2 * i] = h.T @ g
            grads[2 * i + 1] = g.sum(axis=0)
            if i > 0:
                g = g @ self.params[2 * i].T
        return grads


class OddNet:
    """f(x) = (g(x) - g(S x)) / 2, where S flips the sign of the odd inputs.

    Whatever the weights, f changes sign when the odd inputs do and the even inputs stay.
    Inputs are scaled by fixed factors set from training data; odd inputs are not centred, so
    the scaling keeps their sign.
    """

    def __init__(self, n_odd: int, n_even: int, hidden: tuple[int, ...],
                 rng: np.random.Generator):
        self.n_odd = n_odd
        self.n_even = n_even
        self.mlp = MLP((n_odd + n_even, *hidden, 1), rng)
        self.shift = np.zeros(n_odd + n_even)
        self.scale = np.ones(n_odd + n_even)
        self.flip = np.concatenate((-np.ones(n_odd), np.ones(n_even)))

    @property
    def n_params(self) -> int:
        return self.mlp.n_params

    def fit_scaling(self, x: np.ndarray) -> None:
        """Set the input scaling: odd inputs by their RMS, even inputs by mean and std."""
        rms = np.sqrt(np.mean(x[:, : self.n_odd] ** 2, axis=0))
        mean = x[:, self.n_odd:].mean(axis=0)
        sd = x[:, self.n_odd:].std(axis=0)
        self.shift = np.concatenate((np.zeros(self.n_odd), mean))
        self.scale = np.maximum(np.concatenate((rms, sd)), 1e-8)

    def _inputs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = (x - self.shift) / self.scale
        mirror = (x * self.flip - self.shift) / self.scale
        return u, mirror

    def predict(self, x: np.ndarray) -> np.ndarray:
        u, mirror = self._inputs(x)
        return 0.5 * (self.mlp.forward(u)[0][:, 0] - self.mlp.forward(mirror)[0][:, 0])

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """Mean squared error and its gradient for every parameter."""
        u, mirror = self._inputs(x)
        g_pos, c_pos = self.mlp.forward(u)
        g_neg, c_neg = self.mlp.forward(mirror)
        pred = 0.5 * (g_pos[:, 0] - g_neg[:, 0])
        err = pred - y
        loss = float(np.mean(err * err))
        d_pred = (2.0 / err.size) * err[:, None]
        grads_pos = self.mlp.backward(c_pos, 0.5 * d_pred)
        grads_neg = self.mlp.backward(c_neg, -0.5 * d_pred)
        return loss, [a + b for a, b in zip(grads_pos, grads_neg, strict=True)]

    def state(self) -> dict[str, np.ndarray]:
        out = {"shift": self.shift, "scale": self.scale, "n_odd": np.array(self.n_odd),
               "n_even": np.array(self.n_even), "sizes": np.array(self.mlp.sizes)}
        for i, p in enumerate(self.mlp.params):
            out[f"p{i}"] = p
        return out

    @classmethod
    def from_state(cls, state) -> OddNet:
        sizes = tuple(int(s) for s in state["sizes"])
        net = cls(int(state["n_odd"]), int(state["n_even"]), sizes[1:-1],
                  np.random.default_rng(0))
        net.shift = np.array(state["shift"], dtype=float)
        net.scale = np.array(state["scale"], dtype=float)
        net.mlp.params = [np.array(state[f"p{i}"], dtype=float)
                          for i in range(len(net.mlp.params))]
        return net


class AdamW:
    """Adam with decoupled weight decay on the weight matrices only."""

    def __init__(self, params: list[np.ndarray], *, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-4):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray], lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v, strict=True):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            if p.ndim == 2:
                p *= 1.0 - lr * self.wd
            p -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
