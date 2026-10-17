"""Deep parity network in JAX, trained end to end through a differentiable backtest.

The model is a Flax NNX module with two parts:

- **Memory.** Diagonal linear recurrences over the market's normalised excess return z, like a
  state-space layer. Odd channels are EWMAs of z, even channels are EWMAs of z^2 - 1 and of z
  times last week's odd memories. Their decay rates are parameters and are learned.
- **Odd network.** An MLP on the odd and even memories, made odd by construction:
  g(u) = (f(u) - f(S u)) / 2, where S flips the sign of the odd inputs.

The exposure is (1 + tanh(w g)) / 2, where w is the evidence ledger's weight (1 in training).
With no conviction it is one half, and mirroring the market maps it to one minus itself. The
governor caps it on the policy's own drawdown.

``rollout`` runs the weekly backtest of ``softedge.parity.backtest.simulate`` as a ``lax.scan``:
costs on traded units, the 0.05 no-trade band, warm-up and governor included. Its net log growth
is differentiable in every parameter, so the policy is trained to maximise it directly.
``JaxPolicy`` wraps the same step functions for the numpy backtest used in evaluation.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from softedge.parity.deep_network import EvidenceLedger

ODD_SCALES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SQUARE_SCALES = (4, 16, 64)
RELIABILITY_SCALES = (104, 416)
RELIABILITY_MEMORIES = (1, 4, 16, 64)
N_ODD = len(ODD_SCALES) + 1
N_EVEN = len(SQUARE_SCALES) + 2 + len(RELIABILITY_SCALES) * len(RELIABILITY_MEMORIES) + 2
VOL_LAMBDA = 0.94
VOL_SEED = 13
CLIP = 10.0
WARMUP = 52
FIRST_DECISION = WARMUP - 1
BAND = 0.05
FLOOR = 0.78
MULTIPLIER = 4.0
_REL_IDX = np.array([ODD_SCALES.index(s) for s in RELIABILITY_MEMORIES])


class Memory(NamedTuple):
    var: jax.Array
    var_mid: jax.Array
    var_long: jax.Array
    s1: jax.Array
    s2: jax.Array
    m: jax.Array
    q: jax.Array
    rel: jax.Array
    z_sum: jax.Array
    n: jax.Array


def init_memory(batch: tuple[int, ...] = ()) -> Memory:
    z = jnp.zeros(batch)
    return Memory(z, z, z, z, z, jnp.zeros((*batch, len(ODD_SCALES))),
                  jnp.zeros((*batch, len(SQUARE_SCALES))),
                  jnp.zeros((*batch, len(RELIABILITY_SCALES), len(RELIABILITY_MEMORIES))), z, z)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _norm(a):
    return jnp.sqrt((2.0 - a) / a)


class ParityPolicy(nnx.Module):
    def __init__(self, hidden: tuple[int, ...] = (64, 64, 64), *, rngs: nnx.Rngs):
        self.odd_logit = nnx.Param(jnp.array([_logit(min(1.0 / s, 0.95)) for s in ODD_SCALES]))
        self.sq_logit = nnx.Param(jnp.array([_logit(1.0 / s) for s in SQUARE_SCALES]))
        self.rel_logit = nnx.Param(jnp.array([_logit(1.0 / s) for s in RELIABILITY_SCALES]))
        sizes = (N_ODD + N_EVEN, *hidden)
        self.hidden = nnx.List([nnx.Linear(a, b, rngs=rngs)
                                for a, b in zip(sizes[:-1], sizes[1:], strict=True)])
        # No output bias: it cancels in the odd construction and would never be trained.
        self.out = nnx.Linear(sizes[-1], 1, use_bias=False,
                              kernel_init=nnx.initializers.normal(1e-3), rngs=rngs)
        self.shift = nnx.Variable(jnp.zeros(N_ODD + N_EVEN))
        self.scale = nnx.Variable(jnp.ones(N_ODD + N_EVEN))
        self.flip = nnx.Variable(jnp.concatenate((-jnp.ones(N_ODD), jnp.ones(N_EVEN))))

    def decays(self):
        return (jax.nn.sigmoid(self.odd_logit[...]), jax.nn.sigmoid(self.sq_logit[...]),
                jax.nn.sigmoid(self.rel_logit[...]))

    def step(self, mem: Memory, x: jax.Array, t: jax.Array) -> tuple[Memory, jax.Array, jax.Array]:
        """Ingest week t's excess log return. Returns the memory, z and the inputs."""
        a_odd, a_sq, a_rel = self.decays()
        active = t >= VOL_SEED
        seeding = t < VOL_SEED
        safe_var = jnp.where(mem.var > 0.0, mem.var, 1.0)
        z = jnp.where(active, jnp.clip(x / jnp.sqrt(safe_var), -CLIP, CLIP), 0.0)
        s1 = mem.s1 + jnp.where(seeding, x, 0.0)
        s2 = mem.s2 + jnp.where(seeding, x * x, 0.0)
        seed_now = t == VOL_SEED - 1
        seed_var = (s2 - s1 * s1 / VOL_SEED) / (VOL_SEED - 1)

        def ewma(old, lam):
            return jnp.where(seed_now, seed_var,
                             jnp.where(active, lam * old + (1.0 - lam) * x * x, old))

        var = ewma(mem.var, VOL_LAMBDA)
        var_mid = ewma(mem.var_mid, 1.0 - 1.0 / 52.0)
        var_long = ewma(mem.var_long, 1.0 - 1.0 / 260.0)
        act = active[..., None]
        prev = mem.m[..., _REL_IDX] * _norm(a_odd[_REL_IDX])
        rel_new = ((1.0 - a_rel)[:, None] * mem.rel
                   + a_rel[:, None] * (z[..., None, None] * prev[..., None, :]))
        rel = jnp.where(act[..., None], rel_new, mem.rel)
        m = jnp.where(act, (1.0 - a_odd) * mem.m + a_odd * z[..., None], mem.m)
        sq = jnp.minimum(z * z, 25.0) - 1.0
        q = jnp.where(act, (1.0 - a_sq) * mem.q + a_sq * sq[..., None], mem.q)
        n = mem.n + active.astype(x.dtype)
        mem = Memory(var, var_mid, var_long, s1, s2, m, q, rel, mem.z_sum + z, n)
        return mem, z, self.inputs(mem)

    def inputs(self, mem: Memory) -> jax.Array:
        a_odd, a_sq, a_rel = self.decays()
        ready = mem.n > 0
        n = jnp.maximum(mem.n, 1.0)
        sig = jnp.sqrt(jnp.where(mem.var > 0.0, mem.var, 1.0))
        sig_mid = jnp.sqrt(jnp.where(mem.var_mid > 0.0, mem.var_mid, 1.0))
        sig_long = jnp.sqrt(jnp.where(mem.var_long > 0.0, mem.var_long, 1.0))
        odd = jnp.concatenate((mem.m * _norm(a_odd), (mem.z_sum / jnp.sqrt(n))[..., None]), -1)
        rel = mem.rel * _norm(a_rel)[:, None]
        even = jnp.concatenate((
            mem.q * _norm(a_sq) / math.sqrt(2.0),
            jnp.log(sig / sig_mid)[..., None],
            jnp.log(sig_mid / sig_long)[..., None],
            rel.reshape(*rel.shape[:-2], -1),
            jnp.log1p(mem.n / 52.0)[..., None],
            jnp.log(sig)[..., None],
        ), -1)
        u = jnp.clip(jnp.concatenate((odd, even), -1), -CLIP, CLIP)
        return jnp.where(ready[..., None], u, 0.0)

    def _mlp(self, u: jax.Array) -> jax.Array:
        h = u
        for layer in self.hidden:
            h = jax.nn.silu(layer(h))
        return self.out(h)[..., 0]

    def conviction(self, u: jax.Array) -> jax.Array:
        """The odd output g: it changes sign when the odd inputs do."""
        scaled = (u - self.shift[...]) / self.scale[...]
        mirror = (u * self.flip[...] - self.shift[...]) / self.scale[...]
        return 0.5 * (self._mlp(scaled) - self._mlp(mirror))

    def set_scaling(self, u: np.ndarray) -> None:
        """Fix the input scaling from sample inputs: odd by RMS, even by mean and std."""
        u = np.asarray(u, dtype=np.float64)
        rms = np.sqrt(np.mean(u[:, :N_ODD] ** 2, axis=0))
        mean = u[:, N_ODD:].mean(axis=0)
        sd = u[:, N_ODD:].std(axis=0)
        self.shift[...] = jnp.asarray(np.concatenate((np.zeros(N_ODD), mean)))
        self.scale[...] = jnp.asarray(np.maximum(np.concatenate((rms, sd)), 1e-6))


def exposure(g, weight=1.0):
    return 0.5 * (1.0 + jnp.tanh(weight * g))


def governor_cap(wealth, peak):
    return jnp.clip(MULTIPLIER * (wealth - FLOOR * peak) / wealth, 0.0, 1.0)


def rollout(model: ParityPolicy, r: jax.Array, c: jax.Array, cost: float):
    """Backtest one market: weekly log returns r and cash c, both shaped (weeks,).

    Returns the weekly net log growth and the exposure held each week.
    """
    n = r.shape[0]

    def body(carry, inp):
        mem, h, prev_h, w_prev, hi = carry
        r_t, c_t, t = inp
        paid = cost * jnp.abs(h - prev_h)
        w = w_prev * (1.0 - paid) * (h * jnp.exp(r_t) + (1.0 - h) * jnp.exp(c_t))
        hi = jnp.maximum(hi, w)
        mem, _, u = model.step(mem, r_t - c_t, t)
        e = jnp.minimum(exposure(model.conviction(u)), governor_cap(w, hi))
        e = jnp.where(t >= FIRST_DECISION, e, 0.0)
        nxt = jnp.where(t + 1 < WARMUP, 0.0, jnp.where(jnp.abs(e - h) > BAND, e, h))
        return (mem, nxt, h, w, hi), (jnp.log(w / w_prev), h)

    one = jnp.ones(())
    carry0 = (init_memory(), 0.0 * one, 0.0 * one, one, one)
    _, (growth, held) = jax.lax.scan(body, carry0, (r, c, jnp.arange(n)))
    return growth, held


def mean_growth(model: ParityPolicy, r: jax.Array, c: jax.Array, cost: float) -> jax.Array:
    """Mean weekly net log growth after warm-up over a batch of markets (paths, weeks)."""
    growth, _ = jax.vmap(lambda a, b: rollout(model, a, b, cost))(r, c)
    return growth[:, WARMUP:].mean()


class JaxPolicy:
    """The trained model as a policy for ``softedge.parity.backtest.simulate``."""

    def __init__(self, model: ParityPolicy, n_paths: int, *, gate: bool = True,
                 direction: bool = True, governor: bool = True, alpha: float = 0.1,
                 cash: float = 0.0005, trace: bool = False):
        graphdef, state = nnx.split(model)

        def step(state, mem, x, t):
            m = nnx.merge(graphdef, state)
            mem, z, u = m.step(mem, x, t)
            return mem, z, m.conviction(u)

        self._step = jax.jit(step)
        self.state = state
        self.mem = init_memory((n_paths,))
        self.t = -1
        self.cash = cash
        self.gate = gate
        self.direction = direction
        self.governor = governor
        self.ledger = EvidenceLedger(n_paths, alpha=alpha)
        self.g = np.zeros(n_paths)
        self.trace = {"weight": [], "conviction": [], "evidence": []} if trace else None

    def observe(self, logret, cash=None) -> None:
        self.t += 1
        c = self.cash if cash is None else np.asarray(cash, dtype=float)
        x = jnp.asarray(np.asarray(logret, dtype=float) - c)
        self.mem, z, g = self._step(self.state, self.mem, x, jnp.asarray(self.t))
        if self.t >= VOL_SEED:
            self.g = np.asarray(g, dtype=float)
            self.ledger.settle(np.asarray(z, dtype=float))
            self.ledger.place(self.g)

    def target(self, wealth, peak) -> np.ndarray:
        w = self.ledger.weight() if self.gate else np.ones_like(self.g)
        if self.direction:
            e = 0.5 * (1.0 + np.tanh(w * self.g))
        else:
            e = np.full(self.g.shape, 0.5)
        if self.governor:
            wealth = np.asarray(wealth, dtype=float)
            cap = np.clip(MULTIPLIER * (wealth - FLOOR * np.asarray(peak)) / wealth, 0.0, 1.0)
            e = np.minimum(e, cap)
        if self.trace is not None:
            self.trace["weight"].append(np.array(w, copy=True))
            self.trace["conviction"].append(self.g.copy())
            self.trace["evidence"].append(self.ledger.evidence.copy())
        return e


def _key(path) -> str:
    parts = [str(getattr(k, "key", getattr(k, "idx", k))).lstrip(".") for k in path]
    return ".".join(p for p in parts if p != "value")


def save(model: ParityPolicy, path: str) -> None:
    """Every variable of the model in one npz, keyed by its path in the module."""
    leaves = jax.tree_util.tree_flatten_with_path(nnx.state(model))[0]
    np.savez(path, **{_key(p): np.asarray(v) for p, v in leaves})


def load(path: str, hidden: tuple[int, ...] = (64, 64, 64)) -> ParityPolicy:
    model = ParityPolicy(hidden, rngs=nnx.Rngs(0))
    leaves, treedef = jax.tree_util.tree_flatten_with_path(nnx.state(model))
    with np.load(path) as saved:
        values = [jnp.asarray(saved[_key(p)]) for p, _ in leaves]
    nnx.update(model, jax.tree_util.tree_unflatten(treedef, values))
    return model
