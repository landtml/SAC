"""Pooled parity network: the deep parity network for a panel of assets, in JAX.

The deep parity network (``softedge.parity.jaxnet``) reads one market at a time. Its registered
test failed for three reasons: its evidence ledger never reached the line on short or noisy
histories, its drawdown governor was too loose for crashes larger than 25% in a week, and its
default of one half ignored the premium over cash. This version changes each:

- **Pooled inputs.** Every asset's memories are averaged over the assets listed that week. The
  odd network reads its own memories and the panel's, so what the panel shows informs a new or
  short history.
- **Pooled evidence.** A ledger bets on every listed asset each week and settles on the average
  of the bets. Under the sign-flip null each asset's settlement has mean zero given the past, so
  their average does too, and the ledger's capital is a test martingale with any cross-asset
  dependence. This needs the listing of an asset in a week to be known before that week's
  return. The gate reads a CUSUM of log capital, so evidence drains when a learner starts
  losing; its false alarms are bounded per stretch of time, not over all time. An edge too weak
  to prove on one asset passes the gate when it recurs across assets.
- **Premium channel.** A second pooled ledger deploys a half-Kelly premium: each asset's
  Sharpe ratio of excess log return, shrunk towards the panel's by a James-Stein weight. The
  default of one half moves towards cash when the panel shows that the assets lose to cash, and
  towards the asset when they beat it, but only as far as that ledger's evidence allows.
- **Crash governor.** The exposure is capped at (W - f H) / (W c), where c is a crash size
  estimated for each asset: the largest of its own worst weekly loss so far, its current
  volatility times the largest weekly loss in volatility units seen on any asset of the panel so
  far (clipped to [8, 32]), and a minimum. A weekly loss no larger than c cannot take wealth below
  f H, less the cost of the last trade, which stays above the 25% budget.

The exposure is (1 + tanh(w_p p + w_d g)) / 2, where p is the premium logit, g the odd
network's conviction and w_p, w_d the two ledgers' weights. With no evidence it is one half.
Mirroring every excess return maps it to one minus itself when the governor is off.

``rollout`` runs the weekly backtest of ``softedge.parity.backtest.simulate`` for a whole panel
as one ``lax.scan``, with each asset on its own clock from its listing week. One rule differs:
an asset held above its cap is traded down to it even inside the no-trade band. The backtest is
differentiable, and training maximises its net log growth with the gates open.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from softedge.parity import jaxnet as jn

N_OWN = jn.N_ODD + jn.N_EVEN
N_IN_ODD = 2 * jn.N_ODD
N_IN_EVEN = 2 * jn.N_EVEN + 2
N_IN = N_IN_ODD + N_IN_EVEN
WARMUP = jn.WARMUP
FIRST_DECISION = jn.FIRST_DECISION
BAND = jn.BAND
VOL_SEED = jn.VOL_SEED
FLOOR = 0.76
TAIL_PRIOR = 8.0
TAIL_MAX = 32.0
MIN_CRASH = 0.10
V1_FLOOR = 0.78
V1_MULTIPLIER = 4.0
PREMIUM_DECAY = 1.0 / 260.0
PREMIUM_MIN_WEEKS = 26
KELLY_FRACTION = 0.5
LOGIT_CLIP = 3.0
MIN_BETS = 13
MAX_FRACTION = 0.5
N_LEDGERS = 2


class Config(NamedTuple):
    """Switches for evaluation and ablation. Training uses ``gate=False``."""

    gate: bool = True
    premium: bool = True
    direction: bool = True
    governor: str = "crash"
    fixed: float = -1.0
    alpha: float = 0.1


DEFAULT = Config()


class Ledger(NamedTuple):
    evidence: jax.Array
    u_sum: jax.Array
    u_sq: jax.Array
    n_settled: jax.Array
    frac: jax.Array
    bet: jax.Array
    placed: jax.Array
    pred_sq: jax.Array
    n_pred: jax.Array


def init_ledger(n: int) -> Ledger:
    z = jnp.zeros(())
    v = jnp.zeros(n)
    return Ledger(z, z, z, z, z, v, jnp.zeros(n, bool), v, v)


def settle(led: Ledger, z: jax.Array, listed: jax.Array) -> Ledger:
    """Settle last week's bets of the assets listed this week on psi(z) = clip(z, -2, 2) / 2."""
    has = led.placed & listed
    count = has.sum()
    u = jnp.where(has, led.bet * jnp.clip(z, -2.0, 2.0) / 2.0, 0.0)
    ubar = u.sum() / jnp.maximum(count, 1)
    any_bet = count > 0
    step = jnp.where(any_bet, jnp.log1p(led.frac * ubar), 0.0)
    return led._replace(
        evidence=jnp.maximum(0.0, led.evidence + step),
        u_sum=led.u_sum + jnp.where(any_bet, ubar, 0.0),
        u_sq=led.u_sq + jnp.where(any_bet, ubar * ubar, 0.0),
        n_settled=led.n_settled + any_bet,
        placed=jnp.zeros_like(led.placed))


def place(led: Ledger, forecast: jax.Array, valid: jax.Array) -> Ledger:
    """Bet clip(forecast / (2 x RMS of the asset's past forecasts), -1, 1) on each valid asset."""
    scale = 2.0 * jnp.sqrt(led.pred_sq / jnp.maximum(led.n_pred, 1.0))
    ok = valid & (led.n_pred >= MIN_BETS) & (scale > 0.0)
    bet = jnp.where(ok, jnp.clip(forecast / jnp.where(scale > 0.0, scale, 1.0), -1.0, 1.0), 0.0)
    frac = jnp.where(led.n_settled >= MIN_BETS,
                     jnp.clip(led.u_sum / (led.u_sq + 1.0), 0.0, MAX_FRACTION), 0.0)
    return led._replace(
        bet=bet, placed=ok, frac=frac,
        pred_sq=led.pred_sq + jnp.where(valid, forecast * forecast, 0.0),
        n_pred=led.n_pred + valid)


def weight(led: Ledger, alpha: float) -> jax.Array:
    """(1 - E0 / E)+ with E0 = log(K / alpha) for the K ledgers."""
    e0 = math.log(N_LEDGERS / alpha)
    return jnp.where(led.evidence > e0, 1.0 - e0 / jnp.maximum(led.evidence, e0), 0.0)


class Premium(NamedTuple):
    total: jax.Array
    weight: jax.Array
    weight_sq: jax.Array


def premium_logit(prem: Premium, var_long: jax.Array, valid: jax.Array) -> jax.Array:
    """Half-Kelly logit of each asset's excess-return Sharpe ratio, shrunk towards the panel's.

    The Sharpe ratio s_i uses an EWMA mean with decay 1/260 (a half-life of about 3.5 years)
    and the 260-week EWMA volatility. Its sampling variance is 1 / n_i, with n_i the effective number of weeks. The
    shrinkage weight is tau^2 / (tau^2 + 1 / n_i), where tau^2 is the cross-sectional variance of
    s beyond sampling noise. On one asset tau^2 is zero and the target is the asset itself.
    """
    w = jnp.where(prem.weight > 0.0, prem.weight, 1.0)
    mu = prem.total / w
    n_eff = jnp.where(prem.weight_sq > 0.0, prem.weight ** 2 / jnp.maximum(prem.weight_sq, 1e-30),
                      1.0)
    sig = jnp.sqrt(jnp.where(var_long > 0.0, var_long, 1.0))
    s = mu / sig
    k = valid.astype(s.dtype)
    count = jnp.maximum(k.sum(), 1.0)
    pooled = (k * n_eff * s).sum() / jnp.maximum((k * n_eff).sum(), 1e-12)
    spread = (k * (s - pooled) ** 2).sum() / count
    noise = (k / n_eff).sum() / count
    tau2 = jnp.maximum(spread - noise, 0.0)
    shrink = tau2 / (tau2 + 1.0 / n_eff)
    s_star = pooled + shrink * (s - pooled)
    logit = 2.0 * KELLY_FRACTION * s_star / sig
    return jnp.where(valid, jnp.clip(logit, -LOGIT_CLIP, LOGIT_CLIP), 0.0)


class PooledPolicy(jn.ParityPolicy):
    """The odd network on an asset's own memories, the panel's mean memories and two context terms.

    Input layout: own odd, panel odd, own even, panel even, log(1 + listed assets), premium
    weight. The flip S negates the first 2 x 10.
    """

    def __init__(self, hidden: tuple[int, ...] = (64, 64, 64), *, rngs: nnx.Rngs):
        super().__init__(hidden, rngs=rngs)
        sizes = (N_IN, *hidden)
        self.hidden = nnx.List([nnx.Linear(a, b, rngs=rngs)
                                for a, b in zip(sizes[:-1], sizes[1:], strict=True)])
        self.out = nnx.Linear(sizes[-1], 1, use_bias=False,
                              kernel_init=nnx.initializers.normal(1e-3), rngs=rngs)
        self.shift = nnx.Variable(jnp.zeros(N_IN))
        self.scale = nnx.Variable(jnp.ones(N_IN))
        self.flip = nnx.Variable(jnp.concatenate((-jnp.ones(N_IN_ODD), jnp.ones(N_IN_EVEN))))

    def panel_inputs(self, mem: jn.Memory, ready: jax.Array, w_prem: jax.Array) -> jax.Array:
        own = self.inputs(mem)
        k = ready.astype(own.dtype)[:, None]
        count = k.sum()
        pool = jnp.broadcast_to((own * k).sum(0) / jnp.maximum(count, 1.0), own.shape)
        n = own.shape[0]
        return jnp.concatenate((
            own[:, :jn.N_ODD], pool[:, :jn.N_ODD], own[:, jn.N_ODD:], pool[:, jn.N_ODD:],
            jnp.full((n, 1), jnp.log1p(count)), jnp.broadcast_to(w_prem, (n,))[:, None]), -1)

    def set_scaling(self, u: np.ndarray) -> None:
        u = np.asarray(u, dtype=np.float64)
        rms = np.sqrt(np.mean(u[:, :N_IN_ODD] ** 2, axis=0))
        mean = u[:, N_IN_ODD:].mean(axis=0)
        sd = u[:, N_IN_ODD:].std(axis=0)
        self.shift[...] = jnp.asarray(np.concatenate((np.zeros(N_IN_ODD), mean)))
        self.scale[...] = jnp.asarray(np.maximum(np.concatenate((rms, sd)), 1e-6))


def governor_cap(cfg: Config, wealth, peak, crash):
    if cfg.governor == "crash":
        return jnp.clip((wealth - FLOOR * peak) / (wealth * crash), 0.0, 1.0)
    if cfg.governor == "v1":
        return jnp.clip(V1_MULTIPLIER * (wealth - V1_FLOOR * peak) / wealth, 0.0, 1.0)
    return jnp.ones_like(wealth)


def rollout(model: PooledPolicy, r: jax.Array, c: jax.Array, listed: jax.Array, cost,
            cfg: Config = DEFAULT, w_prem_train=1.0, record: bool = False,
            inputs: bool = False) -> dict:
    """Backtest one panel: returns r and listing mask (assets, weeks), cash c (weeks,).

    ``cost`` is the cost per unit traded, a scalar or one value per asset. With ``cfg.gate``
    off the direction weight is 1 and the premium weight is ``w_prem_train``. Returns weekly net
    log growth, the exposure held, the scored mask (listed, past the asset's warm-up) and, with
    ``record``, the targets, caps and ledger weights; with ``inputs``, the network's inputs.
    """
    n_assets, n_steps = r.shape
    cost = jnp.broadcast_to(jnp.asarray(cost, dtype=r.dtype), (n_assets,))
    zero = jnp.zeros(n_assets, dtype=r.dtype)

    def body(carry, inp):
        mem, tau, h, prev_h, w_prev, peak, prem, worst, tail, led_p, led_d = carry
        r_t, c_t, lst = inp
        paid = cost * jnp.abs(h - prev_h)
        w_new = w_prev * (1.0 - paid) * (h * jnp.exp(r_t) + (1.0 - h) * jnp.exp(c_t))
        wealth = jnp.where(lst, w_new, w_prev)
        peak = jnp.maximum(peak, wealth)
        growth = jnp.where(lst, jnp.log(wealth / w_prev), 0.0)
        scored = lst & (tau >= WARMUP)

        sig_prev = jnp.sqrt(jnp.where(mem.var > 0.0, mem.var, 1.0))
        loss = -jnp.expm1(r_t)
        worst = jnp.where(lst, jnp.maximum(worst, loss), worst)
        z_ok = lst & (tau >= VOL_SEED)
        tail = jnp.maximum(tail, jnp.max(jnp.where(z_ok, loss / sig_prev, 0.0)))

        x = r_t - c_t
        new_mem, z, _ = model.step(mem, x, tau)
        mem = jax.tree_util.tree_map(
            lambda a, b: jnp.where(lst.reshape(lst.shape + (1,) * (a.ndim - 1)), a, b),
            new_mem, mem)
        z = jnp.where(z_ok, z, 0.0)
        a = PREMIUM_DECAY
        prem = Premium(jnp.where(lst, (1.0 - a) * prem.total + a * x, prem.total),
                       jnp.where(lst, (1.0 - a) * prem.weight + a, prem.weight),
                       jnp.where(lst, (1.0 - a) ** 2 * prem.weight_sq + a * a, prem.weight_sq))

        led_p = settle(led_p, z, z_ok)
        led_d = settle(led_d, z, z_ok)
        if cfg.gate:
            w_p, w_d = weight(led_p, cfg.alpha), weight(led_d, cfg.alpha)
        else:
            w_p, w_d = jnp.asarray(w_prem_train, dtype=r.dtype), jnp.ones((), dtype=r.dtype)
        if not cfg.premium:
            w_p = jnp.zeros_like(w_p)
        if not cfg.direction:
            w_d = jnp.zeros_like(w_d)

        ready = lst & (mem.n > 0)
        u = model.panel_inputs(mem, ready, w_p)
        g = model.conviction(u)
        p = premium_logit(prem, mem.var_long, ready & (mem.n >= PREMIUM_MIN_WEEKS))
        led_p = place(led_p, p, z_ok & (mem.n >= PREMIUM_MIN_WEEKS))
        led_d = place(led_d, g, z_ok)

        e = 0.5 * (1.0 + jnp.tanh(w_p * p + w_d * g))
        if cfg.fixed >= 0.0:
            e = jnp.full_like(e, cfg.fixed)
        sig = jnp.sqrt(jnp.where(mem.var > 0.0, mem.var, 1.0))
        tail_mult = jnp.clip(tail, TAIL_PRIOR, TAIL_MAX)
        crash = jnp.clip(jnp.maximum(worst, tail_mult * sig), MIN_CRASH, 1.0)
        cap = governor_cap(cfg, wealth, peak, crash)
        e = jnp.where(tau >= FIRST_DECISION, jnp.minimum(e, cap), 0.0)
        move = (jnp.abs(e - h) > BAND) | (h > cap)
        nxt = jnp.where(tau + 1 < WARMUP, 0.0, jnp.where(move, e, h))
        out = {"growth": growth, "held": h, "scored": scored}
        if record:
            out.update(target=e, cap=cap, w_prem=jnp.broadcast_to(w_p, (n_assets,)),
                       w_dir=jnp.broadcast_to(w_d, (n_assets,)), premium=p, conviction=g)
        if inputs:
            out["inputs"] = u
        prev_h = jnp.where(lst, h, prev_h)
        h = jnp.where(lst, nxt, h)
        tau = tau + lst
        return (mem, tau, h, prev_h, wealth, peak, prem, worst, tail, led_p, led_d), out

    ones = jnp.ones(n_assets, dtype=r.dtype)
    carry0 = (jn.init_memory((n_assets,)), jnp.zeros(n_assets, jnp.int32), zero, zero, ones,
              ones, Premium(zero, zero, zero), jnp.full(n_assets, -jnp.inf, dtype=r.dtype),
              jnp.zeros((), r.dtype), init_ledger(n_assets), init_ledger(n_assets))
    _, out = jax.lax.scan(body, carry0, (r.T, c, listed.T))
    return {k: jnp.swapaxes(v, 0, 1) for k, v in out.items()}


def panel_growth(model: PooledPolicy, r, c, listed, cost, w_prem) -> jax.Array:
    """Mean scored weekly net log growth over a batch of panels, gates open (training)."""
    def one(r1, c1, l1, wp):
        out = rollout(model, r1, c1, l1, cost, Config(gate=False), wp)
        return (out["growth"] * out["scored"]).sum(), out["scored"].sum()

    total, count = jax.vmap(one)(r, c, listed, w_prem)
    return total.sum() / jnp.maximum(count.sum(), 1)


def runner(model: PooledPolicy, cfg: Config = DEFAULT, *, solo: bool = False):
    """A jitted evaluation function (r, c, listed, cost) -> recorded rollout, in numpy.

    With ``solo`` every asset runs as a panel of its own, so nothing is pooled.
    """
    graphdef, state = nnx.split(model)

    def run(state, r, c, listed, cost):
        m = nnx.merge(graphdef, state)
        if not solo:
            return rollout(m, r, c, listed, cost, cfg, record=True)
        cost = jnp.broadcast_to(jnp.asarray(cost, dtype=r.dtype), (r.shape[0],))
        out = jax.vmap(lambda a, b, k: rollout(m, a[None], c, b[None], k, cfg, record=True))(
            r, listed, cost)
        return {k: v[:, 0] for k, v in out.items()}

    jitted = jax.jit(run)

    def call(r, c, listed, cost):
        out = jitted(state, jnp.asarray(r), jnp.asarray(c), jnp.asarray(listed),
                     jnp.asarray(cost))
        return {k: np.asarray(v) for k, v in out.items()}

    return call


def save(model: PooledPolicy, path: str) -> None:
    jn.save(model, path)


def load(path: str, hidden: tuple[int, ...] = (64, 64, 64)) -> PooledPolicy:
    model = PooledPolicy(hidden, rngs=nnx.Rngs(0))
    leaves, treedef = jax.tree_util.tree_flatten_with_path(nnx.state(model))
    with np.load(path) as saved:
        values = [jnp.asarray(saved[jn._key(p)]) for p, _ in leaves]
    nnx.update(model, jax.tree_util.tree_unflatten(treedef, values))
    return model
