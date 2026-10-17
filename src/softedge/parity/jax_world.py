"""The training prior of ``softedge.parity.world``, written in JAX.

Same components and parameter ranges as the numpy version: 40% of markets have excess returns
symmetric about zero; the rest switch on each of six drift components with probability one
half, and 30% of those redraw every component at a random week in their last two thirds.
Random draws come from ``jax.random``, the time loop is a ``lax.scan``, and a whole batch is
one jitted call, so training can draw fresh markets at every step.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

NULL_SHARE = 0.4
COMPONENT_ON = 0.5
BREAK_SHARE = 0.3
JUMP_SHARE = 0.3
MAX_DRIFT = 0.5


def _components(key, active):
    """Drift parameters per market; zero where a market or component is off."""
    b = active.shape[0]
    ks = jax.random.split(key, 14)

    def on(k):
        return active & (jax.random.uniform(k, (b,)) < COMPONENT_ON)

    log_theta = jax.random.uniform(ks[0], (b,), minval=jnp.log(0.002), maxval=jnp.log(0.05))
    return {
        "premium": jnp.where(on(ks[1]), 0.05 * jax.random.normal(ks[2], (b,)), 0.0),
        "hidden": jnp.where(on(ks[3]), jax.random.uniform(ks[4], (b,), maxval=0.25), 0.0),
        "rho": jax.random.uniform(ks[5], (b,), minval=0.85, maxval=0.995),
        "auto": jnp.where(on(ks[6]), jax.random.uniform(ks[7], (b,), minval=-0.15, maxval=0.15),
                          0.0),
        "regime": jnp.where(on(ks[8]), jax.random.uniform(ks[9], (b,), maxval=0.2), 0.0),
        "stay": jax.random.uniform(ks[10], (b,), minval=0.95, maxval=0.995),
        "vol_link": jnp.where(on(ks[11]), jax.random.uniform(ks[12], (b,), minval=-0.15,
                                                             maxval=0.15), 0.0),
        "revert": jnp.where(on(ks[13]), jnp.exp(log_theta), 0.0),
    }


@partial(jax.jit, static_argnums=(1, 2))
def simulate(key, n_paths: int, n_steps: int):
    """Weekly log returns of the risky asset and of cash, each shaped (paths, weeks)."""
    b = n_paths
    k = jax.random.split(key, 23)
    cash = jax.random.uniform(k[0], (b,), maxval=0.001)
    sigma0 = jnp.exp(jax.random.uniform(k[1], (b,), minval=jnp.log(0.004), maxval=jnp.log(0.15)))
    phi_h = jax.random.uniform(k[2], (b,), minval=0.85, maxval=0.995)
    s_h = jax.random.uniform(k[3], (b,), maxval=0.6)
    df = jax.random.uniform(k[4], (b,), minval=3.0, maxval=30.0)
    null = jax.random.uniform(k[5], (b,)) < NULL_SHARE
    jumps = jax.random.uniform(k[6], (b,)) < JUMP_SHARE
    jump_rate = jnp.exp(jax.random.uniform(k[7], (b,), minval=jnp.log(0.002),
                                           maxval=jnp.log(0.03)))
    jump_mu = jnp.where(null, 0.0, jax.random.uniform(k[8], (b,), minval=-4.0, maxval=1.0))
    jump_sd = jax.random.uniform(k[9], (b,), minval=0.5, maxval=3.0)
    before = _components(k[10], ~null)
    breaks = ~null & (jax.random.uniform(k[11], (b,)) < BREAK_SHARE)
    after = _components(k[12], breaks)
    break_at = jnp.where(breaks, jax.random.randint(k[13], (b,), n_steps // 3, n_steps), n_steps)

    shape = (n_steps, b)
    noise = {
        "h": jax.random.normal(k[14], shape),
        "hidden": jax.random.normal(k[15], shape),
        "switch": jax.random.uniform(k[16], shape),
        "shock": jax.random.t(k[17], df, shape) / jnp.sqrt(df / (df - 2.0)),
        "jump_hit": jax.random.uniform(k[18], shape),
        "jump_size": jax.random.normal(k[19], shape),
    }
    h0 = s_h * jax.random.normal(k[20], (b,))
    hidden0 = jax.random.normal(k[21], (b,))
    state0 = jnp.where(jax.random.uniform(k[22], (b,)) < 0.5, 1.0, -1.0)

    def body(carry, inp):
        h, hidden, state, dev, prev_z = carry
        t, nz = inp
        comp = jax.tree_util.tree_map(lambda a, c: jnp.where(t >= break_at, c, a), before, after)
        h = phi_h * h + s_h * jnp.sqrt(1.0 - phi_h ** 2) * nz["h"]
        sigma = sigma0 * jnp.exp(h)
        rho = comp["rho"]
        hidden = rho * hidden + jnp.sqrt(1.0 - rho ** 2) * nz["hidden"]
        state = jnp.where(nz["switch"] > comp["stay"], -state, state)
        drift = (comp["premium"] + comp["hidden"] * hidden + comp["auto"] * prev_z
                 + comp["regime"] * state + comp["vol_link"] * h / (s_h + 0.05)
                 - comp["revert"] * dev / sigma)
        drift = jnp.clip(drift, -MAX_DRIFT, MAX_DRIFT)
        jump = jnp.where(jumps & (nz["jump_hit"] < jump_rate),
                         jump_mu + jump_sd * nz["jump_size"], 0.0)
        x = sigma * (drift + nz["shock"] + jump)
        return (h, hidden, state, dev + x, jnp.clip(x / sigma, -4.0, 4.0)), x

    carry0 = (h0, hidden0, state0, jnp.zeros(b), jnp.zeros(b))
    _, x = jax.lax.scan(body, carry0, (jnp.arange(n_steps), noise))
    cash_path = jnp.broadcast_to(cash[:, None], (b, n_steps))
    return cash_path + x.T, cash_path, null
