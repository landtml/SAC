"""Simulated panels of related markets for training the pooled parity network, in JAX.

A panel is one market type observed through several assets that share a calendar and a cash
rate. Each panel draws the drift components of ``softedge.parity.jax_world`` once; its assets
share them, scaled per asset by a heterogeneity factor, and the premium of each asset is the
panel's premium plus its own deviation. Shocks load on a common factor, volatility has a common
and an own part, regimes are panel-wide, and crashes can hit every asset in the same week. Some
assets list late and some never list, so the number of assets varies from panel to panel.

40% of panels are null: every excess return is symmetric about zero given the past.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

NULL_SHARE = 0.4
COMPONENT_ON = 0.5
BREAK_SHARE = 0.3
JUMP_SHARE = 0.4
MAX_DRIFT = 0.5
LATE_SHARE = 0.5
ABSENT_SHARE = 0.3


def _components(key, active):
    """Panel-level drift parameters, zero where a panel or component is off. Shapes (panels,)."""
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


def _per_asset(comp, key, het, active, n_assets):
    """Asset-level parameters (panels, assets): the panel's, scaled by exp(het N(0, 0.5))."""
    b = het.shape[0]
    ks = jax.random.split(key, 2)
    scale = jnp.exp(0.5 * het[:, None] * jax.random.normal(ks[0], (b, n_assets)))
    out = {k: v[:, None] * scale for k, v in comp.items() if k not in ("rho", "stay", "premium")}
    out["rho"] = jnp.broadcast_to(comp["rho"][:, None], (b, n_assets))
    out["stay"] = jnp.broadcast_to(comp["stay"][:, None], (b, n_assets))
    own = 0.05 * het[:, None] * jax.random.normal(ks[1], (b, n_assets))
    out["premium"] = comp["premium"][:, None] + jnp.where(active[:, None], own, 0.0)
    return out


@partial(jax.jit, static_argnums=(1, 2, 3))
def simulate(key, n_panels: int, n_assets: int, n_steps: int):
    """Panels of weekly log returns.

    Returns ``(r, cash, listed, null)``: asset log returns (panels, assets, weeks), zero where an
    asset is not listed; cash log returns (panels, weeks); the listing mask (panels, assets,
    weeks); and the null flag (panels,). Asset 0 of every panel lists in week 0.
    """
    b, n = n_panels, n_assets
    k = jax.random.split(key, 32)
    cash = jax.random.uniform(k[0], (b,), maxval=0.001)
    sigma_panel = jnp.exp(jax.random.uniform(k[1], (b,), minval=jnp.log(0.004),
                                             maxval=jnp.log(0.12)))
    sigma0 = sigma_panel[:, None] * jnp.exp(0.4 * jax.random.normal(k[2], (b, n)))
    sigma0 = jnp.clip(sigma0, 0.003, 0.2)
    phi_h = jax.random.uniform(k[3], (b,), minval=0.85, maxval=0.995)
    s_h = jax.random.uniform(k[4], (b,), maxval=0.6)
    vol_common = jax.random.uniform(k[5], (b,))
    factor = jax.random.uniform(k[6], (b,), maxval=0.8)
    df = jax.random.uniform(k[7], (b,), minval=3.0, maxval=30.0)
    null = jax.random.uniform(k[8], (b,)) < NULL_SHARE
    het = jax.random.uniform(k[9], (b,))
    jumps = jax.random.uniform(k[10], (b,)) < JUMP_SHARE
    jump_rate = jnp.exp(jax.random.uniform(k[11], (b,), minval=jnp.log(0.002),
                                           maxval=jnp.log(0.03)))
    contagion = jax.random.uniform(k[12], (b,))
    jump_mu = jnp.where(null, 0.0, jax.random.uniform(k[13], (b,), minval=-4.0, maxval=1.0))
    jump_sd = jax.random.uniform(k[14], (b,), minval=0.5, maxval=3.0)
    hidden_common = jax.random.uniform(k[15], (b,))
    before = _per_asset(_components(k[16], ~null), k[17], het, ~null, n)
    breaks = ~null & (jax.random.uniform(k[18], (b,)) < BREAK_SHARE)
    after = _per_asset(_components(k[19], breaks), k[20], het, breaks, n)
    break_at = jnp.where(breaks, jax.random.randint(k[21], (b,), n_steps // 3, n_steps), n_steps)

    late = jax.random.uniform(k[22], (b, n)) < LATE_SHARE
    start = jnp.where(late, jax.random.randint(k[23], (b, n), 0, n_steps // 2), 0)
    absent = jax.random.uniform(k[24], (b, n)) < ABSENT_SHARE
    start = jnp.where(absent, n_steps, start).at[:, 0].set(0)

    shape = (n_steps, b, n)
    noise = {
        "h": jax.random.normal(k[25], shape),
        "h_common": jax.random.normal(k[26], (n_steps, b)),
        "hidden": jax.random.normal(k[27], shape),
        "hidden_common": jax.random.normal(k[28], (n_steps, b)),
        "switch": jax.random.uniform(k[29], (n_steps, b)),
        "shock": jax.random.t(k[30], df[None, :, None], shape) / jnp.sqrt(
            df / (df - 2.0))[None, :, None],
        "common": jax.random.t(jax.random.fold_in(k[30], 1), df[None, :], (n_steps, b))
        / jnp.sqrt(df / (df - 2.0))[None, :],
        "jump_hit": jax.random.uniform(k[31], shape),
        "jump_common": jax.random.uniform(jax.random.fold_in(k[31], 1), (n_steps, b)),
        "jump_size": jax.random.normal(jax.random.fold_in(k[31], 2), shape),
    }
    kh = jax.random.split(jax.random.fold_in(key, 99), 4)
    h0 = s_h[:, None] * jax.random.normal(kh[0], (b, n))
    hidden0 = jax.random.normal(kh[1], (b, n))
    hc0 = jax.random.normal(kh[2], (b,))
    state0 = jnp.where(jax.random.uniform(kh[3], (b,)) < 0.5, 1.0, -1.0)
    wc = jnp.sqrt(vol_common)[:, None]
    wo = jnp.sqrt(1.0 - vol_common)[:, None]
    hc_w = jnp.sqrt(hidden_common)[:, None]
    ho_w = jnp.sqrt(1.0 - hidden_common)[:, None]
    f_w = jnp.sqrt(factor)[:, None]
    i_w = jnp.sqrt(1.0 - factor)[:, None]

    def body(carry, inp):
        h_own, h_c, hid_own, hid_c, state, dev, prev_z = carry
        t, nz = inp
        comp = jax.tree_util.tree_map(lambda a, c: jnp.where(t >= break_at[:, None], c, a),
                                      before, after)
        innov = jnp.sqrt(1.0 - phi_h ** 2)
        h_own = phi_h[:, None] * h_own + (s_h * innov)[:, None] * nz["h"]
        h_c = phi_h * h_c + s_h * innov * nz["h_common"]
        h = wc * h_c[:, None] + wo * h_own
        sigma = sigma0 * jnp.exp(h)
        rho = comp["rho"]
        hid_own = rho * hid_own + jnp.sqrt(1.0 - rho ** 2) * nz["hidden"]
        hid_c = rho[:, 0] * hid_c + jnp.sqrt(1.0 - rho[:, 0] ** 2) * nz["hidden_common"]
        hidden = hc_w * hid_c[:, None] + ho_w * hid_own
        state = jnp.where(nz["switch"] > comp["stay"][:, 0], -state, state)
        drift = (comp["premium"] + comp["hidden"] * hidden + comp["auto"] * prev_z
                 + comp["regime"] * state[:, None]
                 + comp["vol_link"] * h / (s_h[:, None] + 0.05)
                 - comp["revert"] * dev / sigma)
        drift = jnp.clip(drift, -MAX_DRIFT, MAX_DRIFT)
        shock = f_w * nz["common"][:, None] + i_w * nz["shock"]
        hit_common = nz["jump_common"][:, None] < (jump_rate * contagion)[:, None]
        hit_own = nz["jump_hit"] < (jump_rate * (1.0 - contagion))[:, None]
        jump = jnp.where(jumps[:, None] & (hit_common | hit_own),
                         jump_mu[:, None] + jump_sd[:, None] * nz["jump_size"], 0.0)
        x = sigma * (drift + shock + jump)
        listed = t >= start
        x = jnp.where(listed, x, 0.0)
        z = jnp.clip(x / sigma, -4.0, 4.0)
        return (h_own, h_c, hid_own, hid_c, state, dev + x, z), x

    carry0 = (h0, jnp.zeros(b), hidden0, hc0, state0, jnp.zeros((b, n)), jnp.zeros((b, n)))
    _, x = jax.lax.scan(body, carry0, (jnp.arange(n_steps), noise))
    x = jnp.moveaxis(x, 0, -1)
    listed = jnp.arange(n_steps)[None, None, :] >= start[:, :, None]
    cash_path = jnp.broadcast_to(cash[:, None], (b, n_steps))
    r = jnp.where(listed, cash_path[:, None, :] + x, 0.0)
    return r, cash_path, listed, null
