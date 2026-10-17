"""Simulated markets for training the deep parity network.

Each path draws its own dynamics from a broad prior. About 40% of paths have no predictable
direction: their excess returns are symmetric about zero. Every other path switches on each of
six components with probability one half:

- a constant premium;
- a persistent hidden drift (AR(1));
- autocorrelation of normalised returns;
- two-state regimes with opposite drifts;
- a drift tied to the volatility state;
- mean reversion of the price.

30% of those paths redraw every parameter at a random week in their last two thirds. Volatility
is a log AR(1), shocks are Student t, and some paths have jumps. Drifts are in units of the
week's volatility. The repository's benchmark generators are not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NULL_SHARE = 0.4
COMPONENT_ON = 0.5
BREAK_SHARE = 0.3
JUMP_SHARE = 0.3
MAX_DRIFT = 0.5


@dataclass
class WorldBatch:
    logret: np.ndarray
    cash: np.ndarray
    null: np.ndarray


def _components(rng: np.random.Generator, active: np.ndarray) -> dict[str, np.ndarray]:
    """Drift parameters for each path; zero where a path or component is off."""
    b = active.size

    def on():
        return active & (rng.random(b) < COMPONENT_ON)

    theta = np.exp(rng.uniform(np.log(0.002), np.log(0.05), b))
    return {
        "premium": np.where(on(), rng.normal(0.0, 0.05, b), 0.0),
        "hidden": np.where(on(), rng.uniform(0.0, 0.25, b), 0.0),
        "rho": rng.uniform(0.85, 0.995, b),
        "auto": np.where(on(), rng.uniform(-0.15, 0.15, b), 0.0),
        "regime": np.where(on(), rng.uniform(0.0, 0.2, b), 0.0),
        "stay": rng.uniform(0.95, 0.995, b),
        "vol_link": np.where(on(), rng.uniform(-0.15, 0.15, b), 0.0),
        "revert": np.where(on(), theta, 0.0),
    }


def simulate_world(n_paths: int, n_steps: int, seed: int) -> WorldBatch:
    """Weekly log returns of the risky asset and of cash for ``n_paths`` simulated markets."""
    rng = np.random.default_rng(seed)
    b = n_paths
    cash = rng.uniform(0.0, 0.001, b)
    sigma0 = np.exp(rng.uniform(np.log(0.004), np.log(0.15), b))
    phi_h = rng.uniform(0.85, 0.995, b)
    s_h = rng.uniform(0.0, 0.6, b)
    df = rng.uniform(3.0, 30.0, b)
    null = rng.random(b) < NULL_SHARE
    jumps = rng.random(b) < JUMP_SHARE
    jump_rate = np.exp(rng.uniform(np.log(0.002), np.log(0.03), b))
    jump_mu = np.where(null, 0.0, rng.uniform(-4.0, 1.0, b))
    jump_sd = rng.uniform(0.5, 3.0, b)
    comp = _components(rng, ~null)
    breaks = ~null & (rng.random(b) < BREAK_SHARE)
    break_at = np.where(breaks, rng.integers(n_steps // 3, n_steps, b), -1)

    h = s_h * rng.standard_normal(b)
    hidden = rng.standard_normal(b)
    state = np.where(rng.random(b) < 0.5, 1.0, -1.0)
    dev = np.zeros(b)
    prev_z = np.zeros(b)
    x = np.zeros((b, n_steps))
    t_scale = np.sqrt(df / (df - 2.0))
    for t in range(n_steps):
        hit = break_at == t
        if hit.any():
            fresh = _components(rng, hit)
            for key, val in fresh.items():
                comp[key] = np.where(hit, val, comp[key])
        h = phi_h * h + s_h * np.sqrt(1.0 - phi_h ** 2) * rng.standard_normal(b)
        sigma = sigma0 * np.exp(h)
        rho = comp["rho"]
        hidden = rho * hidden + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(b)
        state = np.where(rng.random(b) > comp["stay"], -state, state)
        drift = (comp["premium"] + comp["hidden"] * hidden + comp["auto"] * prev_z
                 + comp["regime"] * state + comp["vol_link"] * h / (s_h + 0.05)
                 - comp["revert"] * dev / sigma)
        drift = np.clip(drift, -MAX_DRIFT, MAX_DRIFT)
        shock = rng.standard_t(df) / t_scale
        jump = np.where(jumps & (rng.random(b) < jump_rate),
                        rng.normal(jump_mu, jump_sd), 0.0)
        x[:, t] = sigma * (drift + shock + jump)
        dev += x[:, t]
        prev_z = np.clip(x[:, t] / sigma, -4.0, 4.0)
    cash_path = np.repeat(cash[:, None], n_steps, axis=1)
    return WorldBatch(logret=cash_path + x, cash=cash_path, null=null)
