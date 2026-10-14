"""Ensemble-disagreement shrinkage of the deployed residual.

Each member of a seed ensemble gives a residual ``r_k`` for the bar. Hypothesis, not tested here:
where the members disagree, their mean carries more of the noise in the targets they were all
fitted to. The deployed residual is shrunk by the disagreement:

    rbar = mean_k r_k,   sigma = std_k r_k (population),
    m    = 1 / (1 + (sigma / tau)^2),    deployed = rbar * m.

``tau`` is the median of ``sigma`` over the validation bars of the ensemble's own path, fixed
before the test fold. The exposure-matched control uses the mean of ``m`` over the same bars.
``m`` depends only on the observation at its bar, which is causal. This is
deployment-only: no loss or gradient involves it, and no weight is changed.
"""

from __future__ import annotations

import numpy as np

from softedge.env.vec_env import rollout


def ensemble_stats(agents: list, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation over members of the deterministic residual, (N,)."""
    n = len(obs)
    acts = np.stack([np.asarray(a.act(obs, deterministic=True), dtype=np.float64)
                     .reshape(n, -1)[:, 0] for a in agents])
    return acts.mean(axis=0), acts.std(axis=0)


def shrink_multiplier(sigma: np.ndarray, tau: float) -> np.ndarray:
    """``1 / (1 + (sigma / tau)^2)``: one where the members agree, falling as they disagree."""
    if not tau > 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return 1.0 / (1.0 + (np.asarray(sigma, dtype=np.float64) / tau) ** 2)


def deployed_action(agents: list, obs: np.ndarray, tau: float | None = None,
                    const: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Shrunk residual and its multiplier for one batch of observations.

    ``const`` replaces the disagreement multiplier by a constant (the exposure-matched control).
    """
    rbar, sigma = ensemble_stats(agents, obs)
    if const is not None:
        m = np.full(len(rbar), float(const))
    else:
        m = shrink_multiplier(sigma, tau)
    return rbar * m, m


def validation_tau(agents: list, val_ts) -> float:
    """Median member disagreement over the validation bars, on the ensemble's own path."""
    sigmas = []

    def ensemble(obs: np.ndarray) -> np.ndarray:
        rbar, sigma = ensemble_stats(agents, obs)
        sigmas.append(sigma)
        return rbar

    rollout(val_ts, ensemble)
    tau = float(np.median(np.concatenate(sigmas)))
    if not tau > 0:
        raise ValueError("tau is zero: the members agree on every validation bar, so the shrink "
                         "has no scale")
    return tau


def exposure_matched_policy(agents: list, val_ts, tau: float) -> ShrinkPolicy:
    """Control that holds the multiplier at its mean over the validation bars.

    The constant comes from the shrink's own path on validation, so no test-fold bar enters it.
    """
    shrink = ShrinkPolicy(agents, tau=tau)
    rollout(val_ts, shrink)
    return ShrinkPolicy(agents, const=shrink.mean_multiplier)


class ShrinkPolicy:
    """Rollout policy for the deployed residual. Records the mean multiplier over all calls."""

    def __init__(self, agents: list, tau: float | None = None, const: float | None = None):
        self.agents, self.tau, self.const = agents, tau, const
        self._sum = 0.0
        self._n = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action, m = deployed_action(self.agents, obs, self.tau, self.const)
        self._sum += float(m.sum())
        self._n += m.size
        return action

    @property
    def mean_multiplier(self) -> float:
        if self._n == 0:
            raise ValueError("no bars seen yet")
        return self._sum / self._n


def disagree_shrink_on(deploy_cfg: dict) -> bool:
    """The switch ``deploy.disagree_shrink``: 0 (off, the default) or 1 (on). It has no strength
    to tune, so any other value is an error."""
    value = float(deploy_cfg.get("disagree_shrink", 0.0))
    if value not in (0.0, 1.0):
        raise ValueError(f"deploy.disagree_shrink is a switch: 0 (off) or 1 (on), got {value}")
    return value == 1.0
