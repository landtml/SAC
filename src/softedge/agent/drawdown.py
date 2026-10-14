"""Drawdown-hinged log-growth reward.

The base reward is the per-bar log growth ``R_t``. The log drawdown from the running equity peak,
``u_t = log(P_t / E_t) >= 0``, follows ``u_{t+1} = max(u_t - R_t, 0)`` (``next_drawdown``).
The shaped reward charges time spent below the peak beyond a tolerance:

    r'_t = R_t - dd_penalty * sd * min(max(0, u_{t+1} - dd_tolerance), c),

with the cap ``c = DD_EXCESS_CAP``. ``sd`` is the standard deviation of the base reward in the
training buffer, fixed once before the first gradient step and treated as a constant (no gradient
reaches it). With
``dd_penalty = 0`` the reward is ``R_t`` and nothing here runs.

``u_t`` is part of the state: the observation carries ``log1p(u_t)`` (``obs.drawdown``), and the
replay stores ``u_t`` with each transition. Counterfactual relabelling changes the exposure and the
base reward but keeps the stored ``u_t``; the next drawdown and the shaped reward then follow from
the relabelled ``R_t``, so they match the transition's own dynamics. The peak is taken from the
start of the episode or tape, where ``u`` is zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A wipe-out is not terminal and leaves the book flat, so the same drawdown is charged on every
# later bar. Capping the excess keeps that repeated charge bounded in the critic's Q.
DD_EXCESS_CAP = 1.0
_REWARD_KEYS = {"dd_penalty", "dd_tolerance"}
_OBS_KEYS = {"drawdown"}


def next_drawdown(u: np.ndarray | float, log_growth: np.ndarray | float) -> np.ndarray:
    """Log drawdown from the running equity peak after one step.

    With ``u = log(P / E)`` (peak over equity) and ``log_growth = log(E' / E)``, the new peak is
    ``max(P, E')``, so ``log(P' / E') = max(u - log_growth, 0)``. Only the state ``u`` and the
    step's log growth are needed, so counterfactual replay can recompute it.
    """
    return np.maximum(np.asarray(u, dtype=np.float64) - np.asarray(log_growth, dtype=np.float64),
                      0.0)


@dataclass(frozen=True)
class DrawdownSwitch:
    """Parsed ``reward.dd_penalty``, ``reward.dd_tolerance`` and ``obs.drawdown``."""

    penalty: float = 0.0
    tolerance: float = 0.05
    obs: bool = False


def drawdown_switch(cfg: dict) -> DrawdownSwitch:
    """The switch from a config. Absent sections keep the base agent (penalty 0, no observation)."""
    reward = dict(cfg.get("reward") or {})
    obs = dict(cfg.get("obs") or {})
    if set(reward) - _REWARD_KEYS:
        raise ValueError(f"unknown reward keys: {sorted(set(reward) - _REWARD_KEYS)}")
    if set(obs) - _OBS_KEYS:
        raise ValueError(f"unknown obs keys: {sorted(set(obs) - _OBS_KEYS)}")
    penalty = float(reward.get("dd_penalty", 0.0))
    tolerance = float(reward.get("dd_tolerance", 0.05))
    flag = obs.get("drawdown", False)
    if not isinstance(flag, bool):
        raise ValueError(f"obs.drawdown is a switch: true or false, got {flag!r}")
    if not (np.isfinite(penalty) and penalty >= 0.0):
        raise ValueError(f"reward.dd_penalty must be finite and >= 0, got {penalty}")
    if not (np.isfinite(tolerance) and tolerance >= 0.0):
        raise ValueError(f"reward.dd_tolerance must be finite and >= 0, got {tolerance}")
    if penalty > 0.0 and not flag:
        raise ValueError(
            "reward.dd_penalty > 0 needs obs.drawdown: true, so the policy sees the "
            "drawdown it is charged for"
        )
    return DrawdownSwitch(penalty, tolerance, flag)


class DrawdownShaping:
    """Shaped reward ``R - penalty * sd * min(max(0, u' - tolerance), DD_EXCESS_CAP)``.

    Here ``u' = max(u - R, 0)``.
    """

    def __init__(self, penalty: float, tolerance: float):
        if not penalty > 0.0:
            raise ValueError(f"penalty must be positive, got {penalty}")
        self.penalty = float(penalty)
        self.tolerance = float(tolerance)
        self.sd: float | None = None

    def fit(self, buffer_reward: np.ndarray) -> float:
        """Fix the scale: the standard deviation of the base reward in the training buffer."""
        sd = float(np.std(np.asarray(buffer_reward, dtype=np.float64), ddof=1))
        if not (np.isfinite(sd) and sd > 0.0):
            raise ValueError(f"the buffer reward has no usable scale (sd {sd})")
        self.sd = sd
        return sd

    def shape(self, reward: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Shaped reward for base rewards ``reward`` taken from drawdowns ``u``."""
        if self.sd is None:
            raise ValueError("fit the scale on the training buffer before shaping rewards")
        excess = np.clip(next_drawdown(u, reward) - self.tolerance, 0.0, DD_EXCESS_CAP)
        return np.asarray(reward, dtype=np.float64) - self.penalty * self.sd * excess
