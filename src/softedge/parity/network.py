"""Parity network: a sign-symmetric online learner for one risky asset against cash.

Every quantity the network computes is either even or odd under the sign flip of the excess
log return x = r - c. Even quantities (volatility, the size of moves, the betting fractions)
are unchanged by the flip. Odd quantities (momentum, the premium, every direction forecast)
change sign with it.

- Even part. When excess log returns are symmetric about zero, the growth-optimal exposure is
  exactly one half, whatever their volatility or tails. That is the default exposure.
- Odd part. Direction learners forecast the next vol-normalised excess return z from odd
  features. A forecast moves the exposure away from one half only as far as its evidence
  ledger allows.
- Evidence ledger. Each learner places a bounded bet b * psi(z) on its own forecast. psi is odd,
  so under the null that the sign of the next return is a fair coin given the past and its
  size, every bet is fair. The ledger's capital is then a test martingale for any volatility
  or tail shape. The ledger keeps a CUSUM of its log capital, so evidence earned before a
  break decays once the learner starts losing.
- Governor. The exposure is capped by a cushion on the network's own high-water mark.

All arrays carry a leading batch axis of paths, which share one time index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParityConfig:
    """Constants of the network. Defaults are the registered values."""

    cash: float = 0.0005
    scales: tuple[int, ...] = (1, 4, 13, 52)
    vol_lambda: float = 0.94
    vol_seed: int = 13
    slow_window: int = 52
    standardise_after: int = 52
    half_lives: tuple[float, ...] = (52.0, 260.0, math.inf)
    ridge: float = 52.0
    premium_ridge: float = 52.0
    outcome_clip: float = 2.0
    bet_scale: float = 2.0
    min_history: int = 13
    max_fraction: float = 0.5
    alpha: float = 0.1
    kelly: float = 0.5
    default_exposure: float = 0.5
    floor: float = 0.78
    multiplier: float = 4.0
    direction: bool = True
    gate: bool = True
    governor: bool = True

    @property
    def n_ledgers(self) -> int:
        return 1 + len(self.half_lives)

    @property
    def threshold(self) -> float:
        """Evidence in nats below which a learner is not deployed: log(K / alpha)."""
        return math.log(self.n_ledgers / self.alpha)


class ParityNetwork:
    """Online network over a batch of return paths.

    Call ``observe`` once per closed week with that week's log returns, then ``target`` for
    the exposure to hold over the next week. Nothing else about the paths is ever passed in.
    """

    def __init__(self, n_paths: int, n_steps: int, config: ParityConfig | None = None,
                 *, trace: bool = False):
        cfg = config or ParityConfig()
        self.cfg = cfg
        b = n_paths
        s = len(cfg.scales)
        self.d = 2 * s
        k = len(cfg.half_lives)
        self.t = -1
        self.x = np.zeros((b, n_steps))
        self.var = np.full(b, np.nan)
        self.alpha_ewma = np.array([1.0 / tau for tau in cfg.scales])
        self.norm = np.sqrt((2.0 - self.alpha_ewma) / self.alpha_ewma)
        self.m = np.zeros((b, s))
        self.v_sum = np.zeros(b)
        self.v_sq = np.zeros(b)
        self.v_n = 0
        self.vmod = np.zeros(b)
        self.forget = np.array([1.0 if math.isinf(h) else 0.5 ** (1.0 / h)
                                for h in cfg.half_lives])
        self.A = np.zeros((k, b, self.d, self.d))
        self.bv = np.zeros((k, b, self.d))
        self.phi_prev = None
        self.z_sum = np.zeros(b)
        self.z_n = 0
        n_led = cfg.n_ledgers
        self.pred = np.zeros((b, n_led))
        self.evidence = np.zeros((b, n_led))
        self.log_capital = np.zeros((b, n_led))
        self.max_log_capital = np.zeros((b, n_led))
        self.u_sum = np.zeros((b, n_led))
        self.u_sq = np.zeros((b, n_led))
        self.n_settled = 0
        self.pred_sq = np.zeros((b, n_led))
        self.n_pred = 0
        self.bet = np.zeros((b, n_led))
        self.frac = np.zeros((b, n_led))
        self.bets_active = False
        self.trace = {"evidence": [], "weight": [], "z_deployed": [], "e_star": [],
                      "cap": []} if trace else None

    # ------------------------------------------------------------------ observation
    def observe(self, logret, cash=None) -> None:
        """Ingest one closed week. ``cash`` is that week's cash log return; default ``cfg.cash``."""
        cfg = self.cfg
        self.t += 1
        t = self.t
        c = cfg.cash if cash is None else np.asarray(cash, dtype=float)
        x = np.asarray(logret, dtype=float) - c
        self.x[:, t] = x
        if t + 1 == cfg.vol_seed:
            self.var = np.var(self.x[:, : t + 1], axis=1, ddof=1)
        elif t + 1 > cfg.vol_seed:
            z = np.clip(x / np.sqrt(np.maximum(self.var, 1e-12)), -10.0, 10.0)
            self.var = cfg.vol_lambda * self.var + (1.0 - cfg.vol_lambda) * x * x
            self._ingest(z)

    def _ingest(self, z: np.ndarray) -> None:
        cfg = self.cfg
        self._settle(z)
        if self.phi_prev is not None:
            p = self.phi_prev
            self.A = (self.forget[:, None, None, None] * self.A
                      + p[None, :, :, None] * p[None, :, None, :])
            self.bv = self.forget[:, None, None] * self.bv + p[None, :, :] * z[None, :, None]
        self.m = (1.0 - self.alpha_ewma) * self.m + self.alpha_ewma * z[:, None]
        self._update_modulator()
        odd = self.m * self.norm
        phi = np.concatenate((odd, self.vmod[:, None] * odd), axis=1)
        self.z_sum += z
        self.z_n += 1
        self.pred[:, 0] = self.z_sum / (self.z_n + cfg.premium_ridge)
        eye = cfg.ridge * np.eye(self.d)
        beta = np.linalg.solve(self.A + eye, self.bv[..., None])[..., 0]
        self.pred[:, 1:] = np.einsum("kbd,bd->bk", beta, phi)
        self._place_bets()
        self.phi_prev = phi

    def _update_modulator(self) -> None:
        cfg = self.cfg
        t = self.t
        if t + 1 < cfg.slow_window:
            return
        slow = np.std(self.x[:, t + 1 - cfg.slow_window: t + 1], axis=1, ddof=1)
        v = np.log(np.sqrt(np.maximum(self.var, 1e-24)) / np.maximum(slow, 1e-12))
        if self.v_n >= cfg.standardise_after:
            mean = self.v_sum / self.v_n
            sd = np.sqrt(np.maximum(self.v_sq / self.v_n - mean * mean, 0.0))
            ok = sd > 0.0
            self.vmod = np.where(ok, np.clip((v - mean) / np.where(ok, sd, 1.0), -3.0, 3.0), 0.0)
        self.v_sum += v
        self.v_sq += v * v
        self.v_n += 1

    # ------------------------------------------------------------------ ledgers
    def _settle(self, z: np.ndarray) -> None:
        if not self.bets_active:
            return
        c = self.cfg.outcome_clip
        psi = np.clip(z, -c, c) / c
        u = self.bet * psi[:, None]
        step = np.log1p(self.frac * u)
        self.log_capital += step
        self.max_log_capital = np.maximum(self.max_log_capital, self.log_capital)
        self.evidence = np.maximum(0.0, self.evidence + step)
        self.u_sum += u
        self.u_sq += u * u
        self.n_settled += 1

    def _place_bets(self) -> None:
        cfg = self.cfg
        if self.n_pred >= cfg.min_history:
            scale = cfg.bet_scale * np.sqrt(self.pred_sq / self.n_pred)
            ok = scale > 0.0
            self.bet = np.where(ok, np.clip(self.pred / np.where(ok, scale, 1.0), -1.0, 1.0), 0.0)
            self.bets_active = True
        else:
            self.bet = np.zeros_like(self.pred)
        if self.n_settled >= cfg.min_history:
            self.frac = np.clip(self.u_sum / (self.u_sq + 1.0), 0.0, cfg.max_fraction)
        else:
            self.frac = np.zeros_like(self.pred)
        self.pred_sq += self.pred * self.pred
        self.n_pred += 1

    # ------------------------------------------------------------------ decision
    def weights(self) -> np.ndarray:
        cfg = self.cfg
        if not cfg.gate:
            return np.ones_like(self.evidence)
        e0 = cfg.threshold
        return np.where(self.evidence > e0, 1.0 - e0 / np.maximum(self.evidence, e0), 0.0)

    def deployed_forecast(self) -> np.ndarray:
        if not self.cfg.direction:
            return np.zeros(self.pred.shape[0])
        w = self.weights()
        momentum = w[:, 1:]
        total = np.maximum(1.0, momentum.sum(axis=1))
        return w[:, 0] * self.pred[:, 0] + (momentum * self.pred[:, 1:]).sum(axis=1) / total

    def target(self, wealth, peak) -> np.ndarray:
        """Exposure for the next week, from data up to the last observed week."""
        cfg = self.cfg
        wealth = np.asarray(wealth, dtype=float)
        peak = np.asarray(peak, dtype=float)
        if np.all(np.isnan(self.var)):
            z_dep = np.zeros(wealth.shape)
            e_star = np.full(wealth.shape, cfg.default_exposure)
        else:
            z_dep = self.deployed_forecast()
            sigma = np.sqrt(np.maximum(self.var, 1e-12))
            e_star = np.clip(cfg.default_exposure + cfg.kelly * z_dep / sigma, 0.0, 1.0)
        if cfg.governor:
            cap = np.clip(cfg.multiplier * (wealth - cfg.floor * peak) / wealth, 0.0, 1.0)
        else:
            cap = np.ones(wealth.shape)
        if self.trace is not None:
            self.trace["evidence"].append(self.evidence.copy())
            self.trace["weight"].append(self.weights())
            self.trace["z_deployed"].append(z_dep)
            self.trace["e_star"].append(e_star)
            self.trace["cap"].append(cap)
        return np.minimum(e_star, cap)
