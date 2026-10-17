"""Deep parity network: a pretrained odd network reads a market's history and forecasts direction.

The network's weights are fitted once, on simulated markets (``softedge.parity.world``), and then
frozen. It adapts to a new market only through its inputs: a fixed bank of causal memories of
the market's own history, at time scales from one week to five years. That is in-context
learning: the weights encode how to read a history, and the history says what this market does.

The decision layer is the one of the parity network (``softedge.parity.network``): a default
exposure of one half, the forecast deployed through a sign-test evidence ledger, a Kelly tilt,
and the drawdown governor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from softedge.parity.deep import OddNet

ODD_SCALES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SQUARE_SCALES = (4, 16, 64)
RELIABILITY_SCALES = (104, 416)
RELIABILITY_MEMORIES = (1, 4, 16, 64)
N_ODD = len(ODD_SCALES) + 1
N_EVEN = len(SQUARE_SCALES) + 2 + len(RELIABILITY_SCALES) * len(RELIABILITY_MEMORIES) + 1
FEATURE_CLIP = 10.0


def _norm(a: np.ndarray) -> np.ndarray:
    """Factor that gives an EWMA of unit-variance white noise unit variance."""
    return np.sqrt((2.0 - a) / a)


class MarketMemory:
    """Causal memories of a batch of excess-return paths, split into odd and even inputs.

    Under the sign flip of every excess return, the odd inputs change sign and the even inputs do
    not. Odd: EWMAs of the normalised return z at 1 to 256 weeks, and the t-statistic of its
    running mean. Even: EWMAs of z^2 - 1 at 4, 16 and 64 weeks; two log volatility ratios; the
    running correlation, at 104 and 416 weeks, between this week's z and last week's odd memories
    at 1, 4, 16 and 64 weeks; and the log length of the history.
    """

    def __init__(self, n_paths: int, *, vol_lambda: float = 0.94, vol_seed: int = 13):
        b = n_paths
        self.vol_lambda = vol_lambda
        self.vol_seed = vol_seed
        self.t = -1
        self.seed_x: list[np.ndarray] = []
        self.var = np.full(b, np.nan)
        self.var_mid = np.full(b, np.nan)
        self.var_long = np.full(b, np.nan)
        self.a_odd = 1.0 / np.array(ODD_SCALES, dtype=float)
        self.a_sq = 1.0 / np.array(SQUARE_SCALES, dtype=float)
        self.a_rel = 1.0 / np.array(RELIABILITY_SCALES, dtype=float)
        self.m = np.zeros((b, len(ODD_SCALES)))
        self.q = np.zeros((b, len(SQUARE_SCALES)))
        self.rel = np.zeros((b, len(RELIABILITY_SCALES), len(RELIABILITY_MEMORIES)))
        self.rel_idx = [ODD_SCALES.index(s) for s in RELIABILITY_MEMORIES]
        self.z_sum = np.zeros(b)
        self.n = 0
        self.ready = False

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-12))

    def observe(self, x: np.ndarray) -> np.ndarray | None:
        """Ingest one week of excess log returns. Returns that week's z once volatility exists."""
        self.t += 1
        x = np.asarray(x, dtype=float)
        if self.t + 1 < self.vol_seed:
            self.seed_x.append(x)
            return None
        if self.t + 1 == self.vol_seed:
            self.seed_x.append(x)
            hist = np.stack(self.seed_x, axis=1)
            self.var = np.var(hist, axis=1, ddof=1)
            self.var_mid = self.var.copy()
            self.var_long = self.var.copy()
            self.seed_x = []
            return None
        z = np.clip(x / self.sigma, -FEATURE_CLIP, FEATURE_CLIP)
        lam = self.vol_lambda
        self.var = lam * self.var + (1.0 - lam) * x * x
        self.var_mid = (1.0 - 1.0 / 52.0) * self.var_mid + x * x / 52.0
        self.var_long = (1.0 - 1.0 / 260.0) * self.var_long + x * x / 260.0
        prev = self.m[:, self.rel_idx] * _norm(self.a_odd[self.rel_idx])
        self.rel = ((1.0 - self.a_rel)[None, :, None] * self.rel
                    + self.a_rel[None, :, None] * (z[:, None, None] * prev[:, None, :]))
        self.m = (1.0 - self.a_odd) * self.m + self.a_odd * z[:, None]
        sq = np.minimum(z * z, 25.0) - 1.0
        self.q = (1.0 - self.a_sq) * self.q + self.a_sq * sq[:, None]
        self.z_sum += z
        self.n += 1
        self.ready = True
        return z

    def inputs(self) -> np.ndarray:
        """The (paths, N_ODD + N_EVEN) input matrix after the last observed week."""
        b = self.m.shape[0]
        if not self.ready:
            return np.zeros((b, N_ODD + N_EVEN))
        odd = np.concatenate((self.m * _norm(self.a_odd),
                              (self.z_sum / math.sqrt(self.n))[:, None]), axis=1)
        sig_mid = np.sqrt(np.maximum(self.var_mid, 1e-24))
        sig_long = np.sqrt(np.maximum(self.var_long, 1e-24))
        vol = np.stack((np.log(self.sigma / sig_mid), np.log(sig_mid / sig_long)), axis=1)
        even = np.concatenate((
            self.q * (_norm(self.a_sq) / math.sqrt(2.0)),
            vol,
            (self.rel * _norm(self.a_rel)[None, :, None]).reshape(b, -1),
            np.full((b, 1), math.log1p(self.n / 52.0)),
        ), axis=1)
        return np.clip(np.concatenate((odd, even), axis=1), -FEATURE_CLIP, FEATURE_CLIP)


class EvidenceLedger:
    """One sign-test ledger per path, with the rule of the parity network's ledgers.

    Bet b = clip(forecast / (2 x RMS of past forecasts), -1, 1), settled on clip(z, -2, 2) / 2.
    The fraction is clip(sum u / (sum u^2 + 1), 0, 0.5) from settled bets, the evidence a CUSUM
    of log capital, and the weight 1 - E0 / E above E0 = log(1 / alpha).
    """

    def __init__(self, n_paths: int, *, alpha: float = 0.1, min_history: int = 13):
        self.threshold = math.log(1.0 / alpha)
        self.min_history = min_history
        self.evidence = np.zeros(n_paths)
        self.log_capital = np.zeros(n_paths)
        self.max_log_capital = np.zeros(n_paths)
        self.bet = np.zeros(n_paths)
        self.frac = np.zeros(n_paths)
        self.u_sum = np.zeros(n_paths)
        self.u_sq = np.zeros(n_paths)
        self.pred_sq = np.zeros(n_paths)
        self.n_pred = 0
        self.n_settled = 0
        self.active = False

    def settle(self, z: np.ndarray) -> None:
        if not self.active:
            return
        u = self.bet * np.clip(z, -2.0, 2.0) / 2.0
        step = np.log1p(self.frac * u)
        self.log_capital += step
        self.max_log_capital = np.maximum(self.max_log_capital, self.log_capital)
        self.evidence = np.maximum(0.0, self.evidence + step)
        self.u_sum += u
        self.u_sq += u * u
        self.n_settled += 1

    def place(self, forecast: np.ndarray) -> None:
        if self.n_pred >= self.min_history:
            scale = 2.0 * np.sqrt(self.pred_sq / self.n_pred)
            ok = scale > 0.0
            self.bet = np.where(ok, np.clip(forecast / np.where(ok, scale, 1.0), -1.0, 1.0), 0.0)
            self.active = True
        else:
            self.bet = np.zeros_like(forecast)
        if self.n_settled >= self.min_history:
            self.frac = np.clip(self.u_sum / (self.u_sq + 1.0), 0.0, 0.5)
        else:
            self.frac = np.zeros_like(forecast)
        self.pred_sq += forecast * forecast
        self.n_pred += 1

    def weight(self) -> np.ndarray:
        e = self.evidence
        return np.where(e > self.threshold, 1.0 - self.threshold / np.maximum(e, self.threshold),
                        0.0)


@dataclass(frozen=True)
class DeepConfig:
    """Decision constants. Defaults are the registered values."""

    cash: float = 0.0005
    alpha: float = 0.1
    kelly: float = 0.5
    default_exposure: float = 0.5
    floor: float = 0.78
    multiplier: float = 4.0
    direction: bool = True
    gate: bool = True
    governor: bool = True


class DeepParityNetwork:
    """The policy: memory, frozen odd network, ledger, Kelly tilt and governor."""

    def __init__(self, n_paths: int, model: OddNet, config: DeepConfig | None = None,
                 *, trace: bool = False):
        self.cfg = config or DeepConfig()
        self.model = model
        self.memory = MarketMemory(n_paths)
        self.ledger = EvidenceLedger(n_paths, alpha=self.cfg.alpha)
        self.forecast = np.zeros(n_paths)
        self.trace = {"weight": [], "forecast": [], "evidence": []} if trace else None

    def observe(self, logret, cash=None) -> None:
        c = self.cfg.cash if cash is None else np.asarray(cash, dtype=float)
        z = self.memory.observe(np.asarray(logret, dtype=float) - c)
        if z is None:
            return
        self.ledger.settle(z)
        self.forecast = self.model.predict(self.memory.inputs())
        self.ledger.place(self.forecast)

    def target(self, wealth, peak) -> np.ndarray:
        cfg = self.cfg
        wealth = np.asarray(wealth, dtype=float)
        peak = np.asarray(peak, dtype=float)
        w = self.ledger.weight() if cfg.gate else np.ones_like(self.forecast)
        z_dep = w * self.forecast if cfg.direction else np.zeros_like(self.forecast)
        e_star = np.clip(cfg.default_exposure + cfg.kelly * z_dep / self.memory.sigma, 0.0, 1.0)
        if not self.memory.ready:
            e_star = np.full(wealth.shape, cfg.default_exposure)
        if cfg.governor:
            cap = np.clip(cfg.multiplier * (wealth - cfg.floor * peak) / wealth, 0.0, 1.0)
            e_star = np.minimum(e_star, cap)
        if self.trace is not None:
            self.trace["weight"].append(w.copy())
            self.trace["forecast"].append(self.forecast.copy())
            self.trace["evidence"].append(self.ledger.evidence.copy())
        return e_star


def load_model(path) -> OddNet:
    with np.load(path) as state:
        return OddNet.from_state(state)
