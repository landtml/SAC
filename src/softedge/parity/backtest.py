"""Weekly long-only backtest for one risky asset against cash, batched over paths.

The accounting follows the discovery search registration (docs/preregistration-discovery.md,
Amendments 1 to 3): weeks 1 to 52 are warm-up with exposure 0, the first decision is made at
the close of week 52, a decision is held over the next week through a no-trade band of 0.05,
each unit traded costs 0.0005, and cash earns 0.0005 a week. A path breaches when its wealth
falls more than 25% below its peak in weeks 53 onward.

A policy has two methods: ``observe(logret, cash)`` with the log returns of the risky asset and
of cash in the week just closed, and ``target(wealth, peak)`` for the exposure to hold over the
next week. Real data passes its own weekly cash returns and cost per unit traded; the defaults are
the registered synthetic values.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from softedge.parity.network import ParityConfig, ParityNetwork

WARMUP = 52
FIRST_DECISION = WARMUP - 1
BAND = 0.05
COST = 0.0005
CASH = 0.0005
MAX_DRAWDOWN = 0.25
ANNUAL = 52
FEASIBLE_FRACTION = 0.01


def simulate(logrets, policy, *, cash=None, cost=COST) -> dict:
    """Run a policy on a (paths, weeks) array of weekly log returns.

    ``cash`` holds weekly cash log returns, shaped like ``logrets`` or (weeks,); by default every
    week earns ``CASH``. ``cost`` is the cost per unit traded, a scalar or one value per path.
    """
    r_all = np.asarray(logrets, dtype=float)
    b, n = r_all.shape
    c_all = np.full((b, n), CASH) if cash is None else np.broadcast_to(
        np.asarray(cash, dtype=float), (b, n))
    unit_cost = np.broadcast_to(np.asarray(cost, dtype=float), (b,))
    held = np.zeros((b, n))
    target = np.zeros((b, n))
    wealth = np.ones((b, n))
    peak = np.ones((b, n))
    hi = np.ones(b)
    prev_w = np.ones(b)
    prev_h = np.zeros(b)
    for t in range(n):
        h = held[:, t]
        r = r_all[:, t]
        c = c_all[:, t]
        paid = unit_cost * np.abs(h - prev_h)
        w = prev_w * (1.0 - paid) * (h * np.exp(r) + (1.0 - h) * np.exp(c))
        hi = np.maximum(hi, w)
        wealth[:, t] = w
        peak[:, t] = hi
        policy.observe(r, c)
        if t >= FIRST_DECISION:
            e = np.clip(np.asarray(policy.target(w, hi), dtype=float), 0.0, 1.0)
            target[:, t] = e
        else:
            e = np.zeros(b)
        if t + 1 < n:
            if t + 1 < WARMUP:
                held[:, t + 1] = 0.0
            else:
                held[:, t + 1] = np.where(np.abs(e - h) > BAND, e, h)
        prev_w, prev_h = w, h
    growth = np.empty((b, n))
    growth[:, 0] = np.log(wealth[:, 0])
    growth[:, 1:] = np.log(wealth[:, 1:] / wealth[:, :-1])
    breach = np.any(wealth[:, WARMUP:] < (1.0 - MAX_DRAWDOWN) * peak[:, WARMUP:], axis=1)
    return {"held": held, "target": target, "growth": growth, "breach": breach,
            "wealth": wealth}


# ---------------------------------------------------------------------- benchmark rules
class _History:
    def __init__(self):
        self.cols: list[np.ndarray] = []

    def observe(self, logret, cash=None) -> None:
        self.cols.append(np.asarray(logret, dtype=float))

    def window(self, k: int) -> np.ndarray:
        return np.stack(self.cols[-k:], axis=1)


class BuyAndHold(_History):
    def target(self, wealth, peak):
        return np.ones(np.shape(wealth))


class TrendGate(_History):
    """Exposure 1 when the trailing 10-week log return is positive, else 0."""

    def target(self, wealth, peak):
        return np.where(self.window(10).sum(axis=1) > 0.0, 1.0, 0.0)


class VolTarget(_History):
    """Exposure 0.10 over the trailing 52-week annualised volatility, at most 1."""

    def target(self, wealth, peak):
        vol = math.sqrt(52.0) * np.std(self.window(52), axis=1, ddof=1)
        return np.where(vol > 0.0, np.minimum(1.0, 0.10 / np.where(vol > 0.0, vol, 1.0)), 1.0)


class Cushion(_History):
    """Exposure clip(m (W - f peak) / W, 0, 1): CPPI with f = 0.8, Grossman-Zhou with 0.75."""

    def __init__(self, floor: float, multiplier: float = 2.0):
        super().__init__()
        self.floor = floor
        self.multiplier = multiplier

    def target(self, wealth, peak):
        w = np.asarray(wealth, dtype=float)
        return np.clip(self.multiplier * (w - self.floor * np.asarray(peak)) / w, 0.0, 1.0)


class LogOptimal(_History):
    """Exposure clip(0.5 mean / variance, 0, 1) on the trailing 52 weeks."""

    def target(self, wealth, peak):
        window = self.window(52)
        mean = window.mean(axis=1)
        var = np.var(window, axis=1, ddof=1)
        ok = var > 0.0
        return np.where(ok, np.clip(0.5 * mean / np.where(ok, var, 1.0), 0.0, 1.0), 0.0)


class Governed:
    """A policy whose exposure is capped by the parity network's governor."""

    def __init__(self, inner, floor: float = 0.78, multiplier: float = 4.0):
        self.inner = inner
        self.floor = floor
        self.multiplier = multiplier

    def observe(self, logret, cash=None) -> None:
        self.inner.observe(logret, cash)

    def target(self, wealth, peak):
        w = np.asarray(wealth, dtype=float)
        cap = np.clip(self.multiplier * (w - self.floor * np.asarray(peak)) / w, 0.0, 1.0)
        return np.minimum(np.asarray(self.inner.target(wealth, peak), dtype=float), cap)


SIMPLE_RULES = {
    "buy_and_hold": BuyAndHold,
    "trend_gate": TrendGate,
    "vol_target": VolTarget,
    "cppi": lambda: Cushion(0.8),
    "grossman_zhou": lambda: Cushion(0.75),
    "log_optimal": LogOptimal,
}


def run_rules(logrets, *, cash=None, cost=COST) -> dict:
    """Every simple rule on the same paths."""
    return {name: simulate(logrets, make(), cash=cash, cost=cost)
            for name, make in SIMPLE_RULES.items()}


def run_parity(logrets, config: ParityConfig | None = None, *, trace: bool = False, cash=None,
               cost=COST):
    """The parity network on the paths. Returns the backtest and the network."""
    arr = np.asarray(logrets, dtype=float)
    net = ParityNetwork(arr.shape[0], arr.shape[1], config, trace=trace)
    return simulate(arr, net, cash=cash, cost=cost), net


# ---------------------------------------------------------------------- scoring
def mean_growth(result) -> float:
    """Mean weekly net log growth over the scored weeks, averaged over paths."""
    return float(result["growth"][:, WARMUP:].mean())


def best_rule(results: dict) -> tuple[str, dict]:
    """The feasible rule with the highest mean growth, and a table of every rule."""
    table = {}
    for name, res in results.items():
        n = res["breach"].size
        breaches = int(res["breach"].sum())
        table[name] = {"breaches": breaches, "feasible": breaches <= FEASIBLE_FRACTION * n,
                       "annual_growth": ANNUAL * mean_growth(res)}
    feasible = [name for name, row in table.items() if row["feasible"]]
    if not feasible:
        raise RuntimeError("no simple rule is feasible on these paths")
    return max(feasible, key=lambda name: table[name]["annual_growth"]), table


def paired(candidate, reference) -> dict:
    """Annualised paired difference in net log growth, with a 95% t-interval over paths."""
    diff = ANNUAL * (candidate["growth"][:, WARMUP:] - reference["growth"][:, WARMUP:]).mean(axis=1)
    n = diff.size
    mean = float(diff.mean())
    half = float(stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": int(n)}
