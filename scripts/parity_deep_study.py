"""Registered evaluation of the deep parity network trained in JAX.

The registration is docs/preregistration-parity-deep.md. The script runs the frozen JAX network
(the candidate) and its diagnostics on held-out synthetic families, on the families the first
parity network saw, on a null family, on the Binance spot cross-section (weekly, every cached
coin with two years of returns) and on the currency panel of ``parity_fx.py``. It reads only
cached files and never downloads.

    python scripts/parity_deep_study.py --jax-weights <npz> --numpy-weights <npz> --out <folder>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import warnings
import zipfile

import numpy as np
import pandas as pd
import parity_fx as fx
from scipy import stats

from softedge.data import bench, binance, crypto_xs, synthetic
from softedge.parity import ParityConfig
from softedge.parity import backtest as bt
from softedge.parity import jaxnet as jn
from softedge.parity.deep_network import DeepConfig, DeepParityNetwork, load_model

N_WEEKS = 1040
BAR = "1W"
HELD_OUT = ("regime_switching", "ar_returns", "ou_mean_reverting")
HELD_OUT_SEEDS = range(30000, 31000)
KNOWN = ("noisy_prior", "threshold_signal", "planted_signal", "jump_garch",
         "nonlinear_regime_market", "break_market", "gbm")
KNOWN_SEEDS = range(40000, 40256)
NULL_SEEDS = range(50000, 50256)
BENCH = ("noisy_prior", "threshold_signal", "break_market", "jump_garch")
CRYPTO_COST = 0.0010
MIN_WEEKS = 104
CUTS = (60, 200, 400, 600, 800, 1000)
PROBE_TAPES = 32
MAX_NULL_DEPLOYMENT = 0.02
_MONTH = re.compile(r"-1d-(\d{4}-\d{2})\.zip$")


# ---------------------------------------------------------------------- data
def synthetic_tape(args) -> np.ndarray:
    family, seed = args
    symbol = f"{family}-{seed}"
    if family == "gbm":
        frame = synthetic.gbm(N_WEEKS, mu=0.0, sigma=0.01, seed=seed, symbol=symbol, bar=BAR)
    elif family in BENCH:
        frame = getattr(bench, family)(N_WEEKS, seed=seed, symbol=symbol, bar=BAR)
    else:
        frame = getattr(synthetic, family)(N_WEEKS, seed=seed, symbol=symbol, bar=BAR)
    return np.log(frame["close"].to_numpy(dtype=float) / frame["open"].to_numpy(dtype=float))


def load_synthetic(family: str, seeds: range, cache: str) -> np.ndarray:
    path = os.path.join(cache, f"{family}_{seeds.start}_{seeds.stop}.npy")
    if not os.path.exists(path):
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor() as pool:
            rows = list(pool.map(synthetic_tape, [(family, s) for s in seeds], chunksize=16))
        np.save(path, np.stack(rows))
    return np.load(path)


def cached_daily_closes(root: str) -> dict[str, pd.Series]:
    """Daily closes of every cached USDT spot coin, read from the cached monthly zips only."""
    klines = os.path.join(root, "binance", "spot", "monthly", "klines")
    closes = {}
    for sym in crypto_xs.universe(os.listdir(klines)):
        folder = os.path.join(klines, sym, "1d")
        if not os.path.isdir(folder):
            continue
        parts = []
        for name in sorted(os.listdir(folder)):
            if _MONTH.search(name):
                with open(os.path.join(folder, name), "rb") as fh:
                    with zipfile.ZipFile(io.BytesIO(fh.read())) as zf:
                        parts.append(binance._parse_klines(zf.read(zf.namelist()[0])))
        if not parts:
            continue
        bars = binance._clean(pd.concat(parts).sort_index())
        if len(bars) < 2:
            continue
        for name, part in crypto_xs._split_reused(sym, bars).items():
            closes[name] = part["close"]
    return closes


def crypto_weekly(root: str, cache: str) -> dict[str, pd.DataFrame]:
    """Weekly log returns per coin (weeks end on Sunday, UTC) with the DTB3 cash return.

    A week takes the coin's last daily close in it. A week with no close carries the last close
    forward, so its return is zero and the move is credited when trading resumes. Coins with
    fewer than ``MIN_WEEKS`` weekly returns are dropped.
    """
    path = os.path.join(cache, "crypto_weekly.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    usd = fx.read_series(os.path.join(root, "fred"), fx.USD_CASH)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        closes = cached_daily_closes(root)
    for name, close in closes.items():
        idx = close.index.tz_convert(None) if close.index.tz is not None else close.index
        daily = pd.Series(close.to_numpy(dtype=float), index=idx.normalize())
        daily = daily[daily > 0]
        if daily.size < 2:
            continue
        weekly = daily.resample("W-SUN").last()
        gap = weekly.isna()
        weekly = weekly.ffill()
        ret = np.log(weekly).diff().iloc[1:]
        if ret.size < MIN_WEEKS:
            continue
        start = ret.index - pd.Timedelta(days=7)
        cash = np.log1p(fx.as_of(usd, start).to_numpy() / 100.0 * 7.0 / 365.0)
        out[name] = pd.DataFrame({"risky": ret.to_numpy(), "cash": cash,
                                  "gap": gap.iloc[1:].to_numpy()}, index=ret.index)
    pd.to_pickle(out, path)
    return out


# ---------------------------------------------------------------------- policies
def policy_runs(models, *, with_rules: bool = True) -> dict:
    """Policies as callables of (risky paths, cash, cost) returning (result, network or None).

    ``models`` holds the frozen JAX network (``jax``) and the numpy forecaster (``numpy``).
    """
    def jax_policy(trace=False, **kw):
        def run(r, c, k):
            net = jn.JaxPolicy(models["jax"], r.shape[0], trace=trace, **kw)
            return bt.simulate(r, net, cash=c, cost=k), net
        return run

    def numpy_policy(r, c, k):
        net = DeepParityNetwork(r.shape[0], models["numpy"], DeepConfig())
        return bt.simulate(r, net, cash=c, cost=k), None

    runs = {
        "deep_network": jax_policy(trace=True),
        "deep_ungated": jax_policy(gate=False),
        "deep_direction_off": jax_policy(direction=False),
        "numpy_forecaster": numpy_policy,
        "parity_v1": lambda r, c, k: (bt.run_parity(r, ParityConfig(), cash=c, cost=k)[0], None),
    }
    if with_rules:
        for name, make in bt.SIMPLE_RULES.items():
            runs[name] = lambda r, c, k, make=make: (bt.simulate(r, make(), cash=c, cost=k), None)
        runs["trend_governed"] = lambda r, c, k: (
            bt.simulate(r, bt.Governed(bt.TrendGate()), cash=c, cost=k), None)
    return runs


def deployment(net) -> float:
    w = np.stack(net.trace["weight"], axis=1)
    return float((w > 0).mean())


# ---------------------------------------------------------------------- synthetic
def tape_differences(res, ref) -> np.ndarray:
    return bt.ANNUAL * (res["growth"][:, bt.WARMUP:] - ref["growth"][:, bt.WARMUP:]).mean(axis=1)


def t_interval(diff: np.ndarray) -> dict:
    n = diff.size
    mean = float(diff.mean())
    half = float(stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / math.sqrt(n))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": int(n)}


def scramble(models, r: np.ndarray, cash=None, cost=bt.COST) -> bool:
    model = models["jax"]
    sample = r[:PROBE_TAPES]
    base = bt.simulate(sample, jn.JaxPolicy(model, sample.shape[0]), cash=cash, cost=cost)
    for t0 in CUTS:
        if t0 >= sample.shape[1] - 2:
            continue
        rng = np.random.default_rng(t0)
        flips = np.where(rng.random((sample.shape[0], sample.shape[1] - t0 - 1)) < 0.5, -1, 1)
        other_r = sample.copy()
        other_r[:, t0 + 1:] = flips * sample[:, t0 + 1:]
        other = bt.simulate(other_r, jn.JaxPolicy(model, sample.shape[0]), cash=cash, cost=cost)
        if not np.array_equal(base["target"][:, : t0 + 1], other["target"][:, : t0 + 1]):
            return False
    return True


def synthetic_role(models, families, seeds, cache) -> dict:
    out = {}
    for fam in families:
        r = load_synthetic(fam, seeds, cache)
        results, nets = {}, {}
        for name, run in policy_runs(models).items():
            results[name], nets[name] = run(r, None, bt.COST)
        rules = {k: results[k] for k in bt.SIMPLE_RULES}
        best_name, table = bt.best_rule(rules)
        row = {"best_rule": best_name, "rule_table": table, "policies": {},
               "deep_deployed_share": deployment(nets["deep_network"]),
               "scramble_probe": scramble(models, r)}
        for name, res in results.items():
            row["policies"][name] = {
                "annual_growth": bt.ANNUAL * bt.mean_growth(res),
                "breaches": int(res["breach"].sum()),
                "vs_best": t_interval(tape_differences(res, results[best_name])),
                "deep_minus_this": t_interval(tape_differences(results["deep_network"], res))}
        row["_diff_vs_best"] = tape_differences(results["deep_network"], results[best_name])
        out[fam] = row
    return out


# ---------------------------------------------------------------------- real panels
def real_panel(models, panel: dict, cost_of) -> dict:
    """Every policy on every series of a real panel, with pooled comparisons."""
    runs = policy_runs(models)
    per, growth, deployed = {}, {}, {}
    for name, frame in panel.items():
        r = frame["risky"].to_numpy()[None, :]
        c = frame["cash"].to_numpy()
        per[name] = {}
        for pol, run in runs.items():
            res, net = run(r, c, cost_of(name))
            per[name][pol] = fx.metrics(res, c)
            growth.setdefault(pol, {})[name] = pd.Series(res["growth"][0, bt.WARMUP:],
                                                         index=frame.index[bt.WARMUP:])
            if net is not None and pol == "deep_network":
                deployed[name] = deployment(net)
    calendar = pd.DatetimeIndex(sorted({d for s in growth["deep_network"].values()
                                        for d in s.index}))
    pooled_growth = {p: bt.ANNUAL * float(np.concatenate([s.to_numpy() for s in v.values()]).mean())
                     for p, v in growth.items()}
    breaches = {p: sum(per[s][p]["breach"] for s in panel) for p in runs}
    limit = bt.FEASIBLE_FRACTION * len(panel)
    feasible = [p for p in bt.SIMPLE_RULES if breaches[p] <= limit]
    if feasible:
        best = max(feasible, key=lambda p: pooled_growth[p])
    else:
        best = min(bt.SIMPLE_RULES, key=lambda p: (breaches[p], -pooled_growth[p]))

    def diff(a, b):
        return fx.pooled({s: growth[a][s] - growth[b][s] for s in panel}, calendar)

    comparisons = {p: diff("deep_network", p) for p in runs if p != "deep_network"}
    return {"series": len(panel), "best_rule": best, "feasible_rules": feasible,
            "breaches": breaches, "pooled_annual_log_growth": pooled_growth,
            "deep_minus": comparisons, "deep_deployed_share": deployed, "per_series": per}


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jax-weights", required=True)
    parser.add_argument("--numpy-weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=os.environ.get("SOFTEDGE_DATA") or "data")
    args = parser.parse_args(argv)
    cache = os.path.join(args.out, "tapes")
    os.makedirs(cache, exist_ok=True)
    digests = {}
    for name, path in (("jax", args.jax_weights), ("numpy", args.numpy_weights)):
        with open(path, "rb") as fh:
            digests[name] = hashlib.sha256(fh.read()).hexdigest()
    models = {"jax": jn.load(args.jax_weights), "numpy": load_model(args.numpy_weights)}
    out = {"weights_sha256": digests}

    held = synthetic_role(models, HELD_OUT, HELD_OUT_SEEDS, cache)
    pooled_diff = np.concatenate([held[f].pop("_diff_vs_best") for f in HELD_OUT])
    out["held_out"] = held
    known = synthetic_role(models, KNOWN, KNOWN_SEEDS, cache)
    for f in KNOWN:
        known[f].pop("_diff_vs_best")
    out["known"] = known
    null_r = load_synthetic("gbm", NULL_SEEDS, cache)
    null_net = jn.JaxPolicy(models["jax"], null_r.shape[0], trace=True)
    bt.simulate(null_r, null_net)
    out["null_deployed_share"] = deployment(null_net)

    crypto = crypto_weekly(args.data, cache)
    out["crypto"] = real_panel(models, crypto, lambda _: CRYPTO_COST)
    longest = sorted(crypto, key=lambda s: -len(crypto[s]))[:PROBE_TAPES]
    crypto_probe = all(
        scramble(models, crypto[s]["risky"].to_numpy()[None, :], crypto[s]["cash"].to_numpy(),
                 CRYPTO_COST) for s in longest)
    fx_panel = {c: fx.weekly_panel(c, os.path.join(args.data, "fred")) for c in fx.CURRENCIES}
    out["fx"] = real_panel(models, fx_panel, lambda c: fx.CURRENCIES[c][3])

    dl1 = t_interval(pooled_diff)
    dl1_breach_ok = all(held[f]["policies"]["deep_network"]["breaches"]
                        <= bt.FEASIBLE_FRACTION * len(HELD_OUT_SEEDS) for f in HELD_OUT)
    cr = out["crypto"]
    best = cr["best_rule"]
    dl2 = {"best_rule": best, "difference": cr["deep_minus"][best],
           "deep_breaches": cr["breaches"]["deep_network"], "rule_breaches": cr["breaches"][best]}
    dl2["passed"] = bool(dl2["difference"]["mean"] > 0 and dl2["difference"]["low"] > 0
                         and dl2["deep_breaches"] <= dl2["rule_breaches"])
    dl3 = {"difference": cr["deep_minus"]["parity_v1"],
           "passed": bool(cr["deep_minus"]["parity_v1"]["low"] > 0)}
    probes = {f: held[f]["scramble_probe"] for f in HELD_OUT}
    probes["crypto"] = crypto_probe
    dl4 = {"null_deployed_share": out["null_deployed_share"], "scramble_probe": probes,
           "passed": bool(out["null_deployed_share"] <= MAX_NULL_DEPLOYMENT
                          and all(probes.values()))}
    out["claims"] = {"DL1": {**dl1, "breaches_ok": dl1_breach_ok,
                             "passed": bool(dl1["mean"] > 0 and dl1["low"] > 0 and dl1_breach_ok)},
                     "DL2": dl2, "DL3": dl3, "DL4": dl4}
    with open(os.path.join(args.out, "deep_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out["claims"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
