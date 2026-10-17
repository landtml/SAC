"""Evaluation of the pooled parity network on panels.

``--dev`` runs the development check on sets that earlier registered tests have already used:
the crypto cross-section, the currency panel and the held-out families of the deep test,
grouped into panels of ten. ``--test`` runs the registered evaluation
(docs/preregistration-parity-pooled.md) on the blind synthetic panels and the macro panel. The
script reads only cached files and never downloads.

    python scripts/parity_pooled_study.py --weights <npz> --out <folder> --dev
    python scripts/parity_pooled_study.py --weights <npz> --out <folder> --test --blind <py>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os

import numpy as np
import pandas as pd
import parity_fx as fx
import parity_macro as macro
from scipy import stats

from softedge.parity import ParityConfig
from softedge.parity import backtest as bt
from softedge.parity import jaxnet as jn
from softedge.parity import pooled as pl

CRYPTO_COST = 0.0010
SYNTH_COST = bt.COST
PANEL_SIZE = 10
CUTS = (60, 200, 400, 600, 800, 1000)
MAX_NULL_DEPLOYMENT = 0.02
BLIND_SEEDS = range(1, 101)
NULL_PANELS = 25
PROBE_PANELS = 8
DEV_PANELS = 50
NULL_SEED = 20261017
BLIND_SHA256 = "e3bf00b60b4019861005f381321a8e1c2e2388411cc5b6027f7db8dab4cbea84"
CANDIDATE = "pooled_network"
VARIANTS = {
    "pooled_network": (pl.Config(), False),
    "pooled_ungated": (pl.Config(gate=False), False),
    "pooled_solo": (pl.Config(), True),
    "premium_off": (pl.Config(premium=False), False),
    "direction_off": (pl.Config(direction=False), False),
    "half_crash_governor": (pl.Config(fixed=0.5), False),
    "v1_governor": (pl.Config(governor="v1"), False),
}


# ---------------------------------------------------------------------- policies
def numpy_policies(deep_v1) -> dict:
    """Single-asset policies run by the numpy backtest: simple rules and the earlier networks."""
    makes = dict(bt.SIMPLE_RULES)
    makes["trend_governed"] = lambda: bt.Governed(bt.TrendGate())
    out = {name: (lambda r, c, k, make=make: bt.simulate(r, make(), cash=c, cost=k))
           for name, make in makes.items()}
    out["parity_v1"] = lambda r, c, k: bt.run_parity(r, ParityConfig(), cash=c, cost=k)[0]
    if deep_v1 is not None:
        out["deep_v1"] = lambda r, c, k: bt.simulate(r, jn.JaxPolicy(deep_v1, r.shape[0]),
                                                     cash=c, cost=k)
    return out


def run_numpy(run, R, C, L, cost) -> np.ndarray:
    """Weekly growth (assets, weeks) of a single-asset policy, each asset from its listing week.

    Assets with the same listing span run as one batch. ``C`` is (assets, weeks).
    """
    growth = np.zeros(R.shape)
    spans = {}
    for i in np.flatnonzero(L.any(axis=1)):
        idx = np.flatnonzero(L[i])
        spans.setdefault((idx[0], idx[-1] + 1), []).append(i)
    for (s, e), rows in spans.items():
        if e - s < 2:
            continue
        res = run(R[rows, s:e], C[rows, s:e], cost[rows])
        growth[rows, s:e] = res["growth"]
    return growth


def scored_mask(L) -> np.ndarray:
    return L & (np.cumsum(L, axis=1) > bt.WARMUP)


def breaches(growth, L) -> np.ndarray:
    wealth = np.exp(np.cumsum(np.where(L, growth, 0.0), axis=1))
    peak = np.maximum.accumulate(wealth, axis=1)
    return ((wealth < (1.0 - bt.MAX_DRAWDOWN) * peak) & scored_mask(L)).any(axis=1)


def evaluate_panel(model, deep_v1, R, C, L, cost) -> tuple[dict, dict]:
    """Growth and records of every policy on one panel. ``C`` is the cash path (weeks,)."""
    cost = np.broadcast_to(np.asarray(cost, dtype=float), (R.shape[0],))
    growth, records = {}, {}
    for name, (cfg, solo) in VARIANTS.items():
        out = pl.runner(model, cfg, solo=solo)(R, C, L, cost)
        growth[name] = out["growth"]
        records[name] = out
    C2 = np.broadcast_to(C, R.shape)
    for name, run in numpy_policies(deep_v1).items():
        growth[name] = run_numpy(run, R, C2, L, cost)
    return growth, records


def deployment(rec, scored) -> dict:
    live = scored
    return {"premium": float((rec["w_prem"] > 0)[live].mean()),
            "direction": float((rec["w_dir"] > 0)[live].mean()),
            "either": float(((rec["w_prem"] > 0) | (rec["w_dir"] > 0))[live].mean())}


# ---------------------------------------------------------------------- statistics
def t_interval(x: np.ndarray) -> dict:
    n = x.size
    mean = float(x.mean())
    half = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": int(n)}


def panel_means(growth, scored) -> np.ndarray:
    return bt.ANNUAL * (growth * scored).sum(axis=(1, 2)) / np.maximum(scored.sum(axis=(1, 2)), 1)


def choose_best(table: dict, n_assets: int) -> tuple[str, list]:
    rules = list(bt.SIMPLE_RULES)
    feasible = [p for p in rules if table[p]["breaches"] <= bt.FEASIBLE_FRACTION * n_assets]
    if feasible:
        return max(feasible, key=lambda p: table[p]["annual_growth"]), feasible
    return min(rules, key=lambda p: (table[p]["breaches"], -table[p]["annual_growth"])), feasible


# ---------------------------------------------------------------------- sets
def synthetic_set(model, deep_v1, panels) -> dict:
    """Panels (R, C, L) of one family: per-panel paired differences with a t-interval."""
    growth, scored, deployed, null_live = {}, [], [], []
    for R, C, L in panels:
        g, rec = evaluate_panel(model, deep_v1, R, C, L, SYNTH_COST)
        sc = scored_mask(L)
        for k, v in g.items():
            growth.setdefault(k, []).append(v)
        scored.append(sc)
        deployed.append(deployment(rec[CANDIDATE], sc)["either"] * sc.sum())
        null_live.append(sc.sum())
    sc = np.stack(scored)
    listed = np.stack([p[2] for p in panels])
    n_assets = int(listed.any(axis=2).sum())
    table = {}
    for k, v in growth.items():
        v = np.stack(v)
        table[k] = {"annual_growth": float(bt.ANNUAL * (v * sc).sum() / sc.sum()),
                    "breaches": int(sum(breaches(v[i], listed[i]).sum() for i in range(len(v))))}
        growth[k] = v
    best, feasible = choose_best(table, n_assets)
    diffs = {k: t_interval(panel_means(growth[CANDIDATE] - growth[k], sc))
             for k in growth if k != CANDIDATE}
    return {"assets": n_assets, "panels": len(panels), "best_rule": best, "feasible": feasible,
            "table": table, "candidate_minus": diffs,
            "deployed_share": float(sum(deployed) / sum(null_live)),
            "_diff_vs_best": panel_means(growth[CANDIDATE] - growth[best], sc),
            "_diff_vs_solo": panel_means(growth[CANDIDATE] - growth["pooled_solo"], sc)}


def real_set(model, deep_v1, R, C, L, cost, names, calendar) -> dict:
    g, rec = evaluate_panel(model, deep_v1, R, C, L, cost)
    sc = scored_mask(L)
    table = {k: {"annual_growth": float(bt.ANNUAL * (v * sc).sum() / sc.sum()),
                 "breaches": int(breaches(v, L).sum())} for k, v in g.items()}
    best, feasible = choose_best(table, len(names))

    def series(v, i):
        return pd.Series(v[i, sc[i]], index=calendar[sc[i]])

    diffs = {k: fx.pooled({n: series(g[CANDIDATE] - g[k], i) for i, n in enumerate(names)},
                          calendar) for k in g if k != CANDIDATE}
    per_asset = {n: {k: {"annual_growth": float(bt.ANNUAL * g[k][i, sc[i]].mean()),
                         "breach": bool(breaches(g[k][i:i + 1], L[i:i + 1])[0])}
                     for k in g} for i, n in enumerate(names)}
    return {"series": len(names), "best_rule": best, "feasible": feasible, "table": table,
            "candidate_minus": diffs, "deployment": deployment(rec[CANDIDATE], sc),
            "per_series": per_asset}


def scramble_probe(model, R, C, L) -> bool:
    run = pl.runner(model)
    base = run(R, C, L, 0.0005)
    for cut in CUTS:
        if cut >= R.shape[1] - 2:
            continue
        rng = np.random.default_rng(cut)
        flips = np.where(rng.random(R.shape) < 0.5, -1.0, 1.0)
        x = R - C[None, :]
        other = R.copy()
        other[:, cut + 1:] = (C[None, :] + flips * x)[:, cut + 1:]
        other = np.where(L, other, 0.0)
        out = run(other, C, L, 0.0005)
        if not np.array_equal(base["target"][:, :cut + 1], out["target"][:, :cut + 1]):
            return False
    return True


def sign_null(R, C, L, rng) -> np.ndarray:
    """The panel with every week's excess returns multiplied by one common random sign."""
    s = np.where(rng.random(R.shape[1]) < 0.5, -1.0, 1.0)
    return np.where(L, C[None, :] + s[None, :] * (R - C[None, :]), 0.0)


def null_deployment(model, panels) -> float:
    rng = np.random.default_rng(NULL_SEED)
    run = pl.runner(model)
    deployed, weeks = 0.0, 0.0
    for R, C, L in panels:
        out = run(sign_null(R, C, L, rng), C, L, 0.0005)
        live = out["scored"]
        deployed += ((out["w_prem"] > 0) | (out["w_dir"] > 0))[live].sum()
        weeks += live.sum()
    return float(deployed / weeks)


# ---------------------------------------------------------------------- data
def frames_panel(frames: dict):
    names = sorted(frames)
    cal = pd.DatetimeIndex(sorted({d for f in frames.values() for d in f.index}))
    R = np.zeros((len(names), len(cal)))
    L = np.zeros(R.shape, dtype=bool)
    cash = pd.Series(np.nan, index=cal)
    for i, n in enumerate(names):
        f = frames[n]
        pos = cal.get_indexer(f.index)
        R[i, pos] = f["risky"].to_numpy()
        L[i, pos] = True
        cash.iloc[pos] = f["cash"].to_numpy()
    return R, cash.ffill().bfill().to_numpy(), L, names, cal


def load_blind(path: str):
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != BLIND_SHA256:
        raise RuntimeError(f"blind panel module hash {digest} is not the registered one")
    spec = importlib.util.spec_from_file_location("blind_panels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blind_panels(module, family: str, seeds) -> list:
    out = []
    for seed in seeds:
        logret, cash = module.panel(family, seed, n_assets=PANEL_SIZE, n_weeks=1040)
        L = np.isfinite(logret)
        out.append((np.where(L, logret, 0.0), np.asarray(cash, dtype=float), L))
    return out


def tape_panels(path: str) -> list:
    tapes = np.load(path)
    n = tapes.shape[0] // PANEL_SIZE
    cash = np.full(tapes.shape[1], bt.CASH)
    return [(t, cash, np.ones(t.shape, dtype=bool))
            for t in tapes[: n * PANEL_SIZE].reshape(n, PANEL_SIZE, -1)]


# ---------------------------------------------------------------------- main
def dev(model, deep_v1, args) -> dict:
    tapes = args.dev_tapes
    out = {}
    for fam in ("regime_switching", "ar_returns", "ou_mean_reverting"):
        path = os.path.join(tapes, f"{fam}_30000_31000.npy")
        res = synthetic_set(model, deep_v1, tape_panels(path)[:DEV_PANELS])
        res.pop("_diff_vs_best")
        res.pop("_diff_vs_solo")
        out[fam] = res
    R, C, L, names, cal = frames_panel(pd.read_pickle(os.path.join(tapes, "crypto_weekly.pkl")))
    out["crypto"] = real_set(model, deep_v1, R, C, L, CRYPTO_COST, names, cal)
    frames = {c: fx.weekly_panel(c, os.path.join(args.data, "fred")) for c in fx.CURRENCIES}
    R, C, L, names, cal = frames_panel(frames)
    cost = np.array([fx.CURRENCIES[n][3] for n in names])
    out["currencies"] = real_set(model, deep_v1, R, C, L, cost, names, cal)
    return out


def test(model, deep_v1, args) -> dict:
    module = load_blind(args.blind)
    out = {"blind_sha256": BLIND_SHA256, "families": {}}
    pooled_diff, solo_diff, breach_ok, probes, null_sets = [], [], {}, {}, []
    for fam in module.FAMILIES:
        panels = blind_panels(module, fam, BLIND_SEEDS)
        res = synthetic_set(model, deep_v1, panels)
        pooled_diff.append(res.pop("_diff_vs_best"))
        solo_diff.append(res.pop("_diff_vs_solo"))
        breach_ok[fam] = res["table"][CANDIDATE]["breaches"] <= bt.FEASIBLE_FRACTION * res["assets"]
        probes[fam] = all(scramble_probe(model, *p) for p in panels[:PROBE_PANELS])
        null_sets += panels[:NULL_PANELS]
        out["families"][fam] = res
    frames = {n: macro.weekly_panel(n, os.path.join(args.data, "fred")) for n in macro.SERIES}
    R, C, L, names, cal = frames_panel(frames)
    cost = np.array([macro.COSTS[n] for n in names])
    out["macro"] = real_set(model, deep_v1, R, C, L, cost, names, cal)
    probes["macro"] = scramble_probe(model, R, C, L)
    null_sets.append((R, C, L))
    out["null_deployed_share"] = null_deployment(model, null_sets)

    p1 = t_interval(np.concatenate(pooled_diff))
    mc = out["macro"]
    best = mc["best_rule"]
    p2 = {"best_rule": best, "difference": mc["candidate_minus"][best],
          "candidate_breaches": mc["table"][CANDIDATE]["breaches"],
          "rule_breaches": mc["table"][best]["breaches"]}
    p2["passed"] = bool(p2["difference"]["low"] > 0
                        and p2["candidate_breaches"] <= p2["rule_breaches"])
    p3 = t_interval(np.concatenate(solo_diff))
    out["claims"] = {
        "P1": {**p1, "breaches_ok": breach_ok,
               "passed": bool(p1["low"] > 0 and all(breach_ok.values()))},
        "P2": p2,
        "P3": {**p3, "passed": bool(p3["low"] > 0)},
        "P4": {"null_deployed_share": out["null_deployed_share"], "scramble_probe": probes,
               "passed": bool(out["null_deployed_share"] <= MAX_NULL_DEPLOYMENT
                              and all(probes.values()))},
    }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--deep-v1", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=os.environ.get("SOFTEDGE_DATA") or "data")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true")
    mode.add_argument("--test", action="store_true")
    parser.add_argument("--dev-tapes", default=None)
    parser.add_argument("--blind", default=None)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    with open(args.weights, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    model = pl.load(args.weights)
    deep_v1 = jn.load(args.deep_v1) if args.deep_v1 else None
    out = {"weights_sha256": digest}
    out.update(dev(model, deep_v1, args) if args.dev else test(model, deep_v1, args))
    name = "dev_results.json" if args.dev else "test_results.json"
    with open(os.path.join(args.out, name), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out.get("claims", {}), indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
