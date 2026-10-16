"""Registered evaluation of the parity network.

The registration is docs/preregistration-parity-network.md. The script builds the registered
tapes, runs the simple rules, the network, its ablations and its sensitivity configs, and writes
JSON results and a trial ledger under ``--out``.

    python scripts/parity_study.py --out <folder>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np

from softedge.data import bench, synthetic
from softedge.parity import ParityConfig
from softedge.parity import backtest as bt

N_WEEKS = 1040
BAR = "1W"
BENCH = ("noisy_prior", "threshold_signal", "break_market", "jump_garch")
ROLES = {
    "design": (("noisy_prior", "threshold_signal", "planted_signal"), (0, 256)),
    "selection": (("jump_garch", "nonlinear_regime_market"), (1000, 1256)),
    "null": (("gbm",), (5000, 5256)),
    "test": (("break_market",), (20000, 22000)),
}
CUTS = (60, 200, 400, 600, 800, 1000)
PROBE_TAPES = 32
SENSITIVITY_ROLES = ("design", "test")
MAX_NULL_DEPLOYMENT = 0.02
MIN_GAIN = 0.005


def make_tape(args) -> np.ndarray:
    family, seed = args
    symbol = f"{family}-{seed}"
    if family == "gbm":
        frame = synthetic.gbm(N_WEEKS, mu=0.0, sigma=0.01, seed=seed, symbol=symbol, bar=BAR)
    elif family in BENCH:
        frame = getattr(bench, family)(N_WEEKS, seed=seed, symbol=symbol, bar=BAR)
    else:
        frame = getattr(synthetic, family)(N_WEEKS, seed=seed, symbol=symbol, bar=BAR)
    return np.log(frame["close"].to_numpy(dtype=float) / frame["open"].to_numpy(dtype=float))


def load_family(family: str, seeds: range, cache: str) -> np.ndarray:
    path = os.path.join(cache, f"{family}_{seeds.start}_{seeds.stop}.npy")
    if os.path.exists(path):
        return np.load(path)
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(make_tape, [(family, s) for s in seeds], chunksize=16))
    arr = np.stack(rows)
    np.save(path, arr)
    return arr


def ablations() -> dict:
    base = ParityConfig()
    return {
        "direction_off": replace(base, direction=False),
        "gate_off": replace(base, gate=False),
        "governor_off": replace(base, governor=False),
        "default_one": replace(base, default_exposure=1.0),
    }


def sensitivity() -> dict:
    base = ParityConfig()
    vol_half_life = math.log(0.5) / math.log(base.vol_lambda)
    out = {}
    for k, tag in ((0.5, "half"), (2.0, "double")):
        out[f"kappa_{tag}"] = replace(base, kelly=base.kelly * k)
        out[f"alpha_{tag}"] = replace(base, alpha=base.alpha * k)
        out[f"ridge_{tag}"] = replace(base, ridge=base.ridge * k)
        out[f"half_lives_{tag}"] = replace(base, half_lives=tuple(h * k for h in base.half_lives))
        out[f"vol_half_life_{tag}"] = replace(base, vol_lambda=0.5 ** (1.0 / (vol_half_life * k)))
        out[f"multiplier_{tag}"] = replace(base, multiplier=base.multiplier * k)
    return out


def describe(res: dict, family_best: dict, role_best: dict) -> dict:
    n = int(res["breach"].size)
    breaches = int(res["breach"].sum())
    held = res["held"][:, bt.WARMUP - 1:]
    return {
        "annual_growth": bt.ANNUAL * bt.mean_growth(res),
        "breaches": breaches,
        "feasible": breaches <= bt.FEASIBLE_FRACTION * n,
        "mean_exposure": float(held[:, 1:].mean()),
        "annual_turnover": float(bt.ANNUAL * np.abs(np.diff(held, axis=1)).mean()),
        "vs_family_best": bt.paired(res, family_best),
        "vs_role_best": bt.paired(res, role_best),
    }


def network_diagnostics(net) -> dict:
    weights = np.stack(net.trace["weight"], axis=1)
    evidence = np.stack(net.trace["evidence"], axis=1)
    return {
        "deployed_share_by_ledger": [float((weights[:, :, k] > 0).mean())
                                     for k in range(weights.shape[2])],
        "momentum_deployed_share": float((weights[:, :, 1:] > 0).any(axis=2).mean()),
        "final_evidence_mean": [float(v) for v in evidence[:, -1, :].mean(axis=0)],
        "final_weight_mean": [float(v) for v in weights[:, -1, :].mean(axis=0)],
    }


def scramble_probe(logrets: np.ndarray, config: ParityConfig) -> dict:
    sample = logrets[:PROBE_TAPES]
    base = bt.run_parity(sample, config)[0]
    failures = []
    for t0 in CUTS:
        rng = np.random.default_rng(t0)
        flips = np.where(rng.random((sample.shape[0], sample.shape[1] - t0 - 1)) < 0.5, -1.0, 1.0)
        scrambled = sample.copy()
        scrambled[:, t0 + 1:] = flips * sample[:, t0 + 1:]
        other = bt.run_parity(scrambled, config)[0]
        same = (np.array_equal(base["target"][:, : t0 + 1], other["target"][:, : t0 + 1])
                and np.array_equal(base["held"][:, : t0 + 2], other["held"][:, : t0 + 2]))
        if not same:
            failures.append(t0)
    return {"passed": not failures, "failed_cuts": failures, "tapes": int(sample.shape[0])}


def run_role(role: str, cache: str, ledger) -> dict:
    families, (start, stop) = ROLES[role]
    data = {f: load_family(f, range(start, stop), cache) for f in families}
    rules = {f: bt.run_rules(data[f]) for f in families}
    pooled = {name: {"breach": np.concatenate([rules[f][name]["breach"] for f in families]),
                     "growth": np.concatenate([rules[f][name]["growth"] for f in families])}
              for name in bt.SIMPLE_RULES}
    role_best_name, role_table = bt.best_rule(pooled)
    out = {"role": role, "families": {}, "role_best_rule": role_best_name,
           "role_rule_table": role_table}
    for f in families:
        best_name, table = bt.best_rule(rules[f])
        best = rules[f][best_name]
        role_best = rules[f][role_best_name]
        fam = {"seeds": [start, stop], "best_rule": best_name, "rule_table": table,
               "policies": {}}
        for name, res in rules[f].items():
            fam["policies"][name] = describe(res, best, role_best)
        res, net = bt.run_parity(data[f], trace=True)
        fam["policies"]["parity_network"] = describe(res, best, role_best)
        fam["network"] = network_diagnostics(net)
        ledger.write(json.dumps({"role": role, "family": f, "policy": "parity_network",
                                 "candidate": True,
                                 **fam["policies"]["parity_network"]}) + "\n")
        diagnostics = {name: bt.run_parity(data[f], cfg)[0] for name, cfg in ablations().items()}
        diagnostics["trend_governed"] = bt.simulate(data[f], bt.Governed(bt.TrendGate()))
        if role in SENSITIVITY_ROLES:
            for name, cfg in sensitivity().items():
                diagnostics[name] = bt.run_parity(data[f], cfg)[0]
        for name, res in diagnostics.items():
            fam["policies"][name] = describe(res, best, role_best)
            ledger.write(json.dumps({"role": role, "family": f, "policy": name,
                                     "candidate": False, **fam["policies"][name]}) + "\n")
        fam["scramble_probe"] = scramble_probe(data[f], ParityConfig())
        out["families"][f] = fam
    return out


def claims(results: dict) -> dict:
    test = results["test"]["families"]["break_market"]
    net = test["policies"]["parity_network"]
    best = test["rule_table"][test["best_rule"]]
    p1 = {"best_rule": test["best_rule"], "difference": net["vs_family_best"],
          "network_breaches": net["breaches"], "best_rule_breaches": best["breaches"]}
    p1["passed"] = bool(net["vs_family_best"]["mean"] >= MIN_GAIN
                        and net["vs_family_best"]["low"] > 0.0
                        and net["breaches"] <= best["breaches"] and net["feasible"])
    families = {f: fam for r in results.values() for f, fam in r["families"].items()}
    competitors = list(bt.SIMPLE_RULES) + ["trend_governed"]

    def worst(policy):
        rows = [fam["policies"][policy] for fam in families.values()]
        if not all(row["feasible"] for row in rows):
            return -math.inf
        return min(row["vs_family_best"]["mean"] for row in rows)

    scores = {p: worst(p) for p in ["parity_network"] + competitors}
    p2 = {"worst_family_score": scores,
          "passed": bool(scores["parity_network"] > -math.inf
                         and all(scores["parity_network"] > scores[c] for c in competitors))}
    null = results["null"]["families"]["gbm"]
    probes = {f: fam["scramble_probe"]["passed"] for f, fam in families.items()}
    p3 = {"momentum_deployed_share": null["network"]["momentum_deployed_share"],
          "scramble_probe": probes}
    p3["passed"] = bool(p3["momentum_deployed_share"] <= MAX_NULL_DEPLOYMENT
                        and all(probes.values()))
    return {"P1": p1, "P2": p2, "P3": p3}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="folder for tapes and results")
    parser.add_argument("--roles", nargs="+", default=list(ROLES), choices=list(ROLES))
    args = parser.parse_args(argv)
    cache = os.path.join(args.out, "tapes")
    os.makedirs(cache, exist_ok=True)
    results = {}
    with open(os.path.join(args.out, "trials.jsonl"), "a", encoding="utf-8") as ledger:
        for role in args.roles:
            results[role] = run_role(role, cache, ledger)
            with open(os.path.join(args.out, f"{role}.json"), "w", encoding="utf-8") as fh:
                json.dump(results[role], fh, indent=1, default=float)
    summary = {"config": asdict(ParityConfig()), "threshold": ParityConfig().threshold}
    if set(ROLES) <= set(results):
        summary["claims"] = claims(results)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(json.dumps(summary.get("claims", {}), indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
