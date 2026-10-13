"""Config-driven experiment: train seeds, select on validation, evaluate once on test.

The test period is touched exactly once per run, after checkpoint selection, and every
configuration evaluated is appended to ``trials.jsonl`` in the output root so the deflated Sharpe
ratio can count the trials honestly.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from sactrader.agent.sac import TQCSAC, SACConfig
from sactrader.env.vec_env import EnvConfig, TapeSet, rollout
from sactrader.eval import baselines
from sactrader.eval.report import per_tape_table, rollout_exposures, summary_table, to_markdown
from sactrader.pipeline import Period, SplitSpec, load_frames, split_tapes, to_tapes
from sactrader.tape import DEFAULT_COSTS, MarketTape
from sactrader.train import TrainConfig, train


def _dataclass_from(cls, overrides: dict | None, base=None):
    base = base if base is not None else cls()
    if not overrides:
        return base
    known = {f.name for f in fields(cls)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return replace(base, **overrides)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_train_config(cfg: dict, seed: int) -> TrainConfig:
    env = _dataclass_from(EnvConfig, cfg.get("env"))
    sac = _dataclass_from(SACConfig, cfg.get("sac"))
    tc = _dataclass_from(TrainConfig, cfg.get("train"))
    return replace(tc, env=env, sac=sac, seed=seed)


def build_tapes(cfg: dict) -> dict[str, list[MarketTape]]:
    data = cfg.get("data", {})
    frames = load_frames(cfg["universe"], data.get("start", "2020-01"), data.get("end", "2026-09"),
                         symbols=cfg.get("symbols"), max_markets=data.get("max_markets", 300),
                         min_volume=data.get("min_volume", 1e6))
    overrides = cfg.get("costs") or {}

    def costs(frame):
        return replace(DEFAULT_COSTS[frame.attrs["kind"]], **overrides)

    tapes = to_tapes(frames, costs)
    s = cfg["split"]
    split = SplitSpec(Period(*s["train"]), Period(*s["val"]), Period(*s["test"]))
    return split_tapes(tapes, split)


def ensemble_policy(agents: list[TQCSAC]):
    def act(obs: np.ndarray) -> np.ndarray:
        return np.mean([a.act(obs, deterministic=True) for a in agents], axis=0)
    return act


def run_experiment(cfg: dict, out_root: str | Path = "runs") -> dict:
    name = cfg["name"]
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    parts = build_tapes(cfg)
    if not parts["train"] or not parts["val"] or not parts["test"]:
        raise ValueError({k: len(v) for k, v in parts.items()})
    seeds = cfg.get("seeds", [0])
    agents, histories, feature_names = [], {}, None
    t0 = time.time()
    for seed in seeds:
        tc = build_train_config(cfg, seed)
        res = train(parts["train"], parts["val"], tc,
                    log=lambda r, s=seed: _append_jsonl(out / "train_log.jsonl", {"seed": s, **r}))
        torch.save(res.best_state, out / f"agent_seed{seed}.pt")
        agent = TQCSAC.from_state_dict(res.best_state)
        agents.append(agent)
        histories[seed] = {"best_val_sharpe": res.best_score, "reward_scale": res.reward_scale}
        feature_names = res.feature_names

    window = build_train_config(cfg, seeds[0]).env.window
    start_row = window - 1
    test = parts["test"]
    test_ts = TapeSet(test, window, feature_names)
    runs = {}
    for seed, agent in zip(seeds, agents, strict=True):
        runs[f"sac_seed{seed}"] = rollout(test_ts, lambda o, a=agent: a.act(o, deterministic=True))
    if len(agents) > 1:
        runs["sac_ensemble"] = rollout(test_ts, ensemble_policy(agents))
    runs["flat"] = [rollout_exposures(tp, baselines.flat(tp), start_row) for tp in test]
    runs["buy_and_hold"] = [rollout_exposures(tp, baselines.buy_and_hold(tp), start_row) for tp in test]
    if test[0].kind != "binary":
        trend_kw = cfg.get("trend", {})
        runs["trend"] = [rollout_exposures(tp, baselines.vol_target_trend(tp, **trend_kw), start_row)
                         for tp in test]

    reference = cfg.get("reference", "buy_and_hold")
    n_trials = _count_trials(Path(out_root)) + 1
    summary = summary_table(runs, test, start_row, reference=reference, n_trials=n_trials)
    per_tape = per_tape_table(runs, test, start_row)
    result = {
        "name": name,
        "seeds": histories,
        "minutes": round((time.time() - t0) / 60, 1),
        "n_trials": n_trials,
        "n_tapes": {k: len(v) for k, v in parts.items()},
        "summary": json.loads(summary.set_index("strategy").to_json(orient="index")),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "report.md").write_text(
        f"# {name}\n\nTest tapes: {len(test)}. Trials so far: {n_trials}.\n\n## Pooled\n\n"
        + to_markdown(summary) + "\n\n## Per tape\n\n" + to_markdown(per_tape) + "\n"
    )
    _append_jsonl(Path(out_root) / "trials.jsonl", {"name": name, "config": copy.deepcopy(cfg),
                                                     "summary": result["summary"]})
    return result


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def _count_trials(root: Path) -> int:
    path = root / "trials.jsonl"
    return sum(1 for _ in open(path)) if path.exists() else 0
