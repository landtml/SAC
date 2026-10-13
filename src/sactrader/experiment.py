"""Config-driven experiments: walk-forward folds, validation selection, one look at each test window.

A config holds either one ``split`` or a ``walk_forward`` block. Each fold trains every seed on
data before its validation window, keeps the checkpoint with the best validation Sharpe, picks a
no-trade band on validation, and only then runs the test window. Test windows of successive folds
do not overlap, so their results join into one out-of-sample record that the summary statistics
use. Every experiment is appended to ``trials.jsonl`` in the output root so the deflated Sharpe
ratio counts all configurations tried.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from sactrader.agent.sac import TQCSAC, SACConfig
from sactrader.env.vec_env import N_CONTEXT, EnvConfig, TapeSet, rollout
from sactrader.eval import baselines
from sactrader.eval.metrics import sharpe as annual_sharpe
from sactrader.eval.report import (
    per_tape_table,
    pooled_returns,
    rollout_exposures,
    summary_from_pooled,
    to_markdown,
)
from sactrader.eval.stats import dsr_summary, probabilistic_sharpe_ratio
from sactrader.pipeline import (
    Period,
    SplitSpec,
    load_frames,
    resample_frame,
    split_tapes,
    to_tapes,
)
from sactrader.tape import DEFAULT_COSTS, MarketTape
from sactrader.train import TrainConfig, sharpe, train

NO_TRADE_BANDS = (0.0, 0.05, 0.1, 0.2)


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


def build_all_tapes(cfg: dict) -> list[MarketTape]:
    data = cfg.get("data", {})
    frames = load_frames(cfg["universe"], data.get("start", "2020-01"), data.get("end", "2026-09"),
                         symbols=cfg.get("symbols"), max_markets=data.get("max_markets", 300),
                         min_volume=data.get("min_volume", 1e6))
    if data.get("resample"):
        frames = [resample_frame(f, data["resample"]) for f in frames]
    overrides = cfg.get("costs") or {}

    def costs(frame):
        return replace(DEFAULT_COSTS[frame.attrs["kind"]], **overrides)

    return to_tapes(frames, costs)


def make_splits(cfg: dict) -> list[SplitSpec]:
    """One split from ``split``, or consecutive folds from ``walk_forward``.

    ``walk_forward`` keys: ``first_test`` and ``end`` (dates), ``test`` and ``val`` (durations such
    as "180D"), optional ``embargo`` (default "0D") and ``train_start`` (default: all history).
    Fold k tests ``[first_test + k * test, first_test + (k + 1) * test)``, the last fold clipped at
    ``end``; validation ends one embargo before the test window and training one embargo before
    validation.
    """
    if "walk_forward" not in cfg:
        s = cfg["split"]
        return [SplitSpec(Period(*s["train"]), Period(*s["val"]), Period(*s["test"]))]
    wf = cfg["walk_forward"]
    t = pd.Timestamp(wf["first_test"])
    end = pd.Timestamp(wf["end"])
    test, val = pd.Timedelta(wf["test"]), pd.Timedelta(wf["val"])
    emb = pd.Timedelta(wf.get("embargo", "0D"))
    splits = []
    while t < end:
        v0 = t - emb - val
        splits.append(SplitSpec(Period(wf.get("train_start"), str(v0 - emb)),
                                Period(str(v0), str(t - emb)),
                                Period(str(t), str(min(t + test, end)))))
        t = t + test
    return splits


def ensemble_policy(agents: list[TQCSAC]):
    def act(obs: np.ndarray) -> np.ndarray:
        return np.mean([a.act(obs, deterministic=True) for a in agents], axis=0)
    return act


def banded(act_fn, band: float):
    """Keep the current exposure unless the policy's target differs from it by more than
    ``band`` (in units of max leverage). Small rebalances cost spread and fees for little gain."""
    def act(obs: np.ndarray) -> np.ndarray:
        a = np.asarray(act_fn(obs), dtype=np.float64).reshape(-1)
        current = obs[:, -N_CONTEXT].astype(np.float64)
        return np.where(np.abs(a - current) > band, a, current)
    return act


def choose_band(act_fn, val: TapeSet) -> tuple[float, dict[float, float]]:
    """Band with the best mean validation Sharpe; ties go to the smaller band."""
    scores = {}
    for band in NO_TRADE_BANDS:
        out = rollout(val, banded(act_fn, band))
        scores[band] = float(np.mean([sharpe(o["ret"], tp.bars_per_year)
                                      for o, tp in zip(out, val.tapes, strict=True)]))
    best = max(NO_TRADE_BANDS, key=lambda b: (scores[b], -b))
    return best, scores


def _test_runs(cfg: dict, agents: list[TQCSAC], seeds: list[int], parts: dict, window: int,
               feature_names: tuple[str, ...]) -> tuple[dict, dict]:
    start_row = window - 1
    test = parts["test"]
    test_ts = TapeSet(test, window, feature_names)
    runs = {}
    for seed, agent in zip(seeds, agents, strict=True):
        runs[f"sac_seed{seed}"] = rollout(test_ts, lambda o, a=agent: a.act(o, deterministic=True))
    policy = ensemble_policy(agents)
    if len(agents) > 1:
        runs["sac_ensemble"] = rollout(test_ts, policy)
    val_ts = TapeSet(parts["val"], window, feature_names)
    band, band_scores = choose_band(policy, val_ts)
    runs["sac_banded"] = rollout(test_ts, banded(policy, band))
    # Deploy only on validation evidence: probabilistic Sharpe of the banded policy on the pooled
    # validation record. Below the gate the strategy stays flat for this fold.
    gate_psr = float(cfg.get("deploy", {}).get("gate_psr", 0.8))
    val_run = rollout(val_ts, banded(policy, band))
    val_psr = probabilistic_sharpe_ratio(pooled_returns(val_run, parts["val"], start_row).to_numpy())
    deploy = bool(np.isfinite(val_psr) and val_psr >= gate_psr)
    runs["sac_gated"] = runs["sac_banded"] if deploy else [
        rollout_exposures(tp, baselines.flat(tp), start_row) for tp in test]
    runs["flat"] = [rollout_exposures(tp, baselines.flat(tp), start_row) for tp in test]
    runs["buy_and_hold"] = [rollout_exposures(tp, baselines.buy_and_hold(tp), start_row)
                            for tp in test]
    if test[0].kind != "binary":
        trend_kw = cfg.get("trend", {})
        runs["trend"] = [rollout_exposures(tp, baselines.vol_target_trend(tp, **trend_kw), start_row)
                         for tp in test]
    return runs, {"chosen": band, "val_sharpe": band_scores, "val_psr": float(val_psr),
                  "gate_psr": gate_psr, "deployed": deploy}


def run_experiment(cfg: dict, out_root: str | Path = "runs") -> dict:
    if "cpcv" in cfg:
        from sactrader.cpcv_experiment import run_cpcv

        return run_cpcv(cfg, out_root)
    name = cfg["name"]
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    tapes = build_all_tapes(cfg)
    splits = make_splits(cfg)
    seeds = cfg.get("seeds", [0])
    wf = cfg.get("walk_forward", {})
    window = build_train_config(cfg, seeds[0]).env.window
    start_row = window - 1

    t0 = time.time()
    feature_names: tuple[str, ...] | None = None
    prev_states: dict[int, dict] = {}
    reward_scales: dict[int, float] = {}
    all_runs: dict[str, list[dict]] = {}
    all_tapes: list[MarketTape] = []
    folds, per_tape_tables = [], []
    for k, split in enumerate(splits):
        parts = split_tapes(tapes, split)
        if not parts["train"] or not parts["val"] or not parts["test"]:
            folds.append({"fold": k, "skipped": {n: len(v) for n, v in parts.items()}})
            continue
        agents, seed_info = [], {}
        for seed in seeds:
            tc = build_train_config(cfg, seed)
            init = None
            if wf.get("warm_start") and seed in prev_states:
                init = prev_states[seed]
                tc = replace(tc, total_steps=wf.get("fold_steps", tc.total_steps),
                             warmup_steps=min(tc.warmup_steps, 2048),
                             reward_scale=reward_scales[seed])
            res = train(parts["train"], parts["val"], tc, init_state=init,
                        feature_names=feature_names,
                        log=lambda r, s=seed, f=k: _append_jsonl(out / "train_log.jsonl",
                                                                 {"fold": f, "seed": s, **r}))
            feature_names = res.feature_names
            reward_scales[seed] = res.reward_scale
            prev_states[seed] = res.best_state
            torch.save(res.best_state, out / f"agent_fold{k}_seed{seed}.pt")
            agents.append(TQCSAC.from_state_dict(res.best_state))
            seed_info[seed] = {"best_val_sharpe": res.best_score, "reward_scale": res.reward_scale}
        runs, band = _test_runs(cfg, agents, seeds, parts, window, feature_names)
        test = parts["test"]
        fold_sharpe = {}
        for strat, run in runs.items():
            all_runs.setdefault(strat, []).extend(run)
            r = pooled_returns(run, test, start_row).to_numpy()
            fold_sharpe[strat] = float(annual_sharpe(r, np.median([tp.bars_per_year for tp in test])))
        all_tapes.extend(test)
        table = per_tape_table(runs, test, start_row)
        table.insert(0, "fold", k)
        per_tape_tables.append(table)
        folds.append({"fold": k, "test": [split.test.start, split.test.stop], "seeds": seed_info,
                      "no_trade_band": band, "n_tapes": {n: len(v) for n, v in parts.items()},
                      "pooled_sharpe": fold_sharpe})

    if not all_tapes:
        raise ValueError(f"no fold had train, validation and test data: {folds}")
    pooled = {strat: pooled_returns(run, all_tapes, start_row) for strat, run in all_runs.items()}
    bpy = float(np.median([tp.bars_per_year for tp in all_tapes]))
    reference = cfg.get("reference", "buy_and_hold")
    sac_sr = {s: _per_bar_sharpe(series) for s, series in pooled.items() if s.startswith("sac_")}
    n_trials, sr_var = trial_stats(Path(out_root), cfg["universe"], list(sac_sr.values()))
    summary = summary_from_pooled(pooled, bpy, reference, n_trials=n_trials, sr_variance=sr_var)
    fold_table = pd.DataFrame([{"fold": f["fold"], **f["pooled_sharpe"]} for f in folds
                               if "pooled_sharpe" in f])
    result = {
        "name": name,
        "minutes": round((time.time() - t0) / 60, 1),
        "n_trials": n_trials,
        "n_folds": len(fold_table),
        "folds": folds,
        "summary": json.loads(summary.set_index("strategy").to_json(orient="index")),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    pd.concat(pooled, axis=1).to_csv(out / "pooled_returns.csv")
    (out / "report.md").write_text(
        f"# {name}\n\nFolds: {len(fold_table)}. Out-of-sample bars: {len(pooled[reference])}. "
        f"Trials so far: {n_trials}.\n\n## Out-of-sample summary (all folds)\n\n"
        + to_markdown(summary) + "\n\n## Pooled Sharpe per fold\n\n" + to_markdown(fold_table)
        + "\n\n## Per tape\n\n" + to_markdown(pd.concat(per_tape_tables)) + "\n"
    )
    _append_jsonl(Path(out_root) / "trials.jsonl", {
        "name": name, "universe": cfg["universe"], "config": copy.deepcopy(cfg),
        "sac_sr_bar": sac_sr, "n_bars": len(pooled[reference]), "summary": result["summary"]})
    return result


def _per_bar_sharpe(series: pd.Series) -> float:
    r = series.to_numpy()
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def trial_stats(root: Path, universe: str, current: list[float]) -> tuple[int, float | None]:
    """Trial count and cross-trial Sharpe variance for the deflated Sharpe ratio.

    Every SAC variant reported on ``universe`` counts as one trial: each seed, the ensemble and
    the banded and gated versions, across all earlier experiments in ``root`` plus ``current``.
    The variance is that of their per-bar Sharpe ratios (None below two trials, which leaves the
    zero-skill floor).
    """
    srs = list(current)
    path = root / "trials.jsonl"
    if path.exists():
        for line in open(path):
            row = json.loads(line)
            if row.get("universe") == universe:
                srs.extend(row.get("sac_sr_bar", {}).values())
    n = max(len(srs), 1)
    return n, (float(np.var(srs, ddof=1)) if len(srs) > 1 else None)


def recompute_dsr(root: str | Path) -> pd.DataFrame:
    """DSR table for every run under ``root`` that saved pooled returns, with trials and
    cross-trial variance counted per universe over all those runs."""
    root = Path(root)
    runs = []
    for d in sorted(p for p in root.iterdir() if (p / "pooled_returns.csv").exists()):
        cfg = load_config(d / "config.yaml")
        pooled = pd.read_csv(d / "pooled_returns.csv", index_col=0, parse_dates=True)
        runs.append((d.name, cfg, pooled))
    by_universe: dict[str, list[float]] = {}
    for _, cfg, pooled in runs:
        srs = [_per_bar_sharpe(pooled[c].dropna()) for c in pooled.columns if c.startswith("sac_")]
        by_universe.setdefault(cfg["universe"], []).extend(srs)
    rows = []
    for name, cfg, pooled in runs:
        srs = by_universe[cfg["universe"]]
        var = float(np.var(srs, ddof=1)) if len(srs) > 1 else None
        bpy = _bars_per_year(pooled.index)
        for col in pooled.columns:
            d = dsr_summary(pooled[col].dropna().to_numpy(), bpy, len(srs), var)
            rows.append({"run": name, "universe": cfg["universe"], "strategy": col, **d})
    return pd.DataFrame(rows)


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    span = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
    return (len(index) - 1) / span if span > 0 else float("nan")


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


