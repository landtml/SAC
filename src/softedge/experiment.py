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

from softedge.agent.factory import agent_from_state
from softedge.agent.sac import TQCSAC, SACConfig
from softedge.env.vec_env import EnvConfig, TapeSet, rollout
from softedge.eval import baselines
from softedge.eval.metrics import sharpe as annual_sharpe
from softedge.eval.report import (
    per_tape_table,
    pooled_returns,
    rollout_exposures,
    summary_from_pooled,
    to_markdown,
)
from softedge.eval.stats import (
    deflated_sharpe_ratio,
    dsr_summary,
    null_sr_variance,
    probabilistic_sharpe_ratio,
)
from softedge.features.calibration import CalibConfig
from softedge.pipeline import (
    Period,
    SplitSpec,
    load_frames,
    resample_frame,
    split_tapes,
    to_tapes,
)
from softedge.tape import DEFAULT_COSTS, MarketTape
from softedge.train import TrainConfig, sharpe, train

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
    """Tapes of the config's universe, or of every block in ``sources`` when present.

    A source block takes the keys of a single-universe config (``universe``, ``symbols``,
    ``data``, ``costs``), so one agent can train on several universes with their own costs.
    """
    if "sources" in cfg:
        # The rank is computed across the coins of one frame list. Under sources it would be
        # computed per source, so it is refused rather than silently ranking each source alone.
        if any("xs_funding" in (c.get("data") or {}) for c in [cfg, *cfg["sources"]]):
            raise ValueError("xs_funding ranks the coins of one universe, so it cannot be combined "
                             "with sources; move it to a single-universe config")
        return [tp for src in cfg["sources"] for tp in build_all_tapes(src)]
    data = cfg.get("data", {})
    frames = load_frames(cfg["universe"], data.get("start", "2020-01"), data.get("end", "2026-09"),
                         symbols=cfg.get("symbols"), max_markets=data.get("max_markets", 300),
                         min_volume=data.get("min_volume", 1e6),
                         implied_vol=bool(data.get("implied_vol", False)),
                         premium=bool(data.get("premium", False)),
                         roll_days=data.get("roll_days"), roll_cost=data.get("roll_cost"),
                         family=data.get("family", "all"),
                         seed_offset=int(data.get("seed_offset", 0)))
    if data.get("resample"):
        frames = [resample_frame(f, data["resample"]) for f in frames]
    overrides = cfg.get("costs") or {}

    def costs(frame):
        return replace(DEFAULT_COSTS[frame.attrs["kind"]], **overrides)

    calib = CalibConfig(**data["calib"]) if data.get("calib") else None
    return to_tapes(frames, costs, cross=bool(data.get("cross_features", False)),
                    regime=bool(data.get("regime_features", False)), calib=calib,
                    premia=bool(data.get("premia_features", False)),
                    xs_funding=data.get("xs_funding"))


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


def choose_band(act_fn, val: TapeSet) -> tuple[float, dict[float, float]]:
    """Band with the best mean validation Sharpe; ties go to the smaller band."""
    scores = {}
    for band in NO_TRADE_BANDS:
        out = rollout(val, act_fn, band=band)
        scores[band] = float(np.mean([sharpe(o["ret"], tp.bars_per_year)
                                      for o, tp in zip(out, val.tapes, strict=True)]))
    best = max(NO_TRADE_BANDS, key=lambda b: (scores[b], -b))
    return best, scores


def _test_runs(cfg: dict, agents: list[TQCSAC], seeds: list[int], parts: dict, window: int,
               feature_names: tuple[str, ...], n_val_evals: int = 1) -> tuple[dict, dict]:
    start_row = window - 1
    test = parts["test"]
    action_map = build_train_config(cfg, seeds[0]).env.action_map
    test_ts = TapeSet(test, window, feature_names, action_map)
    runs = {}
    for seed, agent in zip(seeds, agents, strict=True):
        runs[f"sac_seed{seed}"] = rollout(test_ts, lambda o, a=agent: a.act(o, deterministic=True))
    policy = ensemble_policy(agents)
    if len(agents) > 1:
        runs["sac_ensemble"] = rollout(test_ts, policy)
    val_ts = TapeSet(parts["val"], window, feature_names, action_map)
    band, band_scores = choose_band(policy, val_ts)
    runs["sac_banded"] = rollout(test_ts, policy, band=band)
    # Deploy only on validation evidence: probabilistic Sharpe of the banded policy on the pooled
    # validation record. Below the gate the strategy stays flat for this fold. The same validation
    # record picked the checkpoints and the band, so gate "dsr" deflates it by that search:
    # checkpoint evaluations times bands, with the zero-skill variance floor.
    deploy_cfg = cfg.get("deploy", {})
    gate_psr = float(deploy_cfg.get("gate_psr", 0.8))
    val_run = rollout(val_ts, policy, band=band)
    val_r = pooled_returns(val_run, parts["val"], start_row).to_numpy()
    if deploy_cfg.get("gate", "psr") == "dsr":
        n_search = max(1, n_val_evals) * len(NO_TRADE_BANDS)
        val_psr = deflated_sharpe_ratio(val_r, n_search, null_sr_variance(len(val_r)))
    else:
        val_psr = probabilistic_sharpe_ratio(val_r)
    deploy = bool(np.isfinite(val_psr) and val_psr >= gate_psr)
    runs["sac_gated"] = runs["sac_banded"] if deploy else [
        rollout_exposures(tp, baselines.flat(tp), start_row) for tp in test]
    if action_map is not None and action_map.has_prior:
        # The prior alone (zero residual, same sizing): what the policy adds over its prior.
        runs["prior_only"] = rollout(test_ts, lambda o: np.zeros(len(o)))
    runs["flat"] = [rollout_exposures(tp, baselines.flat(tp), start_row) for tp in test]
    runs["buy_and_hold"] = [rollout_exposures(tp, baselines.buy_and_hold(tp), start_row)
                            for tp in test]
    if all("mu_next" in tp.extra for tp in test):
        # Synthetic markets know their expected return: the sign oracle bounds what is learnable.
        runs["oracle"] = [rollout_exposures(tp, np.sign(tp.extra["mu_next"]) * tp.max_leverage,
                                            start_row) for tp in test]
    if test[0].kind == "binary":
        # Holding NO everywhere: the favourite-longshot bias says long shots are overpriced.
        runs["always_no"] = [rollout_exposures(tp, -baselines.buy_and_hold(tp, tp.max_leverage),
                                               start_row) for tp in test]
    else:
        trend_kw = cfg.get("trend", {})
        runs["trend"] = [rollout_exposures(tp, baselines.vol_target_trend(tp, **trend_kw), start_row)
                         for tp in test]
    return runs, {"chosen": band, "val_sharpe": band_scores, "val_psr": float(val_psr),
                  "val_psr_plain": float(probabilistic_sharpe_ratio(val_r)),
                  "gate": deploy_cfg.get("gate", "psr"), "gate_psr": gate_psr, "deployed": deploy}


def run_experiment(cfg: dict, out_root: str | Path = "runs") -> dict:
    if "cpcv" in cfg:
        from softedge.cpcv_experiment import run_cpcv

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
            agents.append(agent_from_state(res.best_state))
            seed_info[seed] = {"best_val_sharpe": res.best_score, "reward_scale": res.reward_scale,
                               "n_val_evals": len(res.history)}
        runs, band = _test_runs(cfg, agents, seeds, parts, window, feature_names,
                                n_val_evals=max(s["n_val_evals"] for s in seed_info.values()))
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
    """DSR table for every saved run under ``root``, deflated by the current trial count.

    Walk-forward and single-split runs (``pooled_returns.csv``) give one row per strategy. CPCV
    runs (``path{j}_returns.csv``) give one row per strategy with the mean over paths of Sharpe,
    PSR and DSR, plus ``dsr_min`` and ``n_paths``. Every SAC strategy of every run is one trial of
    its universe (a CPCV strategy counts once, at its mean per-bar Sharpe across paths); the
    cross-trial variance is that of those per-bar Sharpe ratios. Directories without
    ``config.yaml`` are skipped.
    """
    root = Path(root)
    runs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "config.yaml").exists()):
        cfg = load_config(d / "config.yaml")
        if (d / "pooled_returns.csv").exists():
            pooled = pd.read_csv(d / "pooled_returns.csv", index_col=0, parse_dates=True)
            runs.append((d.name, cfg, "walk_forward", [pooled]))
        else:
            paths = sorted(d.glob("path*_returns.csv"), key=lambda p: int(p.stem[4:-8]))
            if paths:
                frames = [pd.read_csv(p, index_col=0, parse_dates=True) for p in paths]
                runs.append((d.name, cfg, "cpcv", frames))
    by_universe: dict[str, list[float]] = {}
    for _, cfg, _, frames in runs:
        for col in frames[0].columns:
            if col.startswith("sac_"):
                sr = np.mean([_per_bar_sharpe(f[col].dropna()) for f in frames])
                by_universe.setdefault(cfg["universe"], []).append(float(sr))
    rows = []
    for name, cfg, mode, frames in runs:
        srs = by_universe.get(cfg["universe"], [])
        var = float(np.var(srs, ddof=1)) if len(srs) > 1 else None
        n = max(len(srs), 1)
        for col in frames[0].columns:
            ds = [dsr_summary(f[col].dropna().to_numpy(), _bars_per_year(f.index), n, var)
                  for f in frames]
            row = {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}
            row.update(run=name, universe=cfg["universe"], mode=mode, strategy=col,
                       dsr_min=float(np.min([d["dsr"] for d in ds])), n_paths=len(frames))
            rows.append(row)
    return pd.DataFrame(rows)


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    span = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
    return (len(index) - 1) / span if span > 0 else float("nan")


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


