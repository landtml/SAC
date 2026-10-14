"""Combinatorial purged cross-validation of the trading agent.

The ``cpcv`` block of a config cuts ``[start, end)`` into ``n_groups`` time groups and holds out
every combination of ``n_test`` of them (``softedge.eval.cpcv``). Training data around each test
window is purged before it and embargoed after it. Per split:

1. Each seed starts from a base agent pretrained on data before ``start`` (selected on the
   ``pretrain_val`` window just before ``start``), then fine-tunes for ``split_steps`` on the
   split's training windows.
2. Checkpoints, the no-trade band and the deployment gate are chosen on a validation slice: the
   last ``val`` of the latest training window, separated from the rest of training by the purge.
3. The held-out groups are traded once.

Path assembly then stitches, for each of ``C(n_groups - 1, n_test - 1)`` paths, one out-of-sample
result per group into a full-length backtest. The report gives the distribution of Sharpe ratio,
deflated Sharpe ratio, drawdown and expected shortfall across paths, the share of paths in which
each strategy beats the reference, and the probability of backtest overfitting (CSCV) when
picking the best strategy on in-sample Sharpe.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from softedge.agent.factory import agent_from_state
from softedge.eval.cpcv import assemble_paths, cpcv_splits, group_bounds, n_paths
from softedge.eval.metrics import summarize
from softedge.eval.report import pooled_returns, to_markdown
from softedge.eval.stats import dsr_summary, pbo_cscv
from softedge.experiment import (
    _append_jsonl,
    _per_bar_sharpe,
    _series_key,
    _test_runs,
    build_all_tapes,
    build_train_config,
    trial_stats,
)
from softedge.pipeline import window_tapes
from softedge.train import train


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def carve_validation(train_windows, val: pd.Timedelta, purge: pd.Timedelta):
    """Split the latest training window into (training, validation) with a purge gap between.

    Returns ``(train_windows, val_window)``, or ``(train_windows, None)`` when the latest window
    is too short to hold ``val`` plus ``purge`` plus as much again for training.
    """
    wins = sorted(train_windows)
    a, b = wins[-1]
    if b - a < 2 * val + purge:
        return wins, None
    val_window = (b - val, b)
    return wins[:-1] + [(a, b - val - purge)], val_window


def run_cpcv(cfg: dict, out_root: str | Path = "runs") -> dict:
    # Online updates act on the walk-forward deployment only; refuse before any training.
    if build_train_config(cfg, 0).online_every > 0:
        raise ValueError("online deployment updates are not supported under CPCV")
    name = cfg["name"]
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    c = cfg["cpcv"]
    start, end = _utc(c["start"]), _utc(c["end"])
    purge = pd.Timedelta(c.get("purge", "0D"))
    val_len = pd.Timedelta(c.get("val", "90D"))
    splits = cpcv_splits(start, end, c["n_groups"], c["n_test"], c.get("purge", "0D"),
                         c.get("embargo", "0D"))
    groups = group_bounds(start, end, c["n_groups"])
    paths = assemble_paths(splits, c["n_groups"])
    tapes = build_all_tapes(cfg)
    seeds = cfg.get("seeds", [0])
    window = build_train_config(cfg, seeds[0]).env.window
    start_row = window - 1
    t0 = time.time()

    # Base agents: trained only on data before the CPCV span.
    pre_val = (start - pd.Timedelta(c.get("pretrain_val", "180D")), start - purge)
    pre_train = window_tapes(tapes, [(pd.Timestamp("1970-01-01", tz="UTC"), pre_val[0] - purge)])
    pre_val_tapes = window_tapes(tapes, [pre_val])
    if not pre_train or not pre_val_tapes:
        raise ValueError("cpcv needs data before start for the base agent and its validation")
    base, scales, feature_names = {}, {}, None
    for seed in seeds:
        res = train(pre_train, pre_val_tapes, build_train_config(cfg, seed),
                    log=lambda r, s=seed: _append_jsonl(out / "train_log.jsonl",
                                                        {"split": "base", "seed": s, **r}))
        base[seed], scales[seed], feature_names = res.best_state, res.reward_scale, res.feature_names

    split_pooled: dict[int, dict[str, pd.Series]] = {}
    split_info = []
    for sp in splits:
        train_wins, val_win = carve_validation(sp.train, val_len, purge)
        val_tapes = window_tapes(tapes, [val_win]) if val_win else pre_val_tapes
        train_tapes = window_tapes(tapes, train_wins)
        test_tapes = window_tapes(tapes, sp.test)
        agents = []
        for seed in seeds:
            tc = replace(build_train_config(cfg, seed), total_steps=c.get("split_steps", 40000),
                         warmup_steps=2048, reward_scale=scales[seed])
            res = train(train_tapes, val_tapes, tc, init_state=base[seed],
                        feature_names=feature_names,
                        log=lambda r, s=seed, k=sp.index: _append_jsonl(
                            out / "train_log.jsonl", {"split": k, "seed": s, **r}))
            torch.save(res.best_state, out / f"agent_split{sp.index}_seed{seed}.pt")
            agents.append(agent_from_state(res.best_state))
        parts = {"train": train_tapes, "val": val_tapes, "test": test_tapes}
        runs, band = _test_runs(cfg, agents, seeds, parts, window, feature_names)
        split_pooled[sp.index] = {s: pooled_returns(run, test_tapes, start_row)
                                  for s, run in runs.items()}
        split_info.append({"split": sp.index, "test_groups": list(sp.test_groups),
                           "no_trade_band": band,
                           "n_tapes": {k: len(v) for k, v in parts.items()},
                           "val_window": [str(v) for v in val_win] if val_win else "pre_start"})

    strategies = list(split_pooled[splits[0].index])
    path_series: list[dict[str, pd.Series]] = []
    for path in paths:
        series = {}
        for s in strategies:
            pieces = []
            for g, k in enumerate(path):
                ser = split_pooled[k][s]
                a, b = groups[g]
                pieces.append(ser[(ser.index >= a) & (ser.index < b)])
            series[s] = pd.concat(pieces).sort_index()
        path_series.append(series)

    all_tapes = window_tapes(tapes, [(start, end)])
    bpy = float(np.median([tp.bars_per_year for tp in all_tapes]))
    reference = cfg.get("reference", "buy_and_hold")
    sac_sr = {s: float(np.mean([_per_bar_sharpe(p[s]) for p in path_series]))
              for s in strategies if s.startswith("sac_")}
    keys = [_series_key([p[s].dropna().to_numpy() for p in path_series]) for s in sac_sr]
    n_trials, sr_var = trial_stats(Path(out_root), cfg["universe"], list(sac_sr.values()), keys)

    rows = []
    for j, series in enumerate(path_series):
        for s, ser in series.items():
            r = ser.to_numpy()
            m = summarize(r, bpy)
            d = dsr_summary(r, bpy, n_trials, sr_var)
            ref = series[reference].to_numpy()
            rows.append({"path": j, "strategy": s, "sharpe": m["sharpe"], "dsr": d["dsr"],
                         "psr": d["psr"], "max_drawdown": m["max_drawdown"],
                         "cvar_95": m["cvar_95"], "cagr": m["cagr"],
                         "beats_reference": float(m["sharpe"] > summarize(ref, bpy)["sharpe"])})
    path_table = pd.DataFrame(rows)
    agg = path_table.groupby("strategy", sort=False).agg(
        sharpe_mean=("sharpe", "mean"), sharpe_std=("sharpe", "std"),
        sharpe_min=("sharpe", "min"), sharpe_max=("sharpe", "max"),
        share_positive=("sharpe", lambda x: float((x > 0).mean())),
        dsr_mean=("dsr", "mean"), dsr_min=("dsr", "min"), psr_mean=("psr", "mean"),
        max_drawdown_mean=("max_drawdown", "mean"), cvar_95_mean=("cvar_95", "mean"),
        beats_reference_share=("beats_reference", "mean"),
    ).reset_index()
    perf = pd.concat(path_series[0], axis=1).dropna().to_numpy()
    pbo = float(pbo_cscv(perf, n_splits=16)) if perf.shape[0] >= 32 and perf.shape[1] >= 2 else float("nan")

    result = {
        "name": name, "mode": "cpcv", "minutes": round((time.time() - t0) / 60, 1),
        "n_splits": len(splits), "n_paths": n_paths(c["n_groups"], c["n_test"]),
        "n_trials": n_trials, "bars_per_year": bpy, "pbo_strategy_choice": pbo,
        "splits": split_info,
        "aggregate": json.loads(agg.set_index("strategy").to_json(orient="index")),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    path_table.to_csv(out / "paths.csv", index=False)
    for j, series in enumerate(path_series):
        pd.concat(series, axis=1).to_csv(out / f"path{j}_returns.csv")
    (out / "report.md").write_text(
        f"# {name} (CPCV)\n\nGroups: {c['n_groups']}, held out per split: {c['n_test']}, "
        f"splits: {len(splits)}, paths: {len(paths)}, purge {c.get('purge', '0D')}, embargo "
        f"{c.get('embargo', '0D')}. Trials counted for DSR: {n_trials}. PBO of choosing the "
        f"best strategy by in-sample Sharpe: {pbo:.3f}.\n\n## Across paths\n\n"
        + to_markdown(agg) + "\n\n## Per path\n\n" + to_markdown(path_table) + "\n")
    _append_jsonl(Path(out_root) / "trials.jsonl", {
        "name": name, "universe": cfg["universe"], "config": copy.deepcopy(cfg),
        "sac_sr_bar": sac_sr, "mode": "cpcv", "aggregate": result["aggregate"]})
    return result
