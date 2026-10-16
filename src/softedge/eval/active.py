"""Active return of a learned policy over its rule baseline, with its deflated Sharpe ratio.

The active return of a run is ``strategy`` minus its rule baseline, per bar, on the bars both
share. The baseline is the ``prior_only`` column when the run saved one. A run without that
column whose config declares ``env.prior_feature`` gets the baseline rebuilt by
``rule_prior``: the zero-residual policy rolled out on the same test tapes. Runs with neither
are skipped.

``baseline="tuned"`` takes the validation-tuned rule of ``softedge.eval.tuned_rule`` instead, for
every run whose config declares ``env.prior_feature``, whether or not it saved ``prior_only``.

Trial count: ``trial_stats`` for the run's universe (every SAC variant recorded in
``trials.jsonl``).

Cross-trial variance, ``dsr``: the variance (ddof=1) of the active per-bar Sharpe ratios of
the universe's runs that have a baseline. With fewer than ``MIN_BASELINE_TRIALS`` such runs
the null floor is used and the row is flagged ``floor``.

Sensitivity columns: ``dsr_floor`` uses the null floor; ``dsr_abs`` uses the variance of the
absolute SAC Sharpe ratios of the universe, as ``recompute_dsr`` does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from softedge.env.vec_env import TapeSet, rollout
from softedge.eval.report import pooled_returns, to_markdown
from softedge.eval.stats import dsr_summary, null_sr_variance, sharpe_moments
from softedge.eval.tuned_rule import tuned_rule_returns
from softedge.experiment import (
    build_all_tapes,
    build_train_config,
    load_config,
    make_splits,
    recompute_dsr,
    trial_stats,
)
from softedge.pipeline import split_tapes

PRIOR = "prior_only"
BASELINES = ("rule", "tuned")
MIN_BASELINE_TRIALS = 3
# Runs that a later accounting fix replaced (docs/research-log.md rows marked superseded).
SUPERSEDED = ("crypto_perp_1d_wf", "crypto_basis_1d_prior", "hl_basis_1d_prior")
COLUMNS = ["run", "universe", "strategy", "n_bars", "sharpe", "baseline_sharpe", "psr",
           "n_trials", "n_with_baseline", "cross_var", "dsr", "dsr_floor", "dsr_abs",
           "dsr_excl_superseded"]


def active_markdown(runs: str | Path, strategy: str = "sac_banded", baseline: str = "rule") -> str:
    """The active table as a Markdown table, then a line naming the superseded runs left out."""
    root = Path(runs)
    left_out = [name for name in SUPERSEDED if (root / name / "config.yaml").exists()]
    footer = ("Left out of dsr_excl_superseded (trials and variance): "
              + (", ".join(left_out) if left_out else "none") + ".")
    if baseline == "tuned":
        footer = ("Baseline: the validation-tuned rule (theta and band chosen per split).\n"
                  + footer)
    return to_markdown(active_table(root, strategy, baseline)) + "\n\n" + footer


def active_table(runs: str | Path, strategy: str = "sac_banded", baseline: str = "rule"
                 ) -> pd.DataFrame:
    """One row per run with a rule baseline: the DSR of ``strategy`` minus that baseline.

    ``baseline`` is ``"rule"`` (the saved ``prior_only`` column or ``rule_prior``) or ``"tuned"``
    (``tuned_rule_returns``). ``baseline_sharpe`` is the annualised Sharpe of the baseline on the
    bars the active return uses.

    ``dsr`` counts every trial of the universe and takes the variance of all baseline runs.
    ``dsr_excl_superseded`` drops the ``SUPERSEDED`` runs from both (n/a on their own rows).
    """
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}, not {baseline!r}")
    root = Path(runs)
    absolute_var = _absolute_variances(root, strategy)
    entries = []
    for run in sorted(p for p in root.iterdir() if p.is_dir() and (p / "config.yaml").exists()):
        cfg = load_config(run / "config.yaml")
        series = _active_series(run, cfg, strategy, baseline)
        if series is None or len(series[0]) < 3:
            continue
        active, base = series
        bpy = _bars_per_year(active.index)
        entries.append({"run": run.name, "universe": cfg["universe"], "active": active,
                        "bars_per_year": bpy,
                        "sr_bar": sharpe_moments(active.to_numpy())["sr"],
                        "baseline_sharpe": sharpe_moments(base.to_numpy())["sr"] * np.sqrt(bpy)})

    rows = []
    for universe in sorted({e["universe"] for e in entries}):
        group = [e for e in entries if e["universe"] == universe]
        n_trials, _ = trial_stats(root, universe, [])
        srs = [e["sr_bar"] for e in group]
        active_var = (float(np.var(srs, ddof=1))
                      if len(group) >= MIN_BASELINE_TRIALS else None)
        kept = [e for e in group if e["run"] not in SUPERSEDED]
        kept_srs = [e["sr_bar"] for e in kept]
        kept_var = (float(np.var(kept_srs, ddof=1))
                    if len(kept) >= MIN_BASELINE_TRIALS else None)
        n_kept = n_trials - _registry_trials(root, universe, SUPERSEDED)
        for e in group:
            values = e["active"].to_numpy()
            bpy = e["bars_per_year"]
            primary = dsr_summary(values, bpy, n_trials, active_var)
            flagged = active_var is None or active_var < null_sr_variance(len(values))
            excl = (None if e["run"] in SUPERSEDED
                    else dsr_summary(values, bpy, n_kept, kept_var)["dsr"])
            rows.append({
                "run": e["run"], "universe": universe, "strategy": strategy,
                "n_bars": len(values), "sharpe": primary["sharpe"],
                "baseline_sharpe": e["baseline_sharpe"], "psr": primary["psr"],
                "n_trials": n_trials, "n_with_baseline": len(group),
                "cross_var": "floor" if flagged else "active",
                "dsr": primary["dsr"],
                "dsr_floor": dsr_summary(values, bpy, n_trials)["dsr"],
                "dsr_abs": dsr_summary(values, bpy, n_trials,
                                       absolute_var.get(e["run"]))["dsr"],
                "dsr_excl_superseded": excl,
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def rule_prior(run: Path, cfg: dict) -> pd.Series:
    """Per-bar return of the zero-residual policy on the test tapes of each walk-forward split.

    A split contributes only when it has train, validation and test tapes.
    """
    tapes = build_all_tapes(cfg)
    train_cfg = build_train_config(cfg, 0)
    amap, window = train_cfg.env.action_map, train_cfg.env.window
    rollouts, all_test = [], []
    for split in make_splits(cfg):
        parts = split_tapes(tapes, split)
        if parts["train"] and parts["val"] and parts["test"]:
            rollouts += rollout(TapeSet(parts["test"], window, None, amap),
                                lambda obs: np.zeros(len(obs)))
            all_test += parts["test"]
    return pooled_returns(rollouts, all_test, window - 1)


def _active_series(run: Path, cfg: dict, strategy: str, baseline: str = "rule"
                   ) -> tuple[pd.Series, pd.Series] | None:
    """Active return of ``strategy`` and the baseline it is measured against, on common bars."""
    pooled_path = run / "pooled_returns.csv"
    if not pooled_path.exists():
        return None
    pooled = pd.read_csv(pooled_path, index_col=0, parse_dates=True,
                         float_precision="round_trip")
    if strategy not in pooled.columns:
        return None
    has_rule = "prior_feature" in (cfg.get("env") or {})
    if baseline == "tuned":
        if not has_rule:
            return None
        base = tuned_rule_returns(run, cfg)[0]
    elif PRIOR in pooled.columns:
        base = pooled[PRIOR]
    elif has_rule:
        base = rule_prior(run, cfg)
    else:
        return None
    common = pooled.index.intersection(base.index)
    active = (pooled.loc[common, strategy] - base.loc[common]).dropna()
    return active, base.loc[active.index]


def _registry_trials(root: Path, universe: str, names) -> int:
    """SAC trials that ``trials.jsonl`` records for the named runs of ``universe``."""
    path = root / "trials.jsonl"
    if not path.exists():
        return 0
    total = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("universe") == universe and row.get("name") in names:
                total += len(row.get("sac_sr_bar", {}))
    return total


def _absolute_variances(root: Path, strategy: str) -> dict[str, float]:
    """Per run, the cross-trial variance ``recompute_dsr`` uses for its ``strategy`` row."""
    table = recompute_dsr(root)
    if table.empty:
        return {}
    rows = table[table["strategy"] == strategy]
    return {row.run: float(row.sr_variance) for row in rows.itertuples()}


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    span = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
    return (len(index) - 1) / span
