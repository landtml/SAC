"""Validation-tuned rule baseline: the zero-residual prior with a deadzone and a no-trade band.

``rule_prior`` holds the prior's target at every bar. The tuned rule keeps those targets, zeroes
them where the raw prior is weak (deadzone ``theta``) and applies the no-trade band of
``rollout``. The learned policy gets the same band chosen on validation tapes (``choose_band``),
so the rule gets the same tools and the comparison is fair.

Each walk-forward split picks ``(theta, band)`` on its validation tapes only. The test tapes
never enter the choice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from softedge.env.vec_env import ActionMap, TapeSet
from softedge.eval.report import pooled_returns
from softedge.experiment import NO_TRADE_BANDS, build_all_tapes, build_train_config, make_splits
from softedge.pipeline import split_tapes
from softedge.tape import MarketTape
from softedge.train import sharpe

THETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
ROLLOUT_KEYS = ("ret", "target", "w_prev", "cost", "turnover")


def tuned_rule_returns(run: Path, cfg: dict) -> tuple[pd.Series, list[dict]]:
    """Per-bar return of the tuned rule on the test tapes of each walk-forward split.

    A split contributes only when it has train, validation and test tapes. The second value
    holds the chosen ``theta`` and ``band`` of every contributing split, with the validation
    Sharpe they scored.
    """
    tapes = build_all_tapes(cfg)
    train_cfg = build_train_config(cfg, 0)
    amap, window = train_cfg.env.action_map, train_cfg.env.window
    rollouts, all_test, settings = [], [], []
    for i, split in enumerate(make_splits(cfg)):
        parts = split_tapes(tapes, split)
        if parts["train"] and parts["val"] and parts["test"]:
            theta, band, scores = choose_settings(parts["val"], window, amap)
            test_ts = TapeSet(parts["test"], window, None, amap)
            rollouts += rollout_targets(test_ts, deadzone_targets(test_ts, theta), band)
            all_test += parts["test"]
            settings.append({"split": i, "test_start": split.test.start, "theta": theta,
                             "band": band, "val_sharpe": scores[(theta, band)]})
    return pooled_returns(rollouts, all_test, window - 1), settings


def choose_settings(val: list[MarketTape], window: int, amap: ActionMap | None
                    ) -> tuple[float, float, dict[tuple[float, float], float]]:
    """``(theta, band)`` with the best mean validation Sharpe, and the score of every pair."""
    ts = TapeSet(val, window, None, amap)
    rows = {theta: deadzone_targets(ts, theta) for theta in THETAS}
    scores = {}
    for theta in THETAS:
        for band in NO_TRADE_BANDS:
            out = rollout_targets(ts, rows[theta], band)
            scores[(theta, band)] = float(np.mean([sharpe(o["ret"], tp.bars_per_year)
                                                   for o, tp in zip(out, ts.tapes, strict=True)]))
    theta, band = pick_best(scores)
    return theta, band, scores


def pick_best(scores: dict[tuple[float, float], float]) -> tuple[float, float]:
    """Highest score; ties go to the smaller theta, then the smaller band."""
    return max(scores, key=lambda p: (scores[p], -p[0], -p[1]))


def raw_prior(ts: TapeSet) -> np.ndarray:
    """Per-row prior before the clip to [-1, 1], summed over the gains as ``TapeSet`` does."""
    raw = np.zeros(len(ts.features))
    if ts.action_map is None:
        return raw
    col = {name: i for i, name in enumerate(ts.feature_names)}
    for name, gain in ts.action_map.gains:
        if name in col:
            raw += gain * ts.features[:, col[name]].astype(np.float64)
    return raw


def zero_residual_targets(ts: TapeSet) -> np.ndarray:
    """Target of every row (``TapeSet`` order) under the zero-residual policy."""
    k = np.repeat(np.arange(len(ts.tapes)), ts.lengths)
    t = np.arange(len(k)) - ts.offsets[k]
    return ts.targets(k, t, np.zeros(len(k)))


def deadzone_targets(ts: TapeSet, theta: float) -> np.ndarray:
    """Zero-residual targets, zeroed on rows where the raw prior is inside the deadzone."""
    rows = zero_residual_targets(ts)
    rows[np.abs(raw_prior(ts)) < theta] = 0.0
    return rows


def rollout_targets(tapes: TapeSet, rows: np.ndarray, band: float = 0.0
                    ) -> list[dict[str, np.ndarray]]:
    """``rollout`` with the target of every row given in ``rows``, not computed from actions.

    Same lockstep loop, no-trade band and returned keys as ``rollout``. No observation is built
    because the rule reads none.
    """
    n = len(tapes.tapes)
    k = np.arange(n)
    t = np.full(n, tapes.first_row, dtype=np.int64)
    w = np.zeros(n)
    cm = np.ones(n)
    steps = tapes.last_step_row - tapes.first_row + 1
    grid = {key: np.zeros((n, int(steps.max()))) for key in ROLLOUT_KEYS}
    for i in range(int(steps.max())):
        live = np.flatnonzero(i < steps)
        target = rows[tapes.offsets[k[live]] + t[live]]
        if band > 0:
            hold = np.abs(target - w[live]) <= band * tapes.max_leverage[live]
            target = np.where(hold, w[live], target)
        res, _, _ = tapes.transition_target(k[live], t[live], w[live], target, cm[live])
        grid["ret"][live, i] = res.ret
        grid["target"][live, i] = target
        grid["w_prev"][live, i] = w[live]
        grid["cost"][live, i] = res.cost
        grid["turnover"][live, i] = res.turnover
        w[live] = res.w_next
        t[live] += 1
    return [{key: grid[key][j, : steps[j]] for key in ROLLOUT_KEYS} for j in range(n)]
