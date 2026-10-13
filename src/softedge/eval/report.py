"""Strategy reports over rollout outputs: per-tape metrics, pooled series and significance.

A run is one dict per tape, as ``softedge.env.vec_env.rollout`` returns it, with arrays
``ret, target, w_prev, cost, turnover`` of length ``n_steps``. ``runs`` maps a strategy name
to its list of per-tape dicts, in the order of ``tapes``. Step ``i`` of a tape acts at row
``start_row + i``, so its timestamp is ``tape.times[start_row + i]``. ``rollout`` uses
``start_row = window - 1``.

``rollout_exposures`` turns a precomputed exposure path into the same dict, so baselines and
learned policies share one report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from softedge.eval.backtest import backtest_exposures
from softedge.eval.metrics import summarize
from softedge.eval.stats import dsr_summary, sharpe_diff_test
from softedge.tape import MarketTape

Run = list[dict[str, np.ndarray]]

PER_TAPE_COLUMNS = [
    "strategy", "symbol", "kind", "n_bars", "total_return", "cagr", "sharpe", "sortino",
    "max_drawdown", "turnover_per_year", "cost_per_year", "avg_abs_exposure",
]
SUMMARY_COLUMNS = [
    "strategy", "total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
    "ann_vol", "cvar_95", "psr", "sr0", "dsr", "min_trl_years", "sharpe_diff", "diff_lo",
    "diff_hi", "p_value",
]
_METRIC_KEYS = ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar", "ann_vol",
                "cvar_95")
_DIFF_KEYS = ("sharpe_diff", "diff_lo", "diff_hi", "p_value")


def _n_steps(tape: MarketTape, start_row: int) -> int:
    # A terminal tape steps from its last row, which settles the market. Any other tape
    # stops one row early, since its last row has no next bar.
    return len(tape) - start_row - (0 if tape.terminal_at_end else 1)


def _check_run(run: Run, tapes: list[MarketTape], start_row: int) -> None:
    if len(run) != len(tapes):
        raise ValueError(f"run has {len(run)} tape results for {len(tapes)} tapes")
    for tape, res in zip(tapes, run, strict=True):
        want = (_n_steps(tape, start_row),)
        got = np.shape(res["ret"])
        if got != want:
            raise ValueError(f"{tape.symbol}: expected step arrays of shape {want}, got {got}")


def rollout_exposures(
    tape: MarketTape, exposures: np.ndarray, start_row: int
) -> dict[str, np.ndarray]:
    """Backtest an exposure path from ``start_row`` on a flat start.

    ``exposures`` has one value per tape row, as in ``backtest_exposures``. The targets are
    clipped to the tape's leverage limit. The result has the keys of ``rollout``.
    """
    exp = np.asarray(exposures, dtype=np.float64)
    if exp.shape != (len(tape),):
        raise ValueError(f"exposures must have shape ({len(tape)},), got {exp.shape}")
    if not 0 <= start_row < len(tape) - 1:
        raise ValueError(f"start_row must lie in [0, {len(tape) - 1}), got {start_row}")
    # Slicing to the end keeps the terminal flag, so the steps match rollout's.
    res = backtest_exposures(tape.slice(start_row, len(tape)), exp[start_row:])
    return {
        "ret": res.returns,
        "target": res.exposure,
        "w_prev": res.w_prev,
        "cost": res.cost,
        "turnover": res.turnover,
    }


def per_tape_table(
    runs: dict[str, Run], tapes: list[MarketTape], start_row: int
) -> pd.DataFrame:
    """One row per (strategy, tape), with metrics on the tape's own bar clock."""
    rows = []
    for name, run in runs.items():
        _check_run(run, tapes, start_row)
        for tape, res in zip(tapes, run, strict=True):
            m = summarize(res["ret"], tape.bars_per_year)
            rows.append({
                "strategy": name,
                "symbol": tape.symbol,
                "kind": tape.kind,
                "n_bars": int(m["n_bars"]),
                "total_return": m["total_return"],
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "sortino": m["sortino"],
                "max_drawdown": m["max_drawdown"],
                "turnover_per_year": float(np.mean(res["turnover"])) * tape.bars_per_year,
                "cost_per_year": float(np.mean(res["cost"])) * tape.bars_per_year,
                "avg_abs_exposure": float(np.mean(np.abs(res["target"]))),
            })
    return pd.DataFrame(rows, columns=PER_TAPE_COLUMNS)


def pooled_returns(run: Run, tapes: list[MarketTape], start_row: int) -> pd.Series:
    """Equal-weight return at each timestamp, over the tapes that have a step there."""
    bars = {tape.bar for tape in tapes}
    if len(bars) > 1:
        raise ValueError(f"cannot pool tapes with different bar sizes {sorted(bars)}")
    _check_run(run, tapes, start_row)
    times, rets = [], []
    for tape, res in zip(tapes, run, strict=True):
        n = _n_steps(tape, start_row)
        times.append(tape.times[start_row:start_row + n])
        rets.append(np.asarray(res["ret"], dtype=np.float64))
    mean = pd.Series(np.concatenate(rets)).groupby(np.concatenate(times)).mean()
    index = pd.to_datetime(mean.index.to_numpy(), unit="ns", utc=True).rename("time")
    return pd.Series(mean.to_numpy(), index=index, name="ret")


def summary_table(
    runs: dict[str, Run],
    tapes: list[MarketTape],
    start_row: int,
    reference: str,
    n_trials: int = 1,
    sr_variance: float | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per strategy over its pooled series, with tests against ``reference``.

    See ``summary_from_pooled``; bars_per_year is the median over ``tapes``.
    """
    if reference not in runs:
        raise ValueError(f"reference {reference!r} is not a strategy in runs")
    bpy = float(np.median([tape.bars_per_year for tape in tapes]))
    pooled = {name: pooled_returns(run, tapes, start_row) for name, run in runs.items()}
    return summary_from_pooled(pooled, bpy, reference, n_trials, sr_variance, n_boot, seed)


def summary_from_pooled(
    pooled: dict[str, pd.Series],
    bars_per_year: float,
    reference: str,
    n_trials: int = 1,
    sr_variance: float | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per strategy over a pooled return series, with tests against ``reference``.

    Deflated Sharpe ratio framework (Bailey and Lopez de Prado, 2012 and 2014): ``psr`` is the
    probability that the true Sharpe exceeds zero, ``sr0`` the annualized Sharpe that the best of
    ``n_trials`` zero-skill configurations would show by luck, ``dsr`` the probability that the
    true Sharpe exceeds ``sr0``, and ``min_trl_years`` the track record needed for ``psr`` to
    reach 0.95. ``sr_variance`` is the cross-trial variance of per-bar Sharpe ratios, floored
    at the zero-skill estimation variance; ``None`` uses the floor. The diff columns run a paired
    bootstrap of the annualized Sharpe difference on the timestamps both series share. The
    reference row leaves them NaN.
    """
    if reference not in pooled:
        raise ValueError(f"reference {reference!r} is not a strategy in pooled")
    bpy = float(bars_per_year)
    ref = pooled[reference]
    rows = []
    for name, series in pooled.items():
        r = series.to_numpy()
        m = summarize(r, bpy)
        row: dict[str, float | str] = {"strategy": name}
        row.update({k: m[k] for k in _METRIC_KEYS})
        d = dsr_summary(r, bpy, n_trials, sr_variance)
        row.update(psr=d["psr"], sr0=d["sr0"], dsr=d["dsr"], min_trl_years=d["min_trl_years"])
        if name == reference:
            row.update(dict.fromkeys(_DIFF_KEYS, np.nan))
        else:
            a, b = series.align(ref, join="inner")
            d = sharpe_diff_test(a.to_numpy(), b.to_numpy(), bpy, n_boot=n_boot, seed=seed)
            row.update(sharpe_diff=d["diff"], diff_lo=d["lo"], diff_hi=d["hi"],
                       p_value=d["p_value"])
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """GitHub table of ``df``. Floats use ``floatfmt`` and missing values read ``n/a``."""
    header = "| " + " | ".join(_escape(str(c)) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(_cell(v, floatfmt) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def _cell(value, floatfmt: str) -> str:
    if isinstance(value, str):
        return _escape(value)
    if pd.isna(value):
        return "n/a"
    # bool is an int subclass, so it is checked first.
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return format(float(value), floatfmt)


def _escape(text: str) -> str:
    # An unescaped pipe would start a new cell.
    return text.replace("|", "\\|")
