"""Combinatorial purged cross-validation (CPCV): splits and backtest paths.

Time is continuous. ``group_bounds`` cuts ``[start, end)`` into contiguous groups.
``cpcv_splits`` holds out every combination of ``n_test`` groups, and adjacent held-out
groups form one test window. Around each test window, training data is removed in two
zones. The purge zone before it holds samples whose label horizon reaches into the test
window. The embargo zone after it holds samples whose features still carry test-period
information.

Windows are half-open ``[a, b)`` pairs of tz-aware UTC timestamps. Purge and embargo are
duration strings such as ``"5D"`` or ``"12h"``, read as elapsed time.

Reference: Lopez de Prado, Advances in Financial Machine Learning (2018), ch. 7 and 12.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from sactrader.eval.walkforward import Window

_ZERO = pd.Timedelta(0)


@dataclass(frozen=True)
class CPCVSplit:
    index: int  # position in the cpcv_splits output
    test_groups: tuple[int, ...]  # sorted group ids held out
    test: tuple[Window, ...]  # one window per run of adjacent test groups
    train: tuple[Window, ...]  # maximal disjoint windows left for training


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _span(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0, t1 = _utc(start), _utc(end)
    if t0 >= t1:
        raise ValueError(f"start must be before end, got {t0} and {t1}")
    return t0, t1


def _duration(text: str, name: str) -> pd.Timedelta:
    d = pd.Timedelta(text)
    if d < _ZERO:
        raise ValueError(f"{name} must not be negative, got {text!r}")
    return d


def _check_folds(n_groups: int, n_test: int) -> None:
    if not 1 <= n_test < n_groups:
        raise ValueError(f"need 1 <= n_test < n_groups, got n_test={n_test}, n_groups={n_groups}")


def _runs(groups: tuple[int, ...]) -> list[tuple[int, int]]:
    """First and last id of each block of consecutive group ids."""
    runs: list[tuple[int, int]] = []
    for _, block in itertools.groupby(enumerate(groups), key=lambda p: p[1] - p[0]):
        ids = [g for _, g in block]
        runs.append((ids[0], ids[-1]))
    return runs


def _complement(blocked: list[Window], t0: pd.Timestamp, t1: pd.Timestamp) -> tuple[Window, ...]:
    """Maximal intervals of ``[t0, t1)`` that no blocked window covers."""
    kept: list[Window] = []
    cursor = t0
    for lo, hi in sorted(blocked):
        if lo > cursor:
            kept.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < t1:
        kept.append((cursor, t1))
    return tuple(kept)


def group_bounds(start, end, n_groups: int) -> list[Window]:
    """``n_groups`` contiguous windows that cover ``[start, end)`` exactly.

    Edges fall on whole nanoseconds, so durations differ by at most 1 ns when the span
    is not a multiple of ``n_groups`` nanoseconds.
    """
    t0, t1 = _span(start, end)
    if n_groups < 1:
        raise ValueError(f"n_groups must be at least 1, got {n_groups}")
    span_ns = (t1 - t0).value
    if span_ns < n_groups:
        raise ValueError("span must hold at least one nanosecond per group")
    edges = [t0 + pd.Timedelta(span_ns * i // n_groups, unit="ns") for i in range(n_groups + 1)]
    return list(zip(edges[:-1], edges[1:], strict=True))


def cpcv_splits(
    start,
    end,
    n_groups: int,
    n_test: int,
    purge: str = "0D",
    embargo: str = "0D",
) -> list[CPCVSplit]:
    """One split per combination of ``n_test`` groups, in ``itertools.combinations`` order.

    Each training window excludes its test window, the purge zone before it and the
    embargo zone after it. Zones are clipped to ``[start, end)``.
    """
    _check_folds(n_groups, n_test)
    t0, t1 = _span(start, end)
    d_purge = _duration(purge, "purge")
    d_embargo = _duration(embargo, "embargo")
    bounds = group_bounds(t0, t1, n_groups)

    splits: list[CPCVSplit] = []
    for index, held_out in enumerate(itertools.combinations(range(n_groups), n_test)):
        test = tuple((bounds[first][0], bounds[last][1]) for first, last in _runs(held_out))
        # Purge, test and embargo of one window are contiguous, so one interval covers them.
        blocked = [(max(t0, a - d_purge), min(t1, b + d_embargo)) for a, b in test]
        splits.append(
            CPCVSplit(
                index=index,
                test_groups=held_out,
                test=test,
                train=_complement(blocked, t0, t1),
            )
        )
    return splits
