"""Walk-forward train, validation and test windows.

Windows are half-open ``[start, stop)`` intervals in UTC. Durations are pandas
offset strings such as ``"365D"``. Each embargo gap separates train from
validation and validation from test, so no window ends where the next one starts
without a gap, which keeps labels that look ahead from crossing a boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.tseries.frequencies import to_offset

Window = tuple[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class Split:
    train: Window
    val: Window
    test: Window


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def walk_forward_splits(
    start,
    end,
    train: str,
    val: str,
    test: str,
    embargo: str = "0D",
    step: str | None = None,
    anchored: bool = False,
) -> list[Split]:
    """Splits whose test window ends at or before ``end``.

    The first train window starts at ``start``. Each later split moves its origin
    by ``step``, which defaults to ``test``. Rolling splits keep a fixed train
    length. Anchored splits keep the train start at ``start``, so the train window
    grows.
    """
    t_start = _utc(start)
    t_end = _utc(end)
    d_train = to_offset(train)
    d_val = to_offset(val)
    d_test = to_offset(test)
    d_embargo = to_offset(embargo)
    d_step = to_offset(step) if step is not None else d_test

    for name, d in (("train", d_train), ("val", d_val), ("test", d_test), ("step", d_step)):
        if t_start + d <= t_start:
            raise ValueError(f"{name} must be a positive duration, got {d}")
    if t_start + d_embargo < t_start:
        raise ValueError(f"embargo must not be negative, got {d_embargo}")

    splits: list[Split] = []
    origin = t_start
    while True:
        lo_train = t_start if anchored else origin
        hi_train = origin + d_train
        lo_val = hi_train + d_embargo
        hi_val = lo_val + d_val
        lo_test = hi_val + d_embargo
        hi_test = lo_test + d_test
        if hi_test > t_end:
            break
        splits.append(
            Split(
                train=(lo_train, hi_train),
                val=(lo_val, hi_val),
                test=(lo_test, hi_test),
            )
        )
        origin = origin + d_step
    return splits
