"""Cross-asset context: what the rest of a universe is doing at the same bar close.

``cross_features`` takes the per-asset tables from ``compute_features`` and returns, for each
asset, a market factor (``mkt_ret_h{h}``), its 16-bar return minus the mean of the other assets'
(``rel_ret_h16``), the cross-sectional dispersion of one-bar returns (``mkt_dispersion``) and the
breadth, in [-1, 1], of assets with a positive 16-bar return (``mkt_breadth``).

Row ``t`` of every output depends only on row ``t`` of the input tables. Each input row depends
only on prices up to its own bar, so the outputs are causal. Tables are aligned on the union of
their indexes. An asset without a bar at ``t`` is absent from
the cross-section at ``t``.

A timestamp where some needed horizon has fewer than ``min_assets`` finite returns gets zeros in
every output column, so the output is always finite and tapes keep all their rows. A NaN return
(warm-up) drops out of its own horizon only, so one asset with a short history does not blank the
others.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CROSS_HORIZONS = (1, 4, 16, 64)
_DISPERSION_HORIZON = 1
_RELATIVE_HORIZON = 16


def cross_features(tables: dict[str, pd.DataFrame], horizons: tuple[int, ...] = CROSS_HORIZONS,
                   min_assets: int = 3, clip: float = 5.0,
                   min_history: int = 0) -> dict[str, pd.DataFrame]:
    """Cross-asset features for each table, on that table's own index, as float32 columns.

    Columns are ``mkt_ret_h{h}`` for each ``h`` in ``horizons``, then ``rel_ret_h16``,
    ``mkt_dispersion`` and ``mkt_breadth``. The market columns are equal for every asset at a
    timestamp. An asset joins the cross-section at ``t`` only once its table holds at least
    ``min_history`` rows up to and including ``t``.
    """
    if len(tables) < 2:
        raise ValueError("cross_features needs at least two tables")
    if min_assets < 1:
        raise ValueError("min_assets must be at least 1")
    symbols = list(tables)
    # Dispersion and breadth read fixed horizons, so their columns are required even when
    # ``horizons`` omits them.
    needed = sorted({*horizons, _DISPERSION_HORIZON, _RELATIVE_HORIZON})
    for sym in symbols:
        absent = [f"ret_h{h}" for h in needed if f"ret_h{h}" not in tables[sym].columns]
        if absent:
            raise ValueError(f"table {sym!r} lacks columns {absent}")

    union = tables[symbols[0]].index
    for sym in symbols[1:]:
        union = union.union(tables[sym].index)

    # Rows each table holds up to each timestamp: a table counts only once it has ``min_history``.
    seen = (pd.DataFrame({s: pd.Series(1.0, index=tables[s].index) for s in symbols})
            .reindex(union).fillna(0.0).cumsum().to_numpy())
    rets = {h: np.where(seen >= min_history,
                        np.column_stack([tables[s][f"ret_h{h}"].reindex(union).to_numpy(np.float64)
                                         for s in symbols]),
                        np.nan)
            for h in needed}
    finite = {h: np.isfinite(r) for h, r in rets.items()}
    count = {h: f.sum(axis=1) for h, f in finite.items()}
    covered = np.ones(len(union), dtype=bool)
    for h in needed:
        covered &= count[h] >= min_assets

    # Missing values contribute zero to sums; the counts carry the real denominators.
    value = {h: np.where(finite[h], rets[h], 0.0) for h in needed}
    n = {h: np.maximum(count[h], 1) for h in needed}
    mean = {h: value[h].sum(axis=1) / n[h] for h in needed}

    ret_cols = [np.where(covered, np.clip(mean[h] * np.sqrt(n[h]), -clip, clip), 0.0)
                for h in horizons]

    d = _DISPERSION_HORIZON
    # Two-pass variance: the one-pass form can go slightly negative in float64.
    dev = np.where(finite[d], rets[d] - mean[d][:, None], 0.0)
    dispersion = np.where(covered, np.sqrt((dev * dev).sum(axis=1) / n[d]), 0.0)

    r = _RELATIVE_HORIZON
    breadth = np.where(covered, ((value[r] > 0.0).sum(axis=1) / n[r] - 0.5) * 2.0, 0.0)
    # Leave an asset out of its own mean; with no other finite return the relative return is 0.
    others = count[r][:, None] - finite[r].astype(np.int64)
    others_mean = (value[r].sum(axis=1)[:, None] - value[r]) / np.maximum(others, 1)
    rel = np.where(covered[:, None] & finite[r] & (others > 0), value[r] - others_mean, 0.0)

    names = [*(f"mkt_ret_h{h}" for h in horizons), "rel_ret_h16", "mkt_dispersion", "mkt_breadth"]
    out: dict[str, pd.DataFrame] = {}
    for k, sym in enumerate(symbols):
        pos = union.get_indexer(tables[sym].index)
        block = np.column_stack([*(c[pos] for c in ret_cols), rel[pos, k],
                                 dispersion[pos], breadth[pos]])
        out[sym] = pd.DataFrame(block.astype(np.float32), index=tables[sym].index, columns=names)
    return out
