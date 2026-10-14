"""Cross-sectional funding rank for a carry universe: which coins the market pays most to hold.

``xs_funding_features`` takes the daily carry frames of a universe and returns, for each frame,
two columns on its own index:

``funding_xs``
    The coin's trailing mean income over ``window`` bars, minus the cross-sectional median of
    that quantity at the same bar, divided by the cross-sectional median absolute deviation
    (MAD), clipped to ``[-clip, clip]``. A coin one MAD above the median reads 1.0.

``funding_topk``
    The share of the ``topk`` places a coin holds at the bar, from 0.0 to 1.0. A coin with
    ``above`` eligible coins strictly ahead of it and ``tied`` eligible coins at its income
    (itself included) gets ``(topk - above) / tied``, clipped to [0, 1]. Coins above the cut get
    1.0, coins below it get 0.0, and tied coins at the cut split the places left for them: three
    coins tied for places 4 to 6 of a top-five cut get 2/3 each. On a bar with at least ``topk``
    eligible coins the flags sum to ``topk``. With fewer, every eligible coin gets 1.0. The flags
    do not depend on the column order of the frames.

Income is ``-funding_rate``. A carry frame writes its funding in carry sign (see
``softedge.data.basis``), so the income of a short perpetual leg is positive when the perpetual
funding is positive.

Causality. Row ``t`` of every output reads the funding rows ``<= t`` of every frame and nothing
later. A coin joins the cross-section at ``t`` only when it has ``window`` consecutive bars ending
at ``t``, so a coin listed inside a window, or with a gap in it, stays out until its window fills.
A bar with fewer than ``min_coins`` eligible coins, or a zero MAD, gets zeros.
"""

from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd

_MAD_FLOOR = 1e-12


def xs_funding_features(frames: list[pd.DataFrame], window: int = 7, topk: int = 5,
                        min_coins: int = 3, clip: float = 5.0) -> list[pd.DataFrame]:
    """``funding_xs`` and ``funding_topk`` for each frame, on that frame's own index (float32).

    The frames must be daily (``bar == "1D"``) and carry a ``funding_rate`` column. See the
    module docstring for the definitions and the causal rule.
    """
    if window < 1 or topk < 1 or min_coins < 1:
        raise ValueError("window, topk and min_coins must be at least 1")
    for fr in frames:
        if fr.attrs.get("bar") != "1D":
            raise ValueError(f"cross-sectional funding needs daily bars, got {fr.attrs.get('bar')!r}")
        if "funding_rate" not in fr.columns:
            raise ValueError(f"{fr.attrs.get('symbol')!r} has no funding_rate column")
    index = reduce(lambda a, b: a.union(b), (fr.index for fr in frames))
    income = pd.DataFrame({j: -fr["funding_rate"].astype("float64")
                           for j, fr in enumerate(frames)}).reindex(index)

    # A full window: every one of the last ``window`` rows is present, so a gap or a late listing
    # leaves the coin out until ``window`` consecutive bars have passed.
    trail = income.rolling(window, min_periods=window).mean()
    med = trail.median(axis=1)
    mad = trail.sub(med, axis=0).abs().median(axis=1)
    live = ((trail.count(axis=1) >= min_coins) & (mad > _MAD_FLOOR)).to_numpy()[:, None]
    member = trail.notna().to_numpy() & live

    z = trail.sub(med, axis=0).div(mad, axis=0).clip(-clip, clip).to_numpy()
    z = np.where(member, z, 0.0)
    # A coin's ``above`` is its lowest rank minus one, and its ``tied`` is its highest rank minus its
    # lowest plus one. The share of the places left at the cut is clipped to [0, 1].
    eligible = trail.where(member)
    rank_lo = eligible.rank(axis=1, ascending=False, method="min").to_numpy()
    rank_hi = eligible.rank(axis=1, ascending=False, method="max").to_numpy()
    share = np.clip((topk - (rank_lo - 1.0)) / (rank_hi - rank_lo + 1.0), 0.0, 1.0)
    top = np.where(member, share, 0.0)

    out = []
    for j, fr in enumerate(frames):
        pos = index.get_indexer(fr.index)
        block = np.column_stack([z[pos, j], top[pos, j]])
        out.append(pd.DataFrame(block.astype(np.float32), index=fr.index,
                                columns=["funding_xs", "funding_topk"]))
    return out
