"""Market tape: the per-instrument arrays the environment and the replay buffer read.

Timing convention. Row ``t`` is the decision point at the *close* of bar ``t``:
``features[t]`` may use bars ``<= t`` only. An exposure chosen at row ``t`` is
filled at ``close[t]`` (fill costs come from ``cost_long[t]`` / ``cost_short[t]``)
and earns the move from ``close[t]`` to ``close[t + 1]``.

Exposure ``w`` is signed notional as a fraction of equity. ``w > 0`` holds ``w``
of long units, ``w < 0`` holds ``-w`` of short units. For linear instruments
(spot, perp, fx) a short unit is a short sale. For binary markets a short unit is
a long NO token, so both sides are fully collateralized.

All per-step arrays have length ``T`` and describe the interval ``(t, t + 1]``:

* ``long_ret[t]``  simple return of one long unit
* ``short_ret[t]`` simple return of one short unit
* ``short_growth[t]`` growth factor of the short side's notional, used to drift
  exposure after the move (``1 + long_ret`` for a short sale, ``1 + short_ret``
  for a NO token)
* ``long_worst[t]`` / ``short_worst[t]`` worst intra-bar return of each side,
  used for liquidation checks
* ``carry_long[t]`` / ``carry_short[t]`` holding cost rate per unit of exposure
  (funding, swap, borrow). Positive means the position pays.
* ``cost_long[t]`` / ``cost_short[t]`` proportional cost per unit of exposure
  traded at row ``t`` (fee plus half spread)

Episode ends. When ``terminal_at_end`` is false the last row has no next bar: its
returns are zero, an episode may reach row ``T - 1`` but never steps from it, and
reaching it is a truncation. When ``terminal_at_end`` is true (binary markets) the
step from row ``T - 1`` is the settlement: its returns move each side to the
resolved value, and the episode terminates after it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sactrader.data.schema import validate_frame

KIND_INDEX = {"spot": 0, "perp": 1, "fx": 2, "binary": 3}


@dataclass(frozen=True)
class CostSpec:
    """Trading frictions for one instrument.

    Linear instruments quote ``fee_bps`` and ``half_spread_bps`` relative to price.
    Binary markets quote ``half_spread_abs`` in price units (dollars per share),
    so the relative cost grows as the traded side gets cheap.
    """

    fee_bps: float = 5.0
    half_spread_bps: float = 1.0
    half_spread_abs: float = 0.005
    impact: float = 0.0
    borrow_apr: float = 0.0
    swap_apr_long: float = 0.0
    swap_apr_short: float = 0.0
    max_leverage: float = 1.0
    maint_margin: float = 0.005


DEFAULT_COSTS = {
    "spot": CostSpec(fee_bps=7.5, half_spread_bps=1.0, borrow_apr=0.10, max_leverage=1.0),
    "perp": CostSpec(fee_bps=4.5, half_spread_bps=0.5, max_leverage=3.0, maint_margin=0.005),
    "fx": CostSpec(fee_bps=0.5, half_spread_bps=0.7, swap_apr_long=0.01, swap_apr_short=0.01,
                   max_leverage=5.0, maint_margin=0.02),
    "binary": CostSpec(fee_bps=0.0, half_spread_abs=0.005, max_leverage=1.0, maint_margin=0.0),
}


@dataclass(frozen=True)
class MarketTape:
    symbol: str
    kind: str
    bar: str
    times: np.ndarray  # int64 ns since epoch, UTC
    close: np.ndarray
    features: np.ndarray  # float32 (T, F)
    feature_names: tuple[str, ...]
    long_ret: np.ndarray
    short_ret: np.ndarray
    short_growth: np.ndarray
    long_worst: np.ndarray
    short_worst: np.ndarray
    carry_long: np.ndarray
    carry_short: np.ndarray
    cost_long: np.ndarray
    cost_short: np.ndarray
    bars_per_year: float
    max_leverage: float
    maint_margin: float
    impact: float
    terminal_at_end: bool
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.times)

    @property
    def kind_index(self) -> int:
        return KIND_INDEX[self.kind]

    def slice(self, start: int, stop: int) -> MarketTape:
        """Rows ``[start, stop)``. The cut row ``stop - 1`` gets zero returns and
        loses ``terminal_at_end`` unless ``stop`` is the tape end."""
        n = len(self)
        start, stop = max(0, start), min(n, stop)
        if stop - start < 2:
            raise ValueError("slice needs at least 2 rows")
        at_end = stop == n
        arrays = {}
        for name in _STEP_ARRAYS:
            a = getattr(self, name)[start:stop].copy()
            if not at_end:
                a[-1] = 1.0 if name == "short_growth" else 0.0
            arrays[name] = a
        return MarketTape(
            symbol=self.symbol, kind=self.kind, bar=self.bar,
            times=self.times[start:stop], close=self.close[start:stop],
            features=self.features[start:stop], feature_names=self.feature_names,
            bars_per_year=self.bars_per_year, max_leverage=self.max_leverage,
            maint_margin=self.maint_margin, impact=self.impact,
            terminal_at_end=self.terminal_at_end and at_end, extra=dict(self.extra),
            **arrays,
        )

    def time_slice(self, start: str | pd.Timestamp | None, stop: str | pd.Timestamp | None) -> MarketTape:
        """Rows with ``start <= time < stop`` (UTC)."""
        t = self.times
        lo = 0 if start is None else int(np.searchsorted(t, _ns(start), side="left"))
        hi = len(t) if stop is None else int(np.searchsorted(t, _ns(stop), side="left"))
        return self.slice(lo, hi)


_STEP_ARRAYS = (
    "long_ret", "short_ret", "short_growth", "long_worst", "short_worst",
    "carry_long", "carry_short", "cost_long", "cost_short",
)


def _ns(ts: str | pd.Timestamp) -> int:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.as_unit("ns").value)


def estimate_bars_per_year(index: pd.DatetimeIndex) -> float:
    span_years = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
    if span_years <= 0:
        raise ValueError("index must span positive time")
    return (len(index) - 1) / span_years


def build_tape(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    costs: CostSpec | None = None,
    bars_per_year: float | None = None,
) -> MarketTape:
    """Join a bar frame with its causal features into a tape.

    Rows where any feature is NaN (indicator warm-up) are dropped from the start.
    NaNs after the first complete row are an error, since they would hide gaps.
    """
    validate_frame(frame)
    kind = frame.attrs["kind"]
    costs = costs or DEFAULT_COSTS[kind]
    if not features.index.equals(frame.index):
        raise ValueError("features must share the frame index")
    feat = features.to_numpy(dtype=np.float32)
    complete = np.isfinite(feat).all(axis=1)
    if not complete.any():
        raise ValueError("no row has a complete feature vector")
    first = int(np.argmax(complete))
    if not complete[first:].all():
        bad = frame.index[first:][~complete[first:]][0]
        raise ValueError(f"feature NaN/inf after warm-up at {bad}")
    bpy = bars_per_year or estimate_bars_per_year(frame.index)

    close = frame["close"].to_numpy(np.float64)
    high = frame["high"].to_numpy(np.float64)
    low = frame["low"].to_numpy(np.float64)
    n = len(close)
    nxt = slice(1, n)
    cur = slice(0, n - 1)

    long_ret = np.zeros(n)
    short_ret = np.zeros(n)
    short_growth = np.ones(n)
    long_worst = np.zeros(n)
    short_worst = np.zeros(n)
    carry_long = np.zeros(n)
    carry_short = np.zeros(n)

    long_ret[cur] = close[nxt] / close[cur] - 1.0
    long_worst[cur] = np.minimum(low[nxt] / close[cur] - 1.0, long_ret[cur])
    if kind == "binary":
        res = float(frame.attrs["resolution"])
        no_close = 1.0 - close
        short_ret[cur] = no_close[nxt] / no_close[cur] - 1.0
        short_worst[cur] = np.minimum((1.0 - high[nxt]) / no_close[cur] - 1.0, short_ret[cur])
        # Settlement: the last bar's close converts to the resolved value.
        long_ret[n - 1] = res / close[n - 1] - 1.0
        short_ret[n - 1] = (1.0 - res) / no_close[n - 1] - 1.0
        long_worst[n - 1] = long_ret[n - 1]
        short_worst[n - 1] = short_ret[n - 1]
        short_growth = 1.0 + short_ret
        half = costs.half_spread_abs
        fee = costs.fee_bps * 1e-4
        cost_long = fee + half / close
        cost_short = fee + half / no_close
    else:
        short_ret[cur] = -long_ret[cur]
        short_worst[cur] = -(np.maximum(high[nxt] / close[cur] - 1.0, long_ret[cur]))
        short_growth[cur] = 1.0 + long_ret[cur]
        per_bar = 1.0 / bpy
        if kind == "perp":
            funding = frame["funding_rate"].to_numpy(np.float64)
            carry_long[cur] = funding[nxt]
            carry_short[cur] = -funding[nxt]
        elif kind == "spot":
            carry_short[cur] = costs.borrow_apr * per_bar
        elif kind == "fx":
            # A long holds the base currency and borrows the quote currency, so it earns the
            # rate differential; swap_apr_* is the broker's financing markup on top.
            rd = (frame["rate_diff"].to_numpy(np.float64) if "rate_diff" in frame.columns
                  else np.zeros(n))
            carry_long[cur] = (costs.swap_apr_long - rd[cur]) * per_bar
            carry_short[cur] = (costs.swap_apr_short + rd[cur]) * per_bar
        c = (costs.fee_bps + costs.half_spread_bps) * 1e-4
        cost_long = np.full(n, c)
        cost_short = np.full(n, c)

    rows = slice(first, n)
    return MarketTape(
        symbol=frame.attrs["symbol"],
        kind=kind,
        bar=frame.attrs["bar"],
        times=frame.index.as_unit("ns").asi8[rows].copy(),
        close=close[rows].copy(),
        features=np.ascontiguousarray(feat[rows]),
        feature_names=tuple(features.columns),
        long_ret=long_ret[rows],
        short_ret=short_ret[rows],
        short_growth=short_growth[rows],
        long_worst=long_worst[rows],
        short_worst=short_worst[rows],
        carry_long=carry_long[rows],
        carry_short=carry_short[rows],
        cost_long=np.asarray(cost_long, dtype=np.float64)[rows].copy(),
        cost_short=np.asarray(cost_short, dtype=np.float64)[rows].copy(),
        bars_per_year=float(bpy),
        max_leverage=float(costs.max_leverage),
        maint_margin=float(costs.maint_margin),
        impact=float(costs.impact),
        terminal_at_end=kind == "binary",
    )
