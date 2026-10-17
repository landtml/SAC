"""Weekly macro panel for the parity study: equity and oil legs, Treasury total-return proxies.

The loader reads cached FRED files and never downloads. Every input series is cut at END before
any computation, so no row after END can change the panel. Weeks end on Fridays, as in parity_fx.

    SOFTEDGE_DATA=<data folder> python scripts/parity_macro.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from parity_fx import STALE_DAYS, USD_CASH, as_of, read_series

END = pd.Timestamp("2005-12-30")
TRADING_DAYS = 252
CASH_DAYS = 7

# name: (FRED series, kind, modified duration in years for yield legs)
SERIES = {
    "NASDAQ": ("NASDAQCOM", "price", None),
    "NIKKEI": ("NIKKEI225", "price", None),
    "WTI": ("DCOILWTICO", "price", None),
    "BRENT": ("DCOILBRENTEU", "price", None),
    "UST10": ("DGS10", "yield", 8.0),
    "UST5": ("DGS5", "yield", 4.5),
    "UST2": ("DGS2", "yield", 1.9),
}

# Cost per unit traded: equities and oil, then Treasuries.
COSTS = {
    "NASDAQ": 0.0005, "NIKKEI": 0.0005, "WTI": 0.0005, "BRENT": 0.0005,
    "UST10": 0.0002, "UST5": 0.0002, "UST2": 0.0002,
}


def _load(root: str, series_id: str) -> pd.Series:
    s = read_series(root, series_id)
    return s[s.index <= END]


def _grid(name: str, root: str) -> tuple[pd.Series, pd.DatetimeIndex, np.ndarray]:
    """The cut series, the Fridays from its first to its last observation, and the stale flag.

    The stale flag marks a Friday whose last observation on or before it is more than
    STALE_DAYS old. Levels exist on every Friday of the grid, so no week is dropped here.
    """
    s = _load(root, SERIES[name][0])
    fridays = pd.date_range(s.index[0], s.index[-1], freq="W-FRI")
    obs = s.index[s.index.searchsorted(fridays, side="right") - 1]
    stale = np.asarray((fridays - obs).days > STALE_DAYS)
    return s, fridays, stale


def weekly_panel(name: str, root: str) -> pd.DataFrame:
    """Weekly log returns of the risky leg and of US cash, one row per week ending on a Friday.

    A price leg is the log change of the as-of Friday close. A yield leg is the sum of the daily
    total-return proxies log(1 + y[d-1] / 252 - D * (y[d] - y[d-1])) over the trading days after
    the previous Friday, with yields in decimals. Cash is DTB3 on the Friday that starts the week,
    accrued over 7 days. The panel starts at the first week where risky and cash both exist.
    """
    kind, duration = SERIES[name][1], SERIES[name][2]
    s, fridays, stale = _grid(name, root)
    if len(fridays) < 2:
        raise RuntimeError(f"{name}: fewer than two Fridays between its first and last observation")
    start, end = fridays[:-1], fridays[1:]
    if kind == "price":
        risky = np.diff(np.log(as_of(s, fridays).to_numpy()))
    else:
        y = s.to_numpy() / 100.0
        prev, cur = y[:-1], y[1:]
        daily = np.log1p(prev / TRADING_DAYS - duration * (cur - prev))
        bins = fridays.searchsorted(s.index[1:], side="left")
        keep = bins < len(fridays)
        risky = np.bincount(bins[keep], weights=daily[keep], minlength=len(fridays))[1:]
    usd = as_of(_load(root, USD_CASH), start).to_numpy()
    cash = np.log1p(usd / 100.0 * CASH_DAYS / 365.0)
    frame = pd.DataFrame({"risky": risky, "cash": cash, "stale": stale[1:]}, index=end)
    ok = np.isfinite(risky) & np.isfinite(cash)
    if not ok.any():
        raise RuntimeError(f"{name}: no week with both risky and cash")
    frame = frame.iloc[int(np.argmax(ok)):]
    if not np.isfinite(frame[["risky", "cash"]].to_numpy()).all():
        raise RuntimeError(f"{name}: missing values after the first complete week")
    return frame


def coverage(root: str) -> pd.DataFrame:
    """Per series: first and last week with a weekly return, the number of weeks, stale weeks.

    Computes dates and stale flags only, no returns. Cash is not required here.
    """
    rows = []
    for name in SERIES:
        _, fridays, stale = _grid(name, root)
        weeks = fridays[1:]
        rows.append({
            "series": name,
            "first_week": weeks[0].strftime("%Y-%m-%d"),
            "last_week": weeks[-1].strftime("%Y-%m-%d"),
            "weeks": len(weeks),
            "stale_weeks": int(stale[1:].sum()),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = argv[0] if argv else os.path.join(os.environ.get("SOFTEDGE_DATA", "data"), "fred")
    print(coverage(root).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
