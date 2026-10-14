"""Long nominal Treasury against short TIPS of one tenor, as a daily bar frame from FRED.

``load_tips_breakeven`` returns a spot-kind frame with symbol ``BE5`` or ``BE10``. One unit of
the frame is long one unit of the constant-maturity nominal Treasury and short one unit of the
TIPS of the same tenor (the constant-maturity real yield). The close starts at 100 on the first
bar and compounds the daily spread return

    r_t = R_nom_t - R_tips_t

where each leg's total return follows the yield-to-return approximation of ``fred_markets``
with one modified duration D and convexity C per tenor:

    R_nom  = y_nom0 * dt - D * dy_nom + 0.5 * C * dy_nom ** 2
    R_tips = y_real0 * dt + accrual - D * dy_real + 0.5 * C * dy_real ** 2

``accrual`` is the relative change of TIPS reference CPI over the interval, the index-ratio
accrual that TIPS pay on top of the real yield. Reference CPI follows the TIPS convention: for a
day in month K it interpolates from CPI of month K-3 (day 1) toward CPI of month K-2 (last day of
the month). The accrual therefore uses CPI that is at least a month old: no day reads the CPI
of its own month.

The bill financing of the two legs cancels. The long nominal is financed at the bill rate, and
the short TIPS earns the same rebate, so the frame needs no bill rate. The frame omits the
shortfall of the TIPS rebate below the bill rate, a charge on the short TIPS leg. A long spread
holds that leg, so the omission flatters a long spread by the shortfall per year. A short spread
owes no TIPS borrow, so it is not affected. The shortfall is not sourced, and the frame has no
borrow term that a caller could set.

The breakeven is the nominal yield minus the real yield. Its expected inflation part is priced
at ``prior``: the breakeven in percentage points minus the trailing 12-month CPI inflation in
percentage points, using only CPI already released at the bar's close. Each CPI month M is
treated as released on the 16th of M+1, a fixed conservative stamp. The prior is the
``tips_prior`` column, in percentage points. The config sets its gain, and the action map clips
the scaled prior at 1.

A month missing from the FRED CPI series (FRED has no October 2025 value) carries the previous
month's CPI forward, so its accrual is zero. Whether BLS published that month is not verified
here.

The high and low columns are synthetic: the max and min of open and close, not intrabar
prices. The tape's worst-case returns, which liquidation reads, therefore equal the close-to-close
move. At 4x leverage and a 5 percent maintenance margin, liquidation needs a one-day move of about
20 percent, more than 60 daily standard deviations of these spreads, so the synthetic range does
not bind at the measured volatilities.

Approximations: one duration and convexity per tenor (a par bond at a 3 percent yield,
computed for this module, not sourced), constant-maturity yields in place of a bond's own
cash flows, no roll cost (a position in futures or on-the-run issues rolls, and this frame does
not), and no margin or repo haircut on the spread. The FRED CPI series is the current
vintage; no release vintages are loaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from softedge.data.fred_markets import _observations, _window_end
from softedge.data.schema import make_frame

CPI_SERIES = "CPIAUCNS"
# CPI of month M is stamped on this day of month M+1. Fixed, not the BLS calendar.
CPI_RELEASE_DAY = 16

# Modified duration and convexity of a par bond at a 3 percent yield, semiannual coupons.
TENORS: dict[str, dict] = {
    "5": {"nominal": "DGS5", "real": "DFII5", "breakeven": "T5YIE", "duration": 4.6,
          "convexity": 24.4},
    "10": {"nominal": "DGS10", "real": "DFII10", "breakeven": "T10YIE", "duration": 8.6,
           "convexity": 84.6},
}


def load_tips_breakeven(tenor: str, start: str = "2003-01", end: str = "2026-09") -> pd.DataFrame:
    """Daily bars of the nominal-minus-TIPS spread for ``tenor`` (``"5"`` or ``"10"``).

    The series are built on the full FRED history, so the first bar in the window opens at
    the close of the last bar before it. Raises ``ValueError`` for an unknown tenor.
    """
    if tenor not in TENORS:
        raise ValueError(f"unknown tenor {tenor!r}, expected one of {sorted(TENORS)}")
    spec = TENORS[tenor]
    frame = breakeven_frame(
        _observations(spec["nominal"]), _observations(spec["real"]),
        _observations(spec["breakeven"]), _observations(CPI_SERIES),
        duration=spec["duration"], convexity=spec["convexity"], symbol=f"BE{tenor}",
    )
    keep = (frame.index >= pd.Timestamp(start, tz="UTC")) & (frame.index <= _window_end(end))
    out = frame.loc[keep].copy()
    out.attrs = dict(frame.attrs)
    return out


def breakeven_frame(nominal: pd.Series, real: pd.Series, breakeven: pd.Series, cpi: pd.Series,
                    *, duration: float, convexity: float, symbol: str) -> pd.DataFrame:
    """Spread bars from yields in percent (indexed by UTC date) and CPI (by UTC month start).

    Only dates where all three yield series have a value are kept. Raises ``ValueError`` when
    fewer than two dates are common, or when a needed CPI month is missing.
    """
    common = nominal.index.intersection(real.index).intersection(breakeven.index).sort_values()
    if len(common) < 2:
        raise ValueError(f"tenor series share {len(common)} dates, need at least 2")
    yn = nominal.loc[common].to_numpy(dtype=np.float64) / 100.0
    yr = real.loc[common].to_numpy(dtype=np.float64) / 100.0
    dt = np.asarray((common[1:] - common[:-1]).total_seconds()) / (365 * 86400)
    dyn, dyr = np.diff(yn), np.diff(yr)

    ref = reference_cpi(cpi, common)
    accrual = ref[1:] / ref[:-1] - 1.0
    nominal_ret = yn[:-1] * dt - duration * dyn + 0.5 * convexity * dyn**2
    tips_ret = yr[:-1] * dt + accrual - duration * dyr + 0.5 * convexity * dyr**2
    ret = nominal_ret - tips_ret

    close = 100.0 * np.concatenate(([1.0], np.cumprod(1.0 + ret)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    prior = breakeven.loc[common].to_numpy(dtype=np.float64) - trailing_inflation(cpi, common)
    frame = make_frame(
        common, open_, np.maximum(open_, close), np.minimum(open_, close), close,
        symbol=symbol, kind="spot", bar="1D",
    )
    frame["tips_prior"] = prior
    return frame


def reference_cpi(cpi: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """TIPS reference CPI on each date: CPI of month K-3 at day 1, linear toward K-2 at month end."""
    table = _cpi_table(cpi)
    serial = _serial(dates)
    lag3 = _lookup(table, serial - 3)
    lag2 = _lookup(table, serial - 2)
    frac = (np.asarray(dates.day) - 1) / np.asarray(dates.days_in_month)
    return lag3 + frac * (lag2 - lag3)


def trailing_inflation(cpi: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Trailing 12-month CPI inflation in percent, from the CPI released by each date's close."""
    table = _cpi_table(cpi)
    serial = _serial(dates)
    # The month released by the close of day d: M-1 from the release day on, else M-2.
    latest = serial - np.where(np.asarray(dates.day) >= CPI_RELEASE_DAY, 1, 2)
    return 100.0 * (_lookup(table, latest) / _lookup(table, latest - 12) - 1.0)


def _serial(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(dates.year) * 12 + np.asarray(dates.month) - 1


def _cpi_table(cpi: pd.Series) -> pd.Series:
    # A month with no print takes the previous month's value. Interpolating would read the
    # next print, which is not yet released on the days that use the gap month.
    table = pd.Series(cpi.to_numpy(dtype=np.float64), index=_serial(cpi.index))
    return table.reindex(pd.RangeIndex(table.index.min(), table.index.max() + 1)).ffill()


def _lookup(table: pd.Series, serial: np.ndarray) -> np.ndarray:
    values = table.reindex(serial).to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError("CPI is not loaded for a month the TIPS spread needs")
    return values
