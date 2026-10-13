"""Causal short-rate differentials per forex pair, from FRED.

``load_short_rate`` returns one currency's short rate as a decimal per year, on a daily UTC
index. A value takes effect at the first instant it could be known: a monthly average for
month M at 00:00 UTC on the first day of M + 1, a daily rate for day D one business day
after D. ``rate_diff`` reads the value in effect at each timestamp, so a bar only reads
values stamped at or before its open. The stamps assume no publication lag beyond the end
of the period.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

from sactrader.data.cache import cache_dir, http_get

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Currency -> monthly short rate in percent: the 3-month interbank rate where FRED has one,
# otherwise the call money/interbank rate.
SHORT_RATE_SERIES: dict[str, str] = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N",
    "MXN": "IR3TIB01MXM156N",
    "ZAR": "IR3TIB01ZAM156N",
    "BRL": "IRSTCI01BRM156N",
    "INR": "IRSTCI01INM156N",
    "KRW": "IR3TIB01KRM156N",
    "CNY": "IR3TIB01CNM156N",
}

# Currencies with neither a 3-month nor a call money/interbank series on FRED. Their pairs read
# NaN, which attach_rate_diff stores as 0.0.
NO_SHORT_RATE: tuple[str, ...] = ("THB", "SGD")

# Currency -> daily FRED rate that takes over once the monthly series stops early.
SHORT_RATE_EXTENSION: dict[str, str] = {
    "EUR": "ECBESTRVOLWGTTRMDMNRT",  # euro short-term rate, daily from 2019-10
    "GBP": "IUDSOIA",  # sterling overnight index average, daily from 1997-01
}

MAX_CACHE_AGE_S = 86400
# Minimum gap between requests made by this module.
REQUEST_INTERVAL_S = 0.2

_last_request = 0.0


def _pace() -> None:
    global _last_request
    wait = _last_request + REQUEST_INTERVAL_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _fred_csv(series_id: str) -> bytes:
    path = cache_dir("fred") / f"{series_id}.csv"
    if path.exists() and time.time() - path.stat().st_mtime <= MAX_CACHE_AGE_S:
        return path.read_bytes()
    _pace()
    body = http_get(FRED_CSV_URL, params={"id": series_id}).content
    # Write under a temporary name so an interrupted download never leaves a short cache file.
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return body


def _stamped(series_id: str, *, monthly: bool) -> pd.Series:
    """Decimal values indexed by the UTC instant each one takes effect."""
    raw = pd.read_csv(io.BytesIO(_fred_csv(series_id)), na_values=["."])
    # Columns are read by position, so the header names do not matter.
    dates = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0]))
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(np.float64) / 100.0
    if monthly:
        # The average of month M is only complete once M has ended.
        effective = (dates.to_period("M") + 1).to_timestamp()
    else:
        effective = dates + pd.offsets.BDay(1)
    rate = pd.Series(values, index=effective.tz_localize("UTC"), name=series_id)
    rate = rate.dropna().sort_index(kind="stable")
    # Two source days can map to one business day (a weekend value), so keep the later one.
    return rate[~rate.index.duplicated(keep="last")]


def _check_currency(ccy: str) -> None:
    if ccy not in SHORT_RATE_SERIES:
        raise ValueError(f"no short rate for {ccy!r}, expected one of {sorted(SHORT_RATE_SERIES)}")


def load_short_rate(ccy: str) -> pd.Series:
    """Short rate of ``ccy`` as a decimal per year, one value per UTC day from its first stamp.

    Each day holds the last value in effect at 00:00 UTC. Raises ``ValueError`` for a
    currency without a series.
    """
    _check_currency(ccy)
    rate = _stamped(SHORT_RATE_SERIES[ccy], monthly=True)
    if ccy in SHORT_RATE_EXTENSION:
        daily = _stamped(SHORT_RATE_EXTENSION[ccy], monthly=False)
        # Keep the monthly values up to their last stamp, then take the daily values.
        rate = pd.concat([rate, daily[daily.index > rate.index[-1]]])
    days = pd.date_range(rate.index[0], rate.index[-1], freq="D", tz="UTC")
    return rate.reindex(days, method="ffill").rename(ccy)


def _split(pair: str) -> tuple[str, str]:
    if len(pair) != 6:
        raise ValueError(f"pair must be six letters such as 'EURUSD', got {pair!r}")
    base, quote = pair[:3], pair[3:]
    for ccy in (base, quote):
        if ccy not in NO_SHORT_RATE:
            _check_currency(ccy)
    return base, quote


def _as_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def rate_diff(pair: str, index: pd.DatetimeIndex) -> np.ndarray:
    """Base minus quote short rate at each timestamp, as a decimal per year.

    Each timestamp reads the last value stamped at or before it. The result is NaN wherever
    either currency has no value yet, and everywhere when either one is in ``NO_SHORT_RATE``.
    """
    base, quote = _split(pair)
    t = _as_utc(index)
    if base in NO_SHORT_RATE or quote in NO_SHORT_RATE:
        return np.full(len(t), np.nan)
    r_base = load_short_rate(base).reindex(t, method="ffill").to_numpy(np.float64)
    r_quote = load_short_rate(quote).reindex(t, method="ffill").to_numpy(np.float64)
    return r_base - r_quote


def attach_rate_diff(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of an fx bar frame with a ``rate_diff`` column, 0.0 where the rate is undefined."""
    kind = frame.attrs.get("kind")
    if kind != "fx":
        raise ValueError(f"attach_rate_diff needs an fx frame, got kind {kind!r}")
    diff = rate_diff(frame.attrs["symbol"], frame.index)
    out = frame.copy()
    out["rate_diff"] = np.where(np.isnan(diff), 0.0, diff)
    return out
