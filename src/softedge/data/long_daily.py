"""Long daily return panel from FRED: FX, equity indices, a 10-year bond proxy and WTI oil.

``load_daily_panel`` returns a wide frame of daily simple returns with one column per asset,
and a cost series in basis points per unit of turnover, indexed by the same asset names.
Each asset keeps its own history. A date with no observation for an asset is NaN in that
column, and no value is filled across it.

FX returns are price returns in USD terms. FRED quotes JPY, CHF and CAD as units per dollar,
so their levels are inverted first. No interest differential enters the FX returns. The
equity indices are price indices, so dividends are excluded.

The UST10 return is a bond proxy from the yield in percent on DGS10:
``r_t = y_{t-1} / 252 - 8 * (y_t - y_{t-1})`` with yields in decimals. The duration is fixed
at 8 and carries no financing term.

WTI drops non-positive prints before the returns are taken, since FRED's WTI series holds
-36.98 on 2020-04-20.

Series are read through the FRED helper in ``softedge.data.fred_markets``, which caches under
``cache_dir("fred")``.
"""

from __future__ import annotations

import pandas as pd

from softedge.data.fred_markets import _observations

FRED_SERIES: dict[str, str] = {
    "GBP": "DEXUSUK",
    "JPY": "DEXJPUS",
    "CHF": "DEXSZUS",
    "CAD": "DEXCAUS",
    "AUD": "DEXUSAL",
    "EUR": "DEXUSEU",
    "NASDAQ": "NASDAQCOM",
    "NIKKEI": "NIKKEI225",
    "WTI": "DCOILWTICO",
    "UST10": "DGS10",
}

# Series quoted as units of the currency per US dollar, inverted to the dollar value of the currency.
INVERTED_FX: tuple[str, ...] = ("JPY", "CHF", "CAD")

# Cost per unit of turnover in basis points.
COST_BP: dict[str, float] = {
    "GBP": 2.0, "JPY": 2.0, "CHF": 2.0, "CAD": 2.0, "AUD": 2.0, "EUR": 2.0,
    "UST10": 2.0,
    "NASDAQ": 5.0, "NIKKEI": 5.0, "WTI": 5.0,
}

BOND_DURATION = 8.0
TRADING_DAYS = 252

# The registered start date. Observations before it are dropped for every series before any return is taken.
REGISTERED_START = "1971-01-01"


def bond_returns(yield_pct: pd.Series) -> pd.Series:
    """Daily bond proxy return from a yield series in percent, stamped on the later date."""
    y = yield_pct / 100.0
    prev = y.shift(1)
    return (prev / TRADING_DAYS - BOND_DURATION * (y - prev)).dropna()


def asset_returns(asset: str, start: str = REGISTERED_START) -> pd.Series:
    """Daily simple returns of one asset, indexed by UTC date. Raises ``KeyError`` for an unknown asset.

    Observations dated before ``start`` are dropped before the returns are taken, so the first
    return is built from prints on or after ``start`` only.
    """
    values = _observations(FRED_SERIES[asset])
    values = values[values.index >= pd.Timestamp(start, tz="UTC")]
    if asset == "UST10":
        return bond_returns(values)
    if asset == "WTI":
        values = values[values > 0.0]
    if asset in INVERTED_FX:
        values = 1.0 / values
    return values.pct_change().dropna()


def load_daily_panel(start: str = REGISTERED_START) -> tuple[pd.DataFrame, pd.Series]:
    """Wide frame of daily returns (dates by assets) and the cost series in bp per unit turnover.

    Every series is trimmed to observations on or after ``start`` (default: the registered
    1971-01-01) before any return is computed.
    """
    returns = pd.DataFrame({asset: asset_returns(asset, start) for asset in FRED_SERIES}).sort_index()
    returns.index.name = "date"
    costs = pd.Series({asset: COST_BP[asset] for asset in returns.columns}, name="cost_bp")
    return returns, costs
