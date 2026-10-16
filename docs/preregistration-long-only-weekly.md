# Long-only weekly search: pre-registered design

Registered on 2026-10-16, before any result from this search exists. The direction to search for weekly, long-only ideas lifts the 14 October pause on new research for this search. The drawdown-budget checks registered earlier are separate and unchanged.

Question: which long-only, weekly-rebalanced rule or model gives the highest net CAGR with a maximum drawdown of at most 25%, on long history, after costs?

## Constraints

- Long only. Weights are at least 0. Gross exposure is at most 1, and unused weight sits in cash (DTB3).
- Weekly rebalance at the Friday close. A signal uses data up to the close of week t, and the position is held over week t+1.
- No leverage, shorts, options or intraweek trading.
- Turnover band: an asset is rebalanced only when its weight moves by more than 5 percentage points.
- Costs per unit of turnover: NASDAQ and NIKKEI 5 bp, 10-year Treasury 2 bp, WTI 5 bp, cash 0. Crypto 10 bp per side.
- Taxes and financing are not modelled. Short-term gains tax on frequent rotation is a real cost this study leaves out.

## Data

Core, from 1971:

- NASDAQCOM and NIKKEI225: daily price indices. Dividends are excluded, so equity return is understated.
- 10-year Treasury total-return proxy from DGS10, as in the directional study: r_t = y_{t-1}/252 - 8 x (y_t - y_{t-1}), with yields in decimals.
- WTI from DCOILWTICO, from 1986.
- Cash from DTB3.

Features for the ML families only: DGS2, DGS10, DGS5, VIXCLS (from 1990), VXNCLS, OVXCLS, T10YIE and DFII10.

Satellite, exploratory: Binance spot daily closes (USDT pairs) from 2019. The coin list is chosen today, so survivorship bias is present and is flagged in every result.

## Periods

- Common evaluation starts on the first Friday of 1987. ML families train on data before each January and refit every year from 1986.
- Development (selection): 1987 to 2005.
- Test (scored once, for the candidate chosen on development): 2006 to 2016-09.
- Check (reported, never used to choose): 2016-10 to 2026-10. Earlier mix work looked at this window, so it is a check, not a holdout.
- Satellite: 2019 to 2026, exploratory only.
- Stress windows reported: October 1987; 1990; 2000 to 2002; 2008 to 2009; 2011; February and March 2020; 2022.

## Shared rule

Active assets are weighted by inverse trailing 52-week volatility. The book is then scaled so that its trailing 52-week volatility equals the target (10% unless a config says otherwise). Gross exposure is capped at 1, and the rest is cash.

## Families and trial count

Each family's signal decides which assets are active each week. The shared rule sets the weights.

- F1, trend gate. An asset is active when its trailing total return is positive. Configs: 10-month, 12-month, and close above its 200-day average. N = 3.
- F2, dual momentum. Among NASDAQ, NIKKEI, the 10-year and WTI, the asset with the best 12-month return is active, if that return beats cash over the same period. Otherwise all cash. Configs: 12-month and 6-month lookback. N = 2.
- F3, cross-sectional momentum, long only. The top k assets by 12-1-month return (skipping the last month) are active. Configs: k = 1 and k = 2. N = 2.
- F4, risk parity. All four core assets are active. Configs: target 10% and 12%. N = 2.
- F5, ML ranking. Predict each asset's next-week return from: 1-, 3- and 12-month returns; 1-month reversal; 13- and 52-week volatility; distance from the 52-week high; 10-year minus 2-year yield; and 10-year yield change. Models: ridge (alpha 1.0) and shallow gradient-boosted trees (depth 2, 50 trees, learning rate 0.05, at least 200 observations per leaf). Active: the top 2 by predicted return. Configs: one per model. N = 2.
- F6, ML volatility sizing. The core assets are weighted by the shared rule. The gate is the equal-weight core basket's 10-month trend: below its trend, all cash. Above it, the book is scaled so that its forecast next-4-week volatility equals 10%. Forecasts: EWMA with lambda 0.94, and gradient-boosted trees on the F5 features plus VIXCLS, VXNCLS and OVXCLS. N = 2.
- F7, crypto satellite. BTC and ETH with a 10-month trend gate and the shared rule at 10%. Second config: the same on the top 10 coins by volume as of registration. Evaluated on 2019 to 2026 only. A satellite is considered only if it passes the drawdown rule on its own. N = 2.

Total N = 15. Every config is reported, pass or fail.

## Pass rule

1. Development: net maximum drawdown at most 25% over 1987 to 2005, and in every stress window inside it.
2. Among configs that pass 1, the one with the highest net CAGR on development goes to test. Ties go to the lower drawdown.
3. Test: net maximum drawdown at most 25% over 2006 to 2016-09, and in each stress window inside it. Net CAGR above cash. DSR above 0.95 at n_trials = 15. Net CAGR still positive at twice the costs.
4. Check window and satellite: reported with the same metrics. They do not change the choice.

## Metrics for every config

Net CAGR at 1x and 2x costs, annualised volatility, Sharpe against cash, maximum drawdown for the full period and each stress window, average annual turnover, share of weeks in cash, and DSR on test.

## Not covered

- Taxes, financing and ETF tracking error.
- Dividends on equity indices.
- Any family not listed here. A new family needs its own registration.
- Using the check window to choose. It is reported, not used.
