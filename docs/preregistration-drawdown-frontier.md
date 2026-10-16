# Drawdown-budget frontier: pre-registered checks

Registered on 2026-10-16. The linear estimates below were calculated from the stress table in the mixed-portfolio report before this registration, and no exact check had been run. The checks test those estimates. They do not choose between alternatives. Results will be reported against this design, including every check that fails.

Question: what in-sample CAGR does each crash budget buy for the 60/20/20 S&P/BTC/carry mix, and does the estimate hold when it is computed exactly?

Scope: verification only. This design tests no new mix, exposure, rule or threshold. The drawdown brake is excluded. It waits for a decision on the 14 October pause on new research.

## Data

- S&P 500: FRED SP500, price index, 2016-10-13 to 2026-10-12. Dividends are excluded, so the series understates total return.
- BTC: Binance BTCUSDT daily spot close, the series used in the mixed-portfolio report.
- Cash: FRED DTB3, the same series as the report.
- Carry: the cash-and-carry sleeve on 10 coins, 2023-01-04 onward. Stress windows replace carry with cash, because the sleeve has no data there.

No network fetches. Missing data is reported, not filled.

## Mixes and exposures

- Mix A: 60% S&P, 20% BTC, 20% carry.
- Mix B: 60% S&P, 40% carry.
- Each mix is scaled to exposure f, with 1 - f in T-bills, rebalanced monthly.
- Exposures: f = 1.00, 0.93, 0.74, 0.56, 0.37. These are the -25%, -20%, -15% and -10% crash budgets divided by mix A's Covid-window return (-26.9%), rounded to two decimals.

## Checks

**C1, stress returns.** For mix A at each exposure, compute the 2022 calendar return and the February-March 2020 return exactly. Estimates, in percent, at f = 1.00, 0.93, 0.74, 0.56, 0.37: 2022 is -24.1, -22.3, -17.4, -12.8, -8.0; Covid window is -26.9, -25.0, -19.9, -15.1, -10.0. The estimates take cash as 1.5% in 2022 and 0% in the Covid window. Pass: each exact value is within 1.0 point of its estimate.

**C2, CAGR and volatility.** For mix A at each exposure, compute in-sample CAGR and annualised volatility, January 2023 to August 2026. Estimates: CAGR = 4.6 + 20.8 x f, giving 25.4, 23.9, 20.0, 16.2 and 12.3. Volatility = 14.8 x f, giving 14.8, 13.8, 11.0, 8.3 and 5.5. Pass: CAGR within 2.0 points and volatility within 1.0 point of the estimate.

**C3, source of the gap.** At the exposure where mix A's Covid-window return is -15.0% (about f = 0.56), compare mix A's CAGR with the same mix with its BTC weight moved to T-bills, exposure re-solved so the Covid-window return is again -15.0%. Also report mix B at its -15% exposure (about f = 0.74). Descriptive; no pass rule.

**C4, deflated Sharpe ratio.** For mix A and mix B, compute the annualised Sharpe ratio of daily excess return over cash, using 252 bars per year. Compute the deflated Sharpe ratio with `src/softedge/eval/stats.py` at n_trials = 8, the eight mixes in the mixed-portfolio report. Report two values: one with the cross-trial variance the function uses by default, and one with the variance of the eight mixes' Sharpe ratios. Pass: mix A's first value is above 0.95. The second value is a sensitivity check, not a second test.

**C5, pre-FTX data.** List the local cache files that hold Binance perpetual funding and spot closes for the 10 cash-and-carry coins from 2022-06-01 to 2023-01-31, the first available date for each coin, and any gaps in November 2022. Descriptive. The answer decides whether the carry sample can be extended past the FTX collapse (8-11 November 2022). Extending it would be a new study and is not part of this design.

**C6, sourced figures.** Each figure quoted in the study summary is checked against the file it cites. The list stays with the report, outside the repository. Pass: each figure matches its source, or the mismatch is reported.

## Trial count and reporting

- Mixes A and B are two of the eight mixes already evaluated, so C4 uses n_trials = 8.
- The checks add no strategy trials. Their results go in `docs/research-log.md` when they are in, pass or fail.

## Not covered

- The drawdown brake and any threshold for it.
- Any new mix, exposure, venue rule or threshold.
- The 2026 hold-out. The mixed-portfolio backtests used data through August 2026, so 2026 is no longer clean. The only clean out-of-sample test left is a forward paper run.

## Amendments

None.
