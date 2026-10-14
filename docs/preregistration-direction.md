# Directional prediction study: pre-registered design

Registered on 2026-10-14, before any of these tests was run. Results will be reported against this design, including every failed hypothesis.

Question: is there a weak but real directional signal (annual Sharpe 0.3-0.8) that survives costs and honest multiple-testing? Earlier SAC runs had about 3 years of data, too little to detect such an edge (t ~ Sharpe * sqrt(years)). This study buys statistical power with long histories and wide cross-sections, and fixes the set of tests before seeing any result.

## Data

L, long daily panel (FRED, 1971 onward where available):

- FX vs USD: GBP (DEXUSUK), JPY (DEXJPUS), CHF (DEXSZUS), CAD (DEXCAUS), AUD (DEXUSAL), EUR (DEXUSEU, 1999+). Returns include no interest differential (price return only; stated as a limitation).
- Equity: NASDAQCOM (1971+), NIKKEI225 (1971+). Price index, no dividends.
- Bond: US 10y total-return proxy from DGS10: r_t = y_{t-1}/252 - 8 * (y_t - y_{t-1}) (yields in decimals).
- Oil: DCOILWTICO (1986+); drop non-positive prices.

Costs per unit of turnover: FX 2 bp, bond 2 bp, equity 5 bp, oil 5 bp.

X, crypto cross-section: Binance spot USDT pairs, daily close, 2019-01 onward, every symbol available (including delisted ones if the archives have them; if not, state the survivorship bias). Exclude stablecoins and leveraged tokens. A day is used only if at least 30 coins have data. Cost 10 bp per side.

## Hypotheses

11 trials in total: H1a-d, H2a-c, H3a-b, H4a-b. Every one is reported, pass or fail.

- **H1a/H1b/H1c**, time-series momentum on L: position = sign(trailing 21 / 63 / 252-day return) * 10% / (63-day ex-ante annualised vol), capped at 2x per asset, equal risk across assets.
- **H1d**: equal-weight combination of H1a-c.
- **H2a**, crypto 1-day reversal on X: long the bottom quintile, short the top quintile of yesterday's return.
- **H2b/H2c**, crypto momentum on X: same construction on trailing 7 / 28-day return (long top, short bottom). H2 portfolios are dollar-neutral and equal-weighted.
- **H3a**, ridge predictor on L: features = trailing returns over 1/5/21/63/252 days and 21/63-day vol, each z-scored causally per asset; target = next-day return / 63-day vol; pooled across assets; walk-forward with an expanding window, first 10 years burn-in, refit yearly, ridge alpha fixed at 1.0 (not tuned); position = clip(prediction / its trailing std, -2, 2) scaled to 10% vol.
- **H3b**: the same ridge on X, predicting next-day cross-sectional rank; long top, short bottom quintile.
- **H4a/H4b**, SAC check: only if a hypothesis passes, SAC as a residual on that signal (the project's prior mechanism), one run per passing family, at most two. Unused H4 slots still count as trials.

Timing for all: signal from data up to the close of day t; position held over day t+1. No parameter is tuned; the numbers above are the only values tried.

## Statistics and pass rule

Per hypothesis: annualised Sharpe after costs, Newey-West t (5 lags), deflated Sharpe ratio at 11 trials, Sharpe in each half of the sample, and the minimum detectable Sharpe for that sample length (80% power, 5% two-sided).

Pass = Newey-West t > 2.5 AND DSR > 0.95 AND Sharpe > 0 in both halves.

## Controls

- **Null panels**: every hypothesis is also run on 5 random-walk panels with the same assets, dates and vols. The share of null runs that pass is reported as the false-positive rate.
- **Delay check**: each signal delayed by 1 extra day (position held over t+2) should give a similar or weaker result. A delayed signal that does clearly better points to a timing bug.

## Amendments before any run (2026-10-14)

Five changes were made to the design above after a code review. No test had been run on real data when they were made. Each change is listed with its reason.

1. **Delisting.** A coin leaves the cross-section on the first day it has no print. Its signal is missing from that day, so its weight goes to zero and the exit trade pays the cost for its asset class. Reason: a missing print was treated as a zero return, so a delisted coin kept a valid signal and could remain in a quintile after it stopped trading. This rule applies on the crypto panel, where a missing print means the coin has left the market.
2. **Ridge refit timing.** A refit on day d trains only on targets realised before day d. Reason: the earlier code included the target realised on the refit day itself, which is one day later than the timing rule allows.
3. **Missing prints on calendar days.** On the FRED panel, a day on which an asset has no print, such as a holiday in that market, keeps the asset's previous position, and the asset takes no new signal that day. Reason: a holiday in one market would otherwise close and reopen the position, paying costs that are not part of the strategy.
4. **Null panels.** Each hypothesis is run on 100 random-walk panels, replacing the 5 panels stated above. Reason: with 5 panels the false-positive rate can take only six values and cannot measure a 5% rate. The report gives the share of null panels that pass and its 95% Wilson interval.
5. **Deflated Sharpe ratio.** The registered figure stays the primary result: the repo's deflated Sharpe function at 11 trials. A second column uses the variance of the trial Sharpe ratios observed in the run. Reason: the registered figure takes the cross-trial spread from a null variance, and the second column shows whether the conclusion depends on that choice.
