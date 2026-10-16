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
- F8, DALA (drawdown-adaptive loss aversion). Added by Amendment 1 below. Configs: DALA (full), DALA-off (dual fixed at zero), DALA-fixed (dual fixed at one). N = 3.

Total N = 18 after Amendment 1. Every config is reported, pass or fail.

## Pass rule

1. Development: net maximum drawdown at most 25% over 1987 to 2005, and in every stress window inside it.
2. Among configs that pass 1, the one with the highest net CAGR on development goes to test. Ties go to the lower drawdown.
3. Test: net maximum drawdown at most 25% over 2006 to 2016-09, and in each stress window inside it. Net CAGR above cash. DSR above 0.95 at n_trials = 18. Net CAGR still positive at twice the costs.
4. Check window and satellite: reported with the same metrics. They do not change the choice.

## Metrics for every config

Net CAGR at 1x and 2x costs, annualised volatility, Sharpe against cash, maximum drawdown for the full period and each stress window, average annual turnover, share of weeks in cash, and DSR on test.

## Not covered

- Taxes, financing and ETF tracking error.
- Dividends on equity indices.
- Any family not listed here. A new family needs a dated amendment made before any result.
- Using the check window to choose. It is reported, not used.

## Amendment 1 (2026-10-16, before any result from this search)

Family F8 is added. It is a learner invented for this search and is tested on the same protocol as the other families. Its three configs raise the total to N = 18.

Specification. Every quantity is causal and weekly.

- Features at the close of week t: m12, the trailing 52-week return of the equal-weight core basket minus its cash return; v, the EWMA volatility of basket returns (lambda 0.94), annualised; d, the drawdown of the DALA portfolio from its own high-water mark. Each feature is z-scored with the mean and standard deviation of all earlier values. phi_t = (1, z(m12), z(v), z(d)).
- Dual variable: lambda_t = min(20, max(0, lambda_{t-1} + 2.0 (d_t - 0.15))), with lambda_{-1} = 0. DALA-off sets lambda = 0. DALA-fixed sets lambda = 1.
- Exposure: e_t = sigmoid(theta_t . phi_t), held over week t+1, with theta_0 = 0. The risky sleeve holds the four core assets weighted by inverse trailing 52-week volatility, scaled by e_t. The rest is cash.
- Learning, at the close of week t+1. Let R be the basket return of week t+1 with those inverse-volatility shares at gross 1, c the cash return, and P = e_t R + (1 - e_t) c. Then g = (R - c)/(1 + P) - lambda_t max(0, -R), and theta_{t+1} = theta_t + 0.1 clip(g, -1, 1) e_t (1 - e_t) phi_t.
- The update is one step. There is no gradient through the wealth path and no look-ahead.
- Learning starts on the first Friday of 1986, so the learner is warmed up when scoring starts in 1987. theta and lambda carry through development, test and check, with no reset and no refit. Their paths are saved for inspection.
- The constants are fixed here. Changing one needs a new dated amendment and a new trial count.

Novelty claim. The learner's loss aversion is a dual variable on its own realised drawdown. It rises while the portfolio sits more than 15% below its high-water mark, and it falls back to pure log-growth otherwise. The closest prior work, from search snippets rather than the full papers, is: De Franco, Nicolle and Pham (arXiv 2010.15779), who derive a drawdown-based cap on risky weights for an investor who learns an unknown drift; Anderson, Iosifidis and Leith, "Lazy Lagrangians with Predictions for Online Learning" (arXiv 2201.02890), who treat time-varying constraints in online learning; and CPPI variants, which set their multiplier by hand. No learner found updates its exposure rule online through a drawdown dual.

Correction, 2026-10-16. This changes wording only, and no constant changed. The De Franco et al. paper treats the drift as unknown, with a prior, and learns it under partial information. The earlier "known drift" wording was wrong. The Anderson et al. paper is "Lazy Lagrangians with Predictions for Online Learning". Both were checked from abstracts only. The novelty claim is therefore narrower: no learner found combines an online logistic exposure rule with a dual variable on its own drawdown. The Bayesian drawdown work is the nearest prior art and should be read in full before any novelty claim is made.

Correction 2, 2026-10-16. Wording only, no constant changed. DALA's parts are published. They are drawdown-cushion exposure (Grossman and Zhou 1993; CPPI and TIPP); a Lagrangian-style multiplier with projected updates (RCPO, 2019; PID Lagrangian, 2020; virtual queues, Yu and Neely); online exposure from side information (Cover and Ordentlich, 1996); and loss aversion that rises after losses (Barberis, Huang and Santos, 2001). Risk-constrained RL portfolios with an augmented Lagrangian (Enkhsaikhan and Jo, 2025) are a further relevant reference. DALA is therefore a new combination of published parts, not a new part. Its multiplier is updated like a dual variable, but it scales the sleeve's downside return rather than enforcing a constraint, so no duality guarantee applies. Read it as state-dependent loss aversion.
