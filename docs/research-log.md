# Research log

This log lists every market framing and algorithm change the project has built, what happened when
it ran, and where the result is reported. Read it before starting a new run, so a variant that was
already tested is not run again by mistake.

## How to use this log

Before a new market framing or algorithm change, search this page for its universe and its change.
If a row exists, read the cited section first.

Add a row when a framing or change is built. Update its status when it runs, and its outcome when the
result is in. Each outcome with a result cites a section of the [results page](results.md) or a run name.

Run names refer to output folders kept outside the repository (`runs/` is git-ignored). Each run name
matches the `name:` field of a config in [configs/](../configs/README.md), which is the file to rerun.

Status values:

- **built**: the config is on main and has not run.
- **queued**: the config is in the run queue and has no result yet.
- **paused**: the config is queued and not started, because the queue is paused.
- **running**: the run is in progress and has no result file yet.
- **finished**: the run completed and wrote its result file.
- **run**: the run wrote a result.
- **published**: the result appears on the [results page](results.md).
- **stopped** or **failed**: the run ended without a result. See [unfinished runs](#unfinished-runs).

Outcome values:

- **edge**: the paired 95% interval for the difference from the reference lies entirely above zero.
- **no edge**: the interval includes zero, or lies below zero.
- **superseded**: a later config replaces this one. The reason is given in the row.
- **pending**: no result yet.

For the algorithm ablations, the outcome labels follow the results page: **inconclusive** when the
interval includes zero, and **rejected** when the change scores below the base agent.

## Markets and framings

| Framing | Universe | Configs | Status | Outcome | Where |
| --- | --- | --- | --- | --- | --- |
| Planted signal control | Six synthetic markets, 1 year hourly | `synthetic/synthetic_wf.yaml` (run `synthetic_wf`) | published | edge | results.md, Synthetic controls |
| Random walk control | Six synthetic random walks, 1 year hourly | `synthetic/synthetic_null_wf.yaml` (run `synthetic_null_wf`) | published | no edge, below buy-and-hold | results.md, Synthetic controls |
| Hidden-regime market, base agent | Synthetic hidden-regime market, 1 year hourly | `ablation/base.yaml` (run `abl_base`) | published | edge against buy-and-hold; deflated Sharpe low after all variants | results.md, Synthetic controls |
| Crypto perpetuals, hourly, base | 10 coins, hourly | `crypto/crypto_perp.yaml` | built | pending | configs/README.md, Crypto |
| Crypto perpetuals, hourly walk-forward | 10 coins, hourly | `crypto/crypto_perp_wf.yaml` | stopped at fold 0 | pending | runs/crypto_perp_wf_stopped_fold0 |
| Crypto perpetuals, daily walk-forward before the funding fix | 10 coins, daily | `crypto/crypto_perp_1d_wf.yaml` (run `crypto_perp_1d_wf`) | finished | superseded: carried next-bar funding | runs/crypto_perp_1d_wf; configs/README.md |
| Crypto perpetuals, daily, after the funding fix | 10 coins, daily from 2023 | `crypto/crypto_perp_1d_v2.yaml` (run `crypto_perp_1d_v2`) | published | no edge, below trend rule | results.md, Real markets, directional |
| Crypto perpetuals, daily, volatility target | 10 coins, daily | `crypto/crypto_perp_1d_vt.yaml` | failed | pending | runs/crypto_perp_1d_vt.out |
| Crypto perpetuals, trend prior | 10 coins, daily | `crypto/crypto_perp_1d_prior.yaml` (run `crypto_perp_1d_prior`) | finished | not published; no edge | runs/crypto_perp_1d_prior |
| Crypto perpetuals, pathwise objective | 10 coins, daily | `crypto/crypto_perp_1d_path.yaml` (run `crypto_perp_1d_path`) | finished | not published; no edge | runs/crypto_perp_1d_path |
| Crypto perpetuals, implied volatility and regime features | 10 coins, daily | `crypto/crypto_perp_1d_rich.yaml` (run `crypto_perp_1d_rich`) | finished | not published; no edge | runs/crypto_perp_1d_rich |
| Crypto spot, borrow charge | 10 coins, hourly, spot | `crypto/crypto_spot.yaml` | built | pending | configs/README.md, Crypto |
| Crypto cash-and-carry, no prior | 10 coins, daily, 5% borrow | `carry/crypto_basis_1d.yaml` | failed | pending | runs/crypto_basis_1d.out |
| Crypto cash-and-carry, funding prior, no borrow rate | 10 coins, daily | `carry/crypto_basis_1d_prior.yaml` (run `crypto_basis_1d_prior`) | finished | superseded: no borrow charge | runs/crypto_basis_1d_prior |
| Crypto cash-and-carry, funding prior with borrow and deflated gate | 10 coins, daily, 3.7 years | `carry/crypto_basis_1d_prior_v2.yaml` | published | edge over the funding rule | results.md, Real markets, carry |
| Crypto cash-and-carry, premium-index feature | 10 coins, daily | `carry/crypto_basis_1d_prior_prem.yaml` (run `crypto_basis_1d_prior_prem`) | published | no gain over v2 | results.md, Ablations |
| Crypto cash-and-carry, combinatorial purged cross-validation | 10 coins, five paths, 2022-07 to 2026-08 | `carry/crypto_basis_1d_prior_cpcv.yaml` (run `crypto_basis_1d_prior_cpcv`) | published | edge in sign on all five paths; size not established | results.md, Robustness checks |
| Crypto cash-and-carry, wider list | 32 coins, daily | `carry/crypto_basis_wide_prior.yaml` (run `crypto_basis_wide_prior`) | published | edge; deflated Sharpe low | results.md, Robustness checks |
| Hyperliquid basis, funding prior, no borrow rate | Binance spot against Hyperliquid perpetual, daily | `carry/hl_basis_1d_prior.yaml` (run `hl_basis_1d_prior`) | finished | superseded: no borrow charge | runs/hl_basis_1d_prior |
| Hyperliquid basis, funding prior with borrow and deflated gate | Binance spot against Hyperliquid perpetual, daily | `carry/hl_basis_1d_prior_v2.yaml` | published | no edge established | results.md, Real markets, carry |
| Hyperliquid basis, wider list | 32 coins, daily | `carry/hl_basis_wide_prior.yaml` (run `hl_basis_wide_prior`) | published | no gain | results.md, Robustness checks |
| Cross-venue funding spread, no prior | Binance perpetual against Hyperliquid perpetual, daily | `carry/perp_spread_1d.yaml` (run `perp_spread_1d`) | finished | not published; no edge against buy-and-hold | runs/perp_spread_1d |
| Cross-venue funding spread, funding prior | 10 coins, daily, 1.6 years | `carry/perp_spread_1d_prior.yaml` (run `perp_spread_1d_prior`) | published | edge over the funding rule | results.md, Real markets, carry |
| Cross-venue funding spread, wider list | 32 coins, daily | `carry/perp_spread_wide_prior.yaml` (run `perp_spread_wide_prior`) | published | edge | results.md, Robustness checks |
| Dated futures basis, no prior | BTC and ETH quarterly futures against spot, daily | `carry/delivery_basis_1d.yaml` | finished | no demonstrated edge (banded SAC 2.81 vs buy-and-hold 2.57, paired interval crosses zero) | configs/README.md, Carry |
| Dated futures basis, annualized basis prior | BTC and ETH quarterly futures against spot, daily | `carry/delivery_basis_1d_prior.yaml` | finished | no edge | configs/README.md, Carry |
| Pegged pairs, no prior | Stablecoin and liquid-staking spot pairs, hourly | `pegs/pegs_1h.yaml` | running | pending | configs/README.md, Pegs |
| Pegged pairs, peg prior | Stablecoin and liquid-staking spot pairs, hourly | `pegs/pegs_1h_prior.yaml` | running | pending | configs/README.md, Pegs |
| G10 forex, plain | Seven dollar pairs, daily from 2014 | `fx/fx_wf.yaml` (stopped at fold 0); `fx/fx_v2.yaml` (run `fx_v2`) | published | no edge | results.md, Real markets, directional |
| G10 forex, single split | Seven dollar pairs, daily | `fx/fx.yaml` | stopped | pending | runs/fx_v0_single_split_stopped |
| G10 forex, rate-differential prior | Seven dollar pairs, daily from 2014 | `fx/fx_carry.yaml` (run `fx_carry`) | published | no edge | results.md, Real markets, directional |
| G10 forex, carry prior kept only with trend | Seven dollar pairs, daily | `fx/fx_carry_trend.yaml` | built | pending | configs/README.md, Forex |
| Emerging-market dollar pairs, volatility target | EM dollar pairs, daily from 2010 | `fx/fx_em_vt.yaml` (run `fx_em_vt`) | finished | not published; no edge | runs/fx_em_vt |
| Emerging-market dollar pairs, rate-differential prior | EM dollar pairs, daily from 2010 | `fx/fx_em_carry.yaml` (run `fx_em_carry`) | finished | not published; no edge | runs/fx_em_carry |
| Emerging-market dollar pairs, trend-confirmed carry | EM dollar pairs, daily | `fx/fx_em_carry_trend.yaml` | built | pending | configs/README.md, Forex |
| Macro, volatility target | S&P 500, Nasdaq, Nikkei, WTI, Brent, 10y and 2y Treasuries, daily | `macro/macro_1d_vt.yaml` (run `macro_1d_vt`) | published | no edge | results.md, Real markets, directional |
| Macro, trend prior | Same markets, daily | `macro/macro_1d_prior.yaml` (run `macro_1d_prior`) | finished | not published; no edge | runs/macro_1d_prior |
| Macro, implied volatility, regime and cross-asset features | Same markets, daily | `macro/macro_1d_rich.yaml` (run `macro_1d_rich`) | finished | not published; no edge | runs/macro_1d_rich |
| Macro, 12-month time-series momentum prior | Same markets, daily | `macro/macro_1d_tsmom.yaml` | paused | pending | configs/README.md, Macro |
| Macro, yield-curve slope prior | Treasury tapes, daily | `macro/macro_1d_curve.yaml` | paused | pending | configs/README.md, Macro |
| Macro, momentum and curve priors summed | Same markets, daily | `macro/macro_1d_tsmom_curve.yaml` | paused | pending | configs/README.md, Macro |
| FOMC announcement, volatility target | S&P 500, Nasdaq, Nikkei 225, daily | `macro/fomc_1d_vt.yaml` | finished | does not beat the benchmark | configs/README.md, Macro |
| FOMC announcement, scheduled-announcement prior | S&P 500, Nasdaq, Nikkei 225, daily | `macro/fomc_1d_prior.yaml` | finished | does not beat the benchmark | configs/README.md, Macro |
| S&P 500 variance premium, no prior | Short variance index, daily from 2016-10 | `vrp/vrp_spx_1d.yaml` | finished | no edge | runs/vrp_spx_1d.out |
| S&P 500 variance premium, prior with drawdown gate | Short variance index, daily from 2016-10 | `vrp/vrp_spx_1d_prior.yaml` (run `vrp_spx_1d_prior`) | finished | does not beat the benchmark | runs/vrp_spx_1d_prior.out |
| BTC and ETH variance premium, no prior | BTC and ETH, daily from 2023 | `vrp/vrp_btc_eth_1d_plain.yaml` (run `vrp_btc_eth_1d_plain`) | finished | does not beat the benchmark | runs/vrp_btc_eth_1d_plain |
| BTC and ETH variance premium, prior | BTC and ETH, daily from 2023 | `vrp/vrp_btc_eth_1d_prior.yaml` (run `vrp_btc_eth_1d_prior`) | finished | does not beat the benchmark | runs/vrp_btc_eth_1d_prior |
| BTC and ETH variance premium, before costs | BTC and ETH, daily from 2023 | `vrp/vrp_btc_eth_1d_prior_gross.yaml` | built | pending | configs/README.md, Variance |
| BTC and ETH variance premium, three times the cost | BTC and ETH, daily from 2023 | `vrp/vrp_btc_eth_1d_prior_cost3x.yaml` | built | pending | configs/README.md, Variance |
| Kalshi settled macro markets, no prior | 98 settled markets, test from 2026-09-14 | `prediction/kalshi.yaml` (run `kalshi`) | finished | analysis pending | runs/kalshi |
| Kalshi settled macro markets, log-odds prior | Same markets | `prediction/kalshi_prior.yaml` (run `kalshi_prior`) | finished | analysis pending | runs/kalshi_prior |
| Kalshi settled macro markets, calibration prior | Same markets | `prediction/kalshi_calib.yaml` | paused | pending | configs/README.md, Prediction markets |
| Polymarket resolved markets, no prior | Up to 300 markets, split 2024-07 | `prediction/polymarket.yaml` | built | pending | configs/README.md, Prediction markets |
| Polymarket resolved markets, later split | Up to 300 markets, split 2025-07 | `prediction/polymarket_v2.yaml` (run `polymarket_v2`) | finished | not published; no edge; deployed policy flat | runs/polymarket_v2 |
| Polymarket resolved markets, log-odds prior | Up to 300 markets, split 2025-07 | `prediction/polymarket_prior.yaml` (run `polymarket_prior`) | finished | not published; no edge, below flat | runs/polymarket_prior |
| Polymarket resolved markets, calibration prior | Up to 300 markets | `prediction/polymarket_calib.yaml` | paused | pending | configs/README.md, Prediction markets |
| Pooled crypto perpetuals, forex and macro, no prior | Three universes, daily from 2016 | `multi/multi_daily_vt.yaml` | failed | pending | runs/multi_daily_vt.out |
| Pooled crypto perpetuals, forex and macro, trend prior | Three universes, daily from 2016 | `multi/multi_daily_prior.yaml` (run `multi_daily_prior`) | finished | not published; no edge | runs/multi_daily_prior |

The results page has no prediction-market results. The outcomes above come from the run records only.

**2026-10-14, finished runs.** Each line gives the gated SAC Sharpe ratio, the buy-and-hold Sharpe ratio and,
where the run reports one, the deflated Sharpe ratio (DSR). Figures are from each run's `result.json`.

- `vrp_btc_eth_1d_plain`: SAC (gated) Sharpe -2.34, buy-and-hold 0.49, DSR 0.00. Verdict: does not beat the benchmark.
- `vrp_btc_eth_1d_prior`: the gate never trades, Sharpe 0.00. Verdict: does not beat the benchmark.
- `vrp_spx_1d`: SAC Sharpe ranges from 3.61 to 4.91 across the two seeds (gated 4.80), against buy-and-hold 1.59.
  DSR 0.00 under the current pooled-trial definition. The paired gap interval runs from -0.93 to 4.35, so it
  includes zero. Verdict: no edge. The SAC runs have annualized volatility of about 0.1% and CAGR of 0.43% to
  0.49% per seed (0.55% gated), so the high Sharpe ratio reflects a very small exposure, not a large edge.
- `vrp_spx_1d_prior`: the gate never trades, Sharpe 0.00. Verdict: does not beat the benchmark.
- `fomc_1d_prior`: SAC Sharpe 0.14 against buy-and-hold 1.09. Verdict: does not beat the benchmark.
- `fomc_1d_vt`: SAC Sharpe 0.41 against buy-and-hold 1.09. Verdict: does not beat the benchmark.
- `delivery_basis_1d_prior`: SAC Sharpe 2.34 against buy-and-hold 2.57. Verdict: no edge.

**Decided, trial pooling.** Kept pooled trials per universe; identical per-bar return series now count
once. Published DSRs were recomputed with this rule.
Buy-and-hold is a benchmark, not a trial, so its DSR is not a test.

## Agent and algorithm changes

Each change is tested on the hidden-regime market unless the row says otherwise. The results page counts
every variant run on a universe as a trial, including the ablations. The deflated Sharpe ratio on the
results page depends on that count.

| Change | Config | Tested on | Outcome | Trials counted |
| --- | --- | --- | --- | --- |
| Base agent: TQC critics, SimBa encoders, counterfactual replay | `ablation/base.yaml` (run `abl_base`) | Hidden-regime market | reference (results.md, Ablations) | yes |
| Critic dropout 0.1 | `ablation/dropout.yaml` (run `abl_dropout`) | Hidden-regime market | inconclusive | yes |
| Observation noise 0.1 | `ablation/obsnoise.yaml` (run `abl_obsnoise`) | Hidden-regime market | inconclusive | yes |
| Volatility targeting, no prior | `ablation/voltarget.yaml` (run `abl_voltarget`) | Hidden-regime market | inconclusive | yes |
| HL-Gauss categorical critics | `ablation/hlgauss.yaml` (run `abl_hlgauss`) | Hidden-regime market | rejected | yes |
| CVaR actor objective, eta 0.25 | `ablation/cvar.yaml` (run `abl_cvar`) | Hidden-regime market | rejected | yes |
| Pathwise actor objective over 4 steps | `ablation/pathwise.yaml` (run `abl_pathwise`) | Hidden-regime market | rejected | yes |
| Discrete SAC over five position levels | `ablation/discrete.yaml` (run `abl_discrete`) | Hidden-regime market | not published (runs/abl_discrete) | yes |
| Polyak-averaged deployment actor | `ablation/ema.yaml` (run `abl_ema`) | Hidden-regime market | not published (runs/abl_ema) | yes |
| Hyperspherical networks (SimbaV2 style) | `ablation/hyper.yaml` (run `abl_hyper`) | Hidden-regime market | not published (runs/abl_hyper) | yes |
| Counterfactual replay off | `ablation/norelabel.yaml` (run `abl_norelabel`) | Hidden-regime market | not published (runs/abl_norelabel) | yes |
| Trend prior `ret_h64`, volatility target 0.5, action penalty 0.05 | `ablation/prior.yaml` (run `abl_prior`) | Hidden-regime market | not published (runs/abl_prior) | yes |
| Training windows drawn toward recent bars, warm-start shrink | `ablation/recency_shrink.yaml` (run `abl_recency_shrink`) | Hidden-regime market | not published (runs/abl_recency_shrink) | yes |
| Penalty 0.2 on action change between states | `ablation/smooth.yaml` (run `abl_smooth`) | Hidden-regime market | not published (runs/abl_smooth) | yes |
| Confirmation, base agent, three seeds | `ablation/confirm/base.yaml` | Twelve synthetic markets, H0 to H11 | baseline for the confirmation comparisons | no |
| Confirmation, critic dropout 0.1 | `ablation/confirm/dropout.yaml` | Twelve synthetic markets, H0 to H11 | hurt in a confirmation run | no |
| Confirmation, observation noise 0.1 | `ablation/confirm/obsnoise.yaml` | Twelve synthetic markets, H0 to H11 | inconclusive | no |
| Premium-index feature on crypto cash-and-carry | `carry/crypto_basis_1d_prior_prem.yaml` (run `crypto_basis_1d_prior_prem`) | Crypto cash-and-carry, 10 coins | no gain (results.md, Ablations) | not stated in results.md |
| Benchmark, base agent, seed offset 1000 | `bench/bench_base.yaml` | Synthetic benchmark, four failure families | pending | no |
| Benchmark, prior feature, seed offset 2000 | `bench/bench_prior.yaml` | Synthetic benchmark, prior family | pending | no |
| Drawdown-hinged log-growth reward, penalty 1.0 | `bench/drawdown-penalty_on.yaml` (against `_off`) | Synthetic benchmark, four failure families, seed offset 3200 | pending | no |

Outcomes for the published rows follow the results page, Ablations section. Outcomes for the other rows
are read from the run named in the row.

Benchmark rule: each algorithm comparison uses a `seed_offset` that no earlier comparison used, and
reports the share of the sign oracle's Sharpe ratio that the agent captures. Offsets 1000, 2000, 3000,
3100 and 3200 are in use, so the next comparison needs a new offset.

## 2026-10-14: merges, results and decisions

**Merged to main.**

- Ensemble-disagreement shrink is merged behind a deploy switch.
- Causal online deployment updates are merged behind a train switch. The switch is refused under
  combinatorial purged cross-validation (CPCV). A look-ahead across offset tapes was found in review
  and removed before the merge.
- Drawdown-hinged reward is merged behind a switch.
- Trial dedupe is merged. Identical per-bar return series count once.
- Bars per year are stored in every result and backfilled for old runs. Before this change, Sharpe
  ratios recomputed for old runs were about 0.5% low.
- The command-line interface reports the active-over-rule DSR.
- The Kalshi fix drops candles that end after the market close, which were kept before. All 98 cached
  markets closed off the hour, so Kalshi runs made before this change are invalid and were rerun.
- A two-leg cash-and-carry cost config is merged.
- TIPS breakeven carry is merged.
- The open-interest squeeze gate is merged.
- The cross-sectional funding rank is merged.
- The daily Binance spot cross-section now includes delisted coins.
- Cross-section membership is now decided from history up to each bar. Published runs predate this.
- The experiment queue logs a failed run as FAILED.

**Results.** Figures are from the run records.

- Active-over-rule DSR: cash-and-carry scores 0.10 counting all runs and 0.84 excluding the
  superseded run. Cross-exchange spread scores 0.997 and 1.00. Hyperliquid scores 0.59.
- Basis costs are charged per unit of spread. Doubling the costs widens SAC's lead over its rule.
- Critic dropout hurt in a confirmation run. Observation noise is inconclusive. The confirmation set
  overlapped the original on 6 of 12 markets, because `synthetic_hard` ignores `seed_offset`.
- The dated-futures basis plain run shows no edge.

**Directional prediction study.**

- Registered in `docs/preregistration-direction.md`, amended before any run.
- A first run used data from before the registered 1971 start. It is kept as a disclosed deviation.
- The registered run was invalidated by a bug found in review. The 63-day volatility became undefined
  whenever any print was missing in the window, which switched positions off on most days.
- The correction and rerun are pending. No result is claimed.

**Built and not merged.**

- SAC residual on the momentum signal (H4a).
- Regime context with an auxiliary volatility head. The smoke run's auxiliary R-squared was negative.
- Crypto wiring of the directional study runner.

**Paused.** The experiment queue's pending list is saved and not running. It holds TIPS, squeeze,
cross-sectional funding, benchmark pairs for the algorithm switches, the two-leg cost run, and earlier
failed configs.

**Ideas dropped.**

- **Treasury premium in FOMC weeks.** The project has no point-in-time meeting calendar.
- **Funding gap between coin- and dollar-margined perpetuals.** The gap is under 1 bp.
- **FX value from CPI.** The series is too short.

## 2026-10-16: a tuned rule as second baseline

**Why.** The agent's banded policy gets a no-trade band chosen on validation data. The funding rule it
was measured against had no band, so part of the measured gain could come from trading less.

**What.** `src/softedge/eval/tuned_rule.py`: the zero-residual rule with a deadzone and the no-trade
band, both chosen per walk-forward split on validation data. `softedge active --baseline tuned`. No
retraining: the saved SAC returns are compared with the tuned rule on the same test days. Rebuilt
tapes reproduce each run's saved `prior_only` series exactly. `crypto_perp_1d_rich` could not be
rebuilt offline (Deribit cache) and is left out.

**Result.** [Results page, robustness checks](results.md#robustness-checks). The tuned rule beats the
plain rule on every carry run. Against it, the 10-coin cross-venue gain falls from +0.47 to +0.13
(interval includes zero); the 32-coin cross-venue gain falls from +0.66 to +0.27 (interval 0.04 to
0.50). Of the carry runs, only that one and the cash-and-carry run with the premium feature keep an
interval above zero.

**Risk-matched DSR.** With the tuned rule scaled to the agent's volatility, no carry gain of the
agent reaches a DSR of 0.95 (best 0.93, cash-and-carry with the premium feature; 10-coin cross-venue
spread 0.16). The tuned rule alone scores a DSR of 1.00 in every carry market.

**Decision.** The tuned rule is the baseline for any future agent or model on these markets.

## 2026-10-16: drawdown-budget checks, results

Pre-registered in `docs/preregistration-drawdown-frontier.md`. Each check is reported pass or fail, as registered.

- **C1, stress returns: fail on four of ten rows.** All five Covid rows pass. In 2022, the rows at f = 1.00, 0.93, 0.74 and 0.56 miss by 1.5 to 2.1 points, and f = 0.37 passes. The estimates assumed buy-and-hold inside each window. The design rebalances monthly, which keeps BTC at its target weight through the 2022 fall. With buy-and-hold, all ten rows pass.
- **C2, CAGR and volatility: pass on all ten rows.** The largest gap is 0.1 point on both measures.
- **C3, source of the gap (descriptive).** At a Covid-window return of −15.0%, BTC adds 4.0 CAGR points: 15.6% against 11.6% with BTC moved to T-bills. Mix B reaches 11.8%.
- **C4, deflated Sharpe ratio: fail.** Mix A (60/20/20) scores 0.85 at n = 8 with the default variance and 0.79 with the observed variance, against a line of 0.95. Mix B scores 0.75 and 0.67. As a sensitivity outside the registration, n = 4 (the four multi-asset mixes) gives mix A 0.93 and 0.90, and mix B 0.86 and 0.82. Both still fail.
- **C5, pre-FTX data: data exist.** Cached hourly funding and spot data cover all ten coins from 2020. The window from 1 June 2022 to 31 January 2023 has no gap hours, and November 2022 has no gap days for funding or spot. SOL funding moved to 2-hour and 4-hour steps from 9 to 18 November 2022, which the funding accrual model must handle. The carry sample starts in January 2023, but the cache does not force that start. Whether the premium and borrow inputs also reach back is not checked yet. Extending the sample through FTX would be a new registered check.
- **C6, sourced figures: 11 of 12 match.** The mixed-portfolio end value is $22,857, not the $22,858 quoted in chat. Of the report's eight rows, four are multi-asset mixes and four are single-sleeve benchmarks: cash, carry only, S&P only and BTC only.

Conventions and data, from the run record:

- The mixed report's stress table gives labels only. The windows were inferred by reproducing its entries under buy-and-hold. The Covid window runs from the 19 February 2020 close to the 23 March 2020 close, and calendar 2022 from the 31 December 2021 close to the 31 December 2022 close.
- Volatility is calendar-day returns times √365, the report's convention. The registration does not state one.
- The registration says to report missing data, not fill it. To reproduce the report, twelve carry days with no value are set to 0.0. Filling them with the T-bill return moves CAGR by under 0.1 point.
- S&P (46 weekdays) and DTB3 gaps are forward-filled, as in the report. Their match to exchange holidays is not verified against a calendar.

**Decision.** The 60/20/20 mix is a candidate, not an established result: it fails the deflated Sharpe test at the registered count. The frontier moves with the rebalancing rule. At the −15% crash budget the exposure is 53%, not 56%, and the in-sample CAGR is 15.6%, not 16%. The 2022 stress figures in earlier replies were buy-and-hold; monthly rebalancing makes them 1.5 to 2 points worse. Covid still binds.

## 2026-10-16: parity network, registered test

Pre-registered in `docs/preregistration-parity-network.md` (commit eaa02e9) and run once from that
commit with `scripts/parity_study.py`. Before the network was scored, the backtest's six simple
rules reproduced the discovery harness on the 768 design and 256 null tapes: identical tapes, a
largest weekly growth difference of 0.0, and identical breaches.

Paired annual difference in net log growth against the best feasible simple rule of each family,
in points, with 95% intervals. The best rule is the trend gate on `noisy_prior` and CPPI on every
other family.

| Family (role) | Network | Breaches | Direction off | Gate off | Trend gate with governor |
| --- | --- | --- | --- | --- | --- |
| `noisy_prior` (design) | −1.69 [−1.80, −1.58] | 0 | −2.79 | −0.70 | −0.27 |
| `threshold_signal` (design) | −0.27 [−0.31, −0.23] | 0 | −0.33 | +0.07 | −0.10 |
| `planted_signal` (design) | −0.29 [−0.31, −0.26] | 0 | −0.30 | −0.07 | −0.04 |
| `jump_garch` (selection) | −0.32 [−0.36, −0.29] | 0 | −0.36 | +0.19 | −0.16 |
| `nonlinear_regime_market` (selection) | −0.25 [−0.30, −0.20] | 1 | −0.33 | +0.18 | −0.24 |
| `gbm` (null) | −0.31 [−0.34, −0.29] | 0 | −0.32 | −0.05 | −0.31 |
| `break_market` (test, 2,000 tapes) | −0.22 [−0.24, −0.21] | 2 | −0.29 | +0.20 | −0.28 |

- **P1, held-out win: fail.** On the test tapes the network is 0.22 points below CPPI, with 2
  breaches against CPPI's 0.
- **P2, robustness: fail.** The network's worst family is `noisy_prior` at −1.69. That beats CPPI
  (−2.57) and Grossman-Zhou (−2.66), but not the trend gate under the same governor (−0.31). Buy-
  and-hold, the trend gate alone, volatility targeting and the log-optimal rule are infeasible on at
  least one family.
- **P3, null: pass.** On the null tapes the momentum ledgers are deployed in 0.4% of tape-weeks,
  against a limit of 2%. The future-scramble probe passes on all seven families.
- **P4, ablations.** Direction adds 1.10 points on `noisy_prior` (paired interval 0.99 to 1.22)
  and 0.01 to 0.08 points elsewhere. Gate off beats the network on every family, by 0.21 to 0.99
  points. Its mean exposure falls to 0.2 to 0.3 and its turnover rises to 6 to 8 a year. The governor
  is needed: without it, the network breaches on 30 test tapes and 35 `threshold_signal` tapes. A
  default exposure of 1 instead of one half costs 0.31 to 0.48 points on six families and gains 0.21
  on `noisy_prior`.
- **P5, sensitivity.** No constant at half or twice its value changes a conclusion. The largest
  move is the governor multiplier at half (m = 2). It holds more cash, which brings the network to
  within 0.03 to 0.06 points of CPPI where CPPI is best, and lowers `noisy_prior` to −1.92.
- **P6, novelty.** New combination, unchecked. No literature search has been run.
- **Trials.** One candidate. 90 rows in the trial ledger: 7 for the network and 83 diagnostic runs.

**Why it lost.** In every family, buy-and-hold grows between −0.44% and +0.21% a year while cash
earns 2.6%. The excess return is therefore not symmetric about zero, as the default of one half
assumes; its mean is about −0.05 weekly standard deviations. Rules that hold mostly cash win. The
premium learner forecasts this, which is why gate off wins, but its ledger ends the tape with 0.7
to 1.9 nats of evidence on average in the families where CPPI is best, short of the 3.7 the gate
requires. Only `noisy_prior`, with its strong momentum, gets the momentum ledgers past the line
(8.5 to 8.8 nats at the end). The gate did what it was built to do: it blocked edges that one path
cannot confirm. In these families the blocked edge was real.

**Decision.** As registered, the network fails the held-out test. Seeds 20000 to 21999 of
`break_market` are not used again. Every family has now scored this version, so a revised network
needs a new registration and a family it has not seen. The gate-off result is a diagnostic, not a
finding: it was not a candidate, and the test seeds cannot confirm it.

## 2026-10-16: parity network on real currency data

Pre-registered in `docs/preregistration-parity-network-fx.md` (commit d8682c6) and run once from
that commit with `scripts/parity_fx.py`. The network is the synthetic test's version, unchanged.
The market is 13 currency deposits held by a US-dollar investor (weekly FX move plus the foreign
short rate, against Treasury bills), from 1971 to 2026 where the data reach: 26,631 currency-weeks
after warm-up. Pooled differences in annual net log growth, in points, with 95% block-bootstrap
intervals over calendar weeks:

| Network minus | Points | 95% interval |
| --- | --- | --- |
| Grossman-Zhou, the best feasible simple rule | +0.38 | +0.04 to +0.76 |
| CPPI | +0.51 | +0.11 to +0.97 |
| Same network, direction off | +0.49 | +0.11 to +0.91 |
| Trend gate under the network's governor | −0.35 | −0.78 to +0.05 |
| Same network, gate off | −0.10 | −0.41 to +0.21 |
| Grossman-Zhou, both at twice the costs | +0.22 | −0.11 to +0.59 |

- **RD1, beats the best simple rule: pass.** Only CPPI and Grossman-Zhou are feasible on every
  currency. Buy-and-hold, the trend gate, the volatility target and the log-optimal rule breach the
  25% budget on at least one. The network breaches on none; its largest drawdown is 21.3% (ZAR).
- **RD2, direction adds value: pass.**
- **RD3, no leaks: pass** on all 13 currencies.

Limits of the result, all from the same run:

- **Concentrated.** BRL, KRW and CNY carry 79% of the gain over Grossman-Zhou (38%, 28% and 13%).
  Without BRL the interval includes zero (+0.25, −0.07 to +0.61). On the seven G10 currencies the
  gain is +0.13 (−0.23 to +0.53); on the six emerging-market currencies it is +0.72 (+0.13 to
  +1.37). Those three are where the ledgers grew largest (maximum evidence 17 to 50 nats, against
  a threshold of 3.7).
- **A simpler rule did as well.** The trend gate under the same governor breaches on no currency
  and its pooled log growth is 4.72% a year, against the network's 4.37%. It was a reported comparison, not part
  of RD1.
- **Costs.** At twice the costs the gain over Grossman-Zhou is not significant. Turnover is up to
  10 a year on GBP, where the momentum ledgers are deployed.
- **The governor is essential.** Without it, 7 of the 13 currencies breach.
- **Modest returns.** Network CAGR runs from 0.85% (JPY) to 7.78% (BRL). Its Sharpe ratio against
  T-bills is −0.19 to 0.41 on eleven currencies, 0.77 on BRL and 0.95 on CNY.
- On real data the gate cost little (gate off −0.10, not significant), unlike on the synthetic
  families, and a default exposure of 1 made no difference (−0.04).

**Decision.** The network passes the registered real-data check. The gain over the cushion rules
rests mainly on three emerging-market currencies and does not survive twice the costs. The trend
gate under the same governor is at least as good. The network is not changed, and this panel
has now been seen by it.

## 2026-10-17: deep parity network, trained in JAX, registered test

Pre-registered in `docs/preregistration-parity-deep.md` (commit ab72bc0, both models frozen by
SHA-256) and run once from that commit with `scripts/parity_deep_study.py`. The candidate is the
JAX network trained end to end on 640,000 simulated markets; on its own validation markets it
gained 0.24 points a year over the one-half policy. Differences are in points of annual net log
growth.

| Set | Candidate vs best simple rule | 95% interval | Ledger deployed | Breaches |
| --- | --- | --- | --- | --- |
| `regime_switching` (held out) | −0.30 vs CPPI | −0.31 to −0.28 | 2.6% of weeks | 0 |
| `ar_returns` (held out) | −0.31 vs CPPI | −0.32 to −0.29 | 5.5% | 0 |
| `ou_mean_reverting` (held out) | −0.25 vs CPPI | −0.25 to −0.25 | 0.0% | 0 |
| Crypto, 406 coins | −0.16 vs CPPI | −1.44 to +0.84 | 0.1% | 93 (CPPI 9) |
| Currencies, 13 | +0.37 vs Grossman-Zhou | +0.10 to +0.67 | 58.7% | 0 |

- **DL1, held-out synthetic: fail.** −0.28 pooled (−0.29 to −0.28).
- **DL2, crypto: fail.** No simple rule is feasible on crypto; CPPI has the fewest breaches. The
  candidate is 0.16 below it, not significantly, and breaches on 93 coins against CPPI's 9.
- **DL3, depth earns its place: fail.** On crypto the candidate and the first parity network are
  identical (difference 0.00): neither ledger deploys, so both hold one half under the governor.
- **DL4, null and leaks: pass.** Deployment on the null tapes is 1.6% of tape-weeks (limit 2%),
  and the scramble probe passes on every held-out family and on the 32 longest coins.

Diagnostics, from the same run:

- **The gate keeps the network idle.** Crypto histories have a median of 251 weeks and volatility
  near 100% a year, so evidence never reaches the line. On the held-out families the ledger
  deploys in at most 5.5% of weeks. There, and on crypto, the candidate is within 0.02 points of
  its own direction-off version. Where it does deploy, it helps: on `noisy_prior` (deployed in 54%
  of weeks) direction adds 0.99 points, and on the currencies 0.48 (0.21 to 0.77).
- **Ungated, the network has some skill, but not robust skill.** It beats CPPI on `ar_returns`
  (+0.05, +0.01 to +0.08). It earns 3.64% a year on `noisy_prior`, against 2.83% for the first
  parity network and 3.29% for the numpy forecaster. It loses badly on `ou_mean_reverting`
  (−0.55). The trend gate under the same governor beats it on `ar_returns` (+0.16) and
  `noisy_prior` (4.20%).
- **Depth adds nothing on currencies.** The candidate is −0.01 (−0.31 to +0.26) against the first
  parity network and −0.09 against the numpy forecaster. Without its gate it would have earned
  0.22 more (0.06 to 0.40).
- **The governor is too loose for crypto.** Its cap of 0.88 at the peak cannot absorb weekly
  crashes of 30% and more. The governor itself held the rest: buy-and-hold loses 47.8% a year of
  log growth across the panel and breaches on all 406 coins, while the candidate earns 5.5%.

**Decision.** The deep network fails its registered test. Training end to end in JAX produced a
policy that learns real structure in simulation, but the evidence gate keeps it idle on short
or noisy histories, and where it does trade it matches the linear version. The held-out seeds
and the crypto panel are not reused for a revision of this candidate.

## Known data and accounting fixes

Dates are commit dates on main, except where a check date is given.

- **2026-10-13, funding look-ahead.** The perpetual observation used the funding settled during the next
  bar. The context now holds the carry of the interval that just ended. Perpetual runs made before this
  change, including `crypto_perp_1d_wf`, are not comparable with later runs.
- **2026-10-13, spot borrow on short cash-and-carry legs.** A short spread shorts its spot leg, which must
  be borrowed, so the short side pays `borrow_apr`. The v2 carry configs set a 5% rate. The earlier
  cash-and-carry runs without a rate (`crypto_basis_1d_prior`, `hl_basis_1d_prior`) are superseded.
- **2026-10-13, Kalshi candles at the actual close.** Trading ends at the listed close time, and bars at or
  after it are dropped. The scheduled expiration is the only end time the features see.
- **2026-10-14, Binance archive gaps.** A missing archive month leaves one return spanning the gap. Such
  gaps now log a warning that names the first gap, and the frame is still returned.
- **2026-10-14, pegs split at archive gaps.** Each pegged pair is cut at every gap longer than two bars.
  Pieces shorter than 30 days are dropped. The pegs configs are running; their results are not yet analysed.
- **2026-10-14, roll cost on both sides.** The roll cost is charged to long and short carry on the roll
  bar, and a roll on the last bar is not charged. Delivery frames carry the roll cost as a column.
- **2026-10-14, financing above 1x.** A leveraged spot long pays cash financing on exposure above 1x,
  at `costs.financing_apr`. The default is 0, so earlier results are unchanged. The VRP and FOMC configs
  set 5%. The other leveraged spot configs, forex up to 3x and macro up to 4x, keep 0, so their leveraged
  results carry no financing charge.
- **2026-10-14, variance frames at gaps.** Variance frames are split at gaps in the daily opens.
- **2026-10-14, FOMC flag for the Tokyo session.** The statement comes after the Tokyo close, so the
  first session that can react to an announcement is the next one. The NKY flag now sits on that session.
- **Kalshi universe, checked 2026-10-14.** The Kalshi run tested 98 markets: 39 in training, 32 in
  validation and 27 in test (`runs/kalshi/result.json`). The config asks for at most 300 markets with at
  least 20,000 traded volume. The count of events is not verified against the run record.

## Unfinished runs

These runs ended without a result. Their partial output is not an outcome.

- Killed or stopped: `abl_pathwise_killed_1941`, `crypto_basis_1d_prior_cpcv_killed_1941`,
  `crypto_basis_1d_prior_cpcv_noborrow_stopped`, `crypto_basis_1d_prior_v2_killed_1941`,
  `crypto_perp_wf_stopped_fold0`, `fx_v0_single_split_stopped`, `fx_wf_stopped_fold0`,
  `hl_basis_1d_prior_v2_killed_1941`.
- Failed with an error in the output: `crypto_basis_1d` (error in `basis_frame`), `crypto_perp_1d_vt` and
  `multi_daily_vt` (missing frame attribute `kind`). `abl_hyper_failed` holds only its config. The hyperspherical
  run is reported under `abl_hyper`.
- Restarted, with no result in the restart folder: `crypto_basis_1d_prior_cpcv_restart_0425`,
  `crypto_basis_1d_prior_cpcv_restart_2347`, `crypto_basis_wide_prior_restart_0425`,
  `crypto_perp_1d_path_restart_2347`, `macro_1d_prior_restart_2347`, `perp_spread_wide_prior_restart_0425`.
  The result for each config is in the folder under its config name.
- Not finished, no result file yet: `pegs_1h`, `pegs_1h_prior`.

## Ideas considered and not built

This section lists ideas the project judged and dropped.

- **Plain price prediction.** A SAC agent that predicts price direction from price features alone was
  tried on forex, crypto and macro markets and found no edge ([results page](results.md#real-markets-directional)).
  The project uses the agent only as a residual on a structural premium.
- **Live paper trading.** Not planned. The [limits](results.md#limits) section lists what the backtests leave out.
