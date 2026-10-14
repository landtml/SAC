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
| Crypto perpetuals, daily walk-forward before the funding fix | 10 coins, daily | `crypto/crypto_perp_1d_wf.yaml` (run `crypto_perp_1d_wf`) | run | superseded: carried next-bar funding | runs/crypto_perp_1d_wf; configs/README.md |
| Crypto perpetuals, daily, after the funding fix | 10 coins, daily from 2023 | `crypto/crypto_perp_1d_v2.yaml` (run `crypto_perp_1d_v2`) | published | no edge, below trend rule | results.md, Real markets, directional |
| Crypto perpetuals, daily, volatility target | 10 coins, daily | `crypto/crypto_perp_1d_vt.yaml` | failed | pending | runs/crypto_perp_1d_vt.out |
| Crypto perpetuals, trend prior | 10 coins, daily | `crypto/crypto_perp_1d_prior.yaml` (run `crypto_perp_1d_prior`) | run | not published; no edge | runs/crypto_perp_1d_prior |
| Crypto perpetuals, pathwise objective | 10 coins, daily | `crypto/crypto_perp_1d_path.yaml` (run `crypto_perp_1d_path`) | run | not published; no edge | runs/crypto_perp_1d_path |
| Crypto perpetuals, implied volatility and regime features | 10 coins, daily | `crypto/crypto_perp_1d_rich.yaml` (run `crypto_perp_1d_rich`) | run | not published; no edge | runs/crypto_perp_1d_rich |
| Crypto spot, borrow charge | 10 coins, hourly, spot | `crypto/crypto_spot.yaml` | built | pending | configs/README.md, Crypto |
| Crypto cash-and-carry, no prior | 10 coins, daily, 5% borrow | `carry/crypto_basis_1d.yaml` | failed | pending | runs/crypto_basis_1d.out |
| Crypto cash-and-carry, funding prior, no borrow rate | 10 coins, daily | `carry/crypto_basis_1d_prior.yaml` (run `crypto_basis_1d_prior`) | run | superseded: no borrow charge | runs/crypto_basis_1d_prior |
| Crypto cash-and-carry, funding prior with borrow and deflated gate | 10 coins, daily, 3.7 years | `carry/crypto_basis_1d_prior_v2.yaml` | published | edge over the funding rule | results.md, Real markets, carry |
| Crypto cash-and-carry, premium-index feature | 10 coins, daily | `carry/crypto_basis_1d_prior_prem.yaml` (run `crypto_basis_1d_prior_prem`) | published | no gain over v2 | results.md, Ablations |
| Crypto cash-and-carry, combinatorial purged cross-validation | 10 coins, five paths, 2022-07 to 2026-08 | `carry/crypto_basis_1d_prior_cpcv.yaml` (run `crypto_basis_1d_prior_cpcv`) | published | edge in sign on all five paths; size not established | results.md, Robustness checks |
| Crypto cash-and-carry, wider list | 32 coins, daily | `carry/crypto_basis_wide_prior.yaml` (run `crypto_basis_wide_prior`) | published | edge; deflated Sharpe low | results.md, Robustness checks |
| Hyperliquid basis, funding prior, no borrow rate | Binance spot against Hyperliquid perpetual, daily | `carry/hl_basis_1d_prior.yaml` (run `hl_basis_1d_prior`) | run | superseded: no borrow charge | runs/hl_basis_1d_prior |
| Hyperliquid basis, funding prior with borrow and deflated gate | Binance spot against Hyperliquid perpetual, daily | `carry/hl_basis_1d_prior_v2.yaml` | published | no edge established | results.md, Real markets, carry |
| Hyperliquid basis, wider list | 32 coins, daily | `carry/hl_basis_wide_prior.yaml` (run `hl_basis_wide_prior`) | published | no gain | results.md, Robustness checks |
| Cross-venue funding spread, no prior | Binance perpetual against Hyperliquid perpetual, daily | `carry/perp_spread_1d.yaml` (run `perp_spread_1d`) | run | not published; no edge against buy-and-hold | runs/perp_spread_1d |
| Cross-venue funding spread, funding prior | 10 coins, daily, 1.6 years | `carry/perp_spread_1d_prior.yaml` (run `perp_spread_1d_prior`) | published | edge over the funding rule | results.md, Real markets, carry |
| Cross-venue funding spread, wider list | 32 coins, daily | `carry/perp_spread_wide_prior.yaml` (run `perp_spread_wide_prior`) | published | edge | results.md, Robustness checks |
| Dated futures basis, no prior | BTC and ETH quarterly futures against spot, daily | `carry/delivery_basis_1d.yaml` | built | pending | configs/README.md, Carry |
| Dated futures basis, annualized basis prior | BTC and ETH quarterly futures against spot, daily | `carry/delivery_basis_1d_prior.yaml` | built | pending | configs/README.md, Carry |
| Pegged pairs, no prior | Stablecoin and liquid-staking spot pairs, hourly | `pegs/pegs_1h.yaml` | built | pending | configs/README.md, Pegs |
| Pegged pairs, peg prior | Stablecoin and liquid-staking spot pairs, hourly | `pegs/pegs_1h_prior.yaml` | built | pending | configs/README.md, Pegs |
| G10 forex, plain | Seven dollar pairs, daily from 2014 | `fx/fx_wf.yaml` (stopped at fold 0); `fx/fx_v2.yaml` (run `fx_v2`) | published | no edge | results.md, Real markets, directional |
| G10 forex, single split | Seven dollar pairs, daily | `fx/fx.yaml` | stopped | pending | runs/fx_v0_single_split_stopped |
| G10 forex, rate-differential prior | Seven dollar pairs, daily from 2014 | `fx/fx_carry.yaml` (run `fx_carry`) | published | no edge | results.md, Real markets, directional |
| G10 forex, carry prior kept only with trend | Seven dollar pairs, daily | `fx/fx_carry_trend.yaml` | built | pending | configs/README.md, Forex |
| Emerging-market dollar pairs, volatility target | EM dollar pairs, daily from 2010 | `fx/fx_em_vt.yaml` (run `fx_em_vt`) | run | not published; no edge | runs/fx_em_vt |
| Emerging-market dollar pairs, rate-differential prior | EM dollar pairs, daily from 2010 | `fx/fx_em_carry.yaml` (run `fx_em_carry`) | run | not published; no edge | runs/fx_em_carry |
| Emerging-market dollar pairs, trend-confirmed carry | EM dollar pairs, daily | `fx/fx_em_carry_trend.yaml` | built | pending | configs/README.md, Forex |
| Macro, volatility target | S&P 500, Nasdaq, Nikkei, WTI, Brent, 10y and 2y Treasuries, daily | `macro/macro_1d_vt.yaml` (run `macro_1d_vt`) | published | no edge | results.md, Real markets, directional |
| Macro, trend prior | Same markets, daily | `macro/macro_1d_prior.yaml` (run `macro_1d_prior`) | run | not published; no edge | runs/macro_1d_prior |
| Macro, implied volatility, regime and cross-asset features | Same markets, daily | `macro/macro_1d_rich.yaml` (run `macro_1d_rich`) | run | not published; no edge | runs/macro_1d_rich |
| Macro, 12-month time-series momentum prior | Same markets, daily | `macro/macro_1d_tsmom.yaml` | queued | pending | configs/README.md, Macro |
| Macro, yield-curve slope prior | Treasury tapes, daily | `macro/macro_1d_curve.yaml` | queued | pending | configs/README.md, Macro |
| Macro, momentum and curve priors summed | Same markets, daily | `macro/macro_1d_tsmom_curve.yaml` | queued | pending | configs/README.md, Macro |
| FOMC announcement, volatility target | S&P 500, Nasdaq, Nikkei 225, daily | `macro/fomc_1d_vt.yaml` | queued | pending | configs/README.md, Macro |
| FOMC announcement, scheduled-announcement prior | S&P 500, Nasdaq, Nikkei 225, daily | `macro/fomc_1d_prior.yaml` | queued | pending | configs/README.md, Macro |
| S&P 500 variance premium, no prior | Short variance index, daily from 2016-10 | `vrp/vrp_spx_1d.yaml` | queued (running) | pending | runs/vrp_spx_1d.out |
| S&P 500 variance premium, prior with drawdown gate | Short variance index, daily from 2016-10 | `vrp/vrp_spx_1d_prior.yaml` (run `vrp_spx_1d_prior`) | queued (running) | pending | runs/vrp_spx_1d_prior.out |
| BTC and ETH variance premium, no prior | BTC and ETH, daily from 2023 | `vrp/vrp_btc_eth_1d_plain.yaml` (run `vrp_btc_eth_1d_plain`) | run | not published; no edge | runs/vrp_btc_eth_1d_plain |
| BTC and ETH variance premium, prior | BTC and ETH, daily from 2023 | `vrp/vrp_btc_eth_1d_prior.yaml` (run `vrp_btc_eth_1d_prior`) | run | not published; no edge | runs/vrp_btc_eth_1d_prior |
| BTC and ETH variance premium, before costs | BTC and ETH, daily from 2023 | `vrp/vrp_btc_eth_1d_prior_gross.yaml` | built | pending | configs/README.md, Variance |
| BTC and ETH variance premium, three times the cost | BTC and ETH, daily from 2023 | `vrp/vrp_btc_eth_1d_prior_cost3x.yaml` | built | pending | configs/README.md, Variance |
| Kalshi settled macro markets, no prior | 98 settled markets, test from 2026-09-14 | `prediction/kalshi.yaml` (run `kalshi`) | run | not published; no edge against flat | runs/kalshi |
| Kalshi settled macro markets, log-odds prior | Same markets | `prediction/kalshi_prior.yaml` (run `kalshi_prior`) | run | not published; no edge, below flat | runs/kalshi_prior |
| Kalshi settled macro markets, calibration prior | Same markets | `prediction/kalshi_calib.yaml` | queued | pending | configs/README.md, Prediction markets |
| Polymarket resolved markets, no prior | Up to 300 markets, split 2024-07 | `prediction/polymarket.yaml` | built | pending | configs/README.md, Prediction markets |
| Polymarket resolved markets, later split | Up to 300 markets, split 2025-07 | `prediction/polymarket_v2.yaml` (run `polymarket_v2`) | run | not published; no edge; deployed policy flat | runs/polymarket_v2 |
| Polymarket resolved markets, log-odds prior | Up to 300 markets, split 2025-07 | `prediction/polymarket_prior.yaml` (run `polymarket_prior`) | run | not published; no edge, below flat | runs/polymarket_prior |
| Polymarket resolved markets, calibration prior | Up to 300 markets | `prediction/polymarket_calib.yaml` | queued | pending | configs/README.md, Prediction markets |
| Pooled crypto perpetuals, forex and macro, no prior | Three universes, daily from 2016 | `multi/multi_daily_vt.yaml` | failed | pending | runs/multi_daily_vt.out |
| Pooled crypto perpetuals, forex and macro, trend prior | Three universes, daily from 2016 | `multi/multi_daily_prior.yaml` (run `multi_daily_prior`) | run | not published; no edge | runs/multi_daily_prior |

The results page has no prediction-market results. The outcomes above come from the run records only.

**2026-10-14, finished runs.** Each line gives the gated SAC Sharpe ratio, the buy-and-hold Sharpe ratio and,
where the run reports one, the deflated Sharpe ratio (DSR). Figures are from each run's `result.json`.

- `vrp_btc_eth_1d_plain`: SAC (gated) Sharpe -2.34, buy-and-hold 0.49, DSR 0.00. Verdict: failed.
- `vrp_btc_eth_1d_prior`: the gate never trades, Sharpe 0.00. Verdict: failed.
- `vrp_spx_1d`: SAC (gated) Sharpe 4.80, buy-and-hold 1.59, DSR 0.00 under the current pooled-trial definition.
  The paired gap interval runs from -0.93 to 4.35, so it includes zero. Verdict: no edge.
- `vrp_spx_1d_prior`: the gate never trades, Sharpe 0.00. Verdict: failed.
- `fomc_1d_prior`: SAC Sharpe 0.14 against buy-and-hold 1.09. Verdict: failed.
- `fomc_1d_vt`: SAC Sharpe 0.41 against buy-and-hold 1.09. Verdict: failed.
- `delivery_basis_1d_prior`: SAC Sharpe 2.34 against buy-and-hold 2.57. Verdict: no edge.

**Open item, trial pooling.** The DSR pools every SAC trial in a universe, including the prior-residual
variant, which raises the benchmark Sharpe ratio that a run must beat. Excluding the prior family gives a
DSR of 0.94 for `vrp_spx_1d` and 0.44 for `vrp_btc_eth_1d_plain`. `sac_banded` and `sac_gated` are identical
in `vrp_spx_1d`, so that run counts them twice. The decision is pending. The published numbers use the
existing definition. Buy-and-hold is a benchmark, not a trial, so its DSR is not a test.

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
| Confirmation, base agent, three seeds | `ablation/confirm/base.yaml` | Twelve synthetic markets, H0 to H11 | pending | no |
| Confirmation, critic dropout 0.1 | `ablation/confirm/dropout.yaml` | Twelve synthetic markets, H0 to H11 | pending | no |
| Confirmation, observation noise 0.1 | `ablation/confirm/obsnoise.yaml` | Twelve synthetic markets, H0 to H11 | pending | no |
| Premium-index feature on crypto cash-and-carry | `carry/crypto_basis_1d_prior_prem.yaml` (run `crypto_basis_1d_prior_prem`) | Crypto cash-and-carry, 10 coins | no gain (results.md, Ablations) | not stated in results.md |
| Benchmark, base agent, seed offset 1000 | `bench/bench_base.yaml` | Synthetic benchmark, four failure families | pending | no |
| Benchmark, prior feature, seed offset 2000 | `bench/bench_prior.yaml` | Synthetic benchmark, prior family | pending | no |

Outcomes for the published rows follow the results page, Ablations section. Outcomes for the other rows
are read from the run named in the row.

Benchmark rule: each algorithm comparison uses a `seed_offset` that no earlier comparison used, and
reports the share of the sign oracle's Sharpe ratio that the agent captures. Offsets 1000 and 2000 are
in use, so the next comparison needs a new offset.

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
  Pieces shorter than 30 days are dropped. The pegs configs have not run.
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
- Not finished, no result file yet: `pegs_1h`, `pegs_1h_prior`, `delivery_basis_1d`.

## Ideas considered and not built

This section lists ideas the project judged and dropped.

- **Plain price prediction.** A SAC agent that predicts price direction from price features alone was
  tried on forex, crypto and macro markets and found no edge ([results page](results.md#real-markets-directional)).
  The project uses the agent only as a residual on a structural premium.
- **Live paper trading.** Not planned. The [limits](results.md#limits) section lists what the backtests leave out.
