# Configs

Each YAML file describes one experiment. Run one with `softedge run configs/<folder>/<file>.yaml`.
The `name` field sets the output folder, `runs/<name>/`. Files are grouped by market family. A
config uses one of three test designs: walk-forward folds, a single train, validation and test
split, or a `cpcv` block. Walk-forward folds train on the past, pick a checkpoint on a validation
window, and test once on the next window. The [getting-started](../docs/getting-started.md) page
walks through a first run.

Prior and residual settings are explained in the [glossary](../docs/glossary.md#prior-and-residual-policy).
Terms such as "funding prior" refer to a causal signal the policy starts from.

The Results column marks the configs that produce the numbers on the
[results page](../docs/results.md). The other configs do not appear there.

## Controls

Known-answer runs for the pipeline. Run these first.

| File | What it tests | Results |
| --- | --- | --- |
| `synthetic/synthetic_wf.yaml` | Six synthetic markets with a planted return signal, 2021-2023, walk-forward with 120-day test windows. The agent should find the signal. | Yes |
| `synthetic/synthetic_null_wf.yaml` | The same six markets as random walks with no signal. The agent should find nothing. | Yes |

## Ablations

Each file changes one setting from `ablation/base.yaml`. All use the six hidden-regime synthetic
markets (`synthetic_hard`), two seeds, and walk-forward folds. The run name is `abl_<name>`, for
example `abl_dropout`.

| File | Change from base | Results |
| --- | --- | --- |
| `ablation/base.yaml` | No change. The base agent: TQC critics, SimBa encoders, counterfactual replay. | Yes |
| `ablation/cvar.yaml` | CVaR actor objective, eta 0.25. The actor weighs bad outcomes more heavily. | Yes |
| `ablation/discrete.yaml` | Discrete SAC over five position levels, from -1 to 1. | No |
| `ablation/dropout.yaml` | Critic dropout 0.1. | Yes |
| `ablation/ema.yaml` | The deployed actor is a slow average of the trained actor, coefficient 0.995. | No |
| `ablation/hlgauss.yaml` | HL-Gauss categorical critics. | Yes |
| `ablation/hyper.yaml` | Hyperspherical networks (SimbaV2 style). | No |
| `ablation/norelabel.yaml` | Counterfactual replay off (relabel fraction 0). | No |
| `ablation/obsnoise.yaml` | Observation noise 0.1 on the agent's inputs. | Yes |
| `ablation/pathwise.yaml` | Pathwise actor objective over 4 steps. | Yes |
| `ablation/prior.yaml` | Trend prior (`ret_h64`), volatility target 0.5, action penalty 0.05. | No |
| `ablation/recency_shrink.yaml` | Training windows drawn toward recent bars (half-life 2000 bars). Warm-started weights keep 80% of their values and take 20% from a fresh initialization. | No |
| `ablation/smooth.yaml` | Penalty 0.2 on action change between consecutive states. | No |
| `ablation/voltarget.yaml` | Volatility target 0.5, no prior. | Yes |

## Confirm

Re-runs of three ablation settings on twelve synthetic hidden-regime markets (`H0` to `H11`), with
three seeds instead of two. The files are in `ablation/confirm/`.

| File | Change from base | Results |
| --- | --- | --- |
| `ablation/confirm/base.yaml` | No change from the base agent. | No |
| `ablation/confirm/dropout.yaml` | Critic dropout 0.1. | No |
| `ablation/confirm/obsnoise.yaml` | Observation noise 0.1. | No |

## Crypto

Spot and perpetual futures on Binance, the ten coins in the crypto universe.

| File | What it tests | Results |
| --- | --- | --- |
| `crypto/crypto_spot.yaml` | Spot, hourly, no leverage, 10% borrow rate. One split: train to 2024, validation 2024, test from 2025. | No |
| `crypto/crypto_perp.yaml` | Perpetual futures, hourly, up to 2x leverage, cost multiplier randomized between 0.5 and 2. One split: test from 2025. | No |
| `crypto/crypto_perp_wf.yaml` | Hourly perpetual futures, walk-forward from 2023 with 365-day test windows and 180-day validation windows. | No |
| `crypto/crypto_perp_1d_wf.yaml` | Daily perpetual futures, walk-forward from 2023 with 365-day test windows. Smaller networks (hidden 64, actor 32). No prior. Run before the funding look-ahead fix; superseded by `crypto_perp_1d_v2.yaml`. | No |
| `crypto/crypto_perp_1d_v2.yaml` | Same settings as `crypto_perp_1d_wf.yaml`, rerun after the funding look-ahead fix. | Yes |
| `crypto/crypto_perp_1d_vt.yaml` | As `crypto_perp_1d_wf.yaml` with a 40% volatility target. | No |
| `crypto/crypto_perp_1d_prior.yaml` | As `crypto_perp_1d_vt.yaml` with a trend prior (`ret_h64`) and action penalty 0.05. | No |
| `crypto/crypto_perp_1d_path.yaml` | As `crypto_perp_1d_wf.yaml` with the pathwise actor objective over 4 steps. | No |
| `crypto/crypto_perp_1d_rich.yaml` | As `crypto_perp_1d_prior.yaml` with implied volatility (Deribit DVOL), regime features and cross-asset features. | No |

## Carry

Trades whose return is mostly a premium. Basis trades hold spot against a perpetual future. The
funding priors are a causal average of recent funding, with gain -1.

| File | What it tests | Results |
| --- | --- | --- |
| `carry/crypto_basis_1d.yaml` | Crypto cash-and-carry (long spot, short perpetual), daily, walk-forward from 2023, 5% borrow rate, no prior. | No |
| `carry/crypto_basis_1d_prior.yaml` | Daily walk-forward with the funding prior and action penalty 0.05. No borrow rate is set. | No |
| `carry/crypto_basis_1d_prior_v2.yaml` | Funding prior, 5% borrow rate, and the deflated Sharpe ratio gate. | Yes |
| `carry/crypto_basis_1d_prior_prem.yaml` | As `crypto_basis_1d_prior_v2.yaml` plus the Binance perpetual premium index feature. | Yes |
| `carry/crypto_basis_1d_prior_cpcv.yaml` | Funding prior and the deflated Sharpe ratio gate, tested with combinatorial purged cross-validation (CPCV): 6 time groups, 2 held out per split, 20-day purge, 10-day embargo, span from 2022-07 to 2026-09. | No |
| `carry/hl_basis_1d_prior.yaml` | Long Binance spot, short Hyperliquid perpetual, daily, funding prior, 12 bp fee. | No |
| `carry/hl_basis_1d_prior_v2.yaml` | As `hl_basis_1d_prior.yaml` with a 5% borrow rate and the deflated Sharpe ratio gate. | Yes |
| `carry/perp_spread_1d.yaml` | Cross-venue funding spread: long Binance perpetual, short Hyperliquid perpetual. Daily, walk-forward from 2024-07, no prior. | No |
| `carry/perp_spread_1d_prior.yaml` | As `perp_spread_1d.yaml` with the funding prior. | Yes |
| `carry/crypto_basis_wide_prior.yaml` | As `crypto_basis_1d_prior_v2.yaml` on 32 coins instead of ten. | No |
| `carry/hl_basis_wide_prior.yaml` | As `hl_basis_1d_prior_v2.yaml` on the same 32 coins, all listed on Hyperliquid. | No |
| `carry/perp_spread_wide_prior.yaml` | As `perp_spread_1d_prior.yaml` on the same 32 coins, all listed on Hyperliquid. | No |
| `carry/delivery_basis_1d.yaml` | Quarterly futures on Binance (dated contracts that expire and settle to spot): long spot, short the front quarterly, rolled 7 days before expiry. BTC and ETH, daily, walk-forward from 2023, no prior. | No |
| `carry/delivery_basis_1d_prior.yaml` | As `delivery_basis_1d.yaml` with the annualized basis as the prior (`basis_ann`, gain 10, so a 10% annual basis maps to 1.0). | No |

## Pegs

Spot pairs that should trade near a reference value. The stablecoin pairs (USDCUSDT, FDUSDUSDT,
TUSDUSDT) have a peg of 1.0. The liquid-staking pairs (WBETHETH, BNSOLSOL) use the trailing 30-day
median of their own price as the reference. The prior uses the deviation from that reference
(`peg_dev`), with a negative gain, so it leans against a deviation. Hourly bars, normal spot fee,
5% borrow rate. The files are in `pegs/`.

| File | What it tests | Results |
| --- | --- | --- |
| `pegs/pegs_1h.yaml` | Hourly spot pairs, walk-forward from 2024, no prior. The agent sees `peg_dev` as an input. | No |
| `pegs/pegs_1h_prior.yaml` | As `pegs_1h.yaml` with the peg prior (gain -1000: a 10 bp deviation is a full unit). | No |

## Forex

G10 pairs (seven dollar pairs) from 2000, and emerging-market dollar pairs.

| File | What it tests | Results |
| --- | --- | --- |
| `fx/fx.yaml` | G10 forex, daily, up to 3x leverage, swap rates. One split: train to 2016, validation 2016-2019, test from 2019. | No |
| `fx/fx_wf.yaml` | G10 forex, daily walk-forward from 2014 with 730-day validation and test windows. No prior. | No |
| `fx/fx_v2.yaml` | As `fx_wf.yaml` with smaller networks (hidden 64, actor 32) and 256-bar episodes. | Yes |
| `fx/fx_carry.yaml` | As `fx_wf.yaml` with a rate-differential prior (gain 2) and volatility target 0.1. | Yes |
| `fx/fx_em_vt.yaml` | Emerging-market dollar pairs, walk-forward from 2010, 4 bp half-spread, 1% swap rates, volatility target 0.1. | No |
| `fx/fx_em_carry.yaml` | As `fx_em_vt.yaml` with the rate-differential prior (gain 2). | No |
| `fx/fx_carry_trend.yaml` | As `fx_carry.yaml` with the carry prior kept only when the 63-bar price trend agrees (`carry_trend`). | No |
| `fx/fx_em_carry_trend.yaml` | As `fx_em_carry.yaml` with the trend-confirmed carry prior (`carry_trend`). | No |

## Macro

Daily S&P 500, Nasdaq, Nikkei, WTI and Brent price series, and Treasury excess-return indices, from FRED,
from 2000. The equity series are price indices, so their returns exclude dividends, and the buy-and-hold
reference is price-only too.

| File | What it tests | Results |
| --- | --- | --- |
| `macro/macro_1d_vt.yaml` | Volatility target 0.1, no prior, up to 4x leverage. | No |
| `macro/macro_1d_prior.yaml` | As `macro_1d_vt.yaml` with a trend prior (`ret_h64`) and action penalty 0.05. | No |
| `macro/macro_1d_rich.yaml` | As `macro_1d_prior.yaml` with implied volatility, regime features and cross-asset features. | No |
| `macro/macro_1d_tsmom.yaml` | As `macro_1d_vt.yaml` with a 12-month time-series momentum prior (`tsmom_12m`, last month skipped, scaled by volatility). | No |
| `macro/macro_1d_curve.yaml` | As `macro_1d_vt.yaml` with the 10-year minus 2-year yield slope as prior (`curve_slope`) on the Treasury tapes. | No |
| `macro/macro_1d_tsmom_curve.yaml` | As `macro_1d_vt.yaml` with both macro priors summed before clipping (`prior_features`). | No |
| `macro/fomc_1d_vt.yaml` | S&P 500, Nasdaq (FRED NASDAQCOM, the `NDX` symbol) and Nikkei 225, daily, volatility target 0.1, no prior. The plain baseline for the FOMC prior. | No |
| `macro/fomc_1d_prior.yaml` | As `fomc_1d_vt.yaml` with the scheduled-announcement flag as prior (`fomc_next`, gain 1: 1 on a session with a scheduled announcement, else 0) and action penalty 0.05. | No |

## TIPS breakeven

Long nominal Treasury against short TIPS of the same tenor, at 5 and 10 years, daily from 2003 (`softedge.data.tips`).
The spread earns the breakeven inflation rate less realized inflation. The prior is the breakeven minus the trailing
12-month CPI inflation, both in percentage points, using only CPI released by each bar's close; each month's CPI is
taken as released on the 16th of the following month. The TIPS accrual uses CPI lagged three months. Assumptions, not
sourced: the short TIPS leg pays a borrow charge of 0.5 percent a year, and the spread carries no margin or repo
haircut. Each leg's duration and convexity are fixed per tenor. October 2025 has no CPI print in the FRED series, so
that month carries September's value forward.

| File | What it tests | Results |
| --- | --- | --- |
| `tips/tips_breakeven_1d.yaml` | Long nominal, short TIPS, 5 and 10 years. Volatility target 0.05, up to 4x leverage, no prior. The plain baseline for the breakeven prior. | No |
| `tips/tips_breakeven_1d_prior.yaml` | As `tips_breakeven_1d.yaml` with the breakeven prior (`tips_prior`, gain 0.5 per point, clipped at 1). | No |

## Variance risk premium

A short-variance index on the S&P 500, daily, from FRED. The SP500 series starts on 2016-10-13, so the sample is about ten years. Each bar, the index earns today's implied variance minus the next bar's realized variance, divided by 252. The 252 is a scaling convention: the index is a return series, not a traded instrument. See the [data module](../src/softedge/data/vrp.py). Volatility target 0.1, leverage up to 4. The files are in `vrp/`.

The payoff is a proxy. VIX stands in for implied variance and trailing 21-day realized variance is the other leg; no variance-swap quotes are used. The 21-day window smooths crash days, so tail losses are understated.

The SPXVRP index is always on. It is the P&L of a short-variance position held at every bar, so it is the premium itself, with no policy in it. The drawdown gate in the prior file reads that premium. It does not read the strategy's P&L.

Costs are placeholders. Both files use the spot template in `tape.py`: 7.5 bp fee and 1.0 bp half-spread on the index level, and a 10% borrow rate on negative exposure. The borrow rate is a template charge, not a measured cost of a variance position. Replicating short variance carries roll and financing costs, and neither is modelled. Funding does not apply: it is defined for perpetual frames only, and the premium index is a spot frame. Long exposure above 1x pays cash financing at 5% a year (`costs.financing_apr: 0.05`). A cost sensitivity over fee and half-spread levels is required before any result from these files is published. No sensitivity has been run. Training samples a cost multiplier between 0.5 and 2.0 (`cost_mult_low`, `cost_mult_high`); that range is not a reported sensitivity.

Warm-up. The realized window needs 21 returns, so the first 20 bars after the first row move by zero and carry no P&L. No bar after that is zero-filled, and a missing input after warm-up raises. Measured on 2016-10-13 to 2026-09-30: 20 zero-move rows, none after row 20.

Scale. The prior is `vrp_prior = iv - rv`, an annualized variance difference, and `ActionMap` clips `gain * vrp_prior` to [-1, 1]. With gain 50 the clip binds at a gap of 0.02, since 50 x 0.02 = 1. For example, VIX 20 gives implied variance 0.04, and realized volatility of about 14.1% gives 0.02. Measured on the gated `vrp_prior` column after warm-up, |vrp_prior| exceeds 0.02 on 19% of bars (2,483 bars in the window; 27% before the gate). The 63-bar gate window is 252 / 4, one quarter of trading days. The repo does not record whether gain 50 or the 63-bar window was fixed before the first run. Both count as trials until a run records otherwise.

| File | What it tests | Results |
| --- | --- | --- |
| `vrp/vrp_spx_1d.yaml` | Short variance on the S&P 500, daily, vol target 0.1, no prior. | No |
| `vrp/vrp_spx_1d_prior.yaml` | As `vrp_spx_1d.yaml` with the variance risk premium prior (implied minus trailing realized variance, gain 50, so the position is long the index when implied exceeds realized) and a drawdown gate that zeroes the prior while the SPXVRP index (the always-on premium, not the policy's P&L) is below its 63-bar high. | No |

## Variance

A short variance carry on BTC and ETH from 2021-03, rolled daily. Each day is a one-day variance
swap priced from the 30-day Deribit DVOL index: the payoff is DVOL squared over 365 minus the day's
squared log return. It is a proxy scored as a carry, not a held 30-day position. A short 30-day
variance book also loses when implied volatility rises (vega). The proxy charges none of that, so
results on these frames overstate a real 30-day position in volatility spikes. The proxy also omits
gamma path, hedging, and option bid and ask.

The prior is the annualized variance premium (DVOL squared minus trailing 30-day realized variance),
scaled by a 40% volatility. The plain config has no prior and runs the same settings otherwise.

BTC and ETH are the two largest coins today. The choice uses that ranking, so the sample carries
selection hindsight.

Costs: the configs charge 11 bp per unit of exposure change, a fee of 10 bp and a half spread of
1 bp. This replaces the spot default in the tape (7.5 bp fee, 1 bp half spread). The 11 bp level is
an assumption, not a quote: option bid and ask are not in the data, so the cost of the instrument is
unknown.

Scale, measured on the cached data from 2021-03-25 to 2026-09-30 (1h DVOL and 1h spot for BTC and ETH,
2,016 daily rows per coin). Median DVOL is 55.4% for BTC and 69.3% for ETH. Median trailing 30-day
realized volatility is 48.0% and 65.6%. The mean daily payoff before costs is 2.5 bp per unit for BTC,
1.1 bp for ETH and 1.8 bp pooled. With one unit of exposure change per day (full daily re-entry), a
cost of 2.5 bp per unit on BTC, or 1.8 bp pooled, takes the carry to zero. The 11 bp level is about
four times the BTC payoff. The gross config is the payoff before costs and the 3x config is three
times the base level. Neither bounds the real cost.

Borrow, funding and roll are 0 by design. The payoff is a swap on DVOL, so no spot is borrowed to
short it. Funding is defined for perpetual frames only. Each day rolls a fresh one-day swap, so no roll
cost applies (`variance.py`). `max_leverage` is 4, against the spot default of 1:
the payoff index moves far less than a price, so a 0.1 volatility target needs leverage, and the cap
of 4 limits it. Long exposure above 1x pays cash financing at 5% a year (`costs.financing_apr: 0.05`).

The two main configs are two trials. The gross and 3x configs are sensitivity runs on the prior
config. Count all four when judging the Variance results.

| File | What it tests | Results |
| --- | --- | --- |
| `vrp/vrp_btc_eth_1d_plain.yaml` | Two coins, daily, walk-forward from 2023, volatility target 0.1, up to 4x leverage, no prior. | No |
| `vrp/vrp_btc_eth_1d_prior.yaml` | As `vrp_btc_eth_1d_plain.yaml` with the variance premium prior (gain 6.25). | No |
| `vrp/vrp_btc_eth_1d_prior_gross.yaml` | As `vrp_btc_eth_1d_prior.yaml` with fee and spread set to zero: the payoff before costs. | No |
| `vrp/vrp_btc_eth_1d_prior_cost3x.yaml` | As `vrp_btc_eth_1d_prior.yaml` with fee and spread at 33 bp per unit of exposure change, three times the base level. | No |

## Prediction markets

Binary YES and NO markets from Kalshi and Polymarket, each with a single split. The priors use
log-odds with gain 2, which tilts toward the favourite.

| File | What it tests | Results |
| --- | --- | --- |
| `prediction/kalshi.yaml` | Settled Kalshi macro markets (Fed, CPI, payrolls, unemployment), at most 300 markets with at least 20,000 traded volume. Test from 2026-09-14. | No |
| `prediction/kalshi_prior.yaml` | As `kalshi.yaml` with the log-odds prior. | No |
| `prediction/polymarket.yaml` | Up to 300 resolved Polymarket markets with at least 1,000,000 traded volume. Train to 2024-07, test from 2025-01. | No |
| `prediction/polymarket_v2.yaml` | As `polymarket.yaml` with a later split: train to 2025-07, test from 2026-01. | No |
| `prediction/polymarket_prior.yaml` | As `polymarket_v2.yaml` with the log-odds prior. | No |
| `prediction/kalshi_calib.yaml` | As `kalshi_prior.yaml`, with the calibration edge as the prior instead of the log-odds: the YES rate that resolved markets at a similar price earned, minus the price. Fitted only on markets resolved before each bar. | No |
| `prediction/polymarket_calib.yaml` | As `polymarket_prior.yaml`, with the calibration edge as the prior. | No |

## Pooled

One agent trained on several universes at once. Each universe keeps its own costs. The files are
in `multi/`.

| File | What it tests | Results |
| --- | --- | --- |
| `multi/multi_daily_vt.yaml` | Crypto perpetuals, forex and macro, daily, walk-forward from 2016, volatility target 0.1, no prior. | No |
| `multi/multi_daily_prior.yaml` | As `multi_daily_vt.yaml` with a trend prior (`ret_h64`) and action penalty 0.05. | No |

## Benchmark

Synthetic markets with fresh draws, for comparing algorithm changes. The markets come from
`softedge.data.bench`. Four families: `break` (the edge flips or vanishes in the second half),
`jump` (GARCH volatility with crash jumps), `threshold` (the signal predicts only in high-volatility
bars) and `prior` (a linear market whose prior is the failure mode). `data.family: all` mixes the
four. Every market carries `prior_hint`, a prior in units of a 1% bar volatility, so `prior_hint`
works as `prior_feature`.

Each algorithm comparison must use a `seed_offset` that no earlier comparison used, and must report
the share of the sign oracle's Sharpe captured: SAC's Sharpe divided by the `oracle` Sharpe of the
same run. Offsets in use: 1000 (`bench_base`), 2000 (`bench_prior`).

| File | What it tests | Results |
| --- | --- | --- |
| `bench/bench_base.yaml` | All four families, eight markets, seed offset 1000, the base agent of `ablation/base.yaml`. | No |
| `bench/bench_prior.yaml` | The `prior` family, seed offset 2000, the prior `prior_hint` as `prior_feature`. | No |
