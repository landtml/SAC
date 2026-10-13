# Configs

Each YAML file describes one experiment. Run one with `softedge run configs/<file>.yaml`. The
`name` field sets the output folder, `runs/<name>/`. A config uses one of three test designs:
walk-forward folds, a single train, validation and test split, or a `cpcv` block. The
[getting-started](../docs/getting-started.md) page walks through a first run.

Prior and residual settings are explained in the [glossary](../docs/glossary.md#prior-and-residual-policy).
Terms such as "funding prior" refer to a causal signal the policy starts from.

## Controls

Known-answer runs for the pipeline. Run these first.

| File | What it runs |
| --- | --- |
| `synthetic_wf.yaml` | Six synthetic markets with a planted return signal, hourly, 2021-2023. Walk-forward with 120-day test windows. The agent should find the signal. |
| `synthetic_null_wf.yaml` | The same six markets as random walks with no edge. Walk-forward. The agent should find nothing. |

## Ablations

Each file changes one setting from `ablation/base.yaml`. All use the six hidden-regime synthetic
markets (`synthetic_hard`), two seeds, and walk-forward folds. The run name is `abl_<name>`, for
example `abl_dropout`.

| File | Change from base |
| --- | --- |
| `base.yaml` | No change. The base agent: TQC critics, SimBa encoders, counterfactual replay. |
| `cvar.yaml` | CVaR actor objective, eta 0.25. |
| `discrete.yaml` | Discrete SAC over five position levels, from -1 to 1. |
| `dropout.yaml` | Critic dropout 0.1. |
| `ema.yaml` | Polyak-averaged deployment actor, coefficient 0.995. |
| `hlgauss.yaml` | HL-Gauss categorical critics. |
| `hyper.yaml` | Hyperspherical networks (SimbaV2 style). |
| `norelabel.yaml` | Counterfactual replay off (relabel fraction 0). |
| `obsnoise.yaml` | Observation noise 0.1. |
| `pathwise.yaml` | Pathwise actor objective over 4 steps. |
| `prior.yaml` | Trend prior (`ret_h64`), volatility target 0.5, action penalty 0.05. |
| `recency_shrink.yaml` | Training windows drawn toward recent bars (half-life 2000 bars). Warm-started weights keep 80% of their values and take 20% from a fresh initialization. |
| `smooth.yaml` | Penalty 0.2 on action change between consecutive states. |
| `voltarget.yaml` | Volatility target 0.5, no prior. |

## Confirm

Re-runs of three ablation settings on twelve synthetic hidden-regime markets (`H0` to `H11`), with
three seeds instead of two.

| File | Change from base |
| --- | --- |
| `confirm/base.yaml` | No change from the base agent. |
| `confirm/dropout.yaml` | Critic dropout 0.1. |
| `confirm/obsnoise.yaml` | Observation noise 0.1. |

## Crypto

Spot and perpetual futures on Binance, the ten coins in the crypto universe.

| File | What it runs |
| --- | --- |
| `crypto_spot.yaml` | Spot, hourly, no leverage, 10% borrow rate. One split: train to 2024, validation 2024, test from 2025. |
| `crypto_perp.yaml` | Perpetual futures, hourly, up to 2x leverage, cost multiplier randomized between 0.5 and 2. One split: test from 2025. |
| `crypto_perp_wf.yaml` | As `crypto_perp`, hourly, walk-forward with 365-day test windows from 2023 and 180-day validation. |
| `crypto_perp_1d_wf.yaml` | Perpetual futures, daily, walk-forward from 2023 with 365-day test windows. Smaller networks (hidden 64, actor 32). No prior. |
| `crypto_perp_1d_v2.yaml` | Same settings as `crypto_perp_1d_wf.yaml`. |
| `crypto_perp_1d_vt.yaml` | As `crypto_perp_1d_wf.yaml` with a 40% volatility target. |
| `crypto_perp_1d_prior.yaml` | As `crypto_perp_1d_wf.yaml` with a 40% volatility target, a trend prior (`ret_h64`) and action penalty 0.05. |
| `crypto_perp_1d_path.yaml` | As `crypto_perp_1d_wf.yaml` with the pathwise actor objective over 4 steps. |
| `crypto_perp_1d_rich.yaml` | As `crypto_perp_1d_prior.yaml` with implied volatility (Deribit DVOL), regime features and cross-asset features. |

## Carry

Trades whose return is mostly a premium. Basis trades hold spot against a perpetual future. The
funding priors are a causal average of recent funding, with gain -1.

| File | What it runs |
| --- | --- |
| `crypto_basis_1d.yaml` | Crypto cash-and-carry (long spot, short perpetual), daily, walk-forward from 2023, 5% borrow rate, no prior. |
| `crypto_basis_1d_prior.yaml` | Daily walk-forward with the funding prior and action penalty 0.05. No borrow rate is set. |
| `crypto_basis_1d_prior_v2.yaml` | Funding prior, 5% borrow rate, and the deflated Sharpe ratio gate. |
| `crypto_basis_1d_prior_prem.yaml` | As `crypto_basis_1d_prior_v2.yaml` plus the Binance perpetual premium index feature. |
| `crypto_basis_1d_prior_cpcv.yaml` | Funding prior and the deflated Sharpe ratio gate, tested with CPCV: 6 time groups, 2 held out per split, 20-day purge, 10-day embargo, span from 2022-07 to 2026-09. |
| `hl_basis_1d_prior.yaml` | Long Binance spot, short Hyperliquid perpetual, daily, funding prior, 12 bp fee. |
| `hl_basis_1d_prior_v2.yaml` | As `hl_basis_1d_prior.yaml` with a 5% borrow rate and the deflated Sharpe ratio gate. |
| `perp_spread_1d.yaml` | Cross-venue funding spread: long Binance perpetual, short Hyperliquid perpetual. Daily, walk-forward from 2024-07, no prior. |
| `perp_spread_1d_prior.yaml` | As `perp_spread_1d.yaml` with the funding prior. |

## Forex

G10 pairs (seven dollar pairs) from 2000, and emerging-market dollar pairs.

| File | What it runs |
| --- | --- |
| `fx.yaml` | G10 forex, daily, up to 3x leverage, swap rates. One split: train to 2016, validation 2016-2019, test from 2019. |
| `fx_wf.yaml` | G10 forex, daily, walk-forward. |
| `fx_v2.yaml` | G10 forex, daily, walk-forward with smaller networks (hidden 64, actor 32) and 256-bar episodes. |
| `fx_carry.yaml` | G10 forex, walk-forward, rate-differential prior (gain 2), volatility target 0.1. |
| `fx_em_vt.yaml` | Emerging-market dollar pairs, walk-forward with test windows from 2010, 4 bp half-spread, 1% swap rates, volatility target 0.1. |
| `fx_em_carry.yaml` | As `fx_em_vt.yaml` with the rate-differential prior (gain 2). |

## Macro

Daily S&P 500, Nasdaq, Nikkei, WTI, Brent and Treasury excess-return indices from FRED, from 2000.

| File | What it runs |
| --- | --- |
| `macro_1d_vt.yaml` | Volatility target 0.1, no prior, up to 4x leverage. |
| `macro_1d_prior.yaml` | As `macro_1d_vt.yaml` with a trend prior (`ret_h64`) and action penalty 0.05. |
| `macro_1d_rich.yaml` | As `macro_1d_prior.yaml` with implied volatility, regime features and cross-asset features. |

## Prediction markets

Binary YES and NO markets from Kalshi and Polymarket, hourly, with a single split. The priors use
log-odds with gain 2, which tilts toward the favourite.

| File | What it runs |
| --- | --- |
| `kalshi.yaml` | Settled Kalshi macro markets (Fed, CPI, payrolls, unemployment), at most 300 markets with at least 20,000 traded volume. Test from 2026-09-14. |
| `kalshi_prior.yaml` | As `kalshi.yaml` with the log-odds prior. |
| `polymarket.yaml` | Up to 300 resolved Polymarket markets with at least 1,000,000 traded volume. Train to 2024-07, test from 2025-01. |
| `polymarket_v2.yaml` | As `polymarket.yaml` with a later split: train to 2025-07, test from 2026-01. |
| `polymarket_prior.yaml` | As `polymarket_v2.yaml` with the log-odds prior. |

## Pooled

One agent trained on several universes at once. Each universe keeps its own costs.

| File | What it runs |
| --- | --- |
| `multi_daily_vt.yaml` | Crypto perpetuals, forex and macro, daily, walk-forward from 2016, volatility target 0.1, no prior. |
| `multi_daily_prior.yaml` | As `multi_daily_vt.yaml` with a trend prior (`ret_h64`) and action penalty 0.05. |
