# Supervised forecasting architecture

This document outlines a second research track for SoftEdge: forecast returns with supervised models,
then turn forecasts into positions with an explicit, cost-aware portfolio step. It is a design, not a
result. Nothing here has been run.

Why a second track: the SAC experiments found no edge in plain price prediction, and the one edge that
survived honest deflation (the cross-exchange funding spread) came from a known premium with a simple
rule underneath. Because a small trader does not move prices, the trading problem is mostly "predict,
then size". A supervised model solves the first half with lower variance, fewer settings and clearer
attribution. The design below is built around the failures this project has already met; each one has
a named guard (see [Failure modes and their guards](#failure-modes-and-their-guards)).

## Principles

1. **Power before models.** Compute the minimum detectable Sharpe for a study before running it. With
   three years of data an annual Sharpe of 0.5 cannot be told apart from noise; long histories and wide
   cross-sections come first.
2. **Few, registered tests.** Every study has a pre-registration file listing its hypotheses, data,
   parameters, pass rule and trial count before any result exists. Deviations are dated amendments.
3. **Point in time, always.** Every value used at bar t must have been knowable at t, including which
   assets exist, which calendar day it is, and which data vintage was published.
4. **One accounting model.** Positions are scored with the existing tape and portfolio code (costs per
   leg, funding, borrow, financing, liquidation), so supervised and SAC results are comparable.
5. **Simple first.** Rules, then linear models, then shallow trees. A more complex model is tried only
   when a simpler one already shows a signal, and each step counts as a trial.
6. **Report what fails.** Every trial goes into the ledger and the research log, pass or fail.

## Overview

```
 raw sources ─► snapshot store ─► asset registry ─► panel builder ─► features ─┐
 (immutable,      (hashes,          (listings,        (per-asset          (causal,   │
  hashed)          vintages)         delistings,       calendars,          declared   │
                                     ticker reuse)     availability)       lookback)  │
                                                                                      ▼
 evaluation ◄── accounting ◄── portfolio ◄── forecast combiner ◄── models ◄── targets
 (ledger, DSR,   (tape and      (risk model,   (shrinkage,          (rules,    (aligned,
  nulls, lockbox) portfolio)     costs, caps)   calibration)         linear,    horizon-
                                                                     trees)     tagged)
```

## Layers

### 1. Snapshot store
- Every download is written once, never edited, and keyed by a content hash. A run records the hashes
  of every file it read.
- Revised series keep their vintages (for macro data, the vintage published at t, not today's value).
- Cache age never triggers a silent refetch during a study run; refetching is an explicit, logged step.

### 2. Asset registry
- One table per universe: symbol, listing date, delisting date, and events (ticker reuse,
  redenomination, contract roll, symbol change). A reused ticker becomes two assets with no return
  across the junction.
- A validation report runs on every panel build: gaps per asset, extreme daily returns (with a
  classification column), non-positive prices, negative volumes, round-factor level jumps. A study
  cannot run on a panel whose report has unreviewed flags.

### 3. Panel builder
- Long format, one row per (date, asset), each asset on its own trading calendar, with an `available`
  flag. No asset borrows another asset's calendar.
- Universe membership at t is decided from information up to t (listed, enough history, liquidity
  screen computed on past data). Total series length never decides membership.
- Missing prints are missing: no forward or backward fill of prices or returns. A feature that needs a
  full window states what it does on a gap.

### 4. Feature library
- A feature is a pure function of an asset's own observed history up to t, with a declared lookback
  and a declared gap policy. Volatility and other window statistics are computed on the asset's own
  observed returns, then aligned to the panel.
- Cross-sectional features (ranks, z-scores across assets) use only assets that are members at t.
- Every feature gets generated tests: changing or truncating data after t leaves the value at t
  unchanged; inserting a missing print inside the window behaves as declared; the share of days with a
  defined value on the real panel is reported (a feature defined on 4% of days is a bug until shown
  otherwise).

### 5. Targets
- Forward return over horizon h, divided by the ex-ante volatility at t, aligned so the target at row t
  is realised strictly after t. Each target carries its horizon so that training can purge overlap.
- Cross-sectional targets (next-period rank) are computed only among assets available on both days.

### 6. Models
Tried in this order; each tier counts as a trial in the ledger.
- **Rules:** time-series momentum, carry, short-term reversal. They are both baselines and priors.
- **Linear:** ridge or elastic net on standardised features, pooled across assets, regularisation fixed
  in the registration.
- **Shallow trees:** gradient-boosted trees with small depth, few leaves, strong regularisation and
  early stopping on an inner validation window.
- Training is walk-forward: expanding or rolling window, refit on a fixed schedule, purge of training
  rows whose target overlaps the test window, plus an embargo. Any hyperparameter chosen on validation
  is listed in the registration and counted.

### 7. Forecast combiner
- Shrink each forecast toward zero by a factor fitted on validation (regress realised on predicted).
  Low signal-to-noise means raw forecasts are overconfident.
- Combine horizons and models with fixed or validation-fitted weights; track the information
  coefficient per fold so decay is visible.

### 8. Portfolio construction
- Risk model: exponentially weighted covariance with shrinkage toward a diagonal target.
- Positions from a mean-variance step with a turnover penalty and a no-trade band, a volatility target,
  per-asset and gross leverage caps. A plain "forecast times inverse volatility" sizing is always run
  alongside as the baseline.
- Costs are per leg: a basis or spread trade pays both legs. Funding, borrow and financing come from
  the tape.

### 9. Accounting
- Reuse `tape.py` and `portfolio.py`. A run stores its bars per year next to every Sharpe ratio, so a
  recomputed figure always matches the reported one.

### 10. Evaluation
For every study, in one report:
- Sharpe after costs, Newey-West t, Sharpe in each half and per decade, minimum detectable Sharpe.
- Deflated Sharpe ratio at the registered trial count, plus sensitivities: observed cross-trial
  variance, and with superseded runs left out. Identical return series count once.
- Paired bootstrap against the best simple rule on the same days.
- Null panels (Gaussian and block-bootstrap of demeaned returns) for a false-positive rate with its
  interval; a one-bar delay check for timing errors.
- Attribution by asset and by period, with a flag when one asset or one decade carries more than a
  third of the profit.
- Costs at 1x, 2x and 3x.
- A lockbox: the last stretch of each panel (for example the final two years) is never touched until a
  study's registered analysis is complete, then scored once.

### 11. Operations
- A run manifest: commit, config hash, data hashes, start and end time, exit code, and whether the
  result file exists. A run without its result file is a failure, whatever the exit code says.
- Full test suites run one at a time at low priority when they share a machine with training.

## Failure modes and their guards

Each row is a problem this project met; the guard is where it lives in the design above.

| Failure met | What went wrong | Guard |
| --- | --- | --- |
| Settlement leak | Prediction-market candles ending after the close entered the last bar | Snapshot timestamps tested against event times; feature tests truncate after t |
| Membership by future length | Coins were dropped from the cross-section using their total history | Panel builder decides membership from data up to t |
| Volatility undefined on gaps | One missing print in a 63-day window switched positions off on most days | Own-calendar statistics; share-of-days-defined check on the real panel |
| Synthetic-only tests | Every unit test used gap-free panels, so the gap bug passed | Generated tests insert missing prints; checks run on the real panel |
| Ticker reuse | One symbol held two different coins, with a fake jump between them | Asset registry splits reused tickers; extreme-return report |
| Survivorship | Lists of coins chosen today miss delisted ones | Registry keeps delisted assets; exit at last price with costs |
| Data start deviation | A run used years before the registered start | Panel start enforced from the registration file; first valid date printed per asset |
| Costs per spread unit | A two-leg trade was charged one fee | Per-leg cost model; cost sensitivity at 2x and 3x |
| Annualisation mismatch | A recomputed Sharpe used a different bar count | Bars per year stored with each run |
| Lenient deflation | A theoretical variance floor overstated the deflated Sharpe | Observed cross-trial variance as a reported sensitivity; duplicates counted once |
| Hidden concentration | Much of a result came from one asset in one era | Attribution flags by asset and decade |
| Silent failures | A crashed run was logged as successful | Manifest requires the result file |
| Fading effects | Second-half Sharpe well below the first | Per-period reporting; lockbox; minimum detectable Sharpe per period |

## How SAC fits

SAC stays as an optional layer: a residual on top of the supervised portfolio's positions, judged
against two cheaper residual learners on the same task (a ridge regression of the adjustment, and a
small network trained by backpropagating through the realised profit and loss). If SAC does not beat
both, its extra machinery is not earning its keep.

## First studies

1. **Momentum, corrected.** Rerun the registered directional study with volatility computed on each
   asset's own observed returns, as a dated amendment. This is the baseline every model must beat on
   the long panel.
2. **Linear and tree forecasts on the long panel.** The same targets, folds and accounting as the
   rules; registered before running.
3. **Crypto cross-section.** Reversal and momentum ranks on the Binance panel with delisted coins,
   then a ridge rank model, after the panel's validation report is clean.
4. **Funding-spread forecasting.** Forecast next-period funding spreads across venues, the market where
   a robust edge already exists, and test whether forecasts improve on the simple spread rule.

## Out of scope for now

Intraday and order-book data, execution with market impact, and options. Each needs its own data layer
and cost model before it can be tested honestly.
