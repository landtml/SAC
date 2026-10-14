# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Markets: long nominal Treasury against short TIPS at the 5- and 10-year tenors, daily from 2003,
  with a breakeven prior (breakeven minus trailing 12-month CPI inflation, from CPI released by each
  bar's close). Known limitations: the frame omits the TIPS borrow shortfall, which is not sourced and
  flatters a long spread; a short spread's nominal leg carries no specialness or repo cost, also not
  sourced; the per-unit cost of 4 bp is a placeholder, and its gross and 3x cost sensitivity configs exist
  but have not been run; no roll cost is modelled; the spread carries no margin or repo haircut; the high and low columns are synthetic;
  and FRED has no October 2025 CPI value, which carries September forward.

## [0.2.0] - 2026-10-13

### Changed

- The project is named SoftEdge. The package is `softedge`, the command is `softedge`, and the
  data cache is set by `SOFTEDGE_DATA`. Before this release the package was `sactrader`.
  `SACTRADER_DATA` is still read when `SOFTEDGE_DATA` is unset.

### Added

- Markets: crypto spot and perpetual futures; cash-and-carry (long Binance spot, short Binance
  perp); Hyperliquid basis (long Binance spot, short Hyperliquid perp); a cross-venue funding
  spread (long Binance perp, short Hyperliquid perp); G10 and emerging-market forex; macro
  (S&P 500, Nasdaq 100, Nikkei 225, WTI, Brent, 10-year and 2-year US Treasuries); Polymarket and
  Kalshi prediction markets.
- Synthetic markets for controls: a planted signal, a random walk, and a hidden-regime market with
  GARCH volatility and fat tails.
- Data sources, all public and needing no keys: Binance data archive, FRED, Polymarket, Kalshi,
  Hyperliquid and Deribit DVOL.
- Agent: SAC with truncated quantile critics, SimBa and hyperspherical encoders, automatic entropy
  tuning, an optional CVaR actor, HL-Gauss critics, a discrete SAC variant, and counterfactual
  replay that relabels past states with other positions.
- Agent: a residual action map on a prior, with volatility targeting; a Polyak-averaged
  deployment actor; plasticity tools (shrink-and-perturb and ReDo); and a Q-divergence guard.
- Evaluation: walk-forward folds with validation-gated deployment; the probabilistic Sharpe ratio
  (PSR), deflated Sharpe ratio (DSR) and minimum track record length; stationary bootstrap;
  paired tests against simple rules; probability of backtest overfitting (PBO) by CSCV; and
  combinatorial purged cross-validation (CPCV).
- Command-line subcommands: `softedge download`, `run`, `dsr`, `leaderboard`, `summary`,
  `compare` and `pbo`.
- Docs: results with controls, real markets and ablations; glossary; getting started; configs
  index; roadmap; design notes. Result figures and method diagrams are in `assets/`.
- Visuals: an animated README header, an animation of one learning step, PNG renderings for
  viewers without SVG support, and a live in-browser explainer of SAC published with GitHub Pages.

### Known limitations

- On G10 forex and crypto perpetual futures, SAC did not beat the best simple rule with a
  confidence interval above zero. See [the results](docs/results.md).
- Carry results come from a model. Fills happen at bar close with a fixed spread and impact.
  Exchange counterparty risk, intrabar liquidation, and real borrow and margin rates are not
  modelled. Compare strategies with each other, not their absolute Sharpe ratios.
- Carry universes hold 10 coins over 1.6 to 3.7 years. Forex uses one daily fixing, so intraday
  moves are not visible.
- Polymarket markets are selected by traded volume, which is known only after a market closes.
  The sample favours markets that drew attention.
- Synthetic markets contain an edge by construction. They test the pipeline, not real markets.
- Ablations on the hidden-regime market found no change with a significant gain. HL-Gauss
  critics, the CVaR actor and the pathwise objective were rejected.
- The pathwise actor objective is experimental.
- Prediction-market configs exist, but no results are reported for them.
- The numbers come from backtests only. There is no live trading.

## [0.1.0] - 2026-10-13

### Added

- Initial research code, released as `sactrader`: market tape and portfolio accounting, a bar
  frame schema, loaders for Binance, FRED forex and Polymarket data, seeded synthetic market
  generators, walk-forward splits, and the PSR, DSR, stationary bootstrap and PBO statistics.
