# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Training: optional causal online updates during deployment (`train.online_every`, with
  `online_steps` and `online_window`). The deployed agents keep learning from transitions whose
  reward and next exposure are realized by the current bar in calendar time, also across tapes
  that start on different dates. Off by default; the frozen deployment is unchanged.
- Costs: `costs.financing_apr` charges cash financing on long spot exposure above 1x. Zero by default.
- Markets: `delivery_basis` holds long spot against the front quarterly future on BTC and ETH. It
  rolls to the next contract before expiry; `data.roll_days` and `data.roll_cost` set the roll.
- Markets: `pegs` holds stablecoin and liquid-staking spot pairs, with the `peg_dev` feature. Series
  are cut at archive gaps, and each piece is named `<SYMBOL>-<n>`.
- Markets: `vrp` holds daily variance carry on BTC and ETH, and `vrp_spx` an S&P 500 variance risk
  premium index.
- Markets: `macro_fomc` holds equity indices with `fomc_next`, a flag for the scheduled FOMC
  announcement session.
- Synthetic: `synthetic_bench` is a benchmark universe of markets in four failure-mode families,
  chosen with `data.family`.
- Features: `premia_features` adds time-series momentum, trend-confirmed carry and the Treasury
  curve slope to linear markets.
- Features: `data.calib` adds a calibration-edge feature to binary markets, using only markets
  resolved before each bar.
- Agent: `deploy.disagree_shrink: 1` shrinks the deployed residual where ensemble seeds disagree.
  Off by default.
- Command-line: `softedge dsr --out` and `softedge summary --out` write the report to a chosen file.
- Command-line: `softedge download` accepts every universe that `load_frames` loads.

### Changed

- Configs are grouped in folders under `configs/`, such as `configs/carry/` and
  `configs/prediction/`. Update paths that name the flat config files of 0.2.0.
- Binance klines: a series with a gap in its bars emits a `UserWarning` that names the first gap.
- Binance klines: a missing archive month between two listed months raises an error instead of
  being skipped. Frames already in the cache are not rechecked.

### Fixed

- `softedge run` prints the aggregate of a CPCV run instead of failing on the missing
  walk-forward summary.

### Known limitations

- Online deployment updates (`train.online_every`) are refused under CPCV, so they run only in
  walk-forward experiments.
- The `vrp` and `vrp_spx` frames are proxies. They omit gamma, hedging, option spreads and vega,
  and `vrp_spx` does not model roll.

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
