# Design

This page explains the design choices behind the SoftEdge agent, and the known limitations of the
model they run on. Each section below names a choice and the failure it prevents.

## Market tape: one accounting model for four markets

Every instrument becomes a `MarketTape` (`src/softedge/tape.py`): per-step returns, worst
intra-bar returns, carry and trading cost, stored separately for the long and the short side.

| Market | Long unit | Short unit | Carry | Settlement |
| --- | --- | --- | --- | --- |
| Crypto spot | coin | short sale, pays borrow APR | borrow on shorts | none |
| Perpetual futures | contract | contract | funding, charged in the bar it falls in | none |
| Forex | base currency | quote currency | swap APR | none |
| Binary prediction market | YES token | NO token | none | YES pays 1 or 0 on the last step |

`portfolio.step_portfolio` turns a previous exposure and a target exposure into an equity return:
fee plus half spread on traded notional, a `turnover ** 1.5` impact term, carry on held notional,
liquidation when worst intra-bar equity falls to the maintenance margin, and exposure drift after
the move. A binary NO position is a long NO token, so its cost and drift follow the NO price.

Timing: the agent observes features of bar `t` (computed from bars `<= t`), trades at `close[t]`,
and earns the move to `close[t + 1]`.

## Observations without leakage

`features/causal.py` builds scale-free features from causal operations only: volatility-normalized
returns over 1 to 256 bars, EMA gaps, volatility ratios, Parkinson range volatility, candle body,
volume and taker-flow z-scores, funding in basis points, time of day and week, and for binary
markets log-odds, its changes and log hours to the scheduled end date. Tests compare features on
a prefix of a frame with features on the full frame, so any look-ahead fails the suite.

With `data.cross_features: true`, every linear tape also sees its universe at the same close
(`features/cross.py`): the cross-sectional mean of each return horizon scaled by the square root
of the number of assets, the asset's 16-bar return minus the mean of the others, the dispersion
of one-bar returns and the breadth of positive 16-bar returns. Crypto alts follow BTC, forex
pairs share a dollar factor, and equity indices co-move.

No scaler is fit on the full sample. The agent normalizes inputs with running statistics updated
from training batches only.

The observation is a window of feature rows plus context: current exposure over max leverage, log
trading cost on each side, carry on each side, max leverage and a one-hot market type. Tapes with
different feature sets share the union of names, missing columns filled with zeros, so one agent
can train on all four markets.

The carry in the context is the carry of the interval that just ended. The tape's own carry at
row `t` prices the next interval, and for perps that is the funding settled during the next bar,
which the close at `t` does not know. A run that used next-bar funding on perps has a small
look-ahead (a few basis points a day at most), so its perp results are not comparable with the
current ones.

## Action and reward

By default the action `a` in `[-1, 1]` sets the target exposure `a * max_leverage`. An optional
action map (`env.prior_feature`, `env.prior_gain`, `env.residual`, `env.vol_target`) sets the
target to `clip(m_t * (b_t + residual * a), -L, L)`:

* a prior `b_t`, a causal feature clipped to `[-1, 1]` (a trend z-score such as `ret_h64`, the
  forex rate differential, perp funding with a negative gain, binary log-odds for a
  favourite-longshot tilt). The policy becomes a residual on a known premium, and the actor
  penalty `sac.action_l2` keeps it on the prior unless the critic sees a gain worth more than the
  penalty. In a low signal-to-noise market this turns "learn nothing" into "hold the premium"
  instead of "trade noise";
* a volatility target, `m_t = min(target / sigma_ann_t, L)` with `sigma_t` a causal EWMA of
  per-bar volatility over the full history. A unit action then carries the same risk in calm and
  turbulent regimes, which steadies the reward scale the critic has to learn.

`env.prior_features` sums several priors (feature to gain); a feature a market lacks is zero
there, so one pooled agent can hold funding carry on crypto and rate carry on forex. Whenever a
prior is set, the test reports a `prior_only` baseline (the same map with a zero residual), which
isolates what the learned policy adds over its prior.

The prior and `m_t / L` are appended to the observation context. No-trade bands act on target
exposure, so they work the same with and without a map.

The reward is the log
growth of equity after all costs, `log(1 + r)`, which maximizes long-run wealth (Kelly growth).
There is no shaping term: shaped bonuses for trends, streaks or activity pay the agent for
behaviour unrelated to profit.

Rewards are multiplied by a scale computed from the training tapes, `sqrt(1 - gamma^2) / sigma`
with `sigma` the median robust per-bar volatility, so the discounted return of an unlevered
position has a standard deviation near 1. The entropy temperature then means the same thing on
BTC hourly bars and on EURUSD daily bars. Under a volatility target, `sigma` is the target's
per-bar volatility on non-binary tapes.

## Agent

`agent/sac.py` combines four published ideas.

1. **Truncated quantile critics** (Kuznetsov et al., 2020). Each of `E` critics predicts `M`
   quantiles of the soft return. The target pools all `E * M` target quantiles and drops the
   largest `drop` per critic, which controls overestimation more finely than the minimum of two
   critics.
2. **SimBa encoders** (Lee et al., 2025). Running input normalization, pre-LayerNorm residual
   feed-forward blocks and a final LayerNorm. Critics run as one batched ensemble.
3. **Risk-sensitive actor.** With `risk: cvar` the actor maximizes the mean of the lowest
   `cvar_eta` share of pooled quantiles instead of their mean, trading expected growth for a
   thinner left tail.
4. **Automatic entropy tuning** toward `-act_dim * target_entropy_scale`.
5. **Hyperspherical networks** (`sac.network: hyper`, SimbaV2, Lee et al., 2025). Hidden states
   and hidden weight rows stay on the unit sphere (weights are renormalized after every optimizer
   step), with learnable scales and interpolated residuals. Bounded feature norms keep the
   effective learning rate steady when targets are mostly noise.
6. **Pathwise actor objective** (`sac.pathwise_horizon: H`, `agent/pathwise.py`). The market
   path is exogenous, so the next `H` rewards along the recorded tape are exact, differentiable
   functions of the actions (exposure drift, costs, carry and liquidation in torch). The actor
   maximizes the realized `H`-step soft return plus a discounted critic bootstrap, as in
   short-horizon actor-critic (SHAC, Xu et al., 2022) and direct reinforcement (Moody and
   Saffell, 2001). The critic then covers only what lies beyond `H` steps, so less of the policy
   gradient rests on a value estimate fit to noise. Rollouts stop at a binary settlement and
   bootstrap at a tape's last row.

The quantile Huber loss computes its gradient in closed form. On a 4-vCPU machine this takes the
loss from 11 ms to 8 ms per batch of 256.

## Counterfactual replay

A trader small relative to the market does not move prices, so the market half of a transition
depends only on `(tape, row)`. The replay buffer stores `(tape, row, exposure, action, cost
multiplier)` and rebuilds observations on sampling. For a share `relabel_frac` of every batch it
draws a new starting exposure (half uniform over the leverage range, a quarter flat, a quarter at
full long or short) and a new uniform action, then recomputes reward and next exposure exactly
through the tape. The critic sees every position level at every market state, rather than only
the positions its own past policy visited.

## Training and selection

`train.py` runs `n_envs` episodes in parallel on random windows of the training tapes, with an
optional per-episode cost multiplier the agent observes (domain randomization over fees). Every
`eval_every` updates the deterministic policy runs over the full validation tapes; the checkpoint
with the best mean validation Sharpe is kept. The test period is evaluated once, after selection.

A fold deploys only on validation evidence (`sac_gated`): the banded policy trades the test
window if its probabilistic Sharpe on the validation record clears `deploy.gate_psr`, and stays
flat otherwise. That record already chose the checkpoint and the band, so its Sharpe is biased
upward. With `deploy.gate: dsr` the gate deflates it by that search (checkpoint evaluations times
candidate bands, zero-skill variance floor). On a random walk the plain gate let noise folds
through; the deflated gate is the stricter, honest one.

## Evaluation

`experiment.py` trains each seed, evaluates each seed and the seed ensemble (mean action) on the
test tapes, and compares them with flat, buy-and-hold and volatility-targeted trend baselines.
Reports include per-tape metrics, an equal-weight pooled series, the probabilistic Sharpe ratio,
the deflated Sharpe ratio counting every configuration run so far (`runs/trials.jsonl`), and a
paired stationary-bootstrap interval for the Sharpe difference against a reference strategy.

## Deflated Sharpe ratio

A Sharpe ratio reported after trying many configurations is biased upward by the choice of the
best one. Every summary therefore reports the deflated Sharpe ratio framework of Bailey and
Lopez de Prado (2012, 2014) on per-bar returns:

- `psr`: probability that the true Sharpe ratio exceeds zero, correcting the standard error for
  skew and kurtosis of the returns.
- `sr0`: the Sharpe ratio the best of `N` zero-skill configurations would show by luck,
  `sqrt(V) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))` with `g` the Euler-Mascheroni
  constant and `V` the variance of per-bar Sharpe ratios across the `N` trials.
- `dsr`: `psr` measured against `sr0` instead of zero.
- `min_trl_years`: track record length at which `psr` would reach 0.95.

Trial counting is per universe and covers every SAC variant ever reported on it: each seed, the
seed ensemble, the banded and the gated policy, in every experiment recorded in
`runs/trials.jsonl`. `V` is the variance of their per-bar Sharpe ratios, floored at `1/(T - 1)`,
the scatter of Sharpe estimates of zero-skill strategies over `T` bars. `softedge dsr`
recomputes the table over all saved runs, so earlier results deflate further as trials accumulate.

## Comparing variants

Deflated Sharpe ratios ask whether a strategy beats luck. Choosing between variants run on the
same markets and periods is sharper: `softedge compare` bootstraps the Sharpe difference of two
runs on their shared bars (a stationary bootstrap keeps the pairing and the serial dependence),
which cancels the market noise they share. `softedge summary` tests the validation-gated policy
of every run against the best baseline of its market in the same way, and `softedge pbo` gives
the probability of backtest overfitting (CSCV) over every SAC variant tried on a universe.

## Combinatorial purged cross-validation

Walk-forward evaluation yields one out-of-sample path. A `cpcv` config block runs combinatorial
purged cross-validation (Lopez de Prado, 2018): the span is cut into `n_groups` time groups and
every combination of `n_test` groups is held out once. Training data within `purge` before a test
window (whose discounted reward horizon reaches into it) and within `embargo` after it (whose
features still carry test information) is removed. Seeds start from a base agent trained on data
before the span and fine-tune on each split; validation is carved from the latest training window
behind a purge gap. Path assembly stitches `C(n_groups - 1, n_test - 1)` complete backtests, one
out-of-sample result per group, and the report gives the spread of Sharpe, deflated Sharpe,
drawdown and expected shortfall across paths, the share of paths that beat the reference, and the
probability of backtest overfitting when picking the best strategy by in-sample Sharpe.

Set `purge` to at least the discount horizon `1 / (1 - gamma)` bars.

## Data

| Universe | Source | Bars |
| --- | --- | --- |
| `crypto_spot`, `crypto_perp` | data.binance.vision monthly archives, funding events | 1h |
| `fx` | FRED daily reference rates | 1D |
| `crypto_basis` | Spot and perp legs of each symbol: long spot, short perp, funding received | 1h |
| `macro` | FRED: S&P 500, Nasdaq, Nikkei, WTI, Brent; 10y and 2y Treasury excess returns | 1D |
| `polymarket` | Gamma market list, CLOB price history in 14-day windows | 1h |
| `kalshi` | Settled markets of macro series (Fed, CPI, payrolls, unemployment), hourly candles | 1h |
| `hl_basis` | Long Binance spot, short Hyperliquid perp (Hyperliquid funding from May 2023) | 1D |
| `perp_spread` | Long Binance perp, short Hyperliquid perp: the cross-venue funding spread | 1D |

A config may list several universes under `sources`, each with its own costs, so one agent
trains on, for example, crypto perps, forex and macro markets together. Volatility targeting
makes that pooling sensible: a unit action carries the same risk on every market.

Optional exogenous series: `data.implied_vol` joins Deribit's DVOL (30-day implied volatility of
BTC or ETH options, hourly, re-stamped to each candle's end) onto crypto frames, and the features
gain implied minus realized volatility (the variance risk premium) and the change in implied
volatility. `data.regime_features` adds rolling variance ratios, efficiency ratios, lag-1
autocorrelation, volatility of volatility and skewness over 16 and 64 bars, which tell the agent
whether the market is trending or mean-reverting, calm or jumpy.

Treasury markets are excess-return indices built from constant-maturity yields with a
duration-convexity approximation, minus the 3-month bill rate, so a long position does not earn
the cash rate for free. Binary markets stop at their actual close; only the scheduled close
(Kalshi: latest expiration time) feeds the time-to-end feature, since the actual close of an
early-closing market reveals when it resolved.

## Known limitations

- Fills happen at the bar close with a fixed spread and impact model. There is no order book,
  queue position or latency model.
- FRED forex rates are one fixing per day with no intraday high or low, so intra-day liquidation
  risk is invisible for forex.
- Polymarket markets are chosen by total traded volume, a quantity known only after the market
  closes. This selects markets that drew attention and biases the universe.
- Price history at hourly fidelity hides spikes inside the hour, and the spread is a constant
  assumption.
- The basis frame assumes equal leg notionals at every bar and has no intrabar spread range, so
  it ignores second-order drift between the legs and liquidation of the perp leg. A short
  cash-and-carry position borrows spot and pays `costs.borrow_apr` (one flat rate, while real
  margin rates vary by coin and over time). Exchange
  counterparty risk, the main real risk of the carry trade, is outside the model.
- FRED equity indices are price indices without dividends and are not financed, two biases of
  similar size and opposite sign for a long position.

## Recent work this design draws on

Survey of the newest arXiv submissions on Soft Actor-Critic and its applications (query run
through the arXiv API, sorted by submission date, October 2026), with what each changed here.

| Paper | Finding | Use in SoftEdge |
| --- | --- | --- |
| 2609.32819, value divergence in high-UTD SAC | Growing \|Q\| predicts divergence; collapsed representations recover, growing \|Q\| does not | `q_abs_limit` restores the best checkpoint when mean \|Q\| passes a bound |
| 2610.03369, MoE crypto execution | Seeds vary a lot; annealed exploration removed policy collapses | Multiple seeds and an ensemble; `target_entropy_end_scale` anneals the entropy target |
| 2609.31870, actor-critic equity benchmark | Forward retraining before each test window helps SAC | `walk_forward` folds with optional warm start |
| 2606.10448, low-SNR financial SAC | Noisy states plus bootstrapping inflate Q errors ("financial entropy trap") | Clipped scale-free features, truncated quantile critics, validation gate |
| 2609.38903, DAMPER | Actor-critic policies oscillate; temporal-consistency gradients smooth them | `smooth_coef` penalty on action change between consecutive states |
| 2609.09478, tanh Jacobian in SAC | Bypassing the tanh Jacobian for bang-bang actions hurts | Standard tanh-Gaussian kept unchanged |
| 2609.34510, backtest-to-live gap in crypto | Backtests overstate live results under latency and slippage | Cost randomization, impact term; paper trading recommended before capital |
| 2505.23150 (BRC), 2506.03758 (CrossQ + weight norm) | Categorical critics and weight normalization stabilize larger critics | Candidates for future critic variants |

A JAX port of the full update ran 1.16x faster than PyTorch at 128-wide critics and 0.97x at
256-wide critics on a 4-vCPU machine, so the reference implementation stays on PyTorch.
