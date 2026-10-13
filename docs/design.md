# Design

sactrader trains one Soft Actor-Critic agent family across four market types. Each section below
names a choice and the failure it prevents.

## Market tape: one accounting model for four markets

Every instrument becomes a `MarketTape` (`src/sactrader/tape.py`): per-step returns, worst
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

No scaler is fit on the full sample. The agent normalizes inputs with running statistics updated
from training batches only.

The observation is a window of feature rows plus context: current exposure over max leverage, log
trading cost on each side, carry on each side, max leverage and a one-hot market type. Tapes with
different feature sets share the union of names, missing columns filled with zeros, so one agent
can train on all four markets.

## Action and reward

The action `a` in `[-1, 1]` sets the target exposure `a * max_leverage`. The reward is the log
growth of equity after all costs, `log(1 + r)`, which maximizes long-run wealth (Kelly growth).
There is no shaping term: shaped bonuses for trends, streaks or activity pay the agent for
behaviour unrelated to profit.

Rewards are multiplied by a scale computed from the training tapes, `sqrt(1 - gamma^2) / sigma`
with `sigma` the median robust per-bar volatility, so the discounted return of an unlevered
position has a standard deviation near 1. The entropy temperature then means the same thing on
BTC hourly bars and on EURUSD daily bars.

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

## Evaluation

`experiment.py` trains each seed, evaluates each seed and the seed ensemble (mean action) on the
test tapes, and compares them with flat, buy-and-hold and volatility-targeted trend baselines.
Reports include per-tape metrics, an equal-weight pooled series, the probabilistic Sharpe ratio,
the deflated Sharpe ratio counting every configuration run so far (`runs/trials.jsonl`), and a
paired stationary-bootstrap interval for the Sharpe difference against a reference strategy.

## Data

| Universe | Source | Bars |
| --- | --- | --- |
| `crypto_spot`, `crypto_perp` | data.binance.vision monthly archives, funding events | 1h |
| `fx` | FRED daily reference rates | 1D |
| `polymarket` | Gamma market list, CLOB price history in 14-day windows | 1h |

## Known limitations

- Fills happen at the bar close with a fixed spread and impact model. There is no order book,
  queue position or latency model.
- FRED forex rates are one fixing per day with no intraday high or low, so intra-day liquidation
  risk is invisible for forex.
- Polymarket markets are chosen by total traded volume, a quantity known only after the market
  closes. This selects markets that drew attention and biases the universe.
- Price history at hourly fidelity hides spikes inside the hour, and the spread is a constant
  assumption.

## Recent work this design draws on

Survey of the newest arXiv submissions on Soft Actor-Critic and its applications (query run
through the arXiv API, sorted by submission date, October 2026), with what each changed here.

| Paper | Finding | Use in sactrader |
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
