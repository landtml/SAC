# Deep parity network: pre-registered evaluation

Registered on 2026-10-17, before either model below is run on any tape or series listed here.
Both models are frozen in this commit:

| Model | Role | File | SHA-256 |
| --- | --- | --- | --- |
| Deep parity network, trained end to end in JAX | Candidate | `models/parity_deep_jax.npz` | `dc459bcd38089800a47e6cd6f436f814019c311bf6801efaa0588192bc25adba` |
| Supervised forecaster, numpy | Diagnostic | `models/parity_deep_numpy.npz` | `0be9834ea4dde400dae31aa2788ff3a9a5a41c31076b873c212b9fc182f39bad` |

What was seen before registration: training and validation on simulated markets only, from the
project's own simulator; the coverage counts of the crypto panel (coins, weeks, dates, gaps); and
the currency panel, which the first parity network has already been run on. No model, rule or
return average has been computed on any evaluation set below.

## Question

Can a deep network trained only on simulated markets read a real market's history well enough to
beat the simple rules, without trading noise where there is no edge?

## The candidate

Code: `src/softedge/parity/jaxnet.py`, trained by `scripts/parity_jax_train.py`.

- **Memory.** Diagonal linear recurrences over the normalised excess return z: odd EWMAs of z at
  initial time scales of 1 (decay 0.95) to 256 weeks, and the t-statistic of z's running mean;
  even EWMAs of z^2 - 1 at 4, 16 and 64 weeks; even EWMAs, at 104 and 416 weeks, of z times last
  week's odd memories at 1, 4, 16 and 64 weeks; two log volatility ratios; log history length; log
  volatility. The 14 decay rates are learned. Volatility is an EWMA with lambda 0.94.
- **Odd network.** An MLP with three hidden layers of 64 SiLU units on the 25 inputs, made odd:
  g(u) = (f(u) - f(S u)) / 2, where S flips the 10 odd inputs. There is no output bias, since it
  would cancel. Inputs are scaled by fixed factors set from simulated markets before training.
- **Decision.** Exposure (1 + tanh(w g)) / 2, capped by the governor
  clip(4 (W - 0.78 H) / W, 0, 1). The weight w comes from one sign-test evidence ledger that bets
  on g, with the rule of the first parity network and alpha = 0.1, so it deploys above
  E0 = log 10 nats.
- **Training.** Fully in JAX. Each of 2,500 steps draws 256 fresh simulated markets of 1,040
  weeks from `src/softedge/parity/jax_world.py` and takes one step on minus the mean weekly net log
  growth of the differentiable backtest. That backtest charges costs of 10 bp per unit traded and
  includes the 0.05 no-trade band, 52 weeks of warm-up and the governor; the gate is not used in
  training. Optax AdamW: peak learning rate 1e-3, 100 warm-up steps, cosine decay to 1e-5, weight
  decay 1e-4 on kernels, gradient clipping at norm 1. Every 100 steps the policy is scored on
  1,024 fixed simulated markets against the one-half policy under the same governor. The best of
  those checkpoints is the frozen candidate. A unit test confirms that the differentiable backtest
  reproduces `softedge.parity.backtest.simulate` to 1e-10.
- **Training result, on simulated markets only.** The best checkpoint is step 1,800 of 2,500. It
  gains 0.24 points a year of net log growth over the one-half policy on the validation markets:
  0.36 on markets with an edge and 0.05 on markets without one. The model has 10,062 trained
  parameters. Training took 57 minutes on CPU.
- **Simulated markets.** 40% have excess returns symmetric about zero. The rest switch on, each
  with probability one half, a premium, a persistent hidden drift, return autocorrelation,
  two-state regimes, a volatility-linked drift and price mean reversion. 30% of those redraw
  their parameters at a random week. Volatility is a log AR(1) from 0.4% to 15% a week, shocks are
  Student t with 3 to 30 degrees of freedom, and 30% have jumps. Cash earns 0 to 5.2% a year. None
  of the repository's benchmark generators is used.

## Diagnostics

Reported, never selected: the candidate without the gate; the candidate with direction off
(one half under the governor); the numpy forecaster (an odd MLP of the same size, trained by mean
squared error on next-week z from `softedge.parity.world`, deployed through the same kind of
ledger with a Kelly tilt, `src/softedge/parity/deep_network.py`); the first parity network; the
six simple rules; the trend gate under the governor.

## Evaluation sets

| Set | Content | Accounting |
| --- | --- | --- |
| Held-out synthetic | `regime_switching`, `ar_returns`, `ou_mean_reverting`, default parameters, weekly, 1,040 weeks, seeds 30000 to 30999 each | Cash 0.0005 a week, cost 5 bp |
| Known synthetic | The seven families of the first parity test, seeds 40000 to 40255 | Same |
| Null | `gbm`, drift 0, seeds 50000 to 50255 | Same |
| Crypto (primary real) | Every cached Binance USDT spot coin with at least 104 weekly returns | Cash DTB3, cost 10 bp |
| Currencies (secondary real) | The 13 deposits of the currency check | As in that check |

No parity network has been run on the three held-out families. The crypto panel: weeks end on
Sunday (UTC) and take the coin's last daily close. A week with no close carries the price, so
its return is zero and the move is credited when trading resumes. Stablecoin pairs and leveraged
tokens are excluded; delisted coins are included until their last close, and reused tickers are
split. 406 coins qualify, with 100,045 coin-weeks from 2019-01-13 to 2026-10-04 and a median of
251 weeks. 139 stop before the last month, and 67 weeks are gaps. Coins never listed on Binance are
absent. The currency panel was seen by the first parity network, so it is secondary.

Accounting is that of the earlier checks: 52 weeks of warm-up, the 0.05 band, and a breach when
wealth falls more than 25% below its peak.

## Claims

**DL1, held-out synthetic.** Pooled over the 3,000 held-out tapes, the paired annual difference in
net log growth against each family's best feasible simple rule has a mean above zero and a 95%
t-interval above zero. The candidate breaches on at most 1% of each family's tapes.

**DL2, crypto.** The pooled difference against the best simple rule has a 95% block-bootstrap
interval above zero (calendar weeks, blocks of 52, 10,000 draws, as in the currency check). The
best rule is the feasible one, breaching on at most 1% of coins, with the highest pooled growth.
If no rule is feasible, it is the rule with the fewest breaches. The candidate breaches on no more
coins than that rule.

**DL3, depth earns its place.** On crypto, the pooled difference against the first parity network
has its 95% interval above zero.

**DL4, null and leaks.** On the null tapes the ledger deploys in at most 2% of tape-weeks. Flipping
the sign of each week after a cut point leaves every earlier target unchanged, on 32 tapes of each
held-out family and the 32 longest crypto series. The cuts are weeks 60, 200, 400, 600, 800 and
1,000.

Reported with no pass rule: every comparison above for every diagnostic, the known families, the
currency panel, and per-series growth, CAGR, drawdown, Sharpe ratio, exposure and turnover.

## Decision rules

- Neither model, its code nor any constant changes after an evaluation set is scored. A changed
  model is a new registration.
- If DL4's probe fails, the results are void.
- A diagnostic that beats the candidate is reported, not claimed.
- The held-out seeds and the crypto panel are not reused to evaluate a revised version of this
  candidate.

## Trial count

One candidate. Its training choices were fixed before training, and the only selection was the
best validation checkpoint on simulated markets. The numpy forecaster was trained once, before the
JAX work began, and is reported as a diagnostic.

## Software

JAX 0.11.2, Flax 0.12.10 (NNX), Optax 0.2.8, Orbax 0.12.7 and Chex 0.1.92, on CPU.
