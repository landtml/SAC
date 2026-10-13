# Glossary

Terms in alphabetical order. Each entry gives the plain meaning in this project.

### Actor

The part of the agent that chooses a target position from what it observes. It is trained to pick
positions that the critic expects to earn more reward.

### Basis (cash-and-carry)

Buying a coin on the spot market and shorting its perpetual future at the same time. The two legs
mostly cancel the price move, so the return is mostly the funding that the short side receives.

### Bootstrap

A way to measure uncertainty by resampling the return history. This project uses a stationary
bootstrap, which resamples blocks of random length so that short-range patterns in returns stay
intact.

### Carry trade

Holding a position because it pays a steady premium, such as a higher interest rate on one currency
than on another, or the funding on a perpetual future. The risk is a sudden price move against the
position that wipes out the premium.

### Counterfactual replay

Reusing past market moves with positions the agent did not hold. A small trader does not move
prices, so the market part of an old transition still holds. The project recomputes the reward for
the new position and action, so the critic sees more position levels in each market state.

### CPCV

Combinatorial purged cross-validation. The time span is cut into groups, every combination of test
groups is held out once, and training data near each test group is removed. The result is many
backtest paths instead of one.

### Critic

A model that estimates the expected future reward of a state and an action. The critics in this
project predict a spread of possible outcomes, not a single number.

### CVaR

Conditional value at risk, also called expected shortfall. It is the average outcome in the worst
fraction of cases. The CVaR actor gives up some expected reward for a thinner left tail.

### Drawdown

The fall in equity from its highest point to the next low. The maximum drawdown is the largest such
fall over the whole record.

### DSR

Deflated Sharpe ratio. It is the probability that a Sharpe ratio is real after allowing for the
best Sharpe that the same number of trials would reach by luck alone. It uses the [PSR](#psr) with
that higher bar instead of zero.

### Embargo

A gap of time between two windows, such as between training and validation, or between validation
and test. It stops features that look back in time from carrying information across the boundary.

### Entropy

A measure of how spread out the policy's choices are. SAC adds a bonus for keeping choices spread,
which keeps the agent exploring. The entropy temperature sets the size of that bonus, and automatic
tuning moves it toward a target.

### Favourite-longshot bias

The tendency of prediction-market prices to put too high a price on unlikely outcomes and too low a
price on likely ones. The prediction-market prior uses log-odds with a positive gain, which tilts
positions toward the favourite.

### Funding rate

A periodic payment on a perpetual future that keeps its price close to the spot price. When the
rate is positive, longs pay shorts. When it is negative, shorts pay longs.

### HL-Gauss

A critic type that predicts a histogram over bins of possible returns, not a set of quantiles. Each
training target is spread over nearby bins with a Gaussian.

### Look-ahead leakage

A backtest error in which a decision uses information that was not available at the time, such as
a price from a later bar or a funding payment that had not yet settled. The test suite builds
features on a prefix of the data and on the full data, and fails if the two differ.

### MinTRL

Minimum track record length. It is the number of years of out-of-sample returns needed before the
[PSR](#psr) reaches 0.95. A Sharpe ratio with a short record needs a longer record to be trusted.

### Out of sample

Data that the agent did not train on and did not use to choose settings. The results in this
project come only from test windows.

### PBO

Probability of backtest overfitting. It is the chance that the variant that looks best on the
training part of the data ranks below the median on the held-out part. It is computed over every
variant tried on a universe.

### Perpetual future

A futures contract with no expiry date. Funding payments between longs and shorts keep its price
close to the spot price.

### Prediction market

A market in which you buy YES or NO tokens on an event. A YES token pays 1 if the event happens and
0 if it does not. This project uses Polymarket and Kalshi.

### Prior and residual policy

The action is a known premium, the prior, plus a learned adjustment, the residual. Examples of a
prior are a trend signal, a funding signal or a rate differential. The actor is penalized for
moving away from the prior unless the critic sees a gain worth the penalty.

### PSR

Probabilistic Sharpe ratio. It is the probability that the true Sharpe ratio is above a benchmark,
usually zero. It corrects for the skew and fat tails of the returns.

### Purge

Removing training data within a set time before a test window. The agent's reward horizon reaches
into that window, so nearby training data would overlap the test.

### Replay buffer

The memory of past transitions that the agent samples from during training. This project stores
the indices of each transition, such as the market row and the position, and rebuilds the
observation when it samples.

### SAC

Soft Actor-Critic. A reinforcement-learning method with an actor, critics and an entropy bonus. The
version here uses TQC critics, SimBa networks and automatic entropy tuning.

### Sharpe ratio

Average return divided by the volatility of returns. Here it is computed after all costs, from
per-bar returns, and annualized. Higher is better. The scale has no fixed meaning on its own.

### SimBa

A network design that normalizes its inputs and uses residual blocks with a final LayerNorm. The
design keeps larger networks stable.

### TQC

Truncated quantile critics. An ensemble of critics each predicts a spread of outcomes. The pooled
predictions have their largest values dropped, which reduces overestimation.

### Trial

One variant of the agent that was evaluated and reported. The count covers every seed, seed
ensemble, banded policy and gated policy in each experiment recorded in `runs/trials.jsonl`. The
deflated Sharpe ratio penalizes a Sharpe ratio for this count.

### Validation

The period used to choose checkpoints, the no-trade band and the deploy-or-stay-flat gate. It lies
between training and test. The gate deploys the strategy on the test window only if the validation
[PSR](#psr) clears a set threshold. Otherwise the strategy stays flat.

### Volatility targeting

Scaling the position so that its expected risk stays about the same when volatility changes. The
target position is the target risk divided by a volatility estimate that uses only past data, and
it is capped at the maximum leverage.

### Walk-forward

Testing by moving forward in time. Train on the past, choose settings on a validation window, test
once on the next window, then move all the windows forward. The test windows are joined into one
out-of-sample record.
