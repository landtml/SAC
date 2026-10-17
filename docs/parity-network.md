# Parity network

A learning architecture for trading built around one symmetry of returns. Its pre-registered
test is in [the registration](preregistration-parity-network.md). The code is in
`src/softedge/parity/`.

**Result of the registered test: fail.** On the held-out family the network is 0.22 points a year
below the best simple rule (CPPI), and the trend gate under the same governor is more robust
across families. It does no harm where there is no edge, and it passes every leak probe. Details
are in the [research log](research-log.md#2026-10-16-parity-network-registered-test) and in
[What the test showed](#what-the-test-showed) below.

**Result of the registered real-data check: pass, narrowly.** On 13 currency deposits from 1971
to 2026, the same network beats the best feasible simple rule (Grossman-Zhou) by 0.38 points a
year (95% interval 0.04 to 0.76), its learned direction adds 0.49 points (0.11 to 0.91), and it
never breaches the 25% budget. Most of the gain comes from BRL, KRW and CNY. It is not significant
at twice the costs, and the trend gate under the same governor did at least as well. See the
[research log](research-log.md#2026-10-16-parity-network-on-real-currency-data).

## What makes markets hard for learners

| Market fact | What it does to a learner | Where the network handles it |
| --- | --- | --- |
| Direction is barely predictable; size of moves is | Most fitted direction is noise | Direction is learned apart from risk and must earn its use |
| Fat tails and clustered volatility | Gaussian tests and losses are dominated by a few weeks | Bets are bounded and use a sign test that holds for any volatility or tails |
| Edges appear, decay and flip | A model fitted on the past keeps trading a dead edge | Learners forget at several speeds; evidence decays when a learner starts losing |
| Many ideas are tried | The best of many backtests is mostly luck | Each learner's evidence is out of sample by construction and has a known false-positive bound |
| Compounding and a drawdown budget | Average return is the wrong target; one bad path ends the account | Exposure is sized for log growth and capped by a governor |
| A price-taker sees every counterfactual | Reinforcement learning wastes this | Every learner is scored every week, deployed or not |

## The symmetry

Let x be the weekly excess log return, the asset's log return minus cash. Mirror the market by
flipping the sign of every x. Each quantity the network computes is then either **even** (unchanged:
volatility, the size of moves, the evidence) or **odd** (flipped: momentum, the premium, every
direction forecast).

**The even part has an exact answer.** Suppose the network knows nothing about direction, so x is
symmetric about zero given the past. The growth of exposure e against cash is
g(e) = E log(1 + e (exp(x) - 1)). Its slope at e = 1/2 is E[2 tanh(x / 2)], which is zero because
tanh is odd, and g is concave. So **one half is the growth-optimal exposure under directional
ignorance, for any volatility and any tail shape.** The network uses one half as its default and
estimates nothing to get it.

**The odd part is learned and gated.** Direction learners forecast the next normalised excess
return. A forecast moves the exposure away from one half by the Kelly amount
kappa x forecast / volatility, but only in proportion to the evidence its ledger holds.

**Mirror equivariance.** Without the governor, mirroring the market maps the exposure e to 1 - e,
exactly. So the network has no built-in view on direction. A unit test checks this, and also checks
that a default of 1, which is a directional prior, breaks it.

## Parts

```
weekly log return ──► excess return x ──► volatility sigma (even) ──► normalised z = x / sigma
                                                                        │
            ┌───────────────────────────────────────────────────────────┘
            ▼
 odd features: EWMA of z at 1, 4, 13, 52 weeks, and each times the volatility modulator
            │
            ▼
 learners: premium (mean of z) and three ridge regressions forgetting at 52, 260, infinite weeks
            │ forecasts
            ▼
 evidence ledgers: one sign-test bet per learner per week ──► weight w = (1 - E0 / E)+
            │ deployed forecast
            ▼
 exposure = clip(1/2 + kappa x forecast / sigma, 0, 1) ──► governor cap on own drawdown ──► trade
```

### Evidence ledgers

Each week a learner bets on its forecast: b = clip(forecast / scale, -1, 1). The bet settles on
psi(z) = clip(z, -2, 2) / 2, an odd, bounded function of next week's normalised return. The ledger
compounds the capital K by the factor 1 + lambda b psi, where lambda is in [0, 1/2] and is set from
earlier bets only.

Under the null that next week's sign is a fair coin given the past and the size of the move, every
bet has mean zero, because psi is odd. K is then a non-negative martingale, for any volatility
process and any tails. By Ville's inequality, the chance that K ever reaches 1/alpha is at most
alpha. A unit test confirms this on 4,000 paths with GARCH volatility and t tails.

The ledger keeps a CUSUM of log capital: E = max(0, E + log(1 + lambda u)). Evidence rises while a
learner wins and drains when it starts losing, so a learner whose edge died after a break loses
its weight. The deployment weight is the positive-part James-Stein form w = (1 - E0 / E)+, with
E0 = log(K / alpha) for K = 4 ledgers and alpha = 0.1. Nothing is deployed below E0, and the weight
approaches 1 as evidence grows.

Because each learner predicts before the outcome and is scored after it, the evidence is out of
sample by construction. A new learner can be added at any time. It starts with no evidence, so
adding learners does not let noise through, apart from the log K in the threshold.

### Governor

The exposure is capped at clip(m (W - f H) / W, 0, 1), where W is wealth and H its peak, with
m = 4 and f = 0.78. The cap is 0.88 at the peak, about 0.53 at a 10% drawdown, and 0 at 22%. The
floor sits 3 points above the 25% budget, to leave room for a weekly gap. This is the cushion rule
of Grossman and Zhou and of CPPI, applied to the whole network. It is the only part that is not
mirror-equivariant, since the budget treats losses differently from gains.

## The class, beyond this instance

The instance tested here is deliberately small, because one weekly path cannot support more. A
parity network is any learner with these four properties:

1. The default is the growth-optimal exposure under directional ignorance. For many assets with
   symmetric excess returns, it is approximately the portfolio that maximises the excess growth
   rate, w = (1/2) Sigma^-1 diag(Sigma), in the sense of stochastic portfolio theory.
2. Direction comes from a trunk that sees only the past and is mirror-equivariant. A neural trunk
   g can be made so by antisymmetrising it: f(x) = (g(x) - g(-x)) / 2. For many assets, the
   symmetry group flips each asset's sign separately.
3. Every directional deviation is deployed through a sign-test evidence ledger that can decay.
4. A governor enforces the path budget.

## Deep version, trained in JAX

**Result of its registered test: fail** on held-out synthetic families, on crypto, and on the
depth comparison. On the null family it stays out, and every leak probe passes. Its evidence
gate keeps it idle on short or noisy histories, and where it trades it matches the linear
version. See the [research log](research-log.md#2026-10-17-deep-parity-network-trained-in-jax-registered-test).

One market's history cannot fit a deep network, so the deep version learns across markets
instead. It is trained once, on hundreds of thousands of simulated markets, then frozen. On a new
market it adapts only through its inputs: memories of that market's own history. The weights
encode how to read a history, and the history says what this market does. This is in-context
learning, the idea behind prior-fitted networks.

| Part | JAX tool | What it does |
| --- | --- | --- |
| Simulated markets | `jax.random`, `lax.scan`, `jit` | Draws 256 fresh 20-year markets per training step, so no market is seen twice |
| Memory | Flax NNX parameters | Linear recurrences at 1 to 416 weeks whose decay rates are learned, like a state-space layer |
| Odd network | Flax NNX | Three hidden layers of 64 units, odd by construction |
| Backtest | `lax.scan`, `grad`, `vmap` | The weekly backtest with costs, band and governor, differentiable end to end and run over all markets at once |
| Optimiser | Optax | AdamW, warm-up and cosine decay, gradient clipping |
| Checkpoints and tests | Orbax, Chex | Training checkpoints; tree assertions in the unit tests |

Training maximises net log growth directly through the backtest, so costs, the band and the
governor shape the policy. In evaluation, the evidence ledger gates the network's conviction as
in the first version. A unit test confirms that the differentiable backtest matches the numpy
backtest to 1e-10. The registered test is [Deep parity network](preregistration-parity-deep.md).

## Pooled version

**Result of its registered test: fail** on blind synthetic panels, where it breaches the 25%
budget on 15 to 50 of 1,000 assets per family, and on the pooling claim. **Pass** on the macro
panel (equity indices, oil and Treasuries to 2005): 1.80 points a year above Grossman-Zhou
(95% interval 0.90 to 2.74) with no breach. On null panels it stays out and every leak probe
passes. See the [research log](research-log.md#2026-10-17-pooled-parity-network-registered-test).

The deep version could not prove an edge on one history. The pooled version reads a panel of
assets and changes three parts (`src/softedge/parity/pooled.py`):

- **Pooled memory and evidence.** The network reads each asset's memories and the panel's mean
  memories. One ledger bets on every listed asset each week and settles on the average, so an
  edge that recurs across assets builds evidence faster. Its capital stays a test martingale
  under the sign-flip null with any cross-asset dependence.
- **Gated premium.** Each asset's Sharpe ratio over cash, shrunk towards the panel's, gives a
  half-Kelly tilt that a second ledger must earn before it moves the default of one half.
- **Crash governor.** The exposure cap is (W - 0.76 H) / (W c), with c a crash size estimated per
  asset. A weekly loss no larger than c cannot breach the budget. The test showed the weak
  point: c comes from history, so a crash larger than any seen before breaks the guarantee.

It is trained in JAX on simulated panels with shared drifts, common shocks and crashes, and
assets that list late, through the panel backtest with the gates open.

## What is new and what is not

| Part | Prior work |
| --- | --- |
| One half is growth-optimal for symmetric returns | Rebalancing gains, "Shannon's demon" (Luenberger 1998); excess growth rate (Fernholz 2002) |
| Kelly sizing of a forecast | Kelly 1956; Breiman 1961 |
| Sign-flip null | Randomisation and sign-flip tests (Fisher) |
| Betting capital as evidence, Ville's inequality | Ville 1939; Shafer and Vovk 2001, 2019; Waudby-Smith and Ramdas 2024; Ramdas et al. 2023 |
| CUSUM of evidence | Page 1954; e-detectors (Shin, Ramdas and Rinaldo 2024) |
| Positive-part shrinkage | James and Stein 1961 |
| Ridge with forgetting; multiscale momentum | Recursive least squares; Moskowitz, Ooi and Pedersen 2012 |
| Drawdown cushion | Grossman and Zhou 1993; CPPI (Black and Perold 1992) |
| Equivariant networks | Cohen and Welling 2016 |

Every part is published. What may be new is the arrangement: the default comes from the symmetry,
and every directional deviation passes through a sign-test ledger that decays. No literature
search has been run for this page. Until one has, the label is "new combination, unchecked".

## What the test showed

What held up:

- **No harm without an edge.** On the null family the momentum ledgers deploy in 0.4% of
  tape-weeks, and the network then matches its direction-off version.
- **No leaks.** The future-scramble probe passes on all seven families.
- **Feasible everywhere.** At most 2 breaches in 2,000 held-out tapes. Without the governor there
  are 30.
- **It finds a strong edge.** On `noisy_prior`, direction adds 1.10 points a year (paired interval
  0.99 to 1.22) over the same network with direction off.

What did not:

- **The default ignores the cash hurdle.** One half is growth-optimal when the excess return is
  symmetric about zero. In these synthetic families the asset's log drift is near zero and cash pays
  2.6% a year, so cash-heavy cushion rules win.
- **The gate is too strict for one path.** The premium learner sees that cash wins, but its evidence
  stays below the line. Removing the gate beats the network on every family, by 0.21 to 0.99 points.
  On `noisy_prior` the gate deploys momentum years late and costs about 1 point.

Questions for a revised version, which needs its own registration and a family this version has
not seen:

- A default that includes a prior on the premium over cash, rather than assuming none.
- Evidence pooled across many assets or markets, so that an edge too weak to prove on one path can
  pass the gate when it recurs.
- A lower turnover path for deployed forecasts.

## Known limits

- **Slow on weak edges.** Log capital grows by about half the squared correlation between bet and
  outcome each week. At a weekly correlation of 0.2, evidence reaches E0 (3.7 nats) in about three
  to four years. At 0.05 it gains about 1.2 nats over a 20-year tape and is never deployed. This is
  deliberate: an edge that weak cannot be told from luck on one path.
- **The governor locks in cash.** After a 20% drawdown the cap is 0.1, and recovery is slow. This
  is the price of the guaranteed budget.
- **Bang-bang exposure.** At 1% weekly volatility, the Kelly tilt is large for small forecasts, so a
  deployed learner moves the exposure between 0 and 1. The band and costs damp the churn but do not
  remove it.
- **Synthetic only.** Real-data use needs its own registration.
