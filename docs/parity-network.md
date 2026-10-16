# Parity network

A learning architecture for trading built around one symmetry of returns. Its pre-registered
test is in [the registration](preregistration-parity-network.md). The code is in
`src/softedge/parity/`. Until that test has run, this page is a design, not a result.

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
