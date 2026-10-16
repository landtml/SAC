# Parity network: pre-registered evaluation

Registered on 2026-10-16, before the network is run on any tape listed below. The network's code
and constants are fixed in the same commit, in `src/softedge/parity/`. The design is described in
[Parity network](parity-network.md). Synthetic data only. No real-data test is registered here.

What was seen before registration: the simple rules' scores on the discovery search's design and
null tapes. The network itself has been run only on paths generated inside its unit tests.

## Question

The parity network keeps a default exposure of one half and moves away from it only as far as an
evidence ledger allows. Its exposure is capped by a drawdown governor. Two questions:

1. Does it beat the best simple rule on a held-out family?
2. Does it hold up better than every simple rule across all families, including one with no edge?

## Problem and accounting

The problem and accounting are those of the discovery search ([registration](preregistration-discovery.md),
Amendments 1 to 3). There is one risky asset against cash, traded weekly on 1,040-week tapes.
Weeks 1 to 52 are warm-up at exposure 0. The no-trade band is 0.05, costs are 5 bp per unit
traded, and cash earns 0.0005 a week. A breach is wealth more than 25% below its peak after
warm-up. A policy is feasible on a tape set when it breaches on at most 1% of the tapes.

The backtest is `src/softedge/parity/backtest.py`. Before the network is scored, the six simple
rules in that file must reproduce the discovery harness's weekly growth paths on the design and
null tapes, with every weekly difference at most 1e-9. If they do not, the study stops.

## The network, with every constant fixed

| Part | Rule | Constants |
| --- | --- | --- |
| Excess return | x = r - c, the weekly log return minus cash | c = 0.0005 |
| Volatility | EWMA variance of x, seeded with the sample variance of the first 13 weeks | lambda = 0.94 |
| Normalised return | z = x / sigma, where sigma is the volatility before this week, clipped to [-10, 10] | |
| Odd features | EWMA of z at time scales of 1, 4, 13 and 52 weeks, scaled to unit variance under the null, and each multiplied by the volatility modulator | 8 features |
| Volatility modulator | log of fast over 52-week volatility, standardised by its own earlier values, 0 until 52 exist, clipped to [-3, 3] | |
| Direction learners | Ridge regressions of next z on the odd features, with forgetting | half-lives 52, 260 and infinite weeks; ridge 52 |
| Premium learner | Running mean of z, shrunk toward zero | ridge 52 weeks |
| Evidence ledger | One per learner. Bet b = clip(forecast / (2 x RMS of past forecasts), -1, 1). Outcome psi = clip(z, -2, 2) / 2. Fraction lambda = clip(sum u / (sum u^2 + 1), 0, 0.5) from settled bets. Evidence E = max(0, E + log(1 + lambda u)). No bet before 13 forecasts; lambda 0 before 13 settled bets. | |
| Deployment weight | w = 1 - E0 / E when E > E0, else 0 | E0 = log(4 / 0.1) |
| Deployed forecast | w_premium x premium + sum of w_k x forecast_k over max(1, sum of w_k) | |
| Exposure | clip(1/2 + kappa x deployed forecast / sigma, 0, 1) | kappa = 0.5 |
| Governor | min(exposure, clip(m (W - f H) / W, 0, 1)), W wealth, H its peak | f = 0.78, m = 4 |

## Tape sets

| Role | Families | Seeds |
| --- | --- | --- |
| Design | `noisy_prior`, `threshold_signal`, `planted_signal` | 0 to 255 |
| Selection | `jump_garch`, `nonlinear_regime_market` | 1000 to 1255 |
| Test | `break_market` | 20000 to 21999 (2,000 tapes) |
| Null | `gbm`, drift 0 | 5000 to 5255 |

The test seeds are disjoint from the discovery search's test seeds (2000 to 3999), which stay
locked for that search. No tape set is used to choose anything. The network has no tuned constant
and is the only candidate. Design and selection results are reported, not used.

## Claims

**P1, held-out win.** On the test tapes, the annualised paired difference in net log growth against
the best feasible simple rule on those tapes is at least 0.5 points, and its 95% t-interval is above
zero. The network breaches on no more tapes than that rule, and on at most 1% of tapes.

**P2, robustness.** For each of the seven families, take the paired difference against the best
feasible simple rule on that family's tapes. A policy's score is its worst family. A policy that is
infeasible on any family scores minus infinity. P2 passes when the network is feasible on every
family and its score exceeds the score of each of these: the six simple rules, and the trend gate
under the network's governor.

**P3, null.** On the null tapes, momentum ledgers are deployed (w above zero) in at most 2% of
tape-weeks. The future-scramble probe passes: on 32 tapes of each role, flipping the sign of each
week after a cut point, with probability one half, leaves every target up to the cut point unchanged.
The cut points are weeks 60, 200, 400, 600, 800 and 1,000.

**P4, ablations.** Reported on every role, with no pass rule: direction off (exposure one half
under the governor); gate off (every learner fully deployed); governor off; default exposure 1
instead of one half; the trend gate under the governor.

**P5, sensitivity.** Reported on the design and test roles, with no pass rule. Each of these is set
to half and to twice its value: kappa; alpha; the ridge of the direction learners; their two finite
half-lives, together; the volatility half-life (about 11 weeks at lambda 0.94); the governor multiplier.

**P6, novelty.** A label is reported (rediscovery, variant, new combination or new part), with the
prior-art table in the design page.

## Decision rules

- If the backtest does not reproduce the harness, the study stops before the network is scored.
- If the future-scramble probe fails, the results are void.
- If P1 fails, the network is reported as failed on the held-out test. Seeds 20000 to 21999 of
  `break_market` are not used again. A revised network needs a new registration and a held-out
  family on which no version of it has been scored.
- No constant changes after any registered tape is scored. A changed network is a new registration.

## Trial count

One candidate configuration. Ablation and sensitivity runs are diagnostics. They are logged and
their number is reported, but none of them can be selected.

## Not covered

Real data. More than one risky asset. The long-only weekly search: adding the network there needs
a dated amendment to that registration before any of its results exist.
