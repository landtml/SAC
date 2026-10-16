# Discovery search: a learner found by trial and error

Registered on 2026-10-16, before any candidate is generated or scored. The plan is the discussion saved at scratchpad/discussion/algorithm_discovery.md, with the decisions below. Research only, on synthetic data. No real-data test is registered here.

Approval basis: the user's instruction of 2026-10-16 to use that plan. The real-data step (item 5 of the plan) is not registered. It needs its own registration and approval.

## Question

Can slot code written by a language model, searched by elitist evolution, produce a learner that beats the best simple rule on a held-out synthetic family under a 25% drawdown budget? Does it beat random recombination of the same parts?

## Problem

One risky asset against cash, weekly. Each bar is one week. The generators' default sigma of 0.01 per bar is used, about 7% a year. Tapes have 1,040 weeks. Weeks 1 to 52 are warm-up for every candidate and rule, with exposure 0. Scoring covers weeks 53 to 1,040.

- Exposure e_t is in [0, 1], the share in the risky asset. It is held over week t+1. The rest is cash.
- Costs: 5 bp per unit of |change in e| traded.
- No-trade band: trade only when |target - held| > 0.05.
- Drawdown budget: a breach is a tape on which wealth falls more than 25% below its peak after warm-up. A candidate is feasible if it breaches on at most 1% of tapes in the family being scored.

## Slots

A, causal state features; B, exposure map to [0, 1]; C, loss shape; D, update rule (step, normalisation, clipping); E, constraint or gate (dual variable, cushion or brake). Each slot is a pure numpy function of at most 25 lines. A static check rejects forbidden names and any access to rows after t.

Forbidden inputs: `mu_next`, `regime`, `jumps`, and any row after t.

## Seed parts and simple rules

Seeds, from which the candidate pool starts: DALA; DALA-off; a trend gate (exposure 1 when the trailing 10-week return is positive); a volatility target of 10%; a CPPI cushion (multiplier 2, floor at 80% of the high-water mark); the Grossman and Zhou rule, exposure = clip(2 (W - 0.75 peak) / W, 0, 1); and an online log-optimal learner, exposure = clip(0.5 mu_hat / sigma_hat^2, 0, 1) on trailing 52 weeks.

The best simple rule on a tape set is the feasible rule, buy-and-hold included, with the highest mean net log growth on that set. Buy-and-hold holds e = 1, subject to the band.

## Families and seeds

- Design (fitness): `noisy_prior`, `threshold_signal`, `planted_signal`. Seeds 0 to 255 each.
- Selection: `jump_garch`, `nonlinear_regime_market`. Seeds 1000 to 1255.
- Test: `break_market`. Seeds 2000 to 3999, 2,000 tapes. Locked, scored once.
- Null: `gbm` with drift 0, so no edge. Seeds 5000 to 5255.
- Fresh re-scoring in generation g uses seeds from 10000 + 256 g.

Generators: `src/softedge/data/bench.py` and `src/softedge/data/synthetic.py`.

## Score

For a candidate on a tape set, take the weekly net log growth minus that of the best rule on the same tapes, averaged over scored weeks, then over tapes. Annualise by multiplying by 52. Subtract a complexity penalty of 0.0005 per line of slot code above 10 lines. The interval is a t-interval over the per-tape values.

## Search

- **Arm L, language model.** Generations of 50 candidates, 10 at most. Each of four proposal calls receives the top 8 candidates and 4 diverse ones, with code and scores, and proposes about 12 children. Each child changes exactly one slot and states its mechanism in one sentence.
- **Arm R, control.** 500 random draws, each picking one part per slot from the seed parts and a grid of constants. No language model.
- **Ledger.** Every evaluation is appended to `discovery_trials.jsonl` with arm, generation, candidate id, family, seeds and score. Identical exposure paths count once in the statistics.

## Trial policy

- No deflated Sharpe ratio inside the loop. Fresh draws each generation control for noise-driven luck.
- Test: the one candidate picked from the top 10 on the selection family is scored once, on the 2,000 test tapes. Only one candidate reaches the test, so no deflation is applied. The total number of evaluations is reported as context.
- Real data: not registered.

## Success, all required

- **S1, held-out win.** On the test tapes, the annualised paired difference against the best rule on the test family is at least 0.5 points. Its 95% interval is above zero. The candidate has no more breaches than that rule.
- **S2, nulls and leaks.** On the null family, the upper 95% bound of the paired difference is at most zero. The finalist passes every leak probe. The leaky calibration example fails the same probes.
- **S3, ablation.** Resetting the changed slot to its parent's code leaves at most half of the S1 gain. The difference between the candidate and the ablated version has a 95% interval above zero.
- **S4, robustness.** The gain stays positive when each numeric constant in the candidate's code is multiplied by 0.5 and by 2.
- **S5, novelty.** A label is reported: rediscovery of X, variant of X, new combination, or new part, with the search log attached.

## Kill rules

Stop, report, and do not extend.

- **K1.** After generation 5, no candidate beats the best rule on fresh design draws.
- **K2.** The finalist fails S1. No retry on `break_market`. A retry needs a new family and a new registration.
- **K3.** A leak is found. The results are void, and the trials still count.
- **K4.** The finalist is a rediscovery. It is reported with no novelty claim.
- **K5.** Spending reaches 5M tokens, or the run passes 14 days.

## Budget and pilot

The budget cap is 5M tokens, recorded in `scratchpad/discovery/budget.json`. Before the full search, run a pilot of one generation: 50 language-model candidates and 50 random. Its purpose is to test the harness, the leak probes and the cost per generation. Continue only if the leak probes pass, the harness has no open bug, and one generation costs at most 0.4M tokens.

## Prior art

The search methods (AutoML-Zero, FunSearch, AlphaEvolve) and the seed parts listed in the discussion are to be checked before any novelty claim. The checks are not complete.

## Not covered

Real data. Live trading. Any change to the long-only registration. The funding-spread study.

## Amendments

None.
