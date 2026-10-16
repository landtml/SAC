# Results

This page reports what the agent found on synthetic and real markets, with the statistics needed
to read the numbers and their limits.

Measured on 2026-10-13. Every number here is out of sample: the test windows were not used to
train the agent or to pick its settings.

## What we found

- **Where the answer is known, the agent finds it.** On synthetic markets with a planted signal,
  the agent reaches a Sharpe ratio of 15.07. Buy-and-hold reaches 0.04.
- **On random walks, with no edge to find, the agent does worse than buy-and-hold.** Its Sharpe
  ratio is -1.47 against 1.37 for buy-and-hold, and its deflated Sharpe ratio is 0.004.
- **On real directional markets, no edge.** Forex, forex carry, crypto perpetual futures and macro
  markets all show differences from the simple rules whose 95% intervals include zero. The highest
  deflated Sharpe ratio in these rows is 0.37.
- **On carry trades, a small gain over the funding rule.** Two of three carry universes show a
  difference whose 95% interval sits above zero: crypto cash-and-carry (+0.49 Sharpe) and the
  cross-venue funding spread (+0.47). The Hyperliquid basis gain (+0.29) is not statistically
  distinguishable from zero. On a wider list of 32 coins the cross-venue gain holds (+0.66) and
  the cash-and-carry gain shrinks (+0.37, deflated Sharpe 0.13). Under CPCV the cash-and-carry gain
  is positive on all five paths.
- **Against a tuned rule, most of the carry gain goes away.** The funding rule above trades at every
  signal change. A second rule gets the agent's no-trade band and a deadzone, both chosen on
  validation data. Against it, only two carry gains keep a 95% interval above zero: the 32-coin
  cross-venue spread (+0.27) and the cash-and-carry run with the premium feature (+0.31). The
  10-coin cross-venue gain falls from +0.47 to +0.13, with an interval that includes zero.
- **Changes to the agent did not produce a clear win.** On the hidden-regime market, three
  changes raised the Sharpe ratio with intervals that include zero. Three lowered it.

## How to read these numbers

> - [Sharpe ratio](glossary.md#sharpe-ratio): average return divided by its volatility, after all
>   costs, annualized from per-bar returns. Higher is better.
> - 95% CI: the [bootstrap](glossary.md#bootstrap) interval for the Sharpe difference between the
>   agent and the reference rule. If the interval lies entirely above zero, the agent beat the
>   rule in this test.
> - [DSR](glossary.md#dsr): the probability that a Sharpe ratio is real after counting every
>   variant tried, shown in brackets. A DSR of 0.5 means the Sharpe ratio equals the luck
>   benchmark for that many trials. Lower values fall short of it.

Carry Sharpe levels come from a model. They do not include exchange or counterparty risk, so only
comparisons between strategies on the same data mean anything here.

## Synthetic controls

Synthetic markets have a known answer. The planted-signal market should show an edge, and the
random walk should not. The hidden-regime market has changing volatility and fat-tailed shocks.

| Market | SAC Sharpe | Best simple rule | SAC minus rule, 95% CI | DSR (trials) |
| --- | --- | --- | --- | --- |
| Planted signal, 6 synthetic markets, 1 year hourly | 15.07 | buy-and-hold 0.04 | +15.0 [12.7, 17.6] | 1.000 (5) |
| Random walk, 1 year hourly | -1.47 | buy-and-hold 1.37 | -2.8 [-4.8, -0.8] | 0.004 (5) |
| Hidden-regime market, 1 year hourly | 3.76 | buy-and-hold 0.28; sign oracle 8.61 | +3.5 [1.2, 5.9] | 0.14 (70) |

Sharpe is the validation-gated policy's over the full test record, against the best simple rule
of each market (`softedge summary`). Every agent variant tried on the hidden-regime market counts
as a trial, including the ablations below: 70 in all. The comparison with buy-and-hold,
+3.5 [1.2, 5.9], does not depend on that count.

The sign oracle reads each synthetic market's true next expected return and trades its sign. It
shows the ceiling a rule could reach on that market. It is not a strategy.

The planted signal is found with a deflated Sharpe ratio of 1.000. On the random walk, the agent
loses to buy-and-hold. On the hidden-regime market, the agent beats buy-and-hold with an interval
that stays above zero. Its deflated Sharpe ratio is 0.14: once every variant tried on this market is
counted, luck alone could explain a Sharpe ratio this high.

![SAC against the simple rules](../assets/png/figures/sac-vs-rule.png)

## Real markets, directional

These markets have no known answer. The agent is compared with the best simple rule for each
market. Sharpe is the validation-gated policy's over the full test record, against the best
baseline in `softedge summary`; the carry-prior row instead compares with the carry prior alone
on the same days.

| Market | SAC Sharpe | Best simple rule | SAC minus rule, 95% CI | DSR (trials) |
| --- | --- | --- | --- | --- |
| G10 forex, 7 pairs, plain SAC, daily, 12.7 years (2014-2026) | 0.35 | buy-and-hold 0.01 | +0.33 [-0.24, 0.94] | 0.37 (10) |
| G10 forex, SAC on a carry prior | 0.10 | carry rule 0.16 | -0.05 [-0.63, 0.54] | 0.11 (10) |
| Crypto perpetual futures, 10 coins, daily, 3.7 years (2023-2026) | 0.41 | trend 0.64 | -0.24 [-1.54, 1.07] | 0.11 (25) |
| Macro: S&P 500, Nasdaq 100, Nikkei 225, WTI, Brent, 10y and 2y Treasuries, daily, volatility-targeted SAC (validation-gated) | 0.07 | buy-and-hold 0.55; trend 0.30 | -0.49 [-1.00, 0.05] | 0.07 (15) |

All four intervals include zero, so no difference from the simple rule is established.

## Real markets, carry

A [carry trade](glossary.md#carry-trade) earns a premium for holding a position. In these rows the
premium is the [funding rate](glossary.md#funding-rate) on a perpetual future. The agent learns a
residual on top of the funding rule. The "rule" column is the funding rule alone.

| Market | SAC Sharpe | Rule | SAC minus rule, 95% CI, p-value | Active over rule: Sharpe, DSR (trials) |
| --- | --- | --- | --- | --- |
| Crypto [cash-and-carry](glossary.md#basis-cash-and-carry), long Binance spot, short perpetual; 10 coins, daily, 3.7 years | 10.61 | funding rule 10.12; passive carry 8.49 | +0.49 [0.15, 0.95], p=0.002 | 1.52, 0.82 (25) |
| Cross-venue funding spread, long Binance perpetual, short Hyperliquid perpetual; 10 coins, daily, 1.6 years | 6.76 | funding rule 6.29; passive 5.43 | +0.47 [0.14, 0.85], p=0.003 | 3.30, 0.996 (15) |
| Hyperliquid basis, long Binance spot, short Hyperliquid perpetual; daily, 1.6 years | 8.31 | funding rule 8.02 | +0.29 [-0.21, 0.83], p=0.13 | 2.16, 0.85 (15) |

Each row uses the banded policy (the agent without the validation gate), the funding rule (the
prior alone) and buy-and-hold, all on the same test days as the paired test. "Active over rule" is
the Sharpe ratio of the banded policy's returns minus the rule's returns, and the deflated Sharpe
ratio of that difference.

Trial counts include the wider-list and CPCV runs below, which count as further variants on the
same universes.

### Robustness checks

Two checks test whether the carry gains depend on the coin list or on the walk-forward split.

**A wider coin list.** Each carry universe was rerun on 32 coins instead of 10, with the same
configuration. Measures are those of the carry table above: banded policy, funding rule and the
same test days.

| Market, 32 coins, daily | SAC Sharpe | Funding rule | SAC minus rule, 95% CI, p-value | Active over rule: Sharpe, DSR (trials) |
| --- | --- | --- | --- | --- |
| Crypto cash-and-carry, 3.7 years | 10.70 | 10.33 | +0.37 [0.04, 0.76], p=0.015 | 0.46, 0.13 (25) |
| Cross-venue funding spread, 1.7 years | 7.51 | 6.85 | +0.66 [0.39, 0.96], p<0.001 | 6.01, 1.00 (15) |
| Hyperliquid basis, 1.7 years | 8.04 | 8.28 | -0.24 [-1.10, 0.59], p=0.68 | 0.28, 0.08 (15) |

The cross-venue gain holds and grows on the wider list. The cash-and-carry gain shrinks: the paired
interval stays just above zero, but the deflated Sharpe ratio of the active return falls to 0.13.
The Hyperliquid basis shows no gain. The 32 coins are ones listed today, so the list carries
survivorship bias: coins that were delisted are missing.

**Combinatorial purged cross-validation.** The 10-coin cash-and-carry market was also tested with
[CPCV](glossary.md#cpcv), which builds five out-of-sample paths over 2022-07 to 2026-08 from
different train and test splits. The agent's Sharpe ratio is above the funding rule's on all five
paths: 5.04 to 8.36 against 3.44 to 3.47, a difference of +1.6 to +4.9. CPCV trains on data from
both sides of each test window, so these levels are more optimistic than walk-forward. The check
supports the sign of the gain, not its size.

**A tuned rule as second baseline.** The agent's banded policy picks a no-trade band on validation
data. The funding rule above has no band, so it pays costs on every small change of its signal. The
tuned rule is the same funding rule with two settings chosen per walk-forward split on that split's
validation data only: the no-trade band (same grid as the agent) and a deadzone, a funding level
below which the rule stays flat. The test data never enters the choice (`softedge active --baseline
tuned`, `src/softedge/eval/tuned_rule.py`). Rebuilt from the cached data, the plain rule matches each
run's saved `prior_only` series exactly.

| Market, daily | SAC Sharpe | Funding rule | Tuned rule | SAC minus tuned rule, 95% CI, p-value | Active over tuned rule: Sharpe, DSR (trials) |
| --- | --- | --- | --- | --- | --- |
| Crypto cash-and-carry, 10 coins | 10.60 | 10.11 | 10.32 | +0.28 [-0.05, 0.64], p=0.051 | 3.02, 1.00 (22) |
| Crypto cash-and-carry, 10 coins, premium feature | 10.63 | 10.11 | 10.32 | +0.31 [0.05, 0.61], p=0.009 | 3.26, 1.00 (22) |
| Crypto cash-and-carry, 32 coins | 10.69 | 10.33 | 10.46 | +0.23 [-0.13, 0.53], p=0.092 | 2.95, 1.00 (22) |
| Cross-venue funding spread, 10 coins | 6.76 | 6.29 | 6.63 | +0.13 [-0.18, 0.44], p=0.22 | 2.63, 0.96 (12) |
| Cross-venue funding spread, 32 coins | 7.50 | 6.84 | 7.23 | +0.27 [0.04, 0.50], p=0.012 | 4.48, 1.00 (12) |
| Hyperliquid basis, 10 coins | 8.30 | 8.01 | 8.17 | +0.13 [-0.41, 0.72], p=0.32 | 2.19, 0.84 (12) |
| Hyperliquid basis, 32 coins | 8.03 | 8.27 | 8.86 | -0.83 [-2.07, 0.16], p=0.94 | 0.83, 0.22 (12) |

The tuned rule beats the plain rule in every row, by 0.14 to 0.59 Sharpe, so a large part of the
agent's earlier gain came from the no-trade band, not from learning. Two gains keep an interval above
zero. The deflated Sharpe ratio of the active return stays high in most rows because the agent holds
more exposure than the tuned rule and earns a little more return per year (7.3% against 6.6% on the
10-coin spread). Per unit of risk the gap is the Sharpe difference, which is small. Fewer than three
runs per universe compare against this baseline, so the active DSR uses the zero-skill variance floor.

![Carry strategies, equity over time](../assets/png/figures/carry-equity.png)

## Ablations

Each row changes one thing from the base agent on the hidden-regime market, except the last row,
which runs on the crypto carry market. Sharpe is that of the banded policy (`sac_banded`), not of
the seed ensemble, which scores 3.35 for the base.

| Variant | Banded Sharpe | Difference from base, 95% CI | Label |
| --- | --- | --- | --- |
| Base: TQC critics, SimBa encoders, counterfactual replay | 3.76 | reference | reference |
| Critic dropout 0.1 | 4.48 | +0.73 [-0.60, 1.97] | inconclusive |
| Observation noise 0.1 | 4.97 | +1.21 [-0.53, 2.97] | inconclusive |
| Volatility targeting | 4.38 | +0.62 [-1.54, 2.76] | inconclusive |
| [HL-Gauss](glossary.md#hl-gauss) categorical critics | 1.94 | -1.82 [-4.46, 0.95] | rejected |
| [CVaR](glossary.md#cvar) actor | 1.36 | -2.40 [-4.82, 0.10] | rejected |
| Pathwise actor objective (4 steps) | -1.59 | -5.34 [-7.99, -2.72] | rejected |
| Premium-index feature on crypto carry | 10.63 vs 10.61 | +0.03 [-0.10, 0.19] | no gain |

The three rows labelled inconclusive score above the base, and their intervals include zero. The
three labelled rejected score below the base. For the HL-Gauss and CVaR rows, the interval still
includes zero.

![Ablation results](../assets/png/figures/ablations.png)

## Limits

- **Model, not exchange.** Fills happen at the bar close with a fixed spread and impact model.
  There is no order book, queue position or latency model. Exchange counterparty risk is not
  modelled.
- **Borrow and margin are flat.** The short-sale borrow rate is one flat number, and real margin
  rates vary by coin and over time.
- **Hourly bars hide intra-hour moves.** Forex uses one daily fixing with no intraday high or low,
  so intraday liquidation risk is invisible there.
- **Market selection is biased.** Polymarket markets are chosen by total traded volume, which is
  known only after a market closes. The selection favours markets that drew attention.
- **Short records.** The planted and random-walk tests use one test year. The carry rows cover
  1.6 to 3.7 years.
- **Synthetic markets have an answer built in.** They show that the method can find an edge when one
  exists. They do not show that an edge exists in real markets.
- **Deflation grows with trials.** The deflated Sharpe ratio counts every variant recorded on a
  universe. Each new run can lower the DSR of earlier results.
- **Prediction-market results are not in this summary.** The prediction-market configs exist, but
  this page reports no results from them.
- **No live trading.** The numbers above come from backtests only.
