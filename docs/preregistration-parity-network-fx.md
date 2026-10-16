# Parity network on real currency data: pre-registered check

Registered on 2026-10-16, before the network is run on any real data. The network is the one
registered in [Parity network: pre-registered evaluation](preregistration-parity-network.md),
with every constant unchanged. That version failed its synthetic held-out test, and this is a
check of the same fixed network on real markets, not a search. The code is `scripts/parity_fx.py`.

What was seen before registration: the panel's dates, week counts, stale-price counts, largest
weekly moves, and one hand-checked week. No return average, strategy result or rule result on
this panel was computed.

## Why currencies

The long-only weekly search reserves 2006 to 2016-09 of NASDAQ, NIKKEI, the 10-year Treasury proxy
and WTI as its test window, and its pass rule is still undecided. Running anything on those series
now would spend that window. Currency deposits share no series with it, and no registered study
holds a currency window back.

## Market

A US-dollar investor holds either a foreign-currency deposit or US Treasury bills, weekly, long
only.

- **Deposit return**, week ending Friday t: the log change in the dollar price of the currency
  between Friday closes, plus the foreign short rate of the calendar month before the week starts,
  accrued over 7 days as log(1 + rate x 7 / 365). A Friday takes the last FX observation on or
  before it.
- **Cash return**: DTB3 on the Friday the week starts, accrued the same way.
- **Data**: cached FRED files, read without download. FX from the H.10 series; foreign rates are
  OECD 3-month interbank rates (`IR3TIB01*`), or call-money rates (`IRSTCI01*`) for BRL and INR.
  A rate is carried forward after its series ends.
- **Sample**: each currency from its first week with an FX price, a foreign rate and DTB3, to its
  last Friday with an FX price.

| Currency | FX series | Rate series | First week | Weeks | Cost per unit traded |
| --- | --- | --- | --- | --- | --- |
| GBP | DEXUSUK | IR3TIB01GBM156N | 1971-01-15 | 2,909 | 2 bp |
| JPY | DEXJPUS | IR3TIB01JPM156N | 2002-05-10 | 1,275 | 2 bp |
| CHF | DEXSZUS | IR3TIB01CHM156N | 1999-08-13 | 1,418 | 2 bp |
| CAD | DEXCAUS | IR3TIB01CAM156N | 1971-01-15 | 2,909 | 2 bp |
| AUD | DEXUSAL | IR3TIB01AUM156N | 1971-01-15 | 2,909 | 2 bp |
| EUR | DEXUSEU | IR3TIB01EZM156N | 1999-01-15 | 1,448 | 2 bp |
| NZD | DEXUSNZ | IR3TIB01NZM156N | 1974-01-11 | 2,753 | 2 bp |
| ZAR | DEXSFUS | IR3TIB01ZAM156N | 1981-01-09 | 2,387 | 10 bp |
| MXN | DEXMXUS | IR3TIB01MXM156N | 1997-02-14 | 1,547 | 10 bp |
| KRW | DEXKOUS | IR3TIB01KRM156N | 1991-02-08 | 1,861 | 10 bp |
| CNY | DEXCHUS | IR3TIB01CNM156N | 1997-07-11 | 1,526 | 10 bp |
| BRL | DEXBZUS | IRSTCI01BRM156N | 1996-11-08 | 1,561 | 10 bp |
| INR | DEXINUS | IRSTCI01INM156N | 1973-01-12 | 2,804 | 10 bp |

The last week is 2026-10-09 for the first seven and 2026-10-02 for the rest.

## Accounting

As in the synthetic test, except for cash and costs: weeks 1 to 52 of each currency are warm-up
at exposure 0, the no-trade band is 0.05, and a breach is wealth more than 25% below its peak after
warm-up. Cash earns the DTB3 series above. Costs are the per-unit values in the table.

## Policies

- The parity network, registered constants, fed the deposit return and that week's cash return.
- The six simple rules of the synthetic test (buy-and-hold, trend gate, volatility target, CPPI,
  Grossman-Zhou, log-optimal), with the same definitions, run on the deposit return.
- The trend gate under the network's governor.
- Diagnostics: direction off, gate off, governor off, default exposure 1.

## Statistics

Every comparison is a pooled, annualised mean of weekly differences in net log growth over all
currency-weeks after warm-up, each currency-week weighted equally. Its 95% interval comes from a
circular block bootstrap on the shared calendar of weeks: blocks of 52 weeks, 10,000 draws, seed
20261016. Every currency keeps the same resampled weeks, so their correlation through the dollar
is kept.

A simple rule is feasible when it breaches on no currency. The best simple rule is the feasible
rule with the highest pooled growth, chosen after the fact.

## Claims

**RD1, beats the best simple rule.** The network's pooled difference against the best feasible
simple rule is above zero, with its 95% interval above zero. The network breaches on no currency.

**RD2, direction adds value.** The network's pooled difference against the same network with
direction off has its 95% interval above zero.

**RD3, no leaks.** On every currency, flipping the sign of each week after a cut point, with
probability one half, leaves every target up to the cut unchanged. The cuts are weeks 60, 200,
400, 600, 800 and 1,000, where the series is long enough.

Reported with no pass rule: every pooled comparison; per-currency growth, CAGR, maximum drawdown,
Sharpe ratio, exposure, turnover and ledger deployment; and the network and simple rules at twice
the costs.

## Decision rules

- The network is not changed in response to these results. A changed network is a new
  registration, and this currency panel will then have been seen.
- If RD3 fails, the results are void.
- The diagnostics are not candidates. A diagnostic that beats the network is reported, not claimed.

## Trial count

One candidate, already counted in the synthetic registration. Its only real-data run is this one.
