# Roadmap

This page lists the research questions the project has not yet answered. Each one names the part of
the project that would answer it. None of them is a promise. Live paper trading is not planned. The
[limits](results.md#limits) section lists what the model leaves out.

## Do the carry gains persist with more seeds?

The confirmation configs in [`configs/ablation/confirm/`](../configs/README.md#confirm) already train three
seeds on twelve synthetic symbols, for the base agent and two ablations. The same check for the
carry configs would show whether their gains persist.

## Does the carry result hold on wider universes?

The two crypto carry rows in the results use ten coins each. A wider universe would test whether
the gain depends on those coins. The [carry configs](../configs/README.md#carry) give the starting
point.

## Does the carry result hold across more paths?

Walk-forward testing gives one out-of-sample path. The
[CPCV config](../configs/README.md#carry) for crypto cash-and-carry produces many paths from the
same data. Does the gain over the funding rule stay positive across them?

## Does one agent trained on several markets beat single-market agents?

The [pooled configs](../configs/README.md#pooled) train one agent on crypto perpetuals, forex and
macro markets together. The single-market results on the [results page](results.md) give the
comparison.

## Can prediction markets show an edge under the same tests?

Some prediction-market configs use a log-odds prior that tilts positions toward the
[favourite](glossary.md#favourite-longshot-bias). This project reports no prediction-market results
yet. The same pipeline and deflated Sharpe ratio apply to them.

## How much of the carry result survives exchange risk?

The carry model leaves out exchange counterparty risk, and it charges one flat borrow rate for every
coin. Modelling both would show how much of the gain remains after the risks that carry trades are
known for.
