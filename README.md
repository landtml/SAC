<p align="center">
  <img src="assets/header.gif" alt="SoftEdge: Soft Actor-Critic agents in search of a trading edge, measured honestly. The logo is the agent's Gaussian policy over positions; it slides back to zero when there is no edge, while a price line shows the agent staying flat and then trading the out-of-sample window." width="100%">
</p>

<p align="center">
  <a href="https://landtml.github.io/SoftEdge/">Live explainer</a> ·
  <a href="docs/results.md">Results</a> ·
  <a href="docs/getting-started.md">Getting started</a> ·
  <a href="docs/design.md">Design</a> ·
  <a href="docs/glossary.md">Glossary</a> ·
  <a href="docs/roadmap.md">Roadmap</a> ·
  <a href="docs/research-log.md">Research log</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB">
  <img alt="PyTorch CPU" src="https://img.shields.io/badge/PyTorch-CPU-EE4C2C">
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status: open research" src="https://img.shields.io/badge/status-open%20research-blue">
</p>

# SoftEdge

**SoftEdge is an open research project that asks one question: can a modern reinforcement-learning
trading agent find an edge that survives honest testing?**

Many published trading-bot results are measured on the data the bot learned from, ignore fees, or
quietly try hundreds of variants and report the best one. SoftEdge builds the agent carefully and
then tries hard to prove it wrong: every number here comes from data the agent never saw, after
trading costs, and is discounted for every variant we tried.

The agent is a **Soft Actor-Critic (SAC)**, a widely used deep reinforcement-learning method,
updated with ideas from 2024-2026 research. It trades crypto, crypto carry trades, forex, stock
indices, oil, Treasuries and prediction markets, all from free public data.

## What we found so far

Short version: **no free lunch in plain price prediction; a small, real gain on top of known carry
premia.**

| Market | Out-of-sample result | Verdict |
| --- | --- | --- |
| Synthetic market with a hidden signal (control) | Sharpe 15.1, signal found | the method works when an edge exists |
| Random walk (control) | Sharpe -1.5, deflated Sharpe 0.004 | correctly rejected: no false discovery |
| Synthetic market with hidden regimes and fat tails | Sharpe 3.8 (a perfect-foresight sign oracle reaches 8.6) | beats buy-and-hold, but deflated Sharpe is 0.14 after 70 trials |
| Forex, 7 major pairs, 2014-2026 | Sharpe 0.35, deflated Sharpe 0.37 | no edge |
| Crypto perpetual futures, 10 coins, price direction, 2023-2026 | Sharpe 0.41 vs 0.64 for a trend rule | no edge |
| Macro: S&P 500, Nasdaq 100, Nikkei 225, WTI, Brent, 10y and 2y Treasuries, daily | Sharpe 0.07 vs 0.55 for buy-and-hold | no edge |
| Crypto cash-and-carry, 10 coins, 2023-2026 | +0.49 Sharpe over its simple funding rule (95% CI 0.15 to 0.95); positive on all 5 CPCV paths | small, significant gain; shrinks to +0.37 on 32 coins |
| Cross-exchange funding spread, 10 coins, 2024-2026 | +0.47 Sharpe over its simple rule (95% CI 0.14 to 0.85) | small, significant gain; holds on 32 coins (+0.66) |

<p align="center">
  <img src="assets/png/figures/sac-vs-rule.png" alt="SAC minus its simple rule, with 95% intervals, per market" width="90%">
</p>

The carry results are the interesting part. In a carry trade you earn a steady payment for
holding a position; here it is the "funding" that crypto perpetual futures pay. A simple rule,
"hold the carry when funding is positive", already captures most of it. SoftEdge lets the agent
adjust that rule, and the adjustments improve the risk-adjusted return by a statistically
significant margin in two of three carry markets. Absolute Sharpe ratios in carry markets are
inflated because the simulation has no exchange or counterparty risk, so only the comparisons
count. Full tables, caveats and method: [docs/results.md](docs/results.md).

## How it works

<p align="center">
  <img src="assets/animations/sac-loop.gif" alt="Animated loop of a Soft Actor-Critic agent: market state, the actor picks a squashed Gaussian position, the reward is log growth after costs, counterfactual replay stores the moment with other positions, two distributional critics drop their most optimistic guesses, and an entropy thermostat tunes exploration" width="100%">
  <br><em>One learning step of the agent. Want to play with the ideas? Open the <a href="https://landtml.github.io/SoftEdge/">live explainer</a>: soft policies, squashing, twin critics and a toy market, running in your browser.</em>
</p>

<p align="center">
  <img src="assets/png/diagrams/how-it-works.png" alt="Pipeline: public data, causal features, market tape, SAC agent, walk-forward test, honest scorecard" width="100%">
</p>

1. **Public data only.** Binance archives, Hyperliquid, Deribit, FRED, Polymarket and Kalshi. No
   API keys.
2. **No peeking.** Features at each bar use only earlier bars; a test fails the build if any
   feature changes when future prices change.
3. **One accounting model.** Every market becomes a "tape" of returns, trading costs and carry
   (funding, interest-rate differentials, borrow fees, binary settlement, margin liquidation).
4. **The agent** chooses a target position each bar and is rewarded with the growth of its
   equity after all costs.
5. **Walk-forward testing.** Train on the past, choose settings on a validation window, test
   once on the next window, roll forward, and stitch the test windows into one record.
6. **An honest scorecard.** The deflated Sharpe ratio discounts luck from trying many variants;
   paired bootstrap tests compare the agent with simple rules on the same days; combinatorial
   purged cross-validation gives a spread of outcomes instead of one lucky path.

<p align="center">
  <img src="assets/png/diagrams/walk-forward.png" alt="Walk-forward folds: train, validation, embargo, test, stitched into one record" width="100%">
</p>

New to these terms? The [glossary](docs/glossary.md) explains each in a sentence or two.

## Is the agent modern?

Yes. It combines several recent improvements, and each one is tested on its own before it is
kept:

- **Distributional critics** (truncated quantile critics) learn the range of possible outcomes,
  not only the average, which curbs overconfident value estimates.
- **Modern network blocks** (SimBa residual encoders, optional hyperspherical SimbaV2) stay stable
  as networks grow.
- **Counterfactual replay.** A small trader does not move prices, so every stored market moment
  can be replayed with any position and action, with exactly recomputed rewards. This multiplies
  the useful training data.
- **Residual policy on a prior.** The agent can start from a known premium (carry, funding,
  trend) and learn only an adjustment, sized to a volatility target. This is the one idea that
  beat simple rules on real data.
- **Safety rails:** automatic exploration tuning, a value-divergence guard, plasticity resets,
  averaged deployment weights, and a validation gate that keeps the agent flat when the evidence
  is weak.

<p align="center">
  <img src="assets/png/diagrams/residual-policy.png" alt="Target position equals a volatility scale times prior plus the agent's residual" width="85%">
</p>

Not everything helped. On our noisy test market, three popular add-ons made things worse and
were rejected: categorical (HL-Gauss) critics, a tail-risk (CVaR) objective and a pathwise
gradient objective. Dropout, input noise and volatility targeting are still inconclusive.

<p align="center">
  <img src="assets/png/figures/ablations.png" alt="Each add-on compared with the base agent, with 95% intervals" width="90%">
</p>

## Quick start

```bash
git clone https://github.com/landtml/SoftEdge.git && cd SoftEdge
uv venv .venv -p 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/softedge run configs/synthetic/synthetic_wf.yaml   # sanity check on synthetic data, no downloads
.venv/bin/softedge summary --runs runs             # the agent against the best simple rule
```

Then a real market (public data is downloaded on first use):

```bash
.venv/bin/softedge run configs/carry/crypto_basis_1d_prior_v2.yaml
.venv/bin/softedge compare crypto_basis_1d_prior_v2 crypto_basis_1d_prior_prem
.venv/bin/softedge dsr --runs runs                 # deflated Sharpe ratios over every run
```

Step-by-step guide: [docs/getting-started.md](docs/getting-started.md). Every experiment is a YAML
file; the catalogue is in [configs/README.md](configs/README.md).

## Markets

| Family | Universes | Data |
| --- | --- | --- |
| Crypto | `crypto_spot`, `crypto_perp` | Binance public archives, hourly |
| Crypto carry | `crypto_basis`, `hl_basis`, `perp_spread` | Binance and Hyperliquid, daily |
| Forex | `fx` (7 majors), `fx_em` (6 emerging markets) | FRED exchange rates and short rates |
| Macro | `macro`: S&P 500, Nasdaq, Nikkei, WTI, Brent, 10y and 2y Treasuries | FRED |
| Prediction markets | `polymarket`, `kalshi` | public APIs, hourly |
| Controls | `synthetic`, `synthetic_null`, `synthetic_hard` | generated |

One agent can also train on several universes at once (`sources:` in a config).

## Repository map

| Path | What is inside |
| --- | --- |
| `src/softedge/data/` | data loaders and the common bar format |
| `src/softedge/features/` | look-ahead-free features, cross-asset and regime indicators |
| `src/softedge/tape.py`, `portfolio.py` | the market tape and the accounting |
| `src/softedge/env/` | vectorized trading environment |
| `src/softedge/agent/` | SAC, critics, networks, replay, plasticity tools |
| `src/softedge/train.py`, `experiment.py`, `cpcv_experiment.py` | training and evaluation protocols |
| `src/softedge/eval/` | metrics, deflated Sharpe, bootstrap and overfitting tests, reports |
| `configs/` | one YAML file per experiment |
| `docs/` | results, design, glossary, roadmap |
| `scripts/make_figures.py` | regenerates the charts in `assets/figures/` |

## Contributing and citing

SoftEdge is open research: new markets, agent ideas and, above all, attempts to break our results
are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). One rule matters most: every reported result
is out of sample, and every variant you try counts as a trial.

If you use SoftEdge, please cite it with [CITATION.cff](CITATION.cff).

**Research code, not financial advice.** Nothing here has been traded with real money.

Released under the [MIT License](LICENSE).
