# sactrader

Distributional, risk-aware Soft Actor-Critic for trading four market types with one code path:
crypto spot, perpetual futures, forex and binary prediction markets.

- **One accounting model.** Every instrument becomes a tape of long-side and short-side returns,
  carry and costs. Perp funding, forex carry, short borrow, NO-token pricing, binary settlement and
  margin liquidation all go through `portfolio.step_portfolio`.
- **Leakage-tested features.** Scale-free, causal features; tests fail on any look-ahead.
- **Agent.** Truncated quantile critics (TQC) on SimBa residual encoders, a mean or CVaR actor
  objective, automatic entropy tuning, and log-growth reward net of all costs.
- **Counterfactual replay.** Prices do not depend on a small trader's actions, so replayed
  transitions get fresh starting positions and actions with exactly recomputed rewards.
- **Honest evaluation.** Checkpoints, no-trade band and a deploy-or-stay-flat gate are chosen on
  validation; each test window is used once. Walk-forward folds or combinatorial purged
  cross-validation (purge and embargo) produce out-of-sample records; reports give the deflated
  Sharpe ratio with every SAC variant ever tried counted as a trial, minimum track record length,
  expected shortfall, and bootstrap intervals on Sharpe differences against baselines.

See [docs/design.md](docs/design.md) for the reasoning behind each choice and its limitations.

## Install

```bash
uv venv .venv -p 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest            # offline tests
.venv/bin/python -m pytest -m network # tests that download data
```

## Use

```bash
sactrader download crypto_perp          # Binance USD-M hourly klines and funding, 2020 on
sactrader download fx                   # FRED daily rates for 7 major pairs
sactrader download polymarket           # 300 largest resolved YES/NO markets, hourly
sactrader run configs/crypto_perp.yaml  # train, select, test; writes runs/crypto_perp/report.md
sactrader run configs/crypto_perp_wf.yaml   # walk-forward folds, one stitched out-of-sample record
sactrader dsr --runs runs               # deflated Sharpe ratios over every saved run
```

A config evaluates with one `split`, consecutive `walk_forward` folds, or a `cpcv` block for
combinatorial purged cross-validation with purge and embargo, which reports the spread of
out-of-sample Sharpe and deflated Sharpe across backtest paths.

Data is cached under `$SACTRADER_DATA` (default `./data`). A config names a universe, cost
overrides, a train / validation / test split by date, environment, agent and training settings,
and the seeds to train. `configs/` holds one per universe.

## Layout

| Path | Contents |
| --- | --- |
| `src/sactrader/data/` | Bar frame schema, download cache, Binance, FRED forex, Polymarket, synthetic markets |
| `src/sactrader/features/causal.py` | Causal, scale-free features for every market kind |
| `src/sactrader/tape.py` | Market tape and cost model |
| `src/sactrader/portfolio.py` | One-step accounting: costs, impact, carry, liquidation, drift |
| `src/sactrader/env/vec_env.py` | Vectorized environment over many tapes, full-tape rollouts |
| `src/sactrader/agent/` | SimBa networks, TQC-SAC learner, index replay with relabeling |
| `src/sactrader/train.py` | Training loop with validation checkpointing |
| `src/sactrader/eval/` | Backtest, metrics, PSR/DSR/bootstrap/PBO, baselines, walk-forward splits, reports |
| `src/sactrader/experiment.py` | Config-driven experiments, walk-forward folds, the trial ledger |
| `src/sactrader/cpcv_experiment.py` | Combinatorial purged cross-validation of the agent |
