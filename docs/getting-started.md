# Getting started

This page takes you from an empty checkout to a finished run. Each run writes a report you can read.
Terms such as walk-forward and DSR are defined in the [glossary](glossary.md).

## 1. Install

You need Python 3.11 or newer. The setup below uses 3.12 with `uv`.

```bash
uv venv .venv -p 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

The tests run offline by default. Tests that download data are marked `network` and run with
`pytest -m network`.

Activate the environment with `source .venv/bin/activate`, and the `softedge` command is then on
your path. Without activation, call it as `.venv/bin/softedge`.

Downloaded market data goes to `$SOFTEDGE_DATA`, or to `./data` when that variable is unset. Run
outputs go to `runs/` in the current directory. The default `data/` and `runs/` folders are
git-ignored.

## 2. Run the synthetic sanity check

This check needs no network. It plants a return signal in six synthetic markets, and the agent
should find it.

```bash
softedge run configs/synthetic/synthetic_wf.yaml
```

The command trains, selects on validation, tests each window once, and prints a JSON summary. The
report is in `runs/synthetic_wf/report.md`.

Now run the control. It uses the same pipeline on random walks with no signal, and the agent
should find nothing.

```bash
softedge run configs/synthetic/synthetic_null_wf.yaml
```

If the control reports an edge, the check has failed, and results from this pipeline should not
be trusted until it passes.

## 3. Download a real universe

```bash
softedge download crypto_perp
```

This fills the cache for the universe. It prints one line per symbol: the symbol, its kind, its bar
count, and its first and last bar. The options are `--start` and `--end` (year-month), `--symbols`
to pick symbols, and `--max-markets` for the prediction-market universes.

`download` accepts `crypto_perp`, `crypto_spot`, `crypto_basis`, `fx`, `macro`, `polymarket` and
`kalshi`. The other universes in `configs/`, such as `hl_basis` and `perp_spread`, are fetched the
first time a run needs them. You can skip `download` and go straight to a run.

## 4. Run a walk-forward config

```bash
softedge run configs/crypto/crypto_perp_wf.yaml
```

This config trains one agent per fold and seed. Each fold trains on the past, selects its
checkpoint and no-trade band on a validation window, and tests once on the next window. The test
windows of successive folds join into one out-of-sample record. Expect a long run.

Open `runs/crypto_perp_wf/report.md`. It has four parts:

- a header with the number of folds, the out-of-sample bars, and the trials recorded so far;
- an out-of-sample summary for each strategy, including Sharpe ratio, PSR, DSR and maximum
  drawdown;
- the Sharpe ratio of each fold;
- the per-market breakdown.

The strategies to compare are `sac_banded`, the agent's banded policy, and `sac_gated`, the policy
that trades only when validation supports it. The rest are simple rules.

## 5. Compare runs

Summarize every finished run. Each row sets the agent against the best simple rule for its market,
with a paired bootstrap test.

```bash
softedge summary --runs runs
```

This prints the table and writes `runs/summary.md`.

Compare variants on the bars they share. Each name is the `name:` field of a config, and its run
folder is `runs/<name>`.

```bash
softedge compare abl_base abl_dropout
```

The first name is the base run. The command prints the Sharpe difference from the base, its
interval, and a p-value. Add `--strategy` to test a strategy other than `sac_banded`.

Recompute the deflated Sharpe ratio over every saved run:

```bash
softedge dsr --runs runs
```

This writes `runs/dsr.md`. Every run you add is also recorded in `runs/trials.jsonl`, and the trial
count in later DSRs includes it. Earlier DSRs fall as the count grows.

Two more commands rank and test runs. `softedge leaderboard --runs runs --top 10` ranks the runs by
DSR. `softedge pbo crypto_perp` prints the probability of backtest overfitting across the variants
recorded for one universe.
