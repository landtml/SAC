# Contributing to SoftEdge

This page covers setup, the checks a change must pass, and what a result needs before it is
reported.

## Setup

```bash
uv venv .venv -p 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[dev]"
```

The CPU index installs PyTorch without CUDA. `make install` runs these steps.

## Checks

| Command | What it runs |
| --- | --- |
| `make test` | Offline tests. Tests that download data are deselected. |
| `make test-network` | Tests marked `network`, which download market data. |
| `make lint` | `ruff check` with the rules in `pyproject.toml`. |

CI runs the lint and the offline tests on every push and pull request. Run `make lint` and
`make test` before you open one.

## Ground rules

- Every reported result is out of sample. Checkpoints, the no-trade band and the deploy gate are
  chosen on the validation window. Each test window is used once.
- Every SAC variant reported on a universe counts as a trial: each seed, the ensemble, and each
  deployment variant. Trials are appended to `trials.jsonl` in the output root, and the deflated
  Sharpe ratio uses their count. A variant that did poorly still counts. Report it.
- Tests are deterministic (seeded). A test that needs the internet carries `@pytest.mark.network`.

## Adding a market

1. Write a loader in `src/softedge/data/` that returns a bar frame. The contract is in
   `src/softedge/data/schema.py`: a UTC index, `open`, `high`, `low`, `close` and `volume`, plus
   `funding_rate` for perpetual futures, `rate_diff` for forex carry, and
   `df.attrs["resolution"]` for binary markets.
2. Add a branch to `load_frames` in `src/softedge/pipeline.py` that maps the universe name to
   the loader and its symbols.
3. To make the data available to `softedge download`, add the universe name to the `choices` of
   the `download` parser in `src/softedge/cli.py`.
4. Add a config in `configs/` with `universe:` set to the new name, a `walk_forward` or split
   block, and the seeds.
5. Add a test in `tests/data/` that checks the frame's index, dtypes and kind-specific columns.

## Adding an agent variant

1. Add a field to `SACConfig` in `src/softedge/agent/sac.py`. Its default must leave the
   current behavior unchanged.
2. Copy `configs/ablation/base.yaml`, set one key in its `sac:` block, and give the copy a new
   `name:`. Change one thing per variant, or the comparison does not isolate it.
3. Run both configs with `softedge run`, then compare them:

   ```bash
   softedge compare abl_base <your_name>
   ```

4. In the pull request, report the paired Sharpe test and the trial count it was computed with.

## Pull requests

Fill in the pull request template. Commit subjects are imperative and under 72 characters.

## Conduct and license

Participation follows the [Code of Conduct](CODE_OF_CONDUCT.md). Contributions are released
under the MIT License in [LICENSE](LICENSE).
