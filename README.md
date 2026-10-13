# sactrader

Distributional, risk-aware Soft Actor-Critic for trading four market types with one code path:
crypto spot, perpetual futures, forex and binary prediction markets.

## Install

```bash
uv venv .venv -p 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

## Layout

| Path | Contents |
| --- | --- |
| `src/sactrader/data/schema.py` | Bar frame contract shared by every data source |
| `src/sactrader/tape.py` | Market tape: per-step returns, carry and costs for long and short sides |
| `src/sactrader/portfolio.py` | Vectorized one-step accounting: costs, impact, funding, liquidation |
