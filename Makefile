PY ?= .venv/bin/python
UV ?= uv
CPU_INDEX ?= https://download.pytorch.org/whl/cpu

.PHONY: install test test-network lint figures assets

install:
	test -d .venv || $(UV) venv .venv -p 3.12
	$(UV) pip install --python $(PY) torch --index-url $(CPU_INDEX)
	$(UV) pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest -q

test-network:
	$(PY) -m pytest -q -m network

lint:
	$(PY) -m ruff check .

figures:
	$(PY) scripts/make_figures.py

assets:
	node scripts/render_assets.mjs
