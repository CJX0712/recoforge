# RecoForge Makefile —— 本地等价 CI 流程
PY ?= python
VENV ?= .venv

.PHONY: install venv test lint demo ci lock clean

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -r requirements-sota.txt || echo "implicit 安装失败，已降级（不影响 Tier-1）"

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/Scripts/python -m pip install -r requirements.txt
	$(VENV)/Scripts/python -m pip install -r requirements-sota.txt || true

lint:
	$(PY) -m ruff check recoforge tests
	$(PY) -m ruff format --check recoforge tests

test:
	$(PY) -m pytest tests/ -q -W ignore::UserWarning -W ignore::RuntimeWarning

demo:
	$(PY) -m recoforge.examples.run_demo --seed 42 --out benchmark.json

ci: lint test demo

lock:
	$(PY) -m pip freeze > requirements.lock.txt

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache $(VENV)
