.PHONY: help install data pipeline test lint serve clean

PYTHON ?= python
TRANSACTIONS ?= 300000

help:
	@echo "install     install dependencies"
	@echo "data        generate the extract and build the SQL warehouse"
	@echo "pipeline    run the full pipeline end to end (figures, tables, report)"
	@echo "test        run the test suite"
	@echo "lint        run ruff"
	@echo "serve       start the scoring API on :8000"
	@echo "clean       remove generated artefacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/run_pipeline.py --transactions $(TRANSACTIONS)

pipeline: data

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src scripts tests

serve:
	$(PYTHON) scripts/serve.py

clean:
	rm -rf outputs/figures/*.png outputs/tables/*.csv outputs/metrics.json outputs/RESULTS.md
	rm -rf data/processed/*.db .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
