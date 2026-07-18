.DEFAULT_GOAL := help

PYTHON ?= python
RUN_ENV := PYTHONPATH=src PYTHONIOENCODING=utf-8

.PHONY: help install bench bench-real test lint clean

help:
	@echo "Available targets:"
	@echo "  help       - Show this help message (default)"
	@echo "  install    - Install the package in editable mode (pip install -e .)"
	@echo "  bench      - Run the benchmark on the offline stdlib default path"
	@echo "  bench-real - Run the benchmark with real providers (local embed, claude llm, pgvector); requires extras + API keys"
	@echo "  test       - Run the unittest suite"
	@echo "  lint       - Byte-compile all sources under src"
	@echo "  clean      - Remove artifacts and __pycache__ directories"

install:
	$(PYTHON) -m pip install -e .

bench:
	$(RUN_ENV) $(PYTHON) -m mcp_router bench run --out artifacts

bench-real:
	$(RUN_ENV) $(PYTHON) -m mcp_router bench run --out artifacts --embed local --llm claude --vector pgvector

test:
	$(RUN_ENV) $(PYTHON) -m unittest discover -s tests -v

lint:
	$(PYTHON) -m compileall src

clean:
	rm -rf artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
