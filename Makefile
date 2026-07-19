.DEFAULT_GOAL := help

PYTHON ?= python
RUN_ENV := PYTHONPATH=src PYTHONIOENCODING=utf-8

.PHONY: help install bench sweep similarity bench-real test lint clean

help:
	@echo "Available targets:"
	@echo "  help       - Show this help message (default)"
	@echo "  install    - Install the package in editable mode (pip install -e .)"
	@echo "  bench      - Run the benchmark on the offline stdlib default path"
	@echo "  sweep      - core_share sensitivity sweep (cliff is not a constant artifact)"
	@echo "  similarity - measure real MCP tool pairwise similarity vs the synthetic catalog (needs .[local])"
	@echo "  bench-real - Run with real providers (bge-small embed, claude llm); requires extras + API key"
	@echo "  test       - Run the unittest suite"
	@echo "  lint       - Byte-compile all sources under src"
	@echo "  clean      - Remove artifacts and __pycache__ directories"

install:
	$(PYTHON) -m pip install -e .

bench:
	$(RUN_ENV) $(PYTHON) -m mcp_router bench run --out artifacts

sweep:
	$(RUN_ENV) $(PYTHON) -m mcp_router bench sweep --shares 6,7,8,9,10

similarity:
	$(RUN_ENV) $(PYTHON) scripts/pairwise_similarity.py

bench-real:
	$(RUN_ENV) $(PYTHON) -m mcp_router bench run --out artifacts-real --embed local --llm claude

test:
	$(RUN_ENV) $(PYTHON) -m unittest discover -s tests -v

lint:
	$(PYTHON) -m compileall src

clean:
	rm -rf artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
