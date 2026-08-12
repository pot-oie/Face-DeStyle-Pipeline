.PHONY: install install-gpu lint test check

install:
	python -m pip install -e ".[dev]"

install-gpu:
	python -m pip install -e ".[gpu,dev]"

lint:
	ruff check .

test:
	pytest

check: lint test
