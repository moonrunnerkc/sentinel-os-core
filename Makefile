# Author: Bradley R. Kinnard
# Makefile for Sentinel OS Core

.PHONY: all test verify bench clean help

PYTHON := python
PYTEST := $(PYTHON) -m pytest
VENV := .venv

# default target
all: test

# run all tests
test:
	$(PYTEST) tests/ -v --tb=short

# run tests quietly
test-quiet:
	$(PYTEST) tests/ --tb=short -q

# run formal verification tests
verify:
	@echo "=== Running Formal Verification Suite ==="
	$(PYTEST) tests/test_formal_verification.py tests/test_verification.py -v --tb=short
	@echo ""
	@echo "=== Generating Proof Log ==="
	$(PYTHON) -c "from verification.proof_log import run_formal_verification; \
		state = {'beliefs': {'b1': {'confidence': 0.5}}, 'goals': {}, 'episodes': []}; \
		passed, log = run_formal_verification(state, output_path='data/logs/proof_log.json'); \
		print(f'Verification: {\"PASSED\" if passed else \"FAILED\"}')"
	@echo ""
	@echo "=== Verification Complete ==="

# run benchmarks
bench:
	@echo "=== Running Benchmark Suite ==="
	$(PYTHON) -c "import logging; logging.disable(logging.INFO); \
		from benchmarks.e2e_benchmark import E2EBenchmarkSuite; \
		suite = E2EBenchmarkSuite(seed=42); \
		suite.run_standard_suite(); \
		print(suite.generate_report())"

# run crypto benchmarks
bench-crypto:
	$(PYTHON) -m benchmarks.crypto_benchmark

# run property tests with more examples
property-test:
	$(PYTEST) tests/test_formal_verification.py::TestPropertyBasedVerification -v \
		--hypothesis-show-statistics

# run demo
demo:
	$(PYTHON) demo.py

# clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/

# create virtual environment
venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

# install dependencies
install:
	pip install -r requirements.txt

# run linting
lint:
	$(PYTHON) -m flake8 core/ memory/ security/ verification/ --max-line-length=120

# type checking
typecheck:
	$(PYTHON) -m mypy core/ memory/ security/ verification/ --ignore-missing-imports

# full verification pipeline
verify-full: test verify bench
	@echo ""
	@echo "=== Full Verification Pipeline Complete ==="

# docker verification
docker-verify:
	docker-compose run --rm sentinel-os $(MAKE) verify-full

# help
help:
	@echo "Sentinel OS Core - Makefile Targets"
	@echo ""
	@echo "  make test        - Run all tests"
	@echo "  make test-quiet  - Run tests quietly"
	@echo "  make verify      - Run formal verification suite"
	@echo "  make bench       - Run benchmarks"
	@echo "  make bench-crypto - Run crypto benchmarks"
	@echo "  make property-test - Run property-based tests"
	@echo "  make demo        - Run demo"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make venv        - Create virtual environment"
	@echo "  make install     - Install dependencies"
	@echo "  make lint        - Run linting"
	@echo "  make typecheck   - Run type checking"
	@echo "  make verify-full - Full verification pipeline"
	@echo "  make docker-verify - Run verification in Docker"
	@echo "  make help        - Show this help"
