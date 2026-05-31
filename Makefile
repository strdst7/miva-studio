.PHONY: setup install test test-unit test-integration coverage lint format clean evaluate regression serve enroll

PYTHON   := python3
PIP      := pip
PYTEST   := pytest
VENV     := .venv
ACTIVATE := source $(VENV)/bin/activate

# ── Environment ──────────────────────────────────────────────────────────────

setup:
	@echo "→ Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "→ Installing dependencies..."
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	@echo "→ Copying .env template..."
	@[ -f .env ] || cp .env.example .env
	@echo "✓ Setup complete. Run: source $(VENV)/bin/activate"

install:
	$(PIP) install -e ".[dev]"

install-gpu:
	$(PIP) install -e ".[dev,gpu]"

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	PYTHONPATH=. $(PYTEST) tests/ -v -m "not integration"

test-unit:
	PYTHONPATH=. $(PYTEST) tests/unit/ -v

test-integration:
	PYTHONPATH=. $(PYTEST) tests/integration/ -v -m integration

test-all:
	PYTHONPATH=. $(PYTEST) tests/ -v

coverage:
	$(PYTEST) tests/unit/ --cov=miva --cov-report=term-missing --cov-report=html
	@echo "→ HTML report: htmlcov/index.html"

# ── Code Quality ──────────────────────────────────────────────────────────────

lint:
	ruff check miva/ tests/
	mypy miva/ --ignore-missing-imports

format:
	black miva/ tests/
	isort miva/ tests/

check: lint test
	@echo "✓ All checks passed"

# ── Evaluation ────────────────────────────────────────────────────────────────

evaluate:
	@echo "→ Running evaluation against miva_eval_v1..."
	$(PYTHON) -m miva.evaluation.runner \
		--dataset data/eval/eval_dataset_v1.yaml \
		--pipeline-version $$($(PYTHON) -c "import miva; print(miva.__version__)") \
		--output eval_outputs/ \
		--seed 42

regression:
	@echo "→ Running regression detection..."
	$(PYTHON) -m miva.evaluation.regression_detector \
		--baseline eval_outputs/baseline_report.json \
		--current  eval_outputs/latest_report.json

# ── Operations ────────────────────────────────────────────────────────────────

serve:
	uvicorn miva.api.server:app --host 0.0.0.0 --port 8000 --reload

enroll:
	@[ -n "$(SUBJECT)" ] || (echo "Usage: make enroll SUBJECT=id IMAGES=./path/" && exit 1)
	@[ -n "$(IMAGES)"  ] || (echo "Usage: make enroll SUBJECT=id IMAGES=./path/" && exit 1)
	$(PYTHON) -m miva.cli.enroll --subject-id $(SUBJECT) --images-dir $(IMAGES)

# ── Housekeeping ─────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	@echo "✓ Cleaned"

reset: clean
	rm -rf $(VENV) .env
	@echo "✓ Full reset — run make setup to start fresh"
