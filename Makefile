.PHONY: help install install-dev install-all test clean build publish format lint

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with dev dependencies
	pip install -e ".[dev]"

install-all:  ## Install package with all dependencies (including AllenNLP)
	pip install -e ".[all]"

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=causal_narrative --cov-report=html --cov-report=term

format:  ## Format code with black and isort
	black causal_narrative/ tests/
	isort causal_narrative/ tests/

lint:  ## Run linters
	black --check causal_narrative/ tests/
	isort --check-only causal_narrative/ tests/
	flake8 causal_narrative/ tests/ --max-line-length=100

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

check:  ## Check package with twine
	twine check dist/*

publish-test:  ## Upload to Test PyPI
	twine upload --repository testpypi dist/*

publish:  ## Upload to PyPI
	twine upload dist/*

release: clean build check  ## Build and check (but don't publish)
	@echo "Package ready for release. Run 'make publish' to upload to PyPI."

spacy-download:  ## Download spaCy model
	python -m spacy download en_core_web_sm

allennlp-download:  ## Download AllenNLP model (Python 3.9-3.10 only)
	python download_allennlp_model.py

jupyter:  ## Start Jupyter notebook
	jupyter notebook notebooks/

pre-commit-install:  ## Install pre-commit hooks
	pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	pre-commit run --all-files
