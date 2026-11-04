.DEFAULT_GOAL := help
.PHONY: help install dashboard quickstart test lint format typecheck check verify demo \
        pipeline-smoke store-smoke

help:  ## Show this list
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install the package with dev extras (install CPU torch first, see README)
	python -m pip install -e ".[dev]"

dashboard:  ## Launch the Streamlit dashboard (opens on the Possession Inspector)
	python -m streamlit run src/app/app.py

quickstart: verify demo  ## Re-score saved models against reports, then regenerate the demo evidence

test:  ## Run the test suite
	python -m pytest -q

lint:  ## Lint (syntax errors and pyflakes)
	python -m ruff check src tests scripts

format:  ## Check formatting
	python -m black --check src tests scripts

typecheck:  ## Static type check
	python -m mypy src

check: lint format typecheck test  ## Everything CI would run, locally (hosted CI is unavailable)

verify:  ## Re-score saved models on the test split and compare with committed reports
	python scripts/verify_artefacts.py

demo:  ## Regenerate the demo possession figure, CLI transcript and JSON
	python scripts/make_demo_evidence.py

pipeline-smoke:  ## Synthetic end-to-end engineering smoke test (not football evidence)
	python scripts/run_pipeline.py --smoke --synthetic --output-dir data/repro/v1_smoke

store-smoke:  ## Synthetic data-store smoke test (not football evidence)
	python -m src.data.build_store --synthetic --output-dir data/store/smoke
