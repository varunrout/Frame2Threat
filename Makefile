.PHONY: pipeline pipeline-smoke store store-smoke reproduce reproduce-smoke

pipeline:
	python scripts/run_pipeline.py --verbose

pipeline-smoke:
	python scripts/run_pipeline.py --smoke --synthetic --output-dir data/repro/v1_smoke

store:
	python -m src.data.build_store --verbose

store-smoke:
	python -m src.data.build_store --synthetic --output-dir data/store/smoke

reproduce:
	python scripts/run_pipeline.py --verbose

reproduce-smoke:
	python scripts/run_pipeline.py --smoke --synthetic --output-dir data/repro/v1_smoke
