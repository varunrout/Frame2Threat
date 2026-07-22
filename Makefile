.PHONY: pipeline pipeline-smoke reproduce reproduce-smoke

pipeline:
	python scripts/run_pipeline.py --verbose

pipeline-smoke:
	python scripts/run_pipeline.py --smoke --synthetic --output-dir data/repro/v1_smoke

reproduce:
	python scripts/run_pipeline.py --verbose

reproduce-smoke:
	python scripts/run_pipeline.py --smoke --synthetic --output-dir data/repro/v1_smoke
