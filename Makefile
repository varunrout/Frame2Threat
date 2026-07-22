.PHONY: reproduce reproduce-smoke

reproduce:
	python scripts/reproduce_v1.py --verbose

reproduce-smoke:
	python scripts/reproduce_v1.py --smoke --synthetic --output-dir data/repro/v1_smoke
