"""Tests for the single-command v1 reproduction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.reproduce_v1 import reproduce


def _args(output_dir) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(output_dir),
        smoke=True,
        use_processed=False,
        synthetic=True,
        max_matches=6,
        smoke_rows=360,
        require_360=False,
        verbose=False,
    )


def test_reproduce_smoke_writes_scores_summary_model_and_splits(tmp_path):
    summary = reproduce(_args(tmp_path))

    expected_outputs = {
        "pass_instances",
        "split_manifest",
        "train_parquet",
        "val_parquet",
        "test_parquet",
        "model",
        "scored_passes",
        "summary",
    }
    assert expected_outputs.issubset(summary["outputs"])
    for output_name in expected_outputs:
        assert Path(summary["outputs"][output_name]).exists()

    scored = pd.read_csv(tmp_path / "v1_event_only_scored_passes.csv")
    assert not scored.empty
    assert scored["event_only_prob"].between(0, 1).all()
    assert summary["metrics"]["roc_auc"] >= 0
    assert summary["n_train"] > 0
    assert summary["n_val"] > 0
    assert summary["n_test"] > 0
