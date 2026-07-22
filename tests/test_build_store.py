"""Tests for the parquet + SQLite data store builder."""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from src.data.build_store import build_store, read_store_pass_instances


def _args(output_dir) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(output_dir),
        max_matches=4,
        require_360=False,
        from_processed=False,
        synthetic=True,
        verbose=False,
    )


def test_build_store_writes_parquet_partitions_and_sqlite(tmp_path):
    summary = build_store(_args(tmp_path))

    sqlite_path = tmp_path / "metadata.sqlite"
    assert sqlite_path.exists()
    assert summary["n_matches"] == 4
    assert summary["n_passes"] > 0
    assert summary["n_partitions"] > 0

    partition_files = list((tmp_path / "pass_instances").glob("**/pass_instances.parquet"))
    assert partition_files
    assert (tmp_path / "processed" / "train.parquet").exists()
    assert (tmp_path / "processed" / "val.parquet").exists()
    assert (tmp_path / "processed" / "test.parquet").exists()

    with sqlite3.connect(sqlite_path) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
            conn,
        )
        assert {"inventory", "parquet_partitions", "runs"}.issubset(set(tables["name"]))
        runs = pd.read_sql_query("SELECT * FROM runs", conn)
        assert int(runs.iloc[-1]["n_passes"]) == summary["n_passes"]


def test_read_store_pass_instances_filters_partition(tmp_path):
    build_store(_args(tmp_path))

    all_passes = read_store_pass_instances(tmp_path)
    assert not all_passes.empty
    first_competition = int(all_passes["competition_id"].iloc[0])

    filtered = read_store_pass_instances(tmp_path, competition_id=first_competition)

    assert not filtered.empty
    assert set(filtered["competition_id"]) == {first_competition}
    assert len(filtered) < len(all_passes)
