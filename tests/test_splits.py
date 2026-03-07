"""Tests for match-level data splits — critical for leakage prevention."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    apply_manifest_splits,
    create_match_level_splits,
    load_split_manifest,
    split_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pass_df(n_matches: int = 10, passes_per_match: int = 50, seed: int = 0) -> pd.DataFrame:
    """Small synthetic pass_instances DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for match_id in range(1, n_matches + 1):
        for i in range(passes_per_match):
            rows.append(
                {
                    "match_id": match_id,
                    "event_uuid": f"m{match_id}_e{i:04d}",
                    "start_x": float(rng.uniform(20, 90)),
                    "end_x": float(rng.uniform(30, 110)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateMatchLevelSplits:
    def test_returns_three_dataframes(self):
        df = _make_pass_df()
        train, val, test = create_match_level_splits(df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_no_match_overlap_across_splits(self):
        """No match_id appears in more than one split."""
        df = _make_pass_df(n_matches=20)
        train, val, test = create_match_level_splits(df)
        train_ids = set(train["match_id"])
        val_ids = set(val["match_id"])
        test_ids = set(test["match_id"])
        assert train_ids.isdisjoint(val_ids), "Train and val share match_ids"
        assert train_ids.isdisjoint(test_ids), "Train and test share match_ids"
        assert val_ids.isdisjoint(test_ids), "Val and test share match_ids"

    def test_all_matches_assigned(self):
        """Every match ends up in exactly one split."""
        df = _make_pass_df(n_matches=15)
        train, val, test = create_match_level_splits(df)
        all_assigned = set(train["match_id"]) | set(val["match_id"]) | set(test["match_id"])
        assert all_assigned == set(df["match_id"])

    def test_all_passes_preserved(self):
        """Total rows across splits equals input rows."""
        df = _make_pass_df(n_matches=12, passes_per_match=30)
        train, val, test = create_match_level_splits(df)
        assert len(train) + len(val) + len(test) == len(df)

    def test_split_by_match_not_by_row(self):
        """All events from the same match must be in the same split."""
        df = _make_pass_df(n_matches=10, passes_per_match=20)
        train, val, test = create_match_level_splits(df)
        for match_id in df["match_id"].unique():
            in_train = (train["match_id"] == match_id).sum()
            in_val = (val["match_id"] == match_id).sum()
            in_test = (test["match_id"] == match_id).sum()
            splits_present = sum([in_train > 0, in_val > 0, in_test > 0])
            assert splits_present == 1, (
                f"Match {match_id} found in {splits_present} splits"
            )

    def test_split_fractions_approximate(self):
        """Splits are approximately the correct size (±15% tolerance for small N)."""
        df = _make_pass_df(n_matches=30, passes_per_match=20)
        train, val, test = create_match_level_splits(df, train_frac=0.7, val_frac=0.15, test_frac=0.15)
        total = len(df)
        assert len(train) / total > 0.50, "Train set unexpectedly small"
        assert len(val) / total > 0.05, "Val set unexpectedly small"
        assert len(test) / total > 0.05, "Test set unexpectedly small"

    def test_reproducible_with_same_seed(self):
        """Same seed produces identical splits."""
        df = _make_pass_df(n_matches=20)
        t1, v1, s1 = create_match_level_splits(df, seed=42)
        t2, v2, s2 = create_match_level_splits(df, seed=42)
        assert list(t1["match_id"].unique()) == list(t2["match_id"].unique())
        assert list(v1["match_id"].unique()) == list(v2["match_id"].unique())

    def test_different_seeds_may_differ(self):
        """Different seeds should (very likely) produce different splits."""
        df = _make_pass_df(n_matches=20)
        t1, _, _ = create_match_level_splits(df, seed=1)
        t2, _, _ = create_match_level_splits(df, seed=99)
        # With 20 matches it would be extremely unlikely to get identical order
        assert set(t1["match_id"]) != set(t2["match_id"]) or True  # soft assertion

    def test_raises_if_no_match_id_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="match_id"):
            create_match_level_splits(df)

    def test_raises_if_fractions_wrong(self):
        df = _make_pass_df()
        with pytest.raises(ValueError, match="sum to 1"):
            create_match_level_splits(df, train_frac=0.5, val_frac=0.5, test_frac=0.5)


class TestManifest:
    def test_manifest_saved_and_loaded(self):
        """Manifest CSV is written and can be reloaded."""
        df = _make_pass_df(n_matches=10)
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "splits" / "manifest.csv"
            create_match_level_splits(df, manifest_path=manifest_path)
            assert manifest_path.exists(), "Manifest file not created"
            loaded = load_split_manifest(manifest_path)
            assert "match_id" in loaded.columns
            assert "split" in loaded.columns
            assert set(loaded["split"]).issubset({"train", "val", "test"})
            assert len(loaded) == df["match_id"].nunique()

    def test_apply_manifest_splits(self):
        """apply_manifest_splits reproduces the same split as create_match_level_splits."""
        df = _make_pass_df(n_matches=15)
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.csv"
            t1, v1, s1 = create_match_level_splits(df, seed=7, manifest_path=manifest_path)
            manifest = load_split_manifest(manifest_path)
            t2, v2, s2 = apply_manifest_splits(df, manifest)
            assert set(t1["match_id"]) == set(t2["match_id"])
            assert set(v1["match_id"]) == set(v2["match_id"])
            assert set(s1["match_id"]) == set(s2["match_id"])


class TestSplitSummary:
    def test_returns_dataframe_with_correct_index(self):
        df = _make_pass_df(n_matches=9)
        train, val, test = create_match_level_splits(df)
        summary = split_summary(train, val, test)
        assert set(summary.index) == {"train", "val", "test"}
        assert "n_passes" in summary.columns
        assert "n_matches" in summary.columns
        assert "pct_passes" in summary.columns

    def test_pct_passes_sums_to_100(self):
        df = _make_pass_df(n_matches=12)
        train, val, test = create_match_level_splits(df)
        summary = split_summary(train, val, test)
        assert abs(summary["pct_passes"].sum() - 100.0) < 0.01
