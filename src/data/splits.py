"""Match-level train/validation/test splits for Frame2Threat.

Splitting is always done at the **match** level so that every event from the
same match belongs to exactly one split.  This prevents same-possession
leakage and frame-context duplication across folds.

Design principles
-----------------
* Group all events by ``match_id`` first.
* Randomly shuffle *matches* (not events) using a fixed seed.
* Assign matches to train / val / test according to the configured fractions.
* Save a manifest CSV so the exact split is reproducible and auditable.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TRAIN = 0.70
_DEFAULT_VAL = 0.15
_DEFAULT_TEST = 0.15
_DEFAULT_SEED = 42


def create_match_level_splits(
    pass_instances_df: pd.DataFrame,
    train_frac: float = _DEFAULT_TRAIN,
    val_frac: float = _DEFAULT_VAL,
    test_frac: float = _DEFAULT_TEST,
    seed: int = _DEFAULT_SEED,
    manifest_path: Optional[Path | str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *pass_instances_df* into train / val / test at the match level.

    Parameters
    ----------
    pass_instances_df:
        Canonical pass instances table.  Must contain a ``match_id`` column.
    train_frac, val_frac, test_frac:
        Target proportions.  Must sum to ~1.0.
    seed:
        Random seed for reproducibility.
    manifest_path:
        Optional path to save the split manifest CSV.  When given, a file is
        written with columns ``[match_id, split]``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) — subsets of *pass_instances_df*.

    Raises
    ------
    ValueError
        If ``match_id`` column is absent or fractions do not sum to 1.
    """
    if "match_id" not in pass_instances_df.columns:
        raise ValueError("pass_instances_df must contain a 'match_id' column")

    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0, got {total:.4f}")

    match_ids = np.array(sorted(pass_instances_df["match_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(match_ids)

    n = len(match_ids)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    # test gets the remainder to ensure every match is assigned
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n - n_train - n_test)

    train_ids = set(match_ids[:n_train])
    val_ids = set(match_ids[n_train : n_train + n_val])
    test_ids = set(match_ids[n_train + n_val :])

    logger.info(
        "Split: %d train matches, %d val matches, %d test matches",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )

    df = pass_instances_df
    train_df = df[df["match_id"].isin(train_ids)].copy()
    val_df = df[df["match_id"].isin(val_ids)].copy()
    test_df = df[df["match_id"].isin(test_ids)].copy()

    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            [(m, "train") for m in sorted(train_ids)]
            + [(m, "val") for m in sorted(val_ids)]
            + [(m, "test") for m in sorted(test_ids)]
        )
        manifest = pd.DataFrame(rows, columns=["match_id", "split"])
        manifest.to_csv(manifest_path, index=False)
        logger.info("Split manifest saved to %s", manifest_path)

    return train_df, val_df, test_df


def load_split_manifest(manifest_path: Path | str) -> pd.DataFrame:
    """Load a previously saved split manifest.

    Parameters
    ----------
    manifest_path:
        Path to the manifest CSV created by :func:`create_match_level_splits`.

    Returns
    -------
    pd.DataFrame
        Columns: ``[match_id, split]``.
    """
    return pd.read_csv(manifest_path)


def apply_manifest_splits(
    pass_instances_df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply a pre-computed manifest to a DataFrame.

    Useful for re-loading splits after the processed data has been regenerated.

    Parameters
    ----------
    pass_instances_df:
        Canonical pass instances table with ``match_id`` column.
    manifest:
        DataFrame with columns ``[match_id, split]``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    merged = pass_instances_df.merge(manifest, on="match_id", how="left")
    train_df = merged[merged["split"] == "train"].drop(columns=["split"])
    val_df = merged[merged["split"] == "val"].drop(columns=["split"])
    test_df = merged[merged["split"] == "test"].drop(columns=["split"])
    return train_df, val_df, test_df


def materialise_split_parquets(
    pass_instances_df: pd.DataFrame,
    output_dir: Path | str = Path("data/processed"),
    manifest_path: Optional[Path | str] = None,
    train_frac: float = _DEFAULT_TRAIN,
    val_frac: float = _DEFAULT_VAL,
    test_frac: float = _DEFAULT_TEST,
    seed: int = _DEFAULT_SEED,
) -> dict[str, Path]:
    """Write train / val / test parquet files from match-level splits.

    If ``manifest_path`` points to an existing manifest, the persisted splits
    are derived from that manifest. Otherwise a new manifest is created using
    :func:`create_match_level_splits`.

    Parameters
    ----------
    pass_instances_df:
        Canonical pass instances table with ``match_id`` column.
    output_dir:
        Directory where ``train.parquet``, ``val.parquet`` and ``test.parquet``
        should be written.
    manifest_path:
        Existing or target split manifest path. Defaults to
        ``output_dir / "split_manifest.csv"``.
    train_frac, val_frac, test_frac, seed:
        Split parameters used only when a new manifest is created.

    Returns
    -------
    dict[str, Path]
        Paths keyed by ``train``, ``val``, ``test`` and ``manifest``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path) if manifest_path else output_dir / "split_manifest.csv"

    if manifest_path.exists():
        manifest = load_split_manifest(manifest_path)
        train_df, val_df, test_df = apply_manifest_splits(pass_instances_df, manifest)
        logger.info("Applied existing split manifest from %s", manifest_path)
    else:
        train_df, val_df, test_df = create_match_level_splits(
            pass_instances_df,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            manifest_path=manifest_path,
        )

    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    split_paths = {split_name: output_dir / f"{split_name}.parquet" for split_name in split_frames}
    for split_name, split_df in split_frames.items():
        split_df.to_parquet(split_paths[split_name], index=False)
        logger.info("Wrote %s split parquet to %s", split_name, split_paths[split_name])

    return {**split_paths, "manifest": manifest_path}


def split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a summary DataFrame describing the three splits.

    Parameters
    ----------
    train_df, val_df, test_df:
        Split DataFrames as returned by :func:`create_match_level_splits`.

    Returns
    -------
    pd.DataFrame
        Rows: train / val / test.
        Columns: n_matches, n_passes, pct_passes.
    """
    total_passes = len(train_df) + len(val_df) + len(test_df)
    rows = []
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_matches = df["match_id"].nunique() if "match_id" in df.columns else 0
        n_passes = len(df)
        pct = 100.0 * n_passes / total_passes if total_passes > 0 else 0.0
        rows.append(
            {"split": name, "n_matches": n_matches, "n_passes": n_passes, "pct_passes": pct}
        )
    return pd.DataFrame(rows).set_index("split")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialise match-level train/val/test parquet split files."
    )
    parser.add_argument(
        "--input",
        default="data/processed/pass_instances.parquet",
        help="Path to the canonical pass_instances parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for train.parquet, val.parquet and test.parquet.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Existing or target split manifest CSV. Defaults to output-dir/split_manifest.csv.",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--train-frac", type=float, default=_DEFAULT_TRAIN)
    parser.add_argument("--val-frac", type=float, default=_DEFAULT_VAL)
    parser.add_argument("--test-frac", type=float, default=_DEFAULT_TEST)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    pass_instances = pd.read_parquet(args.input)
    paths = materialise_split_parquets(
        pass_instances,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    for split_name, path in paths.items():
        logger.info("%s: %s", split_name, path)


if __name__ == "__main__":
    main()
