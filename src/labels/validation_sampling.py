"""
src/labels/validation_sampling.py
===================================
Manual validation support for label quality assurance.

This module provides a suite of lightweight utility functions for auditing
the quality of computed labels.  The typical workflow is:

1. :func:`label_prevalence_table` – get an overview of all label rates.
2. :func:`sample_positives` / :func:`sample_negatives` – draw random samples
   for manual inspection.
3. :func:`label_by_zone` – look for spatial anomalies in a label's pitch
   distribution.
4. :func:`label_by_pass_type` – check label rates across pass types.
5. :func:`label_sanity_checks` – run automated consistency checks.

All functions accept the canonical ``pass_instances`` DataFrame as input
and are designed to be called interactively in notebooks or from CI scripts.

StatsBomb pitch convention
--------------------------
* x ∈ [0, 120]  (0 = own goal, 120 = opponent goal)
* y ∈ [0, 80]   (0 = near touchline, 80 = far touchline)
* Final third: x ≥ 80
* Penalty box: x ≥ 102, 18 ≤ y ≤ 62
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pitch constants
# ---------------------------------------------------------------------------
_PITCH_X: float = 120.0
_PITCH_Y: float = 80.0


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_positives(
    df: pd.DataFrame,
    label_col: str,
    n: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a random sample of rows where ``label_col`` is truthy.

    Rows where ``label_col`` is ``NaN`` are excluded (treated as missing,
    not positive).

    Parameters
    ----------
    df:
        Pass-instances DataFrame containing ``label_col``.
    label_col:
        Name of the binary label column to filter on.
    n:
        Number of rows to sample.  If fewer positives exist, all are returned.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sample of positive-label rows.  Retains all original columns plus a
        ``_sample_type`` column set to ``"positive"``.
    """
    _validate_label_col(df, label_col)
    positives = df[df[label_col].fillna(False).astype(bool)].copy()
    n_available = len(positives)
    if n_available == 0:
        logger.warning("sample_positives: no positive examples found for '%s'", label_col)
        return positives.assign(_sample_type="positive")

    sample = positives.sample(min(n, n_available), random_state=seed)
    sample["_sample_type"] = "positive"
    logger.debug(
        "sample_positives: sampled %d / %d positives for '%s'",
        len(sample),
        n_available,
        label_col,
    )
    return sample


def sample_negatives(
    df: pd.DataFrame,
    label_col: str,
    n: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a random sample of rows where ``label_col`` is falsy.

    Rows where ``label_col`` is ``NaN`` are excluded (treated as missing,
    not negative).

    Parameters
    ----------
    df:
        Pass-instances DataFrame containing ``label_col``.
    label_col:
        Name of the binary label column to filter on.
    n:
        Number of rows to sample.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sample of negative-label rows with a ``_sample_type`` column set to
        ``"negative"``.
    """
    _validate_label_col(df, label_col)
    # Exclude NaN (unknown); keep only confirmed negatives
    known = df[df[label_col].notna()]
    negatives = known[~known[label_col].astype(bool)].copy()
    n_available = len(negatives)
    if n_available == 0:
        logger.warning("sample_negatives: no negative examples found for '%s'", label_col)
        return negatives.assign(_sample_type="negative")

    sample = negatives.sample(min(n, n_available), random_state=seed)
    sample["_sample_type"] = "negative"
    logger.debug(
        "sample_negatives: sampled %d / %d negatives for '%s'",
        len(sample),
        n_available,
        label_col,
    )
    return sample


# ---------------------------------------------------------------------------
# Prevalence reporting
# ---------------------------------------------------------------------------


def label_prevalence_table(
    df: pd.DataFrame,
    label_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute count and positive rate for each label column.

    Parameters
    ----------
    df:
        Pass-instances DataFrame.
    label_cols:
        List of label column names to summarise.  Defaults to all standard
        Frame2Threat label columns present in ``df``.

    Returns
    -------
    pd.DataFrame
        One row per label with columns:

        * ``label``      – column name.
        * ``n_total``    – total rows (including NaN).
        * ``n_known``    – rows with non-NaN values.
        * ``n_positive`` – rows where label is True.
        * ``n_negative`` – rows where label is False.
        * ``n_nan``      – rows where label is NaN.
        * ``prevalence`` – positive rate among known rows (0–1).
    """
    if label_cols is None:
        label_cols = _default_label_cols(df)

    if not label_cols:
        logger.warning("label_prevalence_table: no label columns found in DataFrame")
        return pd.DataFrame(
            columns=[
                "label",
                "n_total",
                "n_known",
                "n_positive",
                "n_negative",
                "n_nan",
                "prevalence",
            ]
        )

    rows = []
    for col in label_cols:
        if col not in df.columns:
            logger.debug("label_prevalence_table: column '%s' not found, skipping", col)
            continue
        series = df[col]
        n_total = len(series)
        n_nan = int(series.isna().sum())
        n_known = n_total - n_nan
        n_positive = int(series.fillna(0).astype(bool).sum())
        n_negative = n_known - n_positive
        prevalence = n_positive / n_known if n_known > 0 else np.nan
        rows.append(
            {
                "label": col,
                "n_total": n_total,
                "n_known": n_known,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_nan": n_nan,
                "prevalence": prevalence,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spatial and categorical breakdowns
# ---------------------------------------------------------------------------


def label_by_zone(
    df: pd.DataFrame,
    label_col: str,
    zone_grid_x: int = 12,
    zone_grid_y: int = 8,
) -> pd.DataFrame:
    """Compute label prevalence by pitch zone (heatmap-style).

    Divides the pitch into a ``zone_grid_x × zone_grid_y`` grid and
    computes the positive rate of ``label_col`` within each zone.  Zones are
    defined by the pass *start* position (``start_x``, ``start_y``).

    Parameters
    ----------
    df:
        Pass-instances DataFrame.  Must contain ``start_x``, ``start_y``,
        and ``label_col``.
    label_col:
        Binary label column to summarise.
    zone_grid_x:
        Number of zones along the x-axis (default 12 → 10 m per zone).
    zone_grid_y:
        Number of zones along the y-axis (default 8 → 10 m per zone).

    Returns
    -------
    pd.DataFrame
        One row per zone that contains at least one pass with known label.
        Columns: ``zone_x``, ``zone_y``, ``n_passes``, ``n_positive``,
        ``prevalence``.  Suitable for pivot-table visualisation.
    """
    _validate_label_col(df, label_col)
    _validate_columns(df, ["start_x", "start_y"])

    work = df[["start_x", "start_y", label_col]].copy()
    work = work.dropna(subset=["start_x", "start_y", label_col])

    if work.empty:
        logger.warning("label_by_zone: no rows with complete data for '%s'", label_col)
        return pd.DataFrame(columns=["zone_x", "zone_y", "n_passes", "n_positive", "prevalence"])

    work["zone_x"] = np.clip(
        (work["start_x"] / _PITCH_X * zone_grid_x).astype(int), 0, zone_grid_x - 1
    )
    work["zone_y"] = np.clip(
        (work["start_y"] / _PITCH_Y * zone_grid_y).astype(int), 0, zone_grid_y - 1
    )
    work["_positive"] = work[label_col].astype(bool).astype(int)

    zone_stats = (
        work.groupby(["zone_x", "zone_y"], sort=True)
        .agg(n_passes=("_positive", "count"), n_positive=("_positive", "sum"))
        .reset_index()
    )
    zone_stats["prevalence"] = zone_stats["n_positive"] / zone_stats["n_passes"]

    return zone_stats


def label_by_pass_type(
    df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """Compute label prevalence broken down by pass type.

    Parameters
    ----------
    df:
        Pass-instances DataFrame.  Must contain ``pass_type`` and
        ``label_col``.
    label_col:
        Binary label column to summarise.

    Returns
    -------
    pd.DataFrame
        One row per pass type with columns: ``pass_type``, ``n_passes``,
        ``n_positive``, ``prevalence``.  Sorted by descending ``prevalence``.
    """
    _validate_label_col(df, label_col)

    if "pass_type" not in df.columns:
        logger.warning(
            "label_by_pass_type: 'pass_type' column not found; " "returning empty DataFrame."
        )
        return pd.DataFrame(columns=["pass_type", "n_passes", "n_positive", "prevalence"])

    work = df[["pass_type", label_col]].copy()
    work = work.dropna(subset=[label_col])
    work["_positive"] = work[label_col].astype(bool).astype(int)
    # Fill NaN pass types with a placeholder for groupby
    work["pass_type"] = work["pass_type"].fillna("(unknown)")

    stats = (
        work.groupby("pass_type", sort=False)
        .agg(n_passes=("_positive", "count"), n_positive=("_positive", "sum"))
        .reset_index()
    )
    stats["prevalence"] = stats["n_positive"] / stats["n_passes"]

    return stats.sort_values("prevalence", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Automated sanity checks
# ---------------------------------------------------------------------------


def label_sanity_checks(df: pd.DataFrame) -> dict[str, object]:
    """Run a battery of basic sanity checks on the labelled pass-instances table.

    Checks performed
    ----------------
    * ``strict_le_loose``   – strict_line_break rate ≤ loose_line_break rate.
    * ``ft_le_dp``          – final_third_entry_k rate ≤ dangerous_progression_k rate.
    * ``box_le_dp``         – box_entry_k rate ≤ dangerous_progression_k rate.
    * ``shot_le_dp``        – shot_within_k rate ≤ dangerous_progression_k rate.
    * ``dp_or_condition``   – dangerous_progression_k == (ft OR box OR shot).
    * ``threat_gain_range`` – threat_gain ∈ [-1, 1] for non-NaN rows.
    * ``no_nan_known_360``  – no NaN line-break labels for passes with has_360=True.
    * ``strict_lte_loose``  – no row has strict=True and loose=False.
    * ``prevalence_bounds`` – each binary label has prevalence in [0.01, 0.99]
      (warning only; extreme values may be correct for sparse datasets).

    Parameters
    ----------
    df:
        Pass-instances DataFrame with computed label columns.

    Returns
    -------
    dict
        Mapping of check_name → result.  Results are:

        * ``True``  – check passed.
        * ``False`` – check failed (indicates a labelling bug).
        * ``"skip"`` – check skipped because required columns are absent.
        * A descriptive string for informational checks.
    """
    results: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Helper: get prevalence safely
    # ------------------------------------------------------------------
    def _prev(col: str) -> Optional[float]:
        if col not in df.columns:
            return None
        s = df[col].fillna(False).astype(bool)
        n = s.notna().sum()
        return float(s.mean()) if n > 0 else None

    # ------------------------------------------------------------------
    # 1. strict ≤ loose line-break rate
    # ------------------------------------------------------------------
    p_strict = _prev("strict_line_break")
    p_loose = _prev("loose_line_break")
    if p_strict is not None and p_loose is not None:
        results["strict_le_loose"] = bool(p_strict <= p_loose + 1e-9)
    else:
        results["strict_le_loose"] = "skip"

    # ------------------------------------------------------------------
    # 2. Sub-label rates ≤ dangerous_progression rate
    # ------------------------------------------------------------------
    p_dp = _prev("dangerous_progression_k")
    for sub in ("final_third_entry_k", "box_entry_k", "shot_within_k"):
        key = f"{sub.replace('_k', '')}_le_dp"
        p_sub = _prev(sub)
        if p_sub is not None and p_dp is not None:
            results[key] = bool(p_sub <= p_dp + 1e-9)
        else:
            results[key] = "skip"

    # ------------------------------------------------------------------
    # 3. dangerous_progression == ft OR box OR shot (row-level check)
    # ------------------------------------------------------------------
    needed_dp = {"dangerous_progression_k", "final_third_entry_k", "box_entry_k", "shot_within_k"}
    if needed_dp.issubset(df.columns):
        dp_recon = (
            df["final_third_entry_k"].fillna(False).astype(bool)
            | df["box_entry_k"].fillna(False).astype(bool)
            | df["shot_within_k"].fillna(False).astype(bool)
        )
        dp_actual = df["dangerous_progression_k"].fillna(False).astype(bool)
        mismatch = int((dp_recon != dp_actual).sum())
        results["dp_or_condition"] = mismatch == 0
        if mismatch:
            logger.warning("label_sanity_checks: dp_or_condition FAILED for %d rows", mismatch)
    else:
        results["dp_or_condition"] = "skip"

    # ------------------------------------------------------------------
    # 4. threat_gain ∈ [-1, 1]
    # ------------------------------------------------------------------
    if "threat_gain" in df.columns:
        tg = df["threat_gain"].dropna()
        if len(tg) > 0:
            in_range = bool((tg >= -1.0 - 1e-6).all() and (tg <= 1.0 + 1e-6).all())
            results["threat_gain_range"] = in_range
            if not in_range:
                out_low = int((tg < -1.0).sum())
                out_high = int((tg > 1.0).sum())
                logger.warning(
                    "threat_gain_range FAILED: %d below -1, %d above 1",
                    out_low,
                    out_high,
                )
        else:
            results["threat_gain_range"] = "skip"
    else:
        results["threat_gain_range"] = "skip"

    # ------------------------------------------------------------------
    # 5. No NaN line-break labels for passes with has_360=True
    # ------------------------------------------------------------------
    if "has_360" in df.columns and "strict_line_break" in df.columns:
        has_360 = df["has_360"].fillna(False).astype(bool)
        if has_360.any():
            nan_strict = int(df.loc[has_360, "strict_line_break"].isna().sum())
            nan_loose = (
                int(df.loc[has_360, "loose_line_break"].isna().sum())
                if "loose_line_break" in df.columns
                else 0
            )
            results["no_nan_known_360"] = nan_strict == 0 and nan_loose == 0
            if nan_strict or nan_loose:
                logger.warning(
                    "no_nan_known_360 FAILED: %d NaN strict, %d NaN loose for has_360 passes",
                    nan_strict,
                    nan_loose,
                )
        else:
            results["no_nan_known_360"] = "skip"
    else:
        results["no_nan_known_360"] = "skip"

    # ------------------------------------------------------------------
    # 6. No row where strict=True but loose=False
    # ------------------------------------------------------------------
    if "strict_line_break" in df.columns and "loose_line_break" in df.columns:
        strict = df["strict_line_break"].fillna(False).astype(bool)
        loose = df["loose_line_break"].fillna(False).astype(bool)
        impossible = int((strict & ~loose).sum())
        results["strict_lte_loose_rowwise"] = impossible == 0
        if impossible:
            logger.warning(
                "strict_lte_loose_rowwise FAILED: %d rows have strict=True but loose=False",
                impossible,
            )
    else:
        results["strict_lte_loose_rowwise"] = "skip"

    # ------------------------------------------------------------------
    # 7. Prevalence-bounds advisory (informational, not pass/fail)
    # ------------------------------------------------------------------
    binary_labels = [
        "strict_line_break",
        "loose_line_break",
        "dangerous_progression_k",
        "final_third_entry_k",
        "box_entry_k",
        "shot_within_k",
    ]
    low_prevalence = []
    high_prevalence = []
    for col in binary_labels:
        p = _prev(col)
        if p is None:
            continue
        if p < 0.01:
            low_prevalence.append(col)
        elif p > 0.99:
            high_prevalence.append(col)

    if low_prevalence:
        logger.warning("Suspiciously low prevalence (<1%%): %s", low_prevalence)
    if high_prevalence:
        logger.warning("Suspiciously high prevalence (>99%%): %s", high_prevalence)

    results["prevalence_advisory"] = {
        "low_prevalence_cols": low_prevalence,
        "high_prevalence_cols": high_prevalence,
    }

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------
    n_failed = sum(1 for v in results.values() if v is False)
    n_skipped = sum(1 for v in results.values() if v == "skip")
    logger.info(
        "label_sanity_checks: %d checks, %d failed, %d skipped",
        len(results),
        n_failed,
        n_skipped,
    )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_LABEL_COLUMNS: list[str] = [
    "strict_line_break",
    "loose_line_break",
    "dangerous_progression_k",
    "final_third_entry_k",
    "box_entry_k",
    "shot_within_k",
    "threat_gain",
]


def _default_label_cols(df: pd.DataFrame) -> list[str]:
    """Return the subset of default label columns present in ``df``."""
    return [c for c in _DEFAULT_LABEL_COLUMNS if c in df.columns]


def _validate_label_col(df: pd.DataFrame, label_col: str) -> None:
    """Raise ValueError if ``label_col`` is not in ``df``."""
    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )


def _validate_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any of ``cols`` is missing from ``df``."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from DataFrame: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
