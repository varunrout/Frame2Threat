"""
src/labels/line_break.py
========================
Line-break label construction for pass events.

Operational definition
----------------------
A pass "breaks a defensive line" if ALL of the following hold:

1. **Forward progress** – the pass has meaningful positive x-displacement
   toward the opponent goal (end_x - start_x >= min_forward_gain_m, default
   5.0 metres).

2. **Defensive layer present** – at least one (loose) or two (strict) visible
   opponent players are situated *strictly between* the passer and the
   receiver in the x-direction at the moment of the pass (as captured by the
   360 freeze frame).

3. **Pass clears the layer** – because the receiver's x > all opponents in
   the measured band, condition 2 already implies the end position is beyond
   the layer.

Two label variants
------------------
* ``strict_line_break`` – requires ≥ 2 opponents between passer-x and
  receiver-x.
* ``loose_line_break``  – requires ≥ 1 opponent between passer-x and
  receiver-x.

Both variants are computed over open-play passes only (``open_play_only=True``
by default; see note below).

360 data requirement
--------------------
Rows whose ``has_360`` flag is ``False`` receive ``NaN`` for both label
columns rather than ``False``, preserving the distinction between "no 360
data" and "had 360 data but did not break a line".

Note on open_play_only
----------------------
The ``pass_instances`` table produced by :mod:`src.data.join_pass_frames`
is already restricted to open-play passes, so the ``open_play_only`` config
flag acts as a safety guard.  When ``True`` (default) and the DataFrame
contains a ``pass_type`` column, set-piece pass types are additionally
excluded before labelling.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Set-piece pass types to optionally exclude (guard for raw input)
_SET_PIECE_PASS_TYPES: frozenset[str] = frozenset(
    {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_line_break_labels(
    pass_instances_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Compute ``strict_line_break`` and ``loose_line_break`` labels.

    Parameters
    ----------
    pass_instances_df:
        Canonical pass-instances table produced by
        :func:`src.data.join_pass_frames.build_pass_instances`.
        Required columns: ``event_uuid``, ``start_x``, ``end_x``,
        ``has_360``.  Optional: ``pass_type`` (used when
        ``open_play_only=True``).
    frames_df:
        Normalised 360 freeze-frame table produced by
        :func:`src.data.parse_360.parse_360_frames`.
        Required columns: ``event_uuid``, ``teammate``, ``x``.
        Opponents are rows where ``teammate == False``.
    config:
        Dictionary corresponding to the ``line_break`` section of
        ``configs/labels.yaml``.  Recognised keys:

        * ``min_forward_gain_m`` (float, default 5.0) – minimum x-gain.
        * ``defensive_layer_gap_m`` (float, default 3.0) – x-band width for
          clustering; currently used for documentation purposes only; the
          label logic counts any opponent strictly between passer and
          receiver.
        * ``open_play_only`` (bool, default True) – exclude set pieces.

    Returns
    -------
    pd.DataFrame
        Copy of ``pass_instances_df`` with two new (or overwritten) columns:

        * ``strict_line_break`` – bool, NaN where ``has_360`` is False.
        * ``loose_line_break``  – bool, NaN where ``has_360`` is False.
    """
    min_forward_gain_m: float = float(config.get("min_forward_gain_m", 5.0))
    open_play_only: bool = bool(config.get("open_play_only", True))

    result = pass_instances_df.copy()
    # Initialise both label columns to NA using nullable boolean dtype so that
    # True/False/NA can coexist in the same column without dtype-casting errors.
    _na_bool = pd.array([pd.NA] * len(result), dtype="boolean")
    result["strict_line_break"] = _na_bool.copy()
    result["loose_line_break"] = _na_bool.copy()

    if frames_df is None or frames_df.empty:
        logger.warning(
            "compute_line_break_labels: frames_df is empty; "
            "all line-break labels will remain NaN."
        )
        return result

    # ------------------------------------------------------------------
    # Open-play guard (safety; pass_instances should already be filtered)
    # ------------------------------------------------------------------
    working = result.copy()
    if open_play_only and "pass_type" in working.columns:
        is_set_piece = working["pass_type"].isin(_SET_PIECE_PASS_TYPES).fillna(False)
        if is_set_piece.any():
            logger.debug(
                "open_play_only: excluding %d set-piece rows from labelling",
                is_set_piece.sum(),
            )
            working.loc[is_set_piece, ["strict_line_break", "loose_line_break"]] = False

    # ------------------------------------------------------------------
    # Identify passes that have 360 data
    # ------------------------------------------------------------------
    has_360_mask = working["has_360"].fillna(False).astype(bool)
    n_with_360 = int(has_360_mask.sum())
    logger.debug("Passes with 360 data: %d / %d", n_with_360, len(working))

    if n_with_360 == 0:
        logger.info("No passes have 360 data; all line-break labels will remain NaN.")
        result["strict_line_break"] = working["strict_line_break"]
        result["loose_line_break"] = working["loose_line_break"]
        return result

    # ------------------------------------------------------------------
    # Extract opponents from frames
    # ------------------------------------------------------------------
    # teammate is a nullable boolean; False or NA-False rows are opponents
    opp_mask = frames_df["teammate"].fillna(True) == False  # noqa: E712
    opponents = frames_df.loc[opp_mask, ["event_uuid", "x"]].copy()
    opponents = opponents.dropna(subset=["x"])

    logger.debug("Opponent player-frame rows available: %d", len(opponents))

    # ------------------------------------------------------------------
    # Build defender-count table via vectorised merge
    # ------------------------------------------------------------------
    passes_360 = working.loc[
        has_360_mask, ["event_uuid", "start_x", "end_x"]
    ].copy()

    # Left-join: each pass row × all opponents in same freeze frame
    merged = passes_360.merge(opponents, on="event_uuid", how="left")

    # Defender is "between" passer and receiver in x (strict inequalities)
    merged["_between"] = merged["x"].gt(merged["start_x"]) & merged["x"].lt(
        merged["end_x"]
    )

    # Aggregate: total defenders between per pass
    defender_counts = (
        merged.groupby("event_uuid", sort=False)["_between"]
        .sum()
        .astype(int)
        .rename("_n_def_between")
        .reset_index()
    )

    passes_360 = passes_360.merge(defender_counts, on="event_uuid", how="left")
    passes_360["_n_def_between"] = passes_360["_n_def_between"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Apply labelling thresholds
    # ------------------------------------------------------------------
    forward_gain = passes_360["end_x"] - passes_360["start_x"]
    forward_ok = forward_gain >= min_forward_gain_m
    n_def = passes_360["_n_def_between"]

    passes_360["strict_line_break"] = (forward_ok & (n_def >= 2))
    passes_360["loose_line_break"] = (forward_ok & (n_def >= 1))

    # ------------------------------------------------------------------
    # Merge computed labels back into working copy
    # ------------------------------------------------------------------
    # First set all has_360 rows to False (default; will be overridden below)
    working.loc[has_360_mask, "strict_line_break"] = False
    working.loc[has_360_mask, "loose_line_break"] = False

    label_lookup = passes_360.set_index("event_uuid")[
        ["strict_line_break", "loose_line_break"]
    ]
    for col in ("strict_line_break", "loose_line_break"):
        mapped = working.loc[has_360_mask, "event_uuid"].map(label_lookup[col])
        working.loc[has_360_mask, col] = mapped.values

    result["strict_line_break"] = working["strict_line_break"]
    result["loose_line_break"] = working["loose_line_break"]

    # ------------------------------------------------------------------
    # Logging summary
    # ------------------------------------------------------------------
    n_strict = int(result["strict_line_break"].sum(skipna=True))
    n_loose = int(result["loose_line_break"].sum(skipna=True))
    pct_strict = 100.0 * n_strict / n_with_360 if n_with_360 else 0.0
    pct_loose = 100.0 * n_loose / n_with_360 if n_with_360 else 0.0
    logger.info(
        "Line-break labels: strict=%d (%.1f%%), loose=%d (%.1f%%) "
        "out of %d passes with 360 data.",
        n_strict, pct_strict, n_loose, pct_loose, n_with_360,
    )

    return result


# ---------------------------------------------------------------------------
# Utility helpers (also used by tests)
# ---------------------------------------------------------------------------


def _count_defenders_between(
    defenders_df: pd.DataFrame,
    start_x: float,
    end_x: float,
) -> int:
    """Count opponents whose x-position falls strictly between two x-values.

    Parameters
    ----------
    defenders_df:
        DataFrame of opponent player positions for a single event.
        Must contain an ``x`` column with valid float values.
    start_x:
        Passer's x-coordinate (lower bound, exclusive).
    end_x:
        Receiver's x-coordinate (upper bound, exclusive).

    Returns
    -------
    int
        Number of defender rows with ``start_x < x < end_x``.
        Returns 0 for empty input or if end_x <= start_x.
    """
    if defenders_df is None or defenders_df.empty:
        return 0
    if end_x <= start_x:
        return 0

    x_vals = pd.to_numeric(defenders_df["x"], errors="coerce").dropna()
    return int(((x_vals > start_x) & (x_vals < end_x)).sum())


def _detect_defensive_layer(
    defenders_df: pd.DataFrame,
    start_x: float,
    end_x: float,
    layer_gap_m: float = 3.0,
) -> Optional[tuple[float, float]]:
    """Detect whether a defensive layer (cluster) exists between two x-values.

    A defensive layer is defined as a contiguous band of width ≤
    ``layer_gap_m`` in x that contains at least two opponent players, with
    the band situated strictly between ``start_x`` and ``end_x``.

    Parameters
    ----------
    defenders_df:
        DataFrame of opponent player positions for a single event.
        Must contain column ``x``.
    start_x:
        Passer's x-coordinate.
    end_x:
        Receiver's x-coordinate.
    layer_gap_m:
        Maximum x-span for players to be considered in the same layer.

    Returns
    -------
    tuple[float, float] or None
        ``(layer_x_min, layer_x_max)`` of the detected layer, or ``None``
        if no qualifying layer is found.
    """
    if defenders_df is None or defenders_df.empty or end_x <= start_x:
        return None

    x_vals = (
        pd.to_numeric(defenders_df["x"], errors="coerce")
        .dropna()
        .values
    )
    # Keep only opponents strictly between passer and receiver
    x_between = np.sort(x_vals[(x_vals > start_x) & (x_vals < end_x)])

    if len(x_between) < 2:
        return None

    # Sliding window: find first window of width <= layer_gap_m with >= 2 players
    for i in range(len(x_between) - 1):
        if x_between[i + 1] - x_between[i] <= layer_gap_m:
            return (float(x_between[i]), float(x_between[i + 1]))

    return None
