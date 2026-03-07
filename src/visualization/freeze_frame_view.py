"""Freeze-frame visualization for individual pass events using mplsoccer."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from mplsoccer import Pitch

    _MPLSOCCER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPLSOCCER_AVAILABLE = False
    logger.warning(
        "mplsoccer not available; freeze-frame visualization will raise ImportError"
    )

# Player marker sizes
_PLAYER_MS = 120
_PASSER_MS = 220

# Colour scheme
_TEAMMATE_COLOR = "#1f78b4"   # blue
_OPPONENT_COLOR = "#e31a1c"   # red
_KEEPER_COLOR = "#ffff33"     # yellow
_PASSER_COLOR = "#33a02c"     # green star
_ARROW_COLOR = "#ffffff"       # actual pass arrow
_RANKED_CMAP = plt.cm.RdYlGn  # type: ignore[attr-defined]


def plot_freeze_frame(
    event_uuid: str,
    pass_instances_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    scores: np.ndarray | pd.Series | None = None,
    ranked_options: pd.DataFrame | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot a StatsBomb 360 freeze-frame for a single pass event.

    Visualises:

    * All visible players (teammates=blue circles, opponents=red circles,
      keepers=yellow diamonds).
    * Passer location (green star).
    * Actual pass arrow (white).
    * Optional ranked candidate arrows coloured by rank (red → green).
    * Score annotation panel when ``scores`` is provided.

    Parameters
    ----------
    event_uuid:
        UUID string of the pass event to visualise.
    pass_instances_df:
        Pass events DataFrame; must contain ``event_uuid``, ``start_x``,
        ``start_y``, ``end_x``, ``end_y``.
    frames_df:
        360 freeze-frame table; must contain ``event_uuid``, ``x``, ``y``,
        ``teammate``, ``actor``, ``keeper``.
    scores:
        Optional array aligned with ``pass_instances_df`` giving the predicted
        score for each pass.  When provided, the score is annotated on the plot.
    ranked_options:
        Optional DataFrame of candidate pass destinations with columns
        ``end_x``, ``end_y``, ``rank`` (1=best), and optionally ``score``.
    ax:
        Optional existing matplotlib Axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if not _MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is required for plot_freeze_frame")

    # ------------------------------------------------------------------
    # Retrieve the event row
    # ------------------------------------------------------------------
    uuid_col = "event_uuid" if "event_uuid" in pass_instances_df.columns else "id"
    event_rows = pass_instances_df[pass_instances_df[uuid_col] == event_uuid]
    if event_rows.empty:
        raise ValueError(f"event_uuid '{event_uuid}' not found in pass_instances_df")
    event_row = event_rows.iloc[0]

    start_x = float(event_row.get("start_x", 60.0))
    start_y = float(event_row.get("start_y", 40.0))
    end_x = float(event_row.get("end_x", 70.0))
    end_y = float(event_row.get("end_y", 40.0))

    # ------------------------------------------------------------------
    # Retrieve freeze-frame players
    # ------------------------------------------------------------------
    frame_uuid_col = "event_uuid" if "event_uuid" in frames_df.columns else "id"
    frame_rows = frames_df[frames_df[frame_uuid_col] == event_uuid]

    # ------------------------------------------------------------------
    # Draw pitch
    # ------------------------------------------------------------------
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#0d1117", line_color="#aaaaaa")

    if ax is None:
        fig, ax = pitch.draw(figsize=(14, 9))
    else:
        fig = ax.get_figure()
        pitch.draw(ax=ax)

    # ------------------------------------------------------------------
    # Plot players
    # ------------------------------------------------------------------
    if not frame_rows.empty:
        for _, player in frame_rows.iterrows():
            px = float(player.get("x", 0))
            py = float(player.get("y", 0))
            is_actor = bool(player.get("actor", False))
            is_keeper = bool(player.get("keeper", False))
            is_teammate = bool(player.get("teammate", False))

            if is_actor:
                # Passer – plotted separately below
                continue

            if is_keeper:
                color = _KEEPER_COLOR
                marker = "D"
            elif is_teammate:
                color = _TEAMMATE_COLOR
                marker = "o"
            else:
                color = _OPPONENT_COLOR
                marker = "o"

            ax.scatter(
                px,
                py,
                s=_PLAYER_MS,
                c=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
                alpha=0.9,
            )

    # Passer marker
    ax.scatter(
        start_x,
        start_y,
        s=_PASSER_MS,
        c=_PASSER_COLOR,
        marker="*",
        edgecolors="white",
        linewidths=1.0,
        zorder=6,
        label="Passer",
    )

    # ------------------------------------------------------------------
    # Actual pass arrow
    # ------------------------------------------------------------------
    pitch.arrows(
        start_x,
        start_y,
        end_x,
        end_y,
        ax=ax,
        color=_ARROW_COLOR,
        width=2.5,
        headwidth=5,
        headlength=6,
        zorder=5,
        label="Actual pass",
    )

    # ------------------------------------------------------------------
    # Ranked candidate arrows
    # ------------------------------------------------------------------
    if ranked_options is not None and not ranked_options.empty:
        _plot_ranked_arrows(pitch, ax, start_x, start_y, ranked_options)

    # ------------------------------------------------------------------
    # Score annotation
    # ------------------------------------------------------------------
    if scores is not None:
        score_arr = np.asarray(scores, dtype=float)
        idx = pass_instances_df.index[pass_instances_df[uuid_col] == event_uuid]
        if len(idx) > 0:
            row_pos = pass_instances_df.index.get_loc(idx[0])
            if row_pos < len(score_arr):
                score_val = float(score_arr[row_pos])
                ax.text(
                    2,
                    76,
                    f"Score: {score_val:.3f}",
                    fontsize=13,
                    color="white",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#222222", alpha=0.8),
                    zorder=7,
                )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=_PASSER_COLOR, markersize=12, label="Passer", linestyle="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_TEAMMATE_COLOR, markersize=8, label="Teammate", linestyle="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_OPPONENT_COLOR, markersize=8, label="Opponent", linestyle="None"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=_KEEPER_COLOR, markersize=8, label="Keeper", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.7)

    ax.set_title(f"Freeze Frame – {event_uuid[:8]}…", fontsize=12, color="white", pad=8)

    logger.debug("plot_freeze_frame: rendered event %s", event_uuid)
    return fig, ax


def plot_ranked_options(
    event_uuid: str,
    pass_instances_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    ranked_options: pd.DataFrame,
    top_k: int = 5,
) -> Any:
    """Show the top-k ranked pass options for an event on a single figure.

    Draws the freeze frame and overlays the top-k candidate arrows coloured
    from red (rank 1 = best) to green (rank k).

    Parameters
    ----------
    event_uuid:
        UUID of the pass event.
    pass_instances_df:
        Pass events DataFrame.
    frames_df:
        360 freeze-frame DataFrame.
    ranked_options:
        Candidate pass destinations with columns: end_x, end_y, rank, score.
    top_k:
        Number of top-ranked candidates to display.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is required for plot_ranked_options")

    top_options = (
        ranked_options.nsmallest(top_k, "rank") if "rank" in ranked_options.columns
        else ranked_options.head(top_k)
    )

    fig, ax = plot_freeze_frame(
        event_uuid=event_uuid,
        pass_instances_df=pass_instances_df,
        frames_df=frames_df,
        ranked_options=top_options,
    )

    ax.set_title(
        f"Top-{top_k} Ranked Options – {event_uuid[:8]}…",
        fontsize=12,
        color="white",
        pad=8,
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plot_ranked_arrows(
    pitch: Any,
    ax: Any,
    start_x: float,
    start_y: float,
    ranked_options: pd.DataFrame,
) -> None:
    """Draw candidate pass arrows coloured by rank (1=best=green)."""
    if ranked_options.empty:
        return

    n = len(ranked_options)
    cmap = _RANKED_CMAP

    sorted_opts = (
        ranked_options.sort_values("rank") if "rank" in ranked_options.columns
        else ranked_options
    )

    for i, (_, row) in enumerate(sorted_opts.iterrows()):
        ex = float(row.get("end_x", start_x + 10))
        ey = float(row.get("end_y", start_y))
        # Colour: best rank (1) → green, worst → red
        norm_rank = i / max(n - 1, 1)  # 0 (best) → 1 (worst)
        color = cmap(1.0 - norm_rank)

        score_val = row.get("score", None)
        label_text = f"#{i + 1}" if score_val is None else f"#{i + 1} ({score_val:.2f})"

        pitch.arrows(
            start_x,
            start_y,
            ex,
            ey,
            ax=ax,
            color=color,
            width=2.0,
            headwidth=4,
            headlength=5,
            alpha=0.75,
            zorder=5,
        )
        ax.text(
            ex + 0.5,
            ey + 0.5,
            label_text,
            fontsize=8,
            color="white",
            zorder=6,
        )
