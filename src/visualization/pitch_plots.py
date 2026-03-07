"""Pitch visualization utilities using mplsoccer."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from mplsoccer import Pitch, VerticalPitch

    _MPLSOCCER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPLSOCCER_AVAILABLE = False
    logger.warning(
        "mplsoccer not available; pitch visualization functions will raise ImportError"
    )

# StatsBomb pitch dimensions
_PITCH_LENGTH = 120.0
_PITCH_WIDTH = 80.0

# 12 × 8 zone grid
_N_COLS = 12
_N_ROWS = 8
_ZONE_COL_WIDTH = _PITCH_LENGTH / _N_COLS  # 10 units each
_ZONE_ROW_HEIGHT = _PITCH_WIDTH / _N_ROWS  # 10 units each


def plot_pass_map(
    pass_instances_df: pd.DataFrame,
    title: str = "",
    ax: Any | None = None,
    color_col: str | None = None,
) -> tuple[Any, Any]:
    """Plot passes as arrows on a football pitch.

    Parameters
    ----------
    pass_instances_df:
        DataFrame with columns start_x, start_y, end_x, end_y.
        Optionally contains a ``color_col`` column used to colour arrows.
    title:
        Plot title string.
    ax:
        Optional existing matplotlib Axes.  A new figure + Axes is created
        when ``None``.
    color_col:
        Column name in ``pass_instances_df`` to use for colouring arrows via
        a sequential colourmap.  When ``None``, all arrows are the same colour.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if not _MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is required for plot_pass_map")

    pitch = Pitch(pitch_type="statsbomb", pitch_color="grass", line_color="white")

    if ax is None:
        fig, ax = pitch.draw(figsize=(12, 8))
    else:
        fig = ax.get_figure()
        pitch.draw(ax=ax)

    df = pass_instances_df.dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    if df.empty:
        logger.warning("plot_pass_map: no valid passes to plot")
        if title:
            ax.set_title(title, fontsize=13, color="white", pad=10)
        return fig, ax

    xs = df["start_x"].values
    ys = df["start_y"].values
    xe = df["end_x"].values
    ye = df["end_y"].values

    if color_col is not None and color_col in df.columns:
        values = df[color_col].values.astype(float)
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(values))

        for x_s, y_s, x_e, y_e, color in zip(xs, ys, xe, ye, colors):
            pitch.arrows(
                x_s,
                y_s,
                x_e,
                y_e,
                ax=ax,
                color=color,
                alpha=0.6,
                width=1.5,
                headwidth=3,
                headlength=4,
            )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=color_col, fraction=0.03, pad=0.02)
    else:
        pitch.arrows(
            xs,
            ys,
            xe,
            ye,
            ax=ax,
            color="white",
            alpha=0.5,
            width=1.5,
            headwidth=3,
            headlength=4,
        )

    if title:
        ax.set_title(title, fontsize=13, color="white", pad=10)

    logger.debug("plot_pass_map: plotted %d passes", len(df))
    return fig, ax


def plot_zone_heatmap(
    values_df: pd.DataFrame,
    title: str = "",
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot a 12×8 zone-grid heatmap on a football pitch.

    Parameters
    ----------
    values_df:
        DataFrame with columns ``col_idx`` (0-11), ``row_idx`` (0-7), and
        ``value``.  Missing zones default to NaN (shown as neutral colour).
    title:
        Plot title.
    ax:
        Optional existing Axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if not _MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is required for plot_zone_heatmap")

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#cccccc")

    if ax is None:
        fig, ax = pitch.draw(figsize=(12, 8))
    else:
        fig = ax.get_figure()
        pitch.draw(ax=ax)

    grid = np.full((_N_ROWS, _N_COLS), np.nan)
    for _, row in values_df.iterrows():
        c = int(row.get("col_idx", 0))
        r = int(row.get("row_idx", 0))
        if 0 <= c < _N_COLS and 0 <= r < _N_ROWS:
            grid[r, c] = float(row["value"])

    valid = grid[~np.isnan(grid)]
    vmin = float(np.nanmin(valid)) if len(valid) > 0 else 0.0
    vmax = float(np.nanmax(valid)) if len(valid) > 0 else 1.0

    for r in range(_N_ROWS):
        for c in range(_N_COLS):
            if np.isnan(grid[r, c]):
                continue
            x0 = c * _ZONE_COL_WIDTH
            y0 = r * _ZONE_ROW_HEIGHT
            norm_val = (grid[r, c] - vmin) / max(vmax - vmin, 1e-9)
            color = plt.cm.YlOrRd(norm_val)  # type: ignore[attr-defined]
            rect = plt.Rectangle(
                (x0, y0),
                _ZONE_COL_WIDTH,
                _ZONE_ROW_HEIGHT,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.8,
            )
            ax.add_patch(rect)
            ax.text(
                x0 + _ZONE_COL_WIDTH / 2,
                y0 + _ZONE_ROW_HEIGHT / 2,
                f"{grid[r, c]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.YlOrRd,  # type: ignore[attr-defined]
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

    if title:
        ax.set_title(title, fontsize=13, pad=10)

    return fig, ax


def plot_threat_gain_map(
    pass_instances_df: pd.DataFrame,
    title: str = "",
) -> Any:
    """Plot a zone-level heatmap of mean threat_gain values.

    Parameters
    ----------
    pass_instances_df:
        Must contain columns ``start_x``, ``start_y``, and ``threat_gain``.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is required for plot_threat_gain_map")

    df = pass_instances_df.dropna(subset=["start_x", "start_y"])
    if "threat_gain" not in df.columns:
        logger.warning("plot_threat_gain_map: 'threat_gain' column not found")
        df["threat_gain"] = 0.0

    df = df.copy()
    df["col_idx"] = np.clip(
        (df["start_x"] / _PITCH_LENGTH * _N_COLS).astype(int), 0, _N_COLS - 1
    )
    df["row_idx"] = np.clip(
        (df["start_y"] / _PITCH_WIDTH * _N_ROWS).astype(int), 0, _N_ROWS - 1
    )

    zone_means = (
        df.groupby(["col_idx", "row_idx"])["threat_gain"]
        .mean()
        .reset_index()
        .rename(columns={"threat_gain": "value"})
    )

    default_title = title or "Mean Threat Gain by Zone"
    fig, _ = plot_zone_heatmap(zone_means, title=default_title)
    return fig


def plot_player_profile(
    player_name: str,
    pass_instances_df: pd.DataFrame,
    scores: np.ndarray | pd.Series,
) -> Any:
    """Plot a four-panel player pass profile.

    Panels
    ------
    1. All passes coloured by predicted score.
    2. Zone heatmap of mean predicted score.
    3. Score distribution histogram.
    4. Top-10 highest-scoring passes.

    Parameters
    ----------
    player_name:
        Player name string used to filter ``pass_instances_df``.
    pass_instances_df:
        Full pass dataset; must contain ``player_name`` column.
    scores:
        Predicted scores aligned with ``pass_instances_df``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is required for plot_player_profile")

    df = pass_instances_df.copy()
    df["predicted_score"] = np.asarray(scores, dtype=float)

    name_col = "player_name" if "player_name" in df.columns else None
    if name_col is not None:
        player_df = df[df[name_col] == player_name].copy()
    else:
        logger.warning("plot_player_profile: 'player_name' column not found; using all passes")
        player_df = df.copy()

    if player_df.empty:
        logger.warning("plot_player_profile: no passes found for '%s'", player_name)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"No data for {player_name}", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Player Profile: {player_name}", fontsize=16, fontweight="bold")

    # Panel 1 – pass map
    ax1 = fig.add_subplot(2, 2, 1)
    plot_pass_map(
        player_df,
        title=f"{player_name} – Passes (coloured by score)",
        ax=ax1,
        color_col="predicted_score",
    )

    # Panel 2 – zone heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    player_df_zone = player_df.copy()
    player_df_zone["col_idx"] = np.clip(
        (player_df_zone["start_x"] / _PITCH_LENGTH * _N_COLS).astype(int), 0, _N_COLS - 1
    )
    player_df_zone["row_idx"] = np.clip(
        (player_df_zone["start_y"] / _PITCH_WIDTH * _N_ROWS).astype(int), 0, _N_ROWS - 1
    )
    zone_means = (
        player_df_zone.groupby(["col_idx", "row_idx"])["predicted_score"]
        .mean()
        .reset_index()
        .rename(columns={"predicted_score": "value"})
    )
    plot_zone_heatmap(zone_means, title="Mean Predicted Score by Zone", ax=ax2)

    # Panel 3 – score distribution
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(
        player_df["predicted_score"].dropna(),
        bins=20,
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
    )
    ax3.axvline(
        player_df["predicted_score"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean={player_df['predicted_score'].mean():.3f}",
    )
    ax3.set_xlabel("Predicted Score", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Score Distribution", fontsize=12)
    ax3.legend()
    ax3.set_facecolor("#1a1a2e")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("white")

    # Panel 4 – top-10 pass list
    ax4 = fig.add_subplot(2, 2, 4)
    top10 = player_df.nlargest(10, "predicted_score")
    cols_to_show = [c for c in ["start_x", "start_y", "end_x", "end_y", "predicted_score"] if c in top10.columns]
    table_data = top10[cols_to_show].round(2).values
    col_labels = [c.replace("predicted_score", "score") for c in cols_to_show]
    ax4.axis("off")
    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax4.set_title("Top 10 Scoring Passes", fontsize=12)

    plt.tight_layout()
    return fig
