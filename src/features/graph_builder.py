"""
Graph construction from 360 freeze frames for GNN modelling.

Each pass event is represented as a graph:
  - Nodes: all visible players in the freeze frame
  - Node features: [x, y, teammate, is_keeper, is_actor, is_receiver,
                    dist_to_goal, dist_to_passer, local_density]
  - Edges: k-NN by spatial distance (k=5 default) + same-team edges
  - Edge features: [distance, angle_from_goal, same_team]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from torch_geometric.data import Data as PyGData

    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Goal centre on StatsBomb pitch
_GOAL_X: float = 120.0
_GOAL_Y: float = 40.0

# Default k for kNN graph
_DEFAULT_K: int = 5

# Radius for local-density node feature
_DENSITY_RADIUS: float = 10.0


def build_graph(
    event_uuid: str,
    frames_df: pd.DataFrame,
    pass_row: "pd.Series",
    config: dict,
) -> dict:
    """Build a graph representation for a single pass event.

    Parameters
    ----------
    event_uuid:
        Unique identifier for the pass event.
    frames_df:
        Full freeze-frame table (all events).  Filtered internally to the
        rows matching *event_uuid*.
    pass_row:
        Pass event row providing start_x, start_y, end_x, end_y.
    config:
        Configuration dict.  Relevant keys:
        ``k_neighbors`` (int, default 5): number of kNN edges per node.

    Returns
    -------
    dict
        Keys: event_uuid, node_features (N×9 ndarray), edge_index (2×E
        ndarray), edge_attr (E×3 ndarray), n_nodes.  Optional key
        ``labels`` if present in pass_row.
    """
    k = int(config.get("k_neighbors", _DEFAULT_K))

    # ------------------------------------------------------------------ #
    # Gather players for this event
    # ------------------------------------------------------------------ #
    event_frame = frames_df[frames_df["event_uuid"] == event_uuid].copy()

    if event_frame.empty:
        logger.debug("build_graph: no frame data for %s", event_uuid)
        return _empty_graph(event_uuid)

    n = len(event_frame)
    event_frame = event_frame.reset_index(drop=True)

    sx = float(pass_row.get("start_x", 0.0))
    sy = float(pass_row.get("start_y", 0.0))
    ex = float(pass_row.get("end_x", 0.0))
    ey = float(pass_row.get("end_y", 0.0))

    # ------------------------------------------------------------------ #
    # Node feature matrix  (N, 9)
    # ------------------------------------------------------------------ #
    px = event_frame["x"].astype(float).values
    py = event_frame["y"].astype(float).values
    teammate = event_frame["teammate"].fillna(False).astype(float).values
    is_keeper = event_frame["keeper"].fillna(False).astype(float).values
    is_actor = event_frame["actor"].fillna(False).astype(float).values

    # is_receiver: player closest to pass end point among teammates
    is_receiver = np.zeros(n, dtype=float)
    tm_mask = teammate.astype(bool) & ~is_actor.astype(bool)
    if tm_mask.any():
        tm_indices = np.where(tm_mask)[0]
        dist_to_end = np.sqrt((px[tm_indices] - ex) ** 2 + (py[tm_indices] - ey) ** 2)
        receiver_idx = tm_indices[dist_to_end.argmin()]
        is_receiver[receiver_idx] = 1.0

    dist_to_goal = np.sqrt((_GOAL_X - px) ** 2 + (_GOAL_Y - py) ** 2)
    dist_to_passer = np.sqrt((sx - px) ** 2 + (sy - py) ** 2)

    # local density: count of players within radius
    local_density = np.array(
        [
            float(((np.sqrt((px - px[i]) ** 2 + (py - py[i]) ** 2)) < _DENSITY_RADIUS).sum() - 1)
            for i in range(n)
        ]
    )

    node_features = np.stack(
        [
            px,
            py,
            teammate,
            is_keeper,
            is_actor,
            is_receiver,
            dist_to_goal,
            dist_to_passer,
            local_density,
        ],
        axis=1,
    ).astype(
        np.float32
    )  # (N, 9)

    # ------------------------------------------------------------------ #
    # Edge construction
    # ------------------------------------------------------------------ #
    edge_index, edge_attr = _build_edges(px, py, teammate, k)

    graph: dict = {
        "event_uuid": event_uuid,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "n_nodes": n,
    }

    # Attach label columns when present
    for label_col in (
        "line_break",
        "strict_line_break",
        "loose_line_break",
        "dangerous_progression_k",
        "final_third_entry_k",
        "box_entry_k",
        "shot_within_k",
        "threat_gain",
    ):
        val = pass_row.get(label_col, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            graph[label_col] = float(val)

    return graph


def build_graph_dataset(
    pass_instances_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    config: dict,
) -> list[dict]:
    """Build graph dicts for all pass events with 360 data.

    Parameters
    ----------
    pass_instances_df:
        Canonical pass instances table. Must have event_uuid, has_360.
    frames_df:
        Full freeze-frame table.
    config:
        Configuration dict forwarded to :func:`build_graph`.

    Returns
    -------
    list[dict]
        One graph dict per pass with 360 data (``has_360 == True``).
    """
    if pass_instances_df is None or pass_instances_df.empty:
        logger.warning("build_graph_dataset: empty pass_instances_df")
        return []

    subset = pass_instances_df[
        pass_instances_df.get("has_360", pd.Series(False, index=pass_instances_df.index)).astype(
            bool
        )
    ]
    logger.info("Building graph dataset for %d events with 360 data", len(subset))

    graphs: list[dict] = []
    for _, row in subset.iterrows():
        uuid = row["event_uuid"]
        g = build_graph(uuid, frames_df, row, config)
        if g and g["n_nodes"] > 0:
            graphs.append(g)

    logger.info("Built %d graphs", len(graphs))
    return graphs


def to_pytorch_geometric(graph_dict: dict) -> "PyGData":
    """Convert a graph dict to a :class:`torch_geometric.data.Data` object.

    Parameters
    ----------
    graph_dict:
        Dict as returned by :func:`build_graph`.

    Returns
    -------
    torch_geometric.data.Data
        PyTorch Geometric Data object.  ``data.y`` is set to a 1-D tensor
        of label values when label columns are present in *graph_dict*.

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for to_pytorch_geometric()")
    if not _PYG_AVAILABLE:
        raise ImportError("torch_geometric is required for to_pytorch_geometric()")

    x = torch.tensor(graph_dict["node_features"], dtype=torch.float)
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float)

    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.event_uuid = graph_dict["event_uuid"]

    # Attach labels
    _label_keys = [
        "line_break",
        "strict_line_break",
        "loose_line_break",
        "dangerous_progression_k",
        "final_third_entry_k",
        "box_entry_k",
        "shot_within_k",
        "threat_gain",
    ]
    labels = [graph_dict[k] for k in _label_keys if k in graph_dict]
    if labels:
        data.y = torch.tensor(labels, dtype=torch.float)

    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_edges(
    px: np.ndarray,
    py: np.ndarray,
    teammate: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build kNN + same-team edges with edge attributes.

    Parameters
    ----------
    px, py:      Node x/y coordinates.
    teammate:    Float array indicating team membership (1 = teammate).
    k:           Number of nearest neighbours per node.

    Returns
    -------
    edge_index : np.ndarray, shape (2, E)
    edge_attr  : np.ndarray, shape (E, 3)  – [distance, angle_from_goal, same_team]
    """
    n = len(px)
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    # Pairwise distances
    coords = np.stack([px, py], axis=1)  # (N, 2)
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
    dist_mat = np.sqrt((diff**2).sum(-1))  # (N, N)
    np.fill_diagonal(dist_mat, np.inf)

    src_list: list[int] = []
    dst_list: list[int] = []
    added: set[tuple[int, int]] = set()

    # kNN edges
    actual_k = min(k, n - 1)
    for i in range(n):
        nn_indices = np.argsort(dist_mat[i])[:actual_k]
        for j in nn_indices:
            pair = (int(i), int(j))
            if pair not in added:
                src_list.append(i)
                dst_list.append(j)
                added.add(pair)

    # Same-team edges (bidirectional) beyond kNN
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if teammate[i] == teammate[j]:
                pair = (int(i), int(j))
                if pair not in added:
                    src_list.append(i)
                    dst_list.append(j)
                    added.add(pair)

    if not src_list:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    src_arr = np.array(src_list, dtype=np.int64)
    dst_arr = np.array(dst_list, dtype=np.int64)
    edge_index = np.stack([src_arr, dst_arr], axis=0)

    # Edge attributes
    e = len(src_list)
    edge_dist = np.array(
        [
            (
                dist_mat[src_list[i], dst_list[i]]
                if dist_mat[src_list[i], dst_list[i]] != np.inf
                else 0.0
            )
            for i in range(e)
        ],
        dtype=np.float32,
    )

    # angle_from_goal: angle of mid-point between src and dst relative to goal
    mid_x = (px[src_arr] + px[dst_arr]) / 2.0
    mid_y = (py[src_arr] + py[dst_arr]) / 2.0
    angle_from_goal = np.arctan2(mid_y - _GOAL_Y, _GOAL_X - mid_x).astype(np.float32)

    same_team = (teammate[src_arr] == teammate[dst_arr]).astype(np.float32)

    edge_attr = np.stack([edge_dist, angle_from_goal, same_team], axis=1)  # (E, 3)
    return edge_index, edge_attr


def _empty_graph(event_uuid: str) -> dict:
    """Return an empty graph dict for events without frame data."""
    return {
        "event_uuid": event_uuid,
        "node_features": np.zeros((0, 9), dtype=np.float32),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_attr": np.zeros((0, 3), dtype=np.float32),
        "n_nodes": 0,
    }
