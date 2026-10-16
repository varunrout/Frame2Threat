"""
pass_graph.py
=============
Leak-free freeze-frame graphs for the corrected-label PassFrameGNN.

One graph per open-play pass with 360 data.  This follows ``src/features/graph_builder.py``
(nodes = visible players; edges = 5 nearest neighbours + same-team links) with **one deliberate
change**: the original ninth node feature ``is_receiver`` marked the teammate closest to the pass
END location, so it encoded the pass destination.  It is dropped here, leaving 8 node features.

Leakage contract (checked in ``tests/test_pass_gnn.py``)
--------------------------------------------------------
The graph is a function of

* the freeze-frame rows of *this* event (positions and the teammate / keeper / actor flags), and
* the pass **start** location (the event location, known when the pass is played),

and of nothing else.  :func:`build_pass_graph` accepts no pass end location, label or later-event
information, so none can reach a node or edge feature.

Node features (8), all scaled to roughly [0, 1]:
    x/120, y/80, teammate, is_keeper, is_actor, dist_to_goal/120, dist_to_passer/120,
    local_density/10 (other players within 10 m).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X, GOAL_Y = 120.0, 40.0
DENSITY_RADIUS = 10.0
DEFAULT_K = 5

NODE_FEATURES = [
    "x",
    "y",
    "teammate",
    "is_keeper",
    "is_actor",
    "dist_to_goal",
    "dist_to_passer",
    "local_density",
]
FRAME_COLUMNS = ("x", "y", "teammate", "keeper", "actor")


@dataclass
class PassGraph:
    event_uuid: str
    x: np.ndarray  # (N, 8) float32
    edge_index: (
        np.ndarray
    )  # (2, E) int64, row 0 = source, row 1 = target (messages flow source -> target)

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])


def build_edges(
    px: np.ndarray, py: np.ndarray, teammate: np.ndarray, k: int = DEFAULT_K
) -> np.ndarray:
    """kNN edges (each node -> its k nearest nodes) plus same-team links in both directions."""
    n = len(px)
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)
    coords = np.stack([px, py], axis=1)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    kk = min(k, n - 1)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in np.argsort(dist[i], kind="mergesort")[:kk]:
            pairs.add((i, int(j)))
    team = teammate.astype(bool)
    for i in range(n):
        for j in range(n):
            if i != j and team[i] == team[j]:
                pairs.add((i, j))
    ordered = sorted(pairs)
    return np.array(ordered, dtype=np.int64).T


def build_pass_graph(
    event_uuid: str,
    frame: pd.DataFrame,
    start_x: float,
    start_y: float,
    k: int = DEFAULT_K,
) -> PassGraph | None:
    """Graph for one pass from its freeze-frame rows and the pass start location only."""
    if frame is None or len(frame) == 0:
        return None
    f = frame.sort_values(["x", "y"], kind="mergesort").reset_index(drop=True)
    px = f["x"].astype(float).to_numpy()
    py = f["y"].astype(float).to_numpy()
    teammate = f["teammate"].fillna(False).astype(float).to_numpy()
    keeper = f["keeper"].fillna(False).astype(float).to_numpy()
    actor = f["actor"].fillna(False).astype(float).to_numpy()
    dist_goal = np.sqrt((GOAL_X - px) ** 2 + (GOAL_Y - py) ** 2)
    dist_passer = np.sqrt((float(start_x) - px) ** 2 + (float(start_y) - py) ** 2)
    pair = np.sqrt((px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2)
    density = (pair < DENSITY_RADIUS).sum(axis=1) - 1.0
    x = np.stack(
        [
            px / PITCH_LENGTH,
            py / PITCH_WIDTH,
            teammate,
            keeper,
            actor,
            dist_goal / PITCH_LENGTH,
            dist_passer / PITCH_LENGTH,
            density / 10.0,
        ],
        axis=1,
    ).astype(np.float32)
    return PassGraph(event_uuid, x, build_edges(px, py, teammate, k))


def build_graph_dataset(
    passes: pd.DataFrame, frames: pd.DataFrame, k: int = DEFAULT_K
) -> list[PassGraph]:
    """Graphs for every pass in ``passes`` that has freeze-frame rows (order of ``passes``)."""
    wanted = set(passes["event_uuid"])
    sub = frames[frames["event_uuid"].isin(wanted)]
    grouped = {u: g for u, g in sub.groupby("event_uuid", sort=False)}
    out = []
    for u, sx, sy in zip(passes["event_uuid"], passes["start_x"], passes["start_y"]):
        g = grouped.get(u)
        if g is None:
            continue
        graph = build_pass_graph(u, g[list(FRAME_COLUMNS)], sx, sy, k)
        if graph is not None and graph.n_nodes > 0:
            out.append(graph)
    return out
