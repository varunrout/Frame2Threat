"""
pass_gnn.py
===========
PassFrameGNN for the corrected label ``progression_beyond_pass_k``: graph-only, one binary head.

Architecture (as ``configs/model_gnn.yaml``): node projection (8 -> 64, ReLU), 3 x GraphSAGE
(mean aggregation), mean pooling, a fusion block (Linear 64 -> 64, ReLU, dropout) and a single
Linear(64, 1) head.  Unlike the original ``src/models/gnn.py`` there is **no** event-feature branch
and no multitask heads.

GraphSAGE is implemented in plain torch (``SAGEMean``) so no ``torch_geometric`` is needed and,
unlike the old fallback, message passing is never silently skipped.  It equals PyG's
``SAGEConv(aggr="mean")``: ``out_i = W_l * mean_{j: j->i}(h_j) + W_r * h_i`` (``W_l`` with bias,
``W_r`` without; nodes without incoming edges aggregate to zero).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.features.pass_graph import NODE_FEATURES, PassGraph


class SAGEMean(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(h).index_add_(0, dst, h[src])
        deg = torch.zeros(h.shape[0], 1, dtype=h.dtype).index_add_(
            0, dst, torch.ones(len(dst), 1, dtype=h.dtype)
        )
        return self.lin_l(agg / deg.clamp(min=1.0)) + self.lin_r(h)


class PassGraphSAGE(nn.Module):
    def __init__(
        self,
        node_dim: int = len(NODE_FEATURES),
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cfg = {"node_dim": node_dim, "hidden": hidden, "layers": layers, "dropout": dropout}
        self.node_proj = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.convs = nn.ModuleList([SAGEMean(hidden, hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.fusion = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch, n_graphs: int) -> torch.Tensor:
        """Logits, shape (n_graphs,)."""
        h = self.node_proj(x)
        for conv in self.convs:
            h = self.drop(torch.relu(conv(h, edge_index)))
        pooled = torch.zeros(n_graphs, h.shape[1], dtype=h.dtype).index_add_(0, batch, h)
        counts = torch.zeros(n_graphs, 1, dtype=h.dtype).index_add_(
            0, batch, torch.ones(len(batch), 1, dtype=h.dtype)
        )
        return self.head(self.fusion(pooled / counts.clamp(min=1.0))).squeeze(-1)


def collate(graphs: Sequence[PassGraph]):
    """Concatenate graphs into one batch: ``(x, edge_index, batch, n_graphs)``."""
    xs, eis, bs, off = [], [], [], 0
    for i, g in enumerate(graphs):
        xs.append(torch.from_numpy(g.x))
        eis.append(torch.from_numpy(g.edge_index) + off)
        bs.append(torch.full((g.n_nodes,), i, dtype=torch.long))
        off += g.n_nodes
    return torch.cat(xs), torch.cat(eis, dim=1), torch.cat(bs), len(graphs)


@torch.no_grad()
def predict_proba(
    model: PassGraphSAGE, graphs: Sequence[PassGraph], batch_size: int = 256
) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(graphs), batch_size):
        x, ei, b, n = collate(graphs[i : i + batch_size])
        out.append(torch.sigmoid(model(x, ei, b, n)).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def save_model(path: str | Path, model: PassGraphSAGE, meta: dict) -> None:
    """Save weights + architecture config in ``path`` (.pt) and metadata next to it (.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "cfg": model.cfg, "node_features": NODE_FEATURES}, path
    )
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, default=float), encoding="utf-8"
    )


def load_model(path: str | Path) -> PassGraphSAGE:
    ck = torch.load(Path(path), map_location="cpu")
    if ck["node_features"] != NODE_FEATURES:
        raise ValueError(f"node feature list mismatch: {ck['node_features']} != {NODE_FEATURES}")
    model = PassGraphSAGE(**ck["cfg"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model
