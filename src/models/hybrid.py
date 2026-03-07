"""
Hybrid GNN + sequence model.

Extends PassFrameGNN with a GRU sequence encoder for recent event context.
Fuses: graph_embedding + event_context_embedding + sequence_embedding
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available; HybridGNNSeq model will not function")

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

# Default tasks (shared with gnn.py)
_DEFAULT_TASKS = [
    "line_break",
    "dangerous_progression_k",
    "final_third_entry_k",
    "shot_within_k",
]


if _TORCH_AVAILABLE:
    class HybridGNNSeq(nn.Module):
        """Hybrid model fusing GNN graph embedding with a GRU sequence encoder.

        Architecture
        ------------
        1. **Node projection**: Linear → ReLU maps raw node features to
           ``hidden_dim``.
        2. **GraphSAGE layers**: ``num_layers`` message-passing rounds.
        3. **Global mean pooling**: collapses the node set to a graph
           embedding of size ``hidden_dim``.
        4. **Event MLP**: projects tabular event features to ``hidden_dim``.
        5. **GRU encoder**: encodes the recent-event sequence to a
           ``seq_hidden_dim``-dimensional context vector (last hidden state).
        6. **Fusion MLP**: concatenates the three embeddings and projects
           to ``hidden_dim``.
        7. **Multitask heads**: one linear head per task.

        Parameters
        ----------
        node_feat_dim:
            Number of per-node features (9 by default).
        event_feat_dim:
            Dimensionality of the tabular event-context feature vector.
        seq_feat_dim:
            Dimensionality of each step in the input sequence (e.g. number
            of sequence-context features).
        hidden_dim:
            Width of GNN / MLP hidden layers.
        seq_hidden_dim:
            Hidden state size of the GRU sequence encoder.
        num_layers:
            Number of GraphSAGE layers.
        seq_len:
            Maximum sequence length fed to the GRU.
        dropout:
            Dropout probability.
        num_tasks:
            Number of multitask output heads.
        config:
            Optional extra configuration dict.
        """

        def __init__(
            self,
            node_feat_dim: int = 9,
            event_feat_dim: int = 64,
            seq_feat_dim: int = 7,
            hidden_dim: int = 64,
            seq_hidden_dim: int = 32,
            num_layers: int = 2,
            seq_len: int = 5,
            dropout: float = 0.2,
            num_tasks: int = 4,
            config: dict | None = None,
        ) -> None:
            super().__init__()
            self.node_feat_dim = node_feat_dim
            self.event_feat_dim = event_feat_dim
            self.seq_feat_dim = seq_feat_dim
            self.hidden_dim = hidden_dim
            self.seq_hidden_dim = seq_hidden_dim
            self.num_layers = num_layers
            self.seq_len = seq_len
            self.dropout_p = dropout
            self.num_tasks = num_tasks
            self.config = config or {}
            self.task_names: list[str] = self.config.get("task_names", _DEFAULT_TASKS)[:num_tasks]

            # ---- Graph branch ----
            self.node_proj = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
            )

            self.gnn_layers = nn.ModuleList()
            if _PYG_AVAILABLE:
                for _ in range(num_layers):
                    self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                for _ in range(num_layers):
                    self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))

            self.dropout = nn.Dropout(p=dropout)

            # ---- Event context branch ----
            self.event_proj = nn.Sequential(
                nn.Linear(event_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )

            # ---- Sequence branch ----
            self.gru = nn.GRU(
                input_size=seq_feat_dim,
                hidden_size=seq_hidden_dim,
                num_layers=1,
                batch_first=True,
            )

            # ---- Fusion ----
            fusion_in = hidden_dim + hidden_dim + seq_hidden_dim
            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )

            # ---- Multitask heads ----
            self.heads = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
            ])

        def forward(
            self,
            data: Any,  # torch_geometric.data.Data
            event_features: "torch.Tensor",
            sequence: "torch.Tensor",
        ) -> dict[str, "torch.Tensor"]:
            """Forward pass.

            Parameters
            ----------
            data:
                PyG Data batch with x (node features), edge_index, batch.
            event_features:
                Tabular event features of shape (B, event_feat_dim).
            sequence:
                Recent-event sequence of shape (B, seq_len, seq_feat_dim).

            Returns
            -------
            dict[str, torch.Tensor]
                Per-task logit tensors of shape (B,).
            """
            # -- Graph encoding --
            x, edge_index = data.x, data.edge_index
            batch = data.batch if hasattr(data, "batch") and data.batch is not None \
                else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

            h = self.node_proj(x)
            for layer in self.gnn_layers:
                if _PYG_AVAILABLE:
                    h = layer(h, edge_index)
                else:
                    h = layer(h)
                h = F.relu(h)
                h = self.dropout(h)

            if _PYG_AVAILABLE:
                graph_emb = global_mean_pool(h, batch)
            else:
                B = int(batch.max().item()) + 1
                graph_emb = torch.zeros(B, self.hidden_dim, device=h.device)
                counts = torch.zeros(B, 1, device=h.device)
                graph_emb.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
                counts.scatter_add_(0, batch.unsqueeze(1), torch.ones(len(batch), 1, device=h.device))
                graph_emb = graph_emb / counts.clamp(min=1)

            # -- Event context encoding --
            ev_emb = self.event_proj(event_features)  # (B, hidden_dim)

            # -- Sequence encoding --
            _, h_n = self.gru(sequence)           # h_n: (1, B, seq_hidden_dim)
            seq_emb = h_n.squeeze(0)              # (B, seq_hidden_dim)

            # -- Fusion --
            fused = torch.cat([graph_emb, ev_emb, seq_emb], dim=-1)
            fused = self.fusion(fused)            # (B, hidden_dim)

            return {
                name: self.heads[i](fused).squeeze(-1)
                for i, name in enumerate(self.task_names)
            }

        def save(self, path: str | pathlib.Path) -> None:
            """Persist model weights and hyperparameters to *path*."""
            path = pathlib.Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "node_feat_dim": self.node_feat_dim,
                    "event_feat_dim": self.event_feat_dim,
                    "seq_feat_dim": self.seq_feat_dim,
                    "hidden_dim": self.hidden_dim,
                    "seq_hidden_dim": self.seq_hidden_dim,
                    "num_layers": self.num_layers,
                    "seq_len": self.seq_len,
                    "dropout": self.dropout_p,
                    "num_tasks": self.num_tasks,
                    "config": self.config,
                },
                path,
            )
            logger.info("Saved HybridGNNSeq to %s", path)

        @classmethod
        def load(
            cls,
            path: str | pathlib.Path,
            device: str = "cpu",
        ) -> "HybridGNNSeq":
            """Load a :class:`HybridGNNSeq` from a checkpoint file."""
            path = pathlib.Path(path)
            ckpt = torch.load(path, map_location=device)
            model = cls(
                node_feat_dim=ckpt["node_feat_dim"],
                event_feat_dim=ckpt["event_feat_dim"],
                seq_feat_dim=ckpt["seq_feat_dim"],
                hidden_dim=ckpt["hidden_dim"],
                seq_hidden_dim=ckpt["seq_hidden_dim"],
                num_layers=ckpt["num_layers"],
                seq_len=ckpt["seq_len"],
                dropout=ckpt["dropout"],
                num_tasks=ckpt["num_tasks"],
                config=ckpt.get("config"),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            logger.info("Loaded HybridGNNSeq from %s", path)
            return model

else:
    class HybridGNNSeq:  # type: ignore[no-redef]
        """Stub – PyTorch not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for HybridGNNSeq")
