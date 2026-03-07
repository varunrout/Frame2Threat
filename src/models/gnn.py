"""
Graph Neural Network model for pass-frame representation.

Architecture:
  - Node feature projection (Linear + ReLU)
  - 2-3 GraphSAGE / GAT layers
  - Global mean pooling
  - Concatenate with event-context embedding
  - Multitask classification heads (one per label)
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
    logger.warning("PyTorch not available; GNN model will not function")

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    if _TORCH_AVAILABLE:
        logger.warning("torch_geometric not available; falling back to identity layers")

# Task labels supported by the multitask heads
_DEFAULT_TASKS = [
    "line_break",
    "dangerous_progression_k",
    "final_third_entry_k",
    "shot_within_k",
]


if _TORCH_AVAILABLE:
    class PassFrameGNN(nn.Module):
        """GNN model that encodes a pass freeze-frame graph.

        Combines graph-level pooled embeddings with pass event features to
        produce per-task threat predictions.

        Parameters
        ----------
        node_feat_dim:
            Dimensionality of per-node input features (9 by default).
        event_feat_dim:
            Dimensionality of the tabular event-context feature vector.
        hidden_dim:
            Width of hidden layers.
        num_layers:
            Number of GraphSAGE message-passing layers (2 or 3).
        dropout:
            Dropout probability applied after each GNN layer.
        num_tasks:
            Number of output classification heads.
        config:
            Optional extra configuration (currently unused, reserved for
            future architectural variants such as GAT attention heads).
        """

        def __init__(
            self,
            node_feat_dim: int = 9,
            event_feat_dim: int = 64,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            num_tasks: int = 4,
            config: dict | None = None,
        ) -> None:
            super().__init__()
            self.node_feat_dim = node_feat_dim
            self.event_feat_dim = event_feat_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout_p = dropout
            self.num_tasks = num_tasks
            self.config = config or {}
            self.task_names: list[str] = self.config.get("task_names", _DEFAULT_TASKS)[:num_tasks]

            # Node feature projection
            self.node_proj = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
            )

            # GNN layers (GraphSAGE)
            self.gnn_layers = nn.ModuleList()
            if _PYG_AVAILABLE:
                for _ in range(num_layers):
                    self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                # Fallback: simple linear layers (no message passing)
                for _ in range(num_layers):
                    self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))

            self.dropout = nn.Dropout(p=dropout)

            # Event context MLP
            self.event_proj = nn.Sequential(
                nn.Linear(event_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )

            # Fusion + classification heads
            fusion_dim = hidden_dim * 2  # graph_emb || event_emb
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )

            self.heads = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
            ])

        def forward(
            self,
            data: Any,  # torch_geometric.data.Data
            event_features: "torch.Tensor",
        ) -> dict[str, "torch.Tensor"]:
            """Forward pass.

            Parameters
            ----------
            data:
                PyG Data object with attributes: x (N×node_feat_dim),
                edge_index (2×E), batch (N,).
            event_features:
                Batch of event-context features, shape (B, event_feat_dim).

            Returns
            -------
            dict[str, torch.Tensor]
                Mapping of task name → logit tensor of shape (B,).
            """
            x, edge_index = data.x, data.edge_index
            batch = data.batch if hasattr(data, "batch") and data.batch is not None \
                else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

            # Node projection
            h = self.node_proj(x)

            # Message passing
            for layer in self.gnn_layers:
                if _PYG_AVAILABLE:
                    h = layer(h, edge_index)
                else:
                    h = layer(h)
                h = F.relu(h)
                h = self.dropout(h)

            # Global pooling
            if _PYG_AVAILABLE:
                graph_emb = global_mean_pool(h, batch)  # (B, hidden_dim)
            else:
                # Manual mean pooling via batch indices
                B = int(batch.max().item()) + 1
                graph_emb = torch.zeros(B, self.hidden_dim, device=h.device)
                counts = torch.zeros(B, 1, device=h.device)
                graph_emb.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
                counts.scatter_add_(0, batch.unsqueeze(1), torch.ones(len(batch), 1, device=h.device))
                graph_emb = graph_emb / counts.clamp(min=1)

            # Event context embedding
            ev_emb = self.event_proj(event_features)  # (B, hidden_dim)

            # Fuse
            fused = torch.cat([graph_emb, ev_emb], dim=-1)
            fused = self.fusion(fused)  # (B, hidden_dim)

            # Per-task logits
            return {
                name: self.heads[i](fused).squeeze(-1)
                for i, name in enumerate(self.task_names)
            }

    class GNNTrainer:
        """Training and evaluation harness for :class:`PassFrameGNN`.

        Parameters
        ----------
        model:
            Instantiated :class:`PassFrameGNN` model.
        config:
            Training configuration dict.  Keys: lr (float, default 1e-3),
            weight_decay (float, default 1e-4), task_weights (dict mapping
            task name to float), device (str, default 'cpu').
        """

        def __init__(self, model: "PassFrameGNN", config: dict) -> None:
            self.model = model
            self.config = config
            self.device = torch.device(config.get("device", "cpu"))
            self.model.to(self.device)

            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(config.get("lr", 1e-3)),
                weight_decay=float(config.get("weight_decay", 1e-4)),
            )

            # Per-task loss weights (default uniform)
            self.task_weights: dict[str, float] = config.get(
                "task_weights",
                {t: 1.0 for t in model.task_names},
            )

        def train(
            self,
            train_loader: Any,
            val_loader: Any | None = None,
            epochs: int = 50,
        ) -> dict[str, list[float]]:
            """Train the model for *epochs* epochs.

            Parameters
            ----------
            train_loader:
                PyG DataLoader yielding batches with x, edge_index, batch,
                event_features, and per-task label attributes.
            val_loader:
                Optional validation loader.
            epochs:
                Number of training epochs.

            Returns
            -------
            dict[str, list[float]]
                History dict with keys 'train_loss' and (optionally)
                'val_loss', each mapping to a list of per-epoch values.
            """
            history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

            for epoch in range(1, epochs + 1):
                self.model.train()
                epoch_loss = 0.0
                n_batches = 0

                for batch in train_loader:
                    batch = batch.to(self.device)
                    event_feat = batch.event_features.to(self.device)

                    self.optimizer.zero_grad()
                    logits = self.model(batch, event_feat)
                    loss = self._multitask_loss(logits, batch)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_train = epoch_loss / max(n_batches, 1)
                history["train_loss"].append(avg_train)

                if val_loader is not None:
                    val_loss = self._eval_loss(val_loader)
                    history["val_loss"].append(val_loss)
                    logger.info(
                        "Epoch %d/%d – train_loss=%.4f  val_loss=%.4f",
                        epoch, epochs, avg_train, val_loss,
                    )
                else:
                    logger.info("Epoch %d/%d – train_loss=%.4f", epoch, epochs, avg_train)

            return history

        def evaluate(self, loader: Any) -> dict[str, float]:
            """Evaluate the model on *loader*, returning per-task AUC.

            Returns
            -------
            dict[str, float]
                Keys: task_name → AUC score (or -1 if sklearn unavailable).
            """
            from sklearn.metrics import roc_auc_score

            self.model.eval()
            all_logits: dict[str, list[float]] = {t: [] for t in self.model.task_names}
            all_labels: dict[str, list[float]] = {t: [] for t in self.model.task_names}

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    event_feat = batch.event_features.to(self.device)
                    logits = self.model(batch, event_feat)

                    for task in self.model.task_names:
                        probs = torch.sigmoid(logits[task]).cpu().numpy()
                        all_logits[task].extend(probs.tolist())
                        if hasattr(batch, task):
                            labels = getattr(batch, task).cpu().numpy()
                            all_labels[task].extend(labels.tolist())

            metrics: dict[str, float] = {}
            for task in self.model.task_names:
                y_true = np.array(all_labels[task])
                y_score = np.array(all_logits[task])
                if len(y_true) > 0 and len(np.unique(y_true)) == 2:
                    metrics[task] = float(roc_auc_score(y_true, y_score))
                else:
                    metrics[task] = -1.0
            return metrics

        def _multitask_loss(
            self,
            logits: dict[str, "torch.Tensor"],
            batch: Any,
        ) -> "torch.Tensor":
            """Compute weighted sum of BCE losses across tasks."""
            total = torch.tensor(0.0, device=self.device)
            for task, logit in logits.items():
                if not hasattr(batch, task):
                    continue
                y = getattr(batch, task).float().to(self.device)
                mask = ~torch.isnan(y)
                if mask.sum() == 0:
                    continue
                loss = F.binary_cross_entropy_with_logits(
                    logit[mask], y[mask]
                )
                weight = self.task_weights.get(task, 1.0)
                total = total + weight * loss
            return total

        def _eval_loss(self, loader: Any) -> float:
            self.model.eval()
            total = 0.0
            n = 0
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    event_feat = batch.event_features.to(self.device)
                    logits = self.model(batch, event_feat)
                    loss = self._multitask_loss(logits, batch)
                    total += loss.item()
                    n += 1
            return total / max(n, 1)

        def save(self, path: str | pathlib.Path) -> None:
            """Save model weights and config to *path*."""
            path = pathlib.Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                    "node_feat_dim": self.model.node_feat_dim,
                    "event_feat_dim": self.model.event_feat_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "num_layers": self.model.num_layers,
                    "dropout": self.model.dropout_p,
                    "num_tasks": self.model.num_tasks,
                    "model_config": self.model.config,
                },
                path,
            )
            logger.info("Saved GNNTrainer model to %s", path)

        @classmethod
        def load(
            cls,
            path: str | pathlib.Path,
            device: str = "cpu",
        ) -> "GNNTrainer":
            """Load a :class:`GNNTrainer` from a checkpoint file."""
            path = pathlib.Path(path)
            checkpoint = torch.load(path, map_location=device)
            model = PassFrameGNN(
                node_feat_dim=checkpoint["node_feat_dim"],
                event_feat_dim=checkpoint["event_feat_dim"],
                hidden_dim=checkpoint["hidden_dim"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
                num_tasks=checkpoint["num_tasks"],
                config=checkpoint.get("model_config"),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            config = checkpoint["config"]
            config["device"] = device
            trainer = cls(model, config)
            logger.info("Loaded GNNTrainer from %s", path)
            return trainer

else:
    # Stub implementations when PyTorch is not available

    class PassFrameGNN:  # type: ignore[no-redef]
        """Stub – PyTorch not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for PassFrameGNN")

    class GNNTrainer:  # type: ignore[no-redef]
        """Stub – PyTorch not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for GNNTrainer")
