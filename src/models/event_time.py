"""
event_time.py (models)
======================
Models for the event-time (in-play) danger formulation.

* :class:`EventTimeGRU` -- a *causal* GRU (1 layer, hidden 64, as small as the v2 GRU) with
  a learned embedding for the event type (replacing the raw numeric ``type_id`` of v2).
  It emits one logit per event; the output at step ``t`` depends only on events ``0..t``,
  so one forward pass over a possession scores every prefix.
* helpers to train it on eligible snapshot steps and to predict per-step probabilities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from torch.nn.utils.rnn import pad_sequence

from src.data.event_types import N_EVENT_TYPES


class EventTimeGRU(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        embed_dim: int = 8,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(N_EVENT_TYPES, embed_dim)
        self.gru = nn.GRU(embed_dim + n_numeric, hidden_size, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, type_ids: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        """type_ids (B, T) long, numeric (B, T, F) -> logits (B, T)."""
        h, _ = self.gru(torch.cat([self.embed(type_ids), numeric], dim=-1))
        return self.head(self.drop(h)).squeeze(-1)


@dataclass
class SeqSample:
    """One possession: model inputs plus the per-step training mask / label."""

    type_ids: np.ndarray  # (n,)
    numeric: np.ndarray  # (n, F)
    eligible: np.ndarray  # (n,) bool
    label: int


def _collate(batch: Sequence[SeqSample]):
    types = pad_sequence([torch.from_numpy(b.type_ids) for b in batch], batch_first=True)
    num = pad_sequence([torch.from_numpy(b.numeric) for b in batch], batch_first=True)
    mask = pad_sequence([torch.from_numpy(b.eligible) for b in batch], batch_first=True)
    y = torch.tensor([float(b.label) for b in batch])
    return types, num, mask, y


def predict_steps(
    model: EventTimeGRU, samples: Sequence[SeqSample], batch_size: int = 256
) -> list[np.ndarray]:
    """Per-step probabilities (length n each) for every possession."""
    model.eval()
    out: list[np.ndarray] = []
    order = np.argsort([len(s.type_ids) for s in samples])
    res: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            ids = order[i : i + batch_size]
            types, num, _, _ = _collate([samples[j] for j in ids])
            probs = torch.sigmoid(model(types, num)).numpy()
            for row, j in enumerate(ids):
                res[int(j)] = probs[row, : len(samples[j].type_ids)]
    out = [res[j] for j in range(len(samples))]
    return out


def _val_logloss(model: EventTimeGRU, samples: Sequence[SeqSample]) -> float:
    probs = predict_steps(model, samples)
    p = np.concatenate([pr[s.eligible] for pr, s in zip(probs, samples)])
    y = np.concatenate([np.full(int(s.eligible.sum()), s.label) for s in samples])
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def train_event_time_gru(
    train: Sequence[SeqSample],
    val: Sequence[SeqSample],
    n_numeric: int,
    *,
    lr: float = 3e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 40,
    patience: int = 5,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[EventTimeGRU, dict]:
    """Train on eligible steps (BCE); keep the epoch with the lowest validation log loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EventTimeGRU(n_numeric)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best, best_state, bad, hist = np.inf, None, 0, []
    # bucket by length so padding stays small, shuffle bucket order each epoch
    lens = np.array([len(s.type_ids) for s in train])
    for epoch in range(max_epochs):
        model.train()
        idx = np.argsort(lens + rng.uniform(0, 4, size=len(lens)))
        batches = [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]
        rng.shuffle(batches)
        tot = 0.0
        for b in batches:
            types, num, mask, y = _collate([train[j] for j in b])
            logits = model(types, num)
            loss_el = bce(logits, y[:, None].expand_as(logits))
            loss = (loss_el * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        v = _val_logloss(model, val)
        hist.append({"epoch": epoch + 1, "train_loss": tot / len(batches), "val_logloss": v})
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {
        "best_val_logloss": best,
        "history": hist,
        "lr": lr,
        "weight_decay": weight_decay,
    }
