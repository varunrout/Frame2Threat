"""
gru.py
======
GRU-based sequence classifier for possession-level danger prediction.

Architecture
------------
  Input  : padded sequence  (B, T, input_size=8)
  GRU    : hidden_size=64, num_layers=2, dropout=0.2
  Pool   : last valid hidden state (respects variable lengths)
  Head   : Linear(64 → 32) → ReLU → Dropout → Linear(32 → 1)
  Output : logit (use BCEWithLogitsLoss during training)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PossessionGRU(nn.Module):
    """
    Bidirectional-optional GRU for possession sequence classification.

    Parameters
    ----------
    input_size   : number of features per time step (default 8)
    hidden_size  : GRU hidden units per direction (default 64)
    num_layers   : stacked GRU layers (default 2)
    dropout      : dropout between GRU layers and before head (default 0.2)
    bidirectional: use BiGRU (default False — easier to justify in ablation)
    tab_size     : number of possession-level tabular features to concatenate
                   to the GRU’s last hidden state before the classifier head.
                   0 (default) → pure-sequence mode (original behaviour).
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        tab_size: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.tab_size = tab_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        fc_in = hidden_size * self.num_directions + tab_size
        self.head = nn.Sequential(
            nn.Linear(fc_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, T, input_size)
        lengths: torch.Tensor,  # (B,) true sequence lengths (CPU int64)
        x_tab: "torch.Tensor | None" = None,  # (B, tab_size) or None
    ) -> torch.Tensor:  # (B,) logits
        # Pack for efficiency — skip padding
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)  # h_n: (num_layers * directions, B, hidden)

        # Take the last layer's hidden state
        if self.num_directions == 2:
            h_fwd = h_n[-2]  # (B, hidden)
            h_bwd = h_n[-1]
            h_last = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            h_last = h_n[-1]  # (B, hidden)

        # Hybrid mode: append possession-level tabular features
        if self.tab_size > 0 and x_tab is not None:
            h_last = torch.cat([h_last, x_tab], dim=-1)

        logits = self.head(h_last).squeeze(-1)  # (B,)
        return logits


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def make_dataloader(
    X_seq: "np.ndarray",  # (N, T, F)
    lengths: "np.ndarray",  # (N,)
    y: "np.ndarray",  # (N,)
    X_tab: "np.ndarray | None" = None,  # (N, tab_size) or None
    batch_size: int = 256,
    shuffle: bool = True,
) -> "torch.utils.data.DataLoader":
    """Wrap arrays in a DataLoader.  If X_tab is supplied, each batch is
    (X_seq, lengths, y, X_tab); otherwise (X_seq, lengths, y)."""
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader

    X_t = torch.from_numpy(X_seq.astype("float32"))
    L_t = torch.from_numpy(lengths.astype("int64"))
    y_t = torch.from_numpy(y.astype("float32"))

    if X_tab is not None:
        T_t = torch.from_numpy(X_tab.astype("float32"))
        ds = TensorDataset(X_t, L_t, y_t, T_t)
    else:
        ds = TensorDataset(X_t, L_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_epoch(
    model: PossessionGRU,
    loader: "torch.utils.data.DataLoader",
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        if len(batch) == 4:
            X_b, L_b, y_b, T_b = batch
            T_b = T_b.to(device)
        else:
            X_b, L_b, y_b = batch
            T_b = None
        X_b, L_b, y_b = X_b.to(device), L_b.to(device), y_b.to(device)
        optimiser.zero_grad()
        logits = model(X_b, L_b, T_b)
        loss = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: PossessionGRU,
    loader: "torch.utils.data.DataLoader",
    device: torch.device,
) -> tuple[float, "np.ndarray"]:
    """Return (roc_auc, proba_array)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        if len(batch) == 4:
            X_b, L_b, y_b, T_b = batch
            T_b = T_b.to(device)
        else:
            X_b, L_b, y_b = batch
            T_b = None
        logits = model(X_b.to(device), L_b.to(device), T_b)
        all_logits.append(logits.cpu())
        all_y.append(y_b)

    logits_all = torch.cat(all_logits).numpy()
    y_all = torch.cat(all_y).numpy()
    proba = torch.sigmoid(torch.tensor(logits_all)).numpy()
    auc = roc_auc_score(y_all, proba)
    return auc, proba


def train_gru(
    X_train: "np.ndarray",
    lengths_train: "np.ndarray",
    y_train: "np.ndarray",
    X_val: "np.ndarray",
    lengths_val: "np.ndarray",
    y_val: "np.ndarray",
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    tab_size: int = 0,
    X_tab_train: "np.ndarray | None" = None,
    X_tab_val: "np.ndarray | None" = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 50,
    batch_size: int = 256,
    patience: int = 8,
    device_str: str = "cpu",
    verbose: bool = True,
) -> tuple["PossessionGRU", list[dict]]:
    """
    Full training loop with early stopping.

    Parameters
    ----------
    tab_size     : number of possession-level tabular features (0 = pure-seq).
    X_tab_train  : (N_train, tab_size) float32 array, or None.
    X_tab_val    : (N_val,   tab_size) float32 array, or None.

    Returns
    -------
    best_model : PossessionGRU (weights restored to best val AUC)
    history    : list of dicts with epoch-level metrics
    """
    import numpy as np

    device = torch.device(device_str)

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = PossessionGRU(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        tab_size=tab_size,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=4, min_lr=1e-5
    )

    train_loader = make_dataloader(
        X_train, lengths_train, y_train, X_tab=X_tab_train, batch_size=batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        X_val, lengths_val, y_val, X_tab=X_tab_val, batch_size=batch_size, shuffle=False
    )

    best_val_auc = 0.0
    best_state = None
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, criterion, device)
        val_auc, val_prob = evaluate(model, val_loader, device)
        scheduler.step(1 - val_auc)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:3d} | loss={train_loss:.4f} | val_auc={val_auc:.4f}"
                f"  {'*' if no_improve == 0 else ''}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch} (best val_auc={best_val_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
