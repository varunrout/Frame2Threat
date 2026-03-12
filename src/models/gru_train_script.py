"""
gru_train_script.py
===================
Standalone training script for PossessionGRU.
Run from project root:  python src/models/gru_train_script.py

Architecture and training hyperparameters are read from configs/model_gru.yaml.
"""
import sys, warnings, json
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

from src.data.parse_possessions import load_possession_sequences
from src.features.possession_features import build_sequence_tensors, build_tabular_features
from src.labels.possession_labels import attach_possession_labels
from src.models.gru_possession import train_gru, evaluate, make_dataloader

# ── Load config ──────────────────────────────────────────────────────────────
_cfg_path = Path("configs/model_gru.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

ARCH  = _cfg["architecture"]   # input_size, hidden_size, num_layers, dropout, bidirectional
TRAIN = _cfg["training"]       # lr, n_epochs, batch_size, patience, …

LABEL      = TRAIN["label"]
TAB_FEATS  = ARCH.get("tab_features", [])    # list of tabular col names; [] = pure-seq
TAB_SIZE   = ARCH.get("tab_size", 0)         # must equal len(TAB_FEATS) when non-zero
HYBRID     = bool(TAB_FEATS) and TAB_SIZE > 0

print("Loading data …")
poss = load_possession_sequences("data/processed/possession_sequences.parquet")

# Attach new possession labels in case the parquet was built before they existed.
# attach_possession_labels is idempotent: it skips columns already present.
if "poss_tempo" not in poss.columns:
    print("  Attaching possession labels (parquet pre-dates possession_labels.py) …")
    poss = attach_possession_labels(poss)

train_ids = pd.read_parquet("data/processed/train.parquet")["match_id"].unique()
val_ids   = pd.read_parquet("data/processed/val.parquet")["match_id"].unique()
test_ids  = pd.read_parquet("data/processed/test.parquet")["match_id"].unique()

tr = poss[poss["match_id"].isin(train_ids)].reset_index(drop=True)
va = poss[poss["match_id"].isin(val_ids)].reset_index(drop=True)
te = poss[poss["match_id"].isin(test_ids)].reset_index(drop=True)

X_tr, L_tr = build_sequence_tensors(tr);  y_tr = tr[LABEL].astype(int).values
X_va, L_va = build_sequence_tensors(va);  y_va = va[LABEL].astype(int).values
X_te, L_te = build_sequence_tensors(te);  y_te = te[LABEL].astype(int).values

# --- Hybrid tabular features (optional) ---
if HYBRID:
    print(f"Hybrid mode: building {TAB_SIZE} tabular features: {TAB_FEATS}")
    def _tab(df):
        full = build_tabular_features(df)
        return full[TAB_FEATS].fillna(0).values.astype("float32")
    X_tab_tr = _tab(tr)
    X_tab_va = _tab(va)
    X_tab_te = _tab(te)
else:
    X_tab_tr = X_tab_va = X_tab_te = None

print(f"Train {len(tr):,} | Val {len(va):,} | Test {len(te):,}")

print(f"\nTraining GRU … (config: {_cfg_path}, hybrid={HYBRID})")
model, history = train_gru(
    X_tr, L_tr, y_tr,
    X_va, L_va, y_va,
    hidden_size=ARCH["hidden_size"],
    num_layers=ARCH["num_layers"],
    dropout=ARCH["dropout"],
    tab_size=TAB_SIZE,
    X_tab_train=X_tab_tr,
    X_tab_val=X_tab_va,
    lr=TRAIN["lr"],
    n_epochs=TRAIN["n_epochs"],
    batch_size=TRAIN["batch_size"],
    patience=TRAIN["patience"],
    verbose=True,
)

test_loader = make_dataloader(X_te, L_te, y_te, X_tab=X_tab_te,
                              batch_size=TRAIN["test_batch_size"], shuffle=False)
test_auc, test_proba = evaluate(model, test_loader, torch.device("cpu"))
test_ap = average_precision_score(y_te, test_proba)

print(f"\nTest ROC-AUC : {test_auc:.4f}")
print(f"Test AP      : {test_ap:.4f}")
print()
print(classification_report(y_te, (test_proba >= 0.5).astype(int),
                             target_names=["safe", "dangerous"]))

# Save model — architecture config embedded so inference never needs the YAML
out_pt = Path("models/gru_poss_dangerous.pt")
torch.save({
    "model_state": model.state_dict(),
    "config": ARCH,
    "tab_features": TAB_FEATS,
    "test_roc_auc": test_auc,
    "test_ap": test_ap,
}, out_pt)
print(f"Saved → {out_pt}")

# Update results summary
results_path = Path("models/results_summary.json")
results = json.loads(results_path.read_text()) if results_path.exists() else {}
results["gru_poss_dangerous"] = {
    "model": "PossessionGRU",
    "unit": "possession",
    "label": LABEL,
    "config": ARCH,
    "config_file": str(_cfg_path),
    "hybrid": HYBRID,
    "tab_features": TAB_FEATS,
    "n_train": len(tr), "n_val": len(va), "n_test": len(te),
    "val_roc_auc": round(float(max(h["val_auc"] for h in history)), 4),
    "test_roc_auc": round(test_auc, 4),
    "test_ap": round(test_ap, 4),
}
results_path.write_text(json.dumps(results, indent=2))
print("Updated models/results_summary.json")

xgb = results.get("xgboost_poss_dangerous", {}).get("test_roc_auc", "N/A")
print(f"\nv2 Summary:")
print(f"  XGBoost possession tabular : {xgb}")
print(f"  GRU     possession sequence: {round(test_auc, 4)}")
