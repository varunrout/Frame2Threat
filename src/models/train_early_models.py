"""
train_early_models.py
=====================
Train and save v3 early-prediction XGBoost models.

Run from project root:
    python src/models/train_early_models.py

Trains four models:
    - Start-only XGBoost      → models/xgboost_start_only.joblib
    - Cumulative XGBoost @25%  → models/xgboost_cumulative_25pct.joblib
    - Cumulative XGBoost @50%  → models/xgboost_cumulative_50pct.joblib
    - Cumulative XGBoost @75%  → models/xgboost_cumulative_75pct.joblib

(100% is already saved as models/xgboost_poss_dangerous.joblib by NB07.)

Hyperparameters are read from configs/model_possession.yaml → xgboost_early.
"""

import sys, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from src.data.parse_possessions import load_possession_sequences
from src.labels.possession_labels import attach_possession_labels
from src.features.early_features import (
    build_start_features,
    build_cumulative_tabular_features,
)

# ── Load config ──────────────────────────────────────────────────────────────
_cfg_path = Path("configs/model_possession.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

XGB_CFG = _cfg["xgboost_early"]
SEED = XGB_CFG["random_state"]
LABEL = "poss_dangerous"

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data …")
poss = load_possession_sequences("data/processed/possession_sequences.parquet")
if "poss_tempo" not in poss.columns:
    print("  Attaching possession labels …")
    poss = attach_possession_labels(poss)

train_ids = pd.read_parquet("data/processed/train.parquet")["match_id"].unique()
val_ids   = pd.read_parquet("data/processed/val.parquet")["match_id"].unique()
test_ids  = pd.read_parquet("data/processed/test.parquet")["match_id"].unique()

train_poss = poss[poss["match_id"].isin(train_ids)].reset_index(drop=True)
val_poss   = poss[poss["match_id"].isin(val_ids)].reset_index(drop=True)
test_poss  = poss[poss["match_id"].isin(test_ids)].reset_index(drop=True)

y_tr = train_poss[LABEL].astype(int).values
y_va = val_poss[LABEL].astype(int).values
y_te = test_poss[LABEL].astype(int).values

print(f"Train: {len(train_poss):,} | Val: {len(val_poss):,} | Test: {len(test_poss):,}")


# ── Helper ───────────────────────────────────────────────────────────────────
def _make_xgb() -> XGBClassifier:
    """Instantiate an XGBClassifier from the YAML config."""
    return XGBClassifier(
        n_estimators=XGB_CFG["n_estimators"],
        max_depth=XGB_CFG["max_depth"],
        learning_rate=XGB_CFG["learning_rate"],
        subsample=XGB_CFG.get("subsample", 1.0),
        colsample_bytree=XGB_CFG.get("colsample_bytree", 1.0),
        min_child_weight=XGB_CFG.get("min_child_weight", 1),
        reg_lambda=XGB_CFG.get("reg_lambda", 1.0),
        objective="binary:logistic",
        eval_metric=XGB_CFG.get("eval_metric", "auc"),
        random_state=SEED,
        n_jobs=XGB_CFG.get("n_jobs", 0),
        verbosity=0,
    )


def _train_save(
    name: str,
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    out_path: Path,
) -> dict:
    """Train an XGBoost model, save it, return test metrics."""
    model = _make_xgb()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    proba = model.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te, proba)
    ap  = average_precision_score(y_te, proba)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

    print(f"  {name:35s}  ROC-AUC={roc:.4f}  AP={ap:.4f}  → {out_path}")
    return {
        "model": "XGBClassifier",
        "name": name,
        "n_features": X_tr.shape[1],
        "n_train": len(X_tr),
        "test_roc_auc": round(roc, 4),
        "test_ap": round(ap, 4),
        "config": XGB_CFG,
        "config_file": str(_cfg_path),
        "artifact": str(out_path),
    }


# ── 1. Start-only model (EXP-015) ───────────────────────────────────────────
print("\n[1/4] Training start-only XGBoost …")
X_tr_start = build_start_features(train_poss).fillna(0)
X_va_start = build_start_features(val_poss).fillna(0)
X_te_start = build_start_features(test_poss).fillna(0)

start_result = _train_save(
    "xgboost_start_only",
    X_tr_start, X_va_start, X_te_start,
    Path("models/xgboost_start_only.joblib"),
)

# ── 2. Cumulative models at 25 / 50 / 75% (EXP-017) ────────────────────────
cumulative_results = []
for i, frac in enumerate([0.25, 0.50, 0.75], start=2):
    pct = int(frac * 100)
    print(f"\n[{i}/4] Training cumulative XGBoost @{pct}% …")
    X_tr_c = build_cumulative_tabular_features(train_poss, frac=frac).fillna(0)
    X_va_c = build_cumulative_tabular_features(val_poss, frac=frac).fillna(0)
    X_te_c = build_cumulative_tabular_features(test_poss, frac=frac).fillna(0)

    res = _train_save(
        f"xgboost_cumulative_{pct}pct",
        X_tr_c, X_va_c, X_te_c,
        Path(f"models/xgboost_cumulative_{pct}pct.joblib"),
    )
    cumulative_results.append(res)

# ── Update results_summary.json ──────────────────────────────────────────────
results_path = Path("models/results_summary.json")
results = json.loads(results_path.read_text()) if results_path.exists() else {}
results["xgboost_start_only"] = start_result
for res in cumulative_results:
    results[res["name"]] = res
results_path.write_text(json.dumps(results, indent=2))

print("\n" + "=" * 60)
print("All v3 early-prediction models saved:")
print(f"  models/xgboost_start_only.joblib       → {start_result['test_roc_auc']}")
for res in cumulative_results:
    print(f"  {res['artifact']:40s} → {res['test_roc_auc']}")
print(f"\nUpdated {results_path}")
