"""EXP-023: possession-level retrospective models (v2) on the frame-corrected labels.

The possession parquet was regenerated after the opponent-coordinate fix
(``src/data/possession_frame.py``).  This script

1. re-evaluates the *saved* v2 XGBoost / GRU on the old and the corrected labels, and
2. retrains both with the unchanged v2 configuration on the corrected labels,

on the unchanged match-level split.  Retrained models are saved under new names; the
original artefacts are left untouched.  Writes ``reports/audit/exp023_v2_corrected_labels.md``.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parse_possessions import load_possession_sequences  # noqa: E402
from src.evaluation.possession_attribution import load_gru_model  # noqa: E402
from src.features.possession_features import (
    build_sequence_tensors,
    build_tabular_features,
)  # noqa: E402
from src.models.gru_possession import evaluate, make_dataloader, train_gru  # noqa: E402

warnings.filterwarnings("ignore")
LABEL = "poss_dangerous"
PROC = Path("data/processed")


def split(poss):
    ids = {
        s: pd.read_parquet(PROC / f"{s}.parquet")["match_id"].unique()
        for s in ("train", "val", "test")
    }
    return {s: poss[poss.match_id.isin(v)].reset_index(drop=True) for s, v in ids.items()}


def gru_probs(model, df):
    X, L = build_sequence_tensors(df)
    loader = make_dataloader(X, L, df[LABEL].astype(int).values, batch_size=256, shuffle=False)
    return evaluate(model, loader, torch.device("cpu"))[1]


def logit_mean(a, b):
    """Equal-weight logit average: the v2 ensemble rule (reports/03_methodology.md, section 6.7)."""

    def lg(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    return 1 / (1 + np.exp(-(0.5 * lg(a) + 0.5 * lg(b))))


def m(y, p):
    return {
        "roc_auc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def main() -> None:
    old = split(load_possession_sequences(PROC / "possession_sequences_pre_frame_fix.parquet"))
    new = split(load_possession_sequences(PROC / "possession_sequences.parquet"))
    cfg = yaml.safe_load(open("configs/model_possession.yaml"))["xgboost_main"]
    gcfg = yaml.safe_load(open("configs/model_gru.yaml"))
    arch, tr = gcfg["architecture"], gcfg["training"]

    rows = []
    # 1. saved models on old / corrected labels
    xgb_old = joblib.load("models/xgboost_poss_dangerous.joblib")
    gru_old, _ = load_gru_model("models/gru_poss_dangerous.pt")
    for tag, d in (("old labels (as published)", old), ("corrected labels", new)):
        y = d["test"][LABEL].astype(int).values
        px = xgb_old.predict_proba(build_tabular_features(d["test"]))[:, 1]
        pg = gru_probs(gru_old, d["test"])
        rows.append(
            {
                "models": "saved v2 (trained on old labels)",
                "test labels": tag,
                "model": "XGBoost",
                **m(y, px),
            }
        )
        rows.append(
            {
                "models": "saved v2 (trained on old labels)",
                "test labels": tag,
                "model": "GRU",
                **m(y, pg),
            }
        )
        rows.append(
            {
                "models": "saved v2 (trained on old labels)",
                "test labels": tag,
                "model": "Ensemble (logit mean)",
                **m(y, logit_mean(px, pg)),
            }
        )

    # 2. retrain with the unchanged v2 configuration, once per label set (same environment,
    #    so old-vs-corrected is a like-for-like comparison)
    for tag, d, save in (("old labels", old, False), ("corrected labels", new, True)):
        Xtr, ytr = build_tabular_features(d["train"]), d["train"][LABEL].astype(int)
        Xva, yva = build_tabular_features(d["val"]), d["val"][LABEL].astype(int)
        Xte = build_tabular_features(d["test"])
        y = d["test"][LABEL].astype(int).values
        xgb = XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),
            eval_metric=cfg["eval_metric"],
            early_stopping_rounds=cfg["early_stopping_rounds"],
            random_state=cfg["random_state"],
            n_jobs=cfg["n_jobs"],
            verbosity=0,
        )
        xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        px = xgb.predict_proba(Xte)[:, 1]
        Xs = {k: build_sequence_tensors(d[k]) for k in ("train", "val")}
        torch.manual_seed(tr["random_seed"])
        gru, _ = train_gru(
            Xs["train"][0],
            Xs["train"][1],
            ytr.values,
            Xs["val"][0],
            Xs["val"][1],
            yva.values,
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            dropout=arch["dropout"],
            tab_size=0,
            lr=tr["lr"],
            n_epochs=tr["n_epochs"],
            batch_size=tr["batch_size"],
            patience=tr["patience"],
            verbose=False,
        )
        pg = gru_probs(gru, d["test"])
        for name, p in (
            ("XGBoost", px),
            ("GRU", pg),
            ("Ensemble (logit mean)", logit_mean(px, pg)),
        ):
            rows.append(
                {"models": f"retrained on {tag}", "test labels": tag, "model": name, **m(y, p)}
            )
        if save:
            joblib.dump(xgb, "models/v2_corrected_xgboost_poss_dangerous.joblib")
            torch.save(
                {"model_state": gru.state_dict(), "config": arch},
                "models/v2_corrected_gru_poss_dangerous.pt",
            )

    df = pd.DataFrame(rows)
    prev = {"old": float(old["test"][LABEL].mean()), "new": float(new["test"][LABEL].mean())}
    lines = [
        "# EXP-023: v2 possession-level models on frame-corrected labels",
        "",
        f"Test possessions: {len(new['test']):,}. Test prevalence old {prev['old']:.3f} -> corrected {prev['new']:.3f}.",
        "",
        "| models | test labels | model | ROC AUC | AP | Brier |",
        "|---|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r.models} | {r._2} | {r.model} | {r.roc_auc:.4f} | {r.ap:.4f} | {r.brier:.4f} |"
        for r in df.rename(columns={"test labels": "_2"}).itertuples()
    ]
    Path("reports/audit/exp023_v2_corrected_labels.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    Path("reports/audit/exp023_v2_corrected_labels.json").write_text(
        json.dumps(rows, indent=1, default=float)
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
