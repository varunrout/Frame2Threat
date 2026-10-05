"""Reproduction gate for the final pass models on ``progression_beyond_pass_k``.

The final models are meant to be trained with early stopping on validation and no
``scale_pos_weight``, and to reproduce EXP-027 (event-only ~0.728, event+360 ~0.729 test
ROC AUC) within rounding.  This script trains each recipe on the unchanged split and records
the numbers; it saves **no** artefact.

Recipes
-------
* ``EXP-027 recipe``: ``TabularClassifier`` exactly as ``scripts/run_pass_label_ablation.py``
  (``scale_pos_weight`` set, ``early_stopping_rounds`` from the config is *not* applied by
  that class, so all 500 trees are kept).
* ``ES, no spw``: the specified rules (early stopping on validation, no weighting).
* ``no ES, no spw``: all 500 trees, no weighting (isolates the early-stopping effect).

Writes ``reports/pass_label/pass_model_repro_gate.md``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.event_features import build_event_features  # noqa: E402
from src.labels.progression_beyond_pass import compute_progression_beyond_pass  # noqa: E402
from src.models.tabular import TabularClassifier  # noqa: E402

warnings.filterwarnings("ignore")
P = Path("data/processed")
LABEL = "progression_beyond_pass_k"


def main() -> None:
    cfg = yaml.safe_load(open("configs/model_baseline.yaml"))["xgboost"]
    passes = pd.read_parquet(P / "pass_instances.parquet")
    passes = compute_progression_beyond_pass(
        pd.read_parquet(P / "events_parsed.parquet"), passes, {}
    )
    idx = passes.set_index("event_uuid")
    parts = {
        s: idx.loc[pd.read_parquet(P / f"{s}.parquet").event_uuid].reset_index()
        for s in ("train", "val", "test")
    }
    geo = pd.read_parquet(P / "geometry_features_v1.parquet")
    X, X360 = {}, {}
    for s, d in parts.items():
        X[s] = build_event_features(d)
        g = geo.reindex(d["event_uuid"].values).reset_index(drop=True)
        g.index = X[s].index
        X360[s] = pd.concat([X[s], g.fillna(0)], axis=1)
    y = {s: d[LABEL].astype(int).to_numpy() for s, d in parts.items()}
    spw = (y["train"] == 0).sum() / y["train"].sum()

    def xgb(**kw):
        return XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric=cfg["eval_metric"],
            random_state=cfg["random_state"],
            n_jobs=-1,
            **kw,
        )

    rows = []
    for fname, Xs in (("event-only (27)", X), ("event+360 (41)", X360)):
        clf = TabularClassifier(model_type="xgboost", task=LABEL, config=cfg)
        clf.fit(Xs["train"], y["train"], X_val=Xs["val"], y_val=y["val"])
        recipes = {
            "EXP-027 recipe (spw, 500 trees)": (clf.predict_proba(Xs["test"])[:, 1], 500, spw)
        }
        for name, kw in (
            ("ES on validation, no spw", dict(early_stopping_rounds=cfg["early_stopping_rounds"])),
            ("no ES, no spw (500 trees)", {}),
        ):
            m = xgb(**kw)
            m.fit(Xs["train"], y["train"], eval_set=[(Xs["val"], y["val"])], verbose=False)
            best = getattr(m, "best_iteration", None)
            recipes[name] = (
                m.predict_proba(Xs["test"])[:, 1],
                (best + 1) if best is not None else 500,
                1.0,
            )
        for name, (p, trees, w) in recipes.items():
            rows.append(
                {
                    "features": fname,
                    "recipe": name,
                    "trees": trees,
                    "scale_pos_weight": f"{w:.2f}",
                    "test ROC AUC": roc_auc_score(y["test"], p),
                    "test AP": average_precision_score(y["test"], p),
                    "test Brier": brier_score_loss(y["test"], p),
                    "mean prediction": float(np.mean(p)),
                }
            )
    df = pd.DataFrame(rows)
    lines = [
        "# Pass model reproduction gate (`progression_beyond_pass_k`)",
        "",
        "Target (EXP-027): event-only ~0.728, event+360 ~0.729 test ROC AUC.  Test prevalence "
        f"{y['test'].mean():.3f}.  Unchanged split; no artefact saved.",
        "",
        "| " + " | ".join(df.columns) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    for r in df.itertuples(index=False):
        lines.append(
            "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |"
        )
    text = "\n".join(lines) + "\n"
    Path("reports/pass_label/pass_model_repro_gate.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
