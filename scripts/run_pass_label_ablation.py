"""EXP-024/025: pass-level models and the 360 ablation on BOTH labels.

* ``dangerous_progression_k``  -- original label, unchanged (self-determining: see
  reports/audit/defect_audit_before.md, section 4)
* ``progression_beyond_pass_k`` -- new label that ignores the pass's own receipt

Models (unchanged v1 configuration, same match-level split): logistic regression and XGBoost
on the 27 event features, XGBoost on 27 + 14 geometry features (event+360), an end_x-only
logistic baseline and an (end_x, end_y) XGBoost baseline.  Writes ``reports/pass_label/``.

Usage:  python scripts/run_pass_label_ablation.py [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.event_time_eval import bootstrap_by_match, metrics  # noqa: E402
from src.features.event_features import build_event_features  # noqa: E402
from src.features.geometry_features import build_geometry_features  # noqa: E402
from src.labels.progression_beyond_pass import compute_progression_beyond_pass  # noqa: E402
from src.models.tabular import TabularClassifier  # noqa: E402

warnings.filterwarnings("ignore")
PROC = Path("data/processed")
OUT = Path("reports/pass_label")
LABELS = ["dangerous_progression_k", "progression_beyond_pass_k"]
MODELS = ["LR_event", "XGB_event", "XGB_event360", "LR_end_x", "XGB_end_xy"]
DESCR = {
    "LR_event": "Logistic regression, 27 event features",
    "XGB_event": "XGBoost, 27 event features",
    "XGB_event360": "XGBoost, 27 event + 14 geometry (360) features",
    "LR_end_x": "Baseline: logistic regression on end_x only",
    "XGB_end_xy": "Baseline: XGBoost on (end_x, end_y) only",
}


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for r in df.itertuples(index=False):
        out += (
            "| "
            + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r)
            + " |\n"
        )
    return out


def geometry(passes: pd.DataFrame) -> pd.DataFrame:
    cache = PROC / "geometry_features_v1.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = pd.read_parquet(PROC / "frames_360.parquet")
    g = build_geometry_features(passes, frames)
    g.to_parquet(cache)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open("configs/model_baseline.yaml"))
    passes = pd.read_parquet(PROC / "pass_instances.parquet")
    events = pd.read_parquet(PROC / "events_parsed.parquet")
    labels_cfg = yaml.safe_load(open("configs/labels.yaml")).get("progression_beyond_pass", {})
    passes = compute_progression_beyond_pass(events, passes, labels_cfg)

    split_uuids = {
        s: pd.read_parquet(PROC / f"{s}.parquet")["event_uuid"] for s in ("train", "val", "test")
    }
    assert sum(len(v) for v in split_uuids.values()) == len(passes)
    idx = passes.set_index("event_uuid")
    parts = {s: idx.loc[u].reset_index() for s, u in split_uuids.items()}
    tr, va, te = parts["train"], parts["val"], parts["test"]

    X = {s: build_event_features(d) for s, d in parts.items()}
    G = geometry(passes)
    X360 = {}
    for s, d in parts.items():
        geo = G.reindex(d["event_uuid"].values).reset_index(drop=True)
        geo.index = X[s].index
        X360[s] = pd.concat([X[s], geo.fillna(0)], axis=1)
    print("features:", X["train"].shape[1], X360["train"].shape[1])
    ENDX = {s: parts[s][["end_x"]].fillna(parts["train"]["end_x"].median()) for s in parts}
    ENDXY = {
        s: parts[s][["end_x", "end_y"]].fillna(parts["train"][["end_x", "end_y"]].median())
        for s in parts
    }

    results: dict = {
        "n": {s: len(d) for s, d in parts.items()},
        "test_matches": int(te.match_id.nunique()),
    }
    sections: list[str] = []
    for label in LABELS:
        ytr, yva, yte = (d[label].astype(int).to_numpy() for d in (tr, va, te))
        preds: dict[str, np.ndarray] = {}

        lr = make_pipeline(
            StandardScaler(), LogisticRegression(C=cfg["logistic_regression"]["C"], max_iter=5000)
        )
        lr.fit(X["train"], ytr)
        preds["LR_event"] = lr.predict_proba(X["test"])[:, 1]

        for name, Xs in (("XGB_event", X), ("XGB_event360", X360)):
            clf = TabularClassifier(model_type="xgboost", task=label, config=cfg["xgboost"])
            clf.fit(Xs["train"], ytr, X_val=Xs["val"], y_val=yva)
            preds[name] = clf.predict_proba(Xs["test"])[:, 1]

        lr_end = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        lr_end.fit(ENDX["train"], ytr)
        preds["LR_end_x"] = lr_end.predict_proba(ENDX["test"])[:, 1]
        clf_xy = TabularClassifier(
            model_type="xgboost", task=label, config={**cfg["xgboost"], "max_depth": 3}
        )
        clf_xy.fit(ENDXY["train"], ytr, X_val=ENDXY["val"], y_val=yva)
        preds["XGB_end_xy"] = clf_xy.predict_proba(ENDXY["test"])[:, 1]

        rows = te[["match_id"]].copy()
        rows["y"] = yte
        deltas = [
            ("XGB_event360", "XGB_event"),
            ("XGB_event", "LR_end_x"),
            ("XGB_event", "XGB_end_xy"),
        ]
        boot = bootstrap_by_match(rows, preds, deltas=deltas, n_boot=args.n_boot, seed=42)
        sub = te["has_360"].astype(bool).to_numpy()
        boot360 = bootstrap_by_match(
            rows[sub].reset_index(drop=True),
            {m: p[sub] for m, p in preds.items() if m.startswith("XGB_event")},
            deltas=[("XGB_event360", "XGB_event")],
            n_boot=args.n_boot,
            seed=42,
        )
        point = pd.DataFrame([{"model": m, **metrics(yte, preds[m])} for m in MODELS])
        point360 = pd.DataFrame(
            [
                {"model": m, **metrics(yte[sub], preds[m][sub])}
                for m in MODELS
                if m.startswith("XGB_event")
            ]
        )

        endx = te["end_x"].to_numpy()
        hi = endx >= 80
        tauto = {
            "prevalence_test": float(yte.mean()),
            "P(label | end_x >= 80)": float(yte[hi].mean()),
            "P(label | end_x < 80)": float(yte[~hi].mean()),
            "share of positives with end_x >= 80": float(hi[yte == 1].mean()),
            "ROC AUC of end_x alone": float(roc_auc_score(yte, endx)),
        }
        results[label] = {
            "point": point.to_dict("records"),
            "point_360_subset": point360.to_dict("records"),
            "bootstrap": boot.to_dict("records"),
            "bootstrap_360_subset": boot360.to_dict("records"),
            "self_determination": tauto,
            "n_test_360": int(sub.sum()),
        }

        def ci(df, model, stat):
            r = df[(df.model == model) & (df.stat == stat)].iloc[0]
            return f"{r.estimate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]"

        sec = [
            f"## Label: `{label}`",
            "",
            f"Test passes: {len(te):,} ({int(sub.sum()):,} with 360); test matches: {te.match_id.nunique()}.",
            "",
            "How much does the pass's own destination decide the label (test split)?",
            "",
            *[f"- {k}: {v:.3f}" for k, v in tauto.items()],
            "",
            "### Test metrics (all test passes)",
            "",
            md_table(point),
            "",
            f"### 360 delta, event+360 minus event-only XGBoost ({args.n_boot} match-level bootstrap resamples, 95% CI)",
            "",
            "| subset | ROC AUC | average precision | Brier |",
            "|---|---|---|---|",
            "| all test passes | "
            + " | ".join(
                ci(boot, "XGB_event360 - XGB_event", s)
                for s in ("roc_auc", "avg_precision", "brier")
            )
            + " |",
            "| passes with 360 only | "
            + " | ".join(
                ci(boot360, "XGB_event360 - XGB_event", s)
                for s in ("roc_auc", "avg_precision", "brier")
            )
            + " |",
            "",
            "### Test metrics (passes with 360 only)",
            "",
            md_table(point360),
            "",
            "### Event model vs destination-only baselines (paired bootstrap, 95% CI)",
            "",
            "| comparison | ROC AUC | average precision | Brier |",
            "|---|---|---|---|",
        ]
        for a, b in (("XGB_event", "LR_end_x"), ("XGB_event", "XGB_end_xy")):
            sec.append(
                f"| {a} - {b} | "
                + " | ".join(
                    ci(boot, f"{a} - {b}", s) for s in ("roc_auc", "avg_precision", "brier")
                )
                + " |"
            )
        sec += ["", "### 95% intervals per model", ""]
        t = boot[~boot.model.str.contains(" - ")].copy()
        t["95% CI"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t.ci_low, t.ci_high)]
        sec += [md_table(t[["model", "stat", "estimate", "95% CI"]]), ""]
        sections += sec

    head = (
        [
            "# Pass-level models and the 360 ablation on both labels",
            "",
            "Generated by `scripts/run_pass_label_ablation.py`; unchanged match-level split (seed 42); "
            "XGBoost configuration from `configs/model_baseline.yaml` (early stopping on validation).",
            "",
            f"Passes train/val/test: {results['n']}.",
            "",
            "Models:",
            "",
        ]
        + [f"- **{m}**: {DESCR[m]}" for m in MODELS]
        + [""]
    )
    (OUT / "pass_label_results.md").write_text("\n".join(head + sections) + "\n", encoding="utf-8")
    (OUT / "pass_label_results.json").write_text(
        json.dumps(results, indent=1, default=float), encoding="utf-8"
    )
    print("\n".join(head + sections))


if __name__ == "__main__":
    main()
