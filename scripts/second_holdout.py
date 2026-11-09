"""Second sealed holdout (UEFA Euro 2020) and the position-only boosted baseline, scored once.

Protocol: ``reports/repro/second_holdout_protocol.md``; config: ``configs/second_holdout.json``.

One run fits the pre-registered baselines on the original training matches (selection on validation only),
builds the Euro 2020 inputs with the existing pipeline functions, and scores:

* Euro 2020: XGBoost, GRU, ball x/y logistic and the boosted position baseline, all metrics;
* the 15 in-cohort test matches: the same four, plus the precision operating point;
* Euro 2024 (inputs as already scored): the same four, only to pair the models with the new baseline.

No committed model is retrained, refit or recalibrated. Writes ``reports/repro/second_holdout.json``; refuses to run
if that file exists or the git tree is dirty.

``--dry-run`` exercises every step on the validation matches only (used for tuning anyway) and writes nothing.

Usage:  python scripts/second_holdout.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from sealed_holdout import arrays_from, build_holdout_inputs, ci_table  # noqa: E402

from src.app.batch_scoring import load_models  # noqa: E402
from src.evaluation.event_time_eval import (  # noqa: E402
    bootstrap_by_match,
    metrics,
    one_snapshot_per_possession,
)
from src.evaluation.frozen_metrics import (  # noqa: E402
    ECE_DEFINITION,
    FALSE_ALERT_DEFINITION,
    alert_metrics,
    ece_weighted,
)
from src.evaluation.position_baseline import (  # noqa: E402
    POSITION_FEATURES,
    position_features,
    precision_threshold,
)
from src.evaluation.provenance import file_sha256, run_metadata  # noqa: E402
from src.features.event_time import (  # noqa: E402
    FEATURE_NAMES,
    PossessionArrays,
    build_snapshot_table,
    danger_flags,
    sequence_inputs,
    snapshot_labels,
)
from src.models.event_time import SeqSample, predict_steps  # noqa: E402

PROC = ROOT / "data" / "processed"
CONFIG = ROOT / "configs" / "second_holdout.json"
MODELS = ("XGB", "GRU", "LR_xy", "GBM_pos")
DELTAS = [
    ("XGB", "LR_xy"),
    ("XGB", "GBM_pos"),
    ("GRU", "LR_xy"),
    ("GRU", "GBM_pos"),
    ("GRU", "XGB"),
]


def eligible_rows(arrs: list[PossessionArrays]) -> pd.DataFrame:
    return build_snapshot_table(arrs).query("not terminal").reset_index(drop=True)


def fit_gbm_position(tr: pd.DataFrame, va: pd.DataFrame, cfg: dict, seed: int):
    """Same budget as fit_xgb in scripts/run_event_time.py, on position-only features."""
    xtr, xva = position_features(tr)[POSITION_FEATURES], position_features(va)[POSITION_FEATURES]
    best = None
    grid = cfg["grid"]
    for depth in grid["max_depth"]:
        for mcw in grid["min_child_weight"]:
            m = XGBClassifier(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=depth,
                min_child_weight=mcw,
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                eval_metric="logloss",
                early_stopping_rounds=cfg["early_stopping_rounds"],
                random_state=seed,
                n_jobs=cfg["threads"],
            )
            m.fit(xtr, tr.y, eval_set=[(xva, va.y)], verbose=False)
            if best is None or m.best_score < best[0]:
                best = (
                    float(m.best_score),
                    {
                        "max_depth": depth,
                        "min_child_weight": mcw,
                        "n_estimators": int(m.best_iteration + 1),
                    },
                    m,
                )
    assert best is not None
    return best[2], {**best[1], "val_logloss": best[0]}


def predict(arrs, rows, art, net, lr_xy, gbm) -> dict[str, np.ndarray]:
    samples = []
    for a in arrs:
        eligible, label, _ = snapshot_labels(a)
        tid, num = sequence_inputs(a)
        samples.append(SeqSample(tid, num, eligible, int(label[0])))
    steps = predict_steps(net, samples)
    preds = {
        "XGB": art["model"].predict_proba(rows[FEATURE_NAMES][art["features"]])[:, 1],
        "GRU": np.array([steps[i][t] for i, t in zip(rows.arr_idx, rows.t)]),
        "LR_xy": lr_xy.predict_proba(rows[["cur_x", "cur_y"]])[:, 1],
        "GBM_pos": gbm.predict_proba(position_features(rows)[POSITION_FEATURES])[:, 1],
    }
    for name, p in preds.items():
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"invalid probabilities from {name}")
    return preds


def evaluate(arrs, rows, preds, cfg, thresholds: dict | None) -> dict:
    y = rows.y.to_numpy()
    seed, n_boot = cfg["seed"], cfg["n_boot"]
    one = one_snapshot_per_possession(rows, seed)
    rows_one = rows.iloc[one].reset_index(drop=True)
    preds_one = {m: p[one] for m, p in preds.items()}
    out = {
        "matches": int(rows.match_id.nunique()),
        "eligible_snapshots": int(len(rows)),
        "possessions_with_snapshots": int(rows.groupby(["match_id", "possession_id"]).ngroups),
        "snapshot_prevalence": float(y.mean()),
        "all_snapshots": {m: metrics(y, preds[m]) for m in MODELS},
        "one_per_possession": {m: metrics(rows_one.y.to_numpy(), preds_one[m]) for m in MODELS},
        "bootstrap_all": ci_table(
            bootstrap_by_match(rows, preds, deltas=DELTAS, n_boot=n_boot, seed=seed)
        ),
        "bootstrap_one_per_possession": ci_table(
            bootstrap_by_match(rows_one, preds_one, deltas=DELTAS, n_boot=n_boot, seed=seed)
        ),
        "ece": {m: ece_weighted(y, preds[m], cfg["ece_bins"]) for m in MODELS},
    }
    if thresholds is not None:
        danger_at = {
            (a.match_id, a.possession_id): int(np.argmax(danger_flags(a)))
            for a in arrs
            if danger_flags(a).any()
        }
        out["alerts_validation_prevalence"] = {
            m: alert_metrics(rows, preds[m], thresholds["prevalence"][m.lower()], danger_at)
            for m in ("XGB", "GRU")
        }
        out["alerts_precision_operating_point"] = {
            m: (
                alert_metrics(rows, preds[m], thresholds["precision"][m]["threshold"], danger_at)
                if thresholds["precision"][m]["reachable"]
                else None
            )
            for m in ("XGB", "GRU")
        }
    return out


def git_clean() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip() == ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out = ROOT / cfg["output"]
    if not args.dry_run:
        if out.exists():
            raise SystemExit(f"{out} exists: this holdout is scored once only")
        if not git_clean():
            raise SystemExit("git working tree is dirty: commit or stash before scoring")
    pinned = {**cfg["models"], **cfg["training_inputs_sha256"], **cfg["euro2024_inputs_sha256"]}
    for rel, digest in pinned.items():
        if file_sha256(ROOT / rel) != digest:
            raise SystemExit(f"{rel} differs from the protocol (SHA-256 mismatch)")
    torch.set_num_threads(2)
    art, net, meta = load_models(ROOT)
    prevalence_thr = {"xgb": float(meta["xgb_threshold"]), "gru": float(meta["gru_threshold"])}
    if prevalence_thr != cfg["thresholds_validation_prevalence"]:
        raise SystemExit("model thresholds differ from the protocol")

    # ---- baselines: fit on the 69 training matches, select on the 15 validation matches
    manifest = pd.read_csv(ROOT / "configs/research_split_manifest.csv")
    split_of = dict(zip(manifest.match_id, manifest.split))
    research = arrays_from(PROC)
    by_split = {
        s: [a for a in research if split_of[a.match_id] == s] for s in ("train", "val", "test")
    }
    tr, va = eligible_rows(by_split["train"]), eligible_rows(by_split["val"])
    lr_cfg = cfg["baseline_lr_xy"]
    lr_xy = make_pipeline(StandardScaler(), LogisticRegression(C=lr_cfg["C"], max_iter=2000))
    lr_xy.fit(tr[lr_cfg["features"]], tr.y)
    gbm, gbm_tuned = fit_gbm_position(tr, va, cfg["baseline_gbm_position"], cfg["seed"])
    print("GBM_pos selected on validation:", gbm_tuned, flush=True)

    # ---- precision operating point, chosen on validation only
    p_val = predict(by_split["val"], va, art, net, lr_xy, gbm)
    op = cfg["precision_operating_point"]
    precision_thr = {
        m: precision_threshold(
            va.y.to_numpy(), p_val[m], op["target_precision"], op["min_validation_alert_share"]
        )
        for m in ("XGB", "GRU")
    }
    print("precision operating point (validation):", precision_thr, flush=True)
    thresholds = {"prevalence": prevalence_thr, "precision": precision_thr}

    if args.dry_run:
        res = evaluate(by_split["val"], va, p_val, {**cfg, "n_boot": 20}, thresholds)
        print(
            "dry run on VALIDATION only:",
            {m: round(v["roc_auc"], 4) for m, v in res["all_snapshots"].items()},
        )
        return 0

    targets: dict = {}
    # Euro 2020: build inputs once, then score everything.
    e20_dir = build_holdout_inputs(cfg)
    e20 = arrays_from(e20_dir)
    rows = eligible_rows(e20)
    targets["euro2020"] = evaluate(
        e20, rows, predict(e20, rows, art, net, lr_xy, gbm), cfg, thresholds
    )
    # In-cohort test matches.
    te = by_split["test"]
    rows = eligible_rows(te)
    targets["test"] = evaluate(te, rows, predict(te, rows, art, net, lr_xy, gbm), cfg, thresholds)
    # Euro 2024, inputs exactly as already scored; no new operating point.
    e24 = arrays_from(ROOT / "data/holdout/euro2024")
    rows = eligible_rows(e24)
    targets["euro2024"] = evaluate(e24, rows, predict(e24, rows, art, net, lr_xy, gbm), cfg, None)

    # The models' earlier results must reproduce exactly.
    checks = {}
    for name, rel in (
        ("test", "reports/repro/frozen_metrics_test.json"),
        ("euro2024", "reports/repro/sealed_holdout.json"),
    ):
        old = json.loads((ROOT / rel).read_text(encoding="utf-8"))["results"]["all_snapshots"]
        checks[name] = {
            m: {
                "earlier": old[m]["roc_auc"],
                "now": targets[name]["all_snapshots"][m]["roc_auc"],
                "identical": abs(old[m]["roc_auc"] - targets[name]["all_snapshots"][m]["roc_auc"])
                < 1e-12,
            }
            for m in ("XGB", "GRU", "LR_xy")
        }

    record = {
        "data": {
            "euro2020": {
                "data_dir": cfg["holdout_data_dir"],
                "competition": cfg["competition"],
                "competition_id": cfg["competition_id"],
                "season_id": cfg["season_id"],
                "events_parsed_sha256": file_sha256(e20_dir / "events_parsed.parquet"),
                "possession_sequences_sha256": file_sha256(
                    e20_dir / "possession_sequences.parquet"
                ),
            },
            "test": "15 in-cohort test matches, configs/research_split_manifest.csv, data/processed",
            "euro2024": "data/holdout/euro2024, as scored in reports/repro/sealed_holdout.json",
        },
        "baselines": {
            "LR_xy": {**lr_cfg, "train_snapshots": int(len(tr))},
            "GBM_pos": {
                "features": POSITION_FEATURES,
                "selected_on_validation": gbm_tuned,
                "train_snapshots": int(len(tr)),
                "validation_snapshots": int(len(va)),
                "added": "after the Euro 2024 scoring; frozen before it was scored on any target",
            },
        },
        "definitions": {"false_alert_rate": FALSE_ALERT_DEFINITION, "ece": ECE_DEFINITION},
        "thresholds": thresholds,
        "precision_operating_point_rule": op,
        "reference_auc": cfg["reference_auc"],
        "reproduces_earlier_model_results": checks,
        "results": targets,
        "provenance": {
            "command": "python scripts/second_holdout.py",
            "config": str(args.config.resolve().relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "open_data_revision": cfg["open_data_revision"],
            "seed": cfg["seed"],
            "n_boot": cfg["n_boot"],
            **run_metadata(ROOT, [ROOT / p for p in cfg["models"]]),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                t: {m: round(v["all_snapshots"][m]["roc_auc"], 4) for m in MODELS}
                for t, v in targets.items()
            }
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
