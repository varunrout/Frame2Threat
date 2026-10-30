"""Check that the saved model artefacts reproduce the numbers in the committed reports.

Reloads each saved model, scores the held-out test matches and compares the point estimate with the
value recorded in ``reports/`` (tolerance 5e-4 ROC AUC).  Writes ``reports/repro/artefact_check.md``
and exits non-zero on any mismatch.  Needs the gitignored ``data/processed`` and ``models`` files.

Usage:  python scripts/verify_artefacts.py [--tol 5e-4]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.app.batch_scoring import load_models  # noqa: E402
from src.features.event_time import (  # noqa: E402
    FEATURE_NAMES,
    build_snapshot_table,
    sequence_inputs,
    snapshot_labels,
)
from src.models.event_time import SeqSample, predict_steps  # noqa: E402

PROC = ROOT / "data" / "processed"


def check_event_time(tol: float) -> list[tuple[str, float, float, bool]]:
    test_ids = set(pd.read_csv(PROC / "split_manifest.csv").query("split == 'test'")["match_id"])
    arrs = pickle.loads((PROC / "event_time_arrays.pkl").read_bytes())
    arrs = [a for a in arrs if a.match_id in test_ids]
    art, net, _ = load_models(ROOT)
    snaps = build_snapshot_table(arrs)
    keep = ~snaps["terminal"].to_numpy().astype(bool)
    snaps = snaps[keep].reset_index(drop=True)
    y = snaps["y"].astype(int).to_numpy()
    p_x = art["model"].predict_proba(snaps[FEATURE_NAMES][art["features"]])[:, 1]
    samples = []
    for p in arrs:
        elig, lab, _ = snapshot_labels(p)
        tid, num = sequence_inputs(p)
        samples.append(SeqSample(tid, num, elig, int(lab[0])))
    steps = predict_steps(net, samples)
    p_g = np.array([steps[i][t] for i, t in zip(snaps["arr_idx"], snaps["t"])])
    results = json.loads((ROOT / "reports/event_time/event_time_results.json").read_text())
    doc = {r["model"]: r["roc_auc"] for r in results["pooled_all"]}
    rows = []
    for name, p in (("XGB", p_x), ("GRU", p_g)):
        got = float(roc_auc_score(y, p))
        rows.append(
            (f"event-time {name}, all test snapshots", got, doc[name], abs(got - doc[name]) <= tol)
        )
    return rows


def check_pass_models(tol: float) -> list[tuple[str, float, float, bool]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_pass_models as tpm  # noqa: PLC0415

    from src.models.pass_progression import EVENT_360, EVENT_ONLY, load_model  # noqa: PLC0415

    cwd = Path.cwd()
    import os

    os.chdir(ROOT)
    try:
        _, parts = tpm.load_data()
        geo = pd.read_parquet(PROC / "geometry_features_v1.parquet")
        te = parts["test"]
        y = te[tpm.LABEL].astype(int).to_numpy()
        feats = {
            EVENT_ONLY: tpm.build_feature_frame(te),
            EVENT_360: tpm.build_feature_frame(te, geo, with_360=True),
        }
        doc = json.loads((ROOT / "reports/pass_label/pass_model_final.json").read_text())
        point = doc["views"]["all test passes"]["point"]
        rows = []
        for name, key in ((EVENT_ONLY, "event_only"), (EVENT_360, "event_360")):
            got = float(roc_auc_score(y, load_model(name).predict_proba_frame(feats[name])))
            ref = point[key]["roc_auc"]
            rows.append((f"pass model {key}, all test passes", got, ref, abs(got - ref) <= tol))
    finally:
        os.chdir(cwd)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=5e-4)
    args = ap.parse_args()
    rows = check_pass_models(args.tol) + check_event_time(args.tol)
    lines = [
        "# Artefact check",
        "",
        "Saved models re-scored on the held-out test matches versus the value recorded in `reports/`.",
        "",
        "| artefact | recomputed ROC AUC | documented | within tolerance |",
        "|---|---|---|---|",
    ]
    ok = True
    for name, got, doc, good in rows:
        ok &= good
        lines.append(f"| {name} | {got:.4f} | {doc:.4f} | {'PASS' if good else 'FAIL'} |")
    out = ROOT / "reports" / "repro" / "artefact_check.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
