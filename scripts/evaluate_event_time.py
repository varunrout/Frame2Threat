"""Evaluate saved models on rebuilt held-out arrays, including calibration and a constant baseline."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.app.batch_scoring import load_models  # noqa: E402
from src.features.event_time import (  # noqa: E402
    FEATURE_NAMES,
    build_possession_arrays,
    build_snapshot_table,
    sequence_inputs,
    snapshot_labels,
)
from src.models.event_time import SeqSample, predict_steps  # noqa: E402
from src.evaluation.event_time_eval import (
    metrics,
    reliability_table,
    one_snapshot_per_possession,
)  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--run-root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, default=Path("reports/repro/repair_evaluation.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    manifest = pd.read_csv(args.data_dir / "split_manifest.csv")
    if manifest.match_id.duplicated().any():
        raise ValueError("overlapping splits")
    events = pd.read_parquet(args.data_dir / "events_parsed.parquet")
    keys = pd.read_parquet(
        args.data_dir / "possession_sequences.parquet", columns=["match_id", "possession_id"]
    )
    arrs = build_possession_arrays(events, keys)
    ids = set(manifest.query("split == 'test'").match_id)
    train_ids = set(manifest.query("split == 'train'").match_id)
    train = build_snapshot_table([a for a in arrs if a.match_id in train_ids])
    prevalence = float(train.loc[~train.terminal, "y"].mean())
    arrs = [a for a in arrs if a.match_id in ids]
    rows = build_snapshot_table(arrs).query("not terminal").reset_index(drop=True)
    art, net, _ = load_models(args.run_root)
    samples = []
    for a in arrs:
        eligible, label, _ = snapshot_labels(a)
        tid, num = sequence_inputs(a)
        samples.append(SeqSample(tid, num, eligible, int(label[0])))
    steps = predict_steps(net, samples)
    preds = {
        "prevalence": np.full(len(rows), prevalence),
        "XGB": art["model"].predict_proba(rows[FEATURE_NAMES])[:, 1],
        "GRU": np.array([steps[i][t] for i, t in zip(rows.arr_idx, rows.t)]),
    }
    result = {"test_matches": len(ids), "test_snapshots": len(rows), "train_prevalence": prevalence}
    for view, ix in (
        ("all", np.arange(len(rows))),
        ("one_per_possession", one_snapshot_per_possession(rows, 42)),
    ):
        result[view] = {}
        y = rows.y.to_numpy()[ix]
        for name, pred in preds.items():
            p = pred[ix]
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError("invalid model probabilities")
            rel = reliability_table(y, p).dropna()
            ece = float((rel.n * (rel.mean_pred - rel.observed).abs()).sum() / len(y))
            result[view][name] = {**metrics(y, p), "ece_10_equal_width": ece}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
