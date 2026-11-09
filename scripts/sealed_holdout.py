"""Score the committed event-time models with the frozen sealed-holdout protocol.

Protocol: ``reports/repro/sealed_holdout_protocol.md``; config: ``configs/sealed_holdout.json``.

``--target test``     original 15 test matches (in-cohort counterpart, same definitions);
                      writes ``reports/repro/frozen_metrics_test.json``. Safe to re-run.
``--target holdout``  UEFA Euro 2024. Builds inputs with the existing pipeline functions into the config's
                      ``holdout_data_dir`` (refuses to overwrite), scores once and writes
                      ``reports/repro/sealed_holdout.json``. Refuses to run if that file exists.

No model is retrained or refit. The ball x/y baseline is refit on the original training matches only.

Usage:  python scripts/sealed_holdout.py --target {test,holdout}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.app.batch_scoring import load_models  # noqa: E402
from src.data.ingest import get_events  # noqa: E402
from src.data.parse_events import parse_events  # noqa: E402
from src.data.parse_possessions import (  # noqa: E402
    build_possession_sequences,
    save_possession_sequences,
)
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
from src.evaluation.provenance import file_sha256, run_metadata  # noqa: E402
from src.features.event_time import (  # noqa: E402
    FEATURE_NAMES,
    PossessionArrays,
    build_possession_arrays,
    build_snapshot_table,
    danger_flags,
    sequence_inputs,
    snapshot_labels,
)
from src.models.event_time import SeqSample, predict_steps  # noqa: E402

PROC = ROOT / "data" / "processed"
CONFIG = ROOT / "configs" / "sealed_holdout.json"
MODELS = ("XGB", "GRU", "LR_xy")
DELTAS = [("XGB", "LR_xy"), ("GRU", "LR_xy"), ("GRU", "XGB")]


def arrays_from(data_dir: Path) -> list[PossessionArrays]:
    events = pd.read_parquet(data_dir / "events_parsed.parquet")
    keys = pd.read_parquet(
        data_dir / "possession_sequences.parquet", columns=["match_id", "possession_id"]
    )
    return build_possession_arrays(events, keys)


def build_holdout_inputs(cfg: dict) -> Path:
    """Same functions and order as scripts/prepare_event_time.py, for one competition-season."""
    out = ROOT / cfg["holdout_data_dir"]
    if (out / "events_parsed.parquet").exists():
        raise SystemExit(f"{out} already holds holdout inputs; refusing to rebuild")
    rev = cfg["open_data_revision"]
    url = (
        f"https://raw.githubusercontent.com/statsbomb/open-data/{rev}/data/matches/"
        f"{cfg['competition_id']}/{cfg['season_id']}.json"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    match_ids = sorted(int(m["match_id"]) for m in response.json())
    cohort = set(pd.read_csv(ROOT / "configs/research_split_manifest.csv").match_id)
    if cohort & set(match_ids):
        raise ValueError("holdout overlaps the research cohort")
    parts = []
    for mid in match_ids:
        frame = parse_events(get_events(mid, revision=rev))
        if set(frame.match_id) != {mid}:
            raise ValueError(f"raw data match identity mismatch for {mid}")
        parts.append(frame)
        print(f"Parsed match {mid}: {len(frame)} events", flush=True)
    events = pd.concat(parts, ignore_index=True)
    if events.event_uuid.duplicated().any():
        raise ValueError("duplicate event UUIDs")
    poss = build_possession_sequences(events)
    out.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out / "events_parsed.parquet", index=False)
    save_possession_sequences(poss, out / "possession_sequences.parquet")
    pd.DataFrame({"match_id": match_ids, "split": "holdout"}).to_csv(
        out / "match_manifest.csv", index=False
    )
    print(f"Wrote {len(events)} events and {len(poss)} possessions to {out}")
    return out


def score(arrs: list[PossessionArrays], art, net, lr_xy) -> tuple[pd.DataFrame, dict]:
    rows = build_snapshot_table(arrs).query("not terminal").reset_index(drop=True)
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
    }
    for name, p in preds.items():
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"invalid probabilities from {name}")
    return rows, preds


def ci_table(boot: pd.DataFrame) -> dict:
    out: dict = {}
    for r in boot.itertuples(index=False):
        out.setdefault(r.model, {})[r.stat] = {
            "estimate": float(r.estimate),
            "ci_low": float(r.ci_low),
            "ci_high": float(r.ci_high),
        }
    return out


def evaluate(arrs, art, net, lr_xy, thresholds: dict, cfg: dict) -> dict:
    rows, preds = score(arrs, art, net, lr_xy)
    y = rows.y.to_numpy()
    seed, n_boot = cfg["seed"], cfg["n_boot"]
    one = one_snapshot_per_possession(rows, seed)
    rows_one = rows.iloc[one].reset_index(drop=True)
    preds_one = {m: p[one] for m, p in preds.items()}
    danger_at = {
        (a.match_id, a.possession_id): int(np.argmax(danger_flags(a)))
        for a in arrs
        if danger_flags(a).any()
    }
    result = {
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
        "alerts": {
            m: alert_metrics(rows, preds[m], thresholds[m.lower()], danger_at)
            for m in ("XGB", "GRU")
        },
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=("test", "holdout"), required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out = ROOT / (
        cfg["output"] if args.target == "holdout" else "reports/repro/frozen_metrics_test.json"
    )
    if args.target == "holdout" and out.exists():
        raise SystemExit(f"{out} exists: the sealed holdout is scored once only")
    for rel, digest in {**cfg["models"], **cfg["training_inputs_sha256"]}.items():
        if file_sha256(ROOT / rel) != digest:
            raise SystemExit(f"{rel} differs from the protocol (SHA-256 mismatch)")
    torch.set_num_threads(2)
    art, net, meta = load_models(ROOT)
    thresholds = {"xgb": float(meta["xgb_threshold"]), "gru": float(meta["gru_threshold"])}
    if thresholds != cfg["thresholds"]:
        raise SystemExit("model thresholds differ from the protocol")

    manifest = pd.read_csv(ROOT / "configs/research_split_manifest.csv")
    research = arrays_from(PROC)
    split_of = dict(zip(manifest.match_id, manifest.split))
    train = build_snapshot_table([a for a in research if split_of[a.match_id] == "train"])
    train = train.query("not terminal")
    base = cfg["baseline"]
    lr_xy = make_pipeline(StandardScaler(), LogisticRegression(C=base["C"], max_iter=2000))
    lr_xy.fit(train[base["features"]], train.y)

    if args.target == "test":
        arrs = [a for a in research if split_of[a.match_id] == "test"]
        data_desc = {"data_dir": "data/processed", "split": "test (15 matches)"}
    else:
        data_dir = build_holdout_inputs(cfg)
        arrs = arrays_from(data_dir)
        data_desc = {
            "data_dir": cfg["holdout_data_dir"],
            "competition": cfg["competition"],
            "competition_id": cfg["competition_id"],
            "season_id": cfg["season_id"],
            "events_parsed_sha256": file_sha256(data_dir / "events_parsed.parquet"),
            "possession_sequences_sha256": file_sha256(data_dir / "possession_sequences.parquet"),
        }

    result = evaluate(arrs, art, net, lr_xy, thresholds, cfg)
    record = {
        "target": args.target,
        "data": data_desc,
        "baseline_train_snapshots": int(len(train)),
        "definitions": {"false_alert_rate": FALSE_ALERT_DEFINITION, "ece": ECE_DEFINITION},
        "thresholds": thresholds,
        "reference_auc": cfg["reference_auc"],
        "results": result,
        "provenance": {
            "command": f"python scripts/sealed_holdout.py --target {args.target}",
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
    summary = {m: round(result["all_snapshots"][m]["roc_auc"], 4) for m in MODELS} | {
        "snapshots": result["eligible_snapshots"]
    }
    print(json.dumps(summary))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
