"""EXP-032: PassFrameGNN on ``progression_beyond_pass_k`` and a fair comparison with XGBoost.

* Graphs: one per open-play pass with 360 data (``src/features/pass_graph.py``, 8 node features,
  no pass-destination information).
* Split: the saved match-level split (train / val / test parquet event_uuids); never regenerated.
* Model: graph-only PassFrameGNN (3 x GraphSAGE mean, 64 hidden, mean pooling, one binary head),
  plain BCE (no class weighting).
* Tuning: a fixed 8-configuration grid (lr, dropout, weight decay) on **validation** with seed 0;
  then 5 seeds with the chosen configuration; the reported model is the seed with the best
  validation ROC AUC.  Test is read once, after selection.
* Comparison on the identical 360 test subset: GNN, XGBoost event-only, XGBoost event+360 (the
  EXP-027 recipe, unchanged), the end_x-only baseline, plus the early-stopped / unweighted XGBoost
  variants as clearly labelled reference rows.

Usage:  python scripts/train_pass_gnn.py [--n-boot 1000] [--max-epochs 40]
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.event_time_eval import (  # noqa: E402
    bootstrap_by_match,
    metrics,
    reliability_table,
)
from src.features.event_features import build_event_features  # noqa: E402
from src.features.pass_graph import NODE_FEATURES, build_graph_dataset  # noqa: E402
from src.labels.progression_beyond_pass import compute_progression_beyond_pass  # noqa: E402
from src.models.pass_gnn import PassGraphSAGE, collate, predict_proba, save_model  # noqa: E402
from src.models.tabular import TabularClassifier  # noqa: E402

warnings.filterwarnings("ignore")
PROC = Path("data/processed")
OUT = Path("reports/pass_label")
LABEL = "progression_beyond_pass_k"
SEEDS = [0, 1, 2, 3, 4]
GRID = {"lr": [1e-3, 3e-3], "dropout": [0.1, 0.3], "weight_decay": [0.0, 1e-4]}
EXPECTED_360 = {"train": 18234, "val": 4545, "test": 3736}
LABEL_DEFINITION = (
    "Within the next 5 events of the same possession, ignoring the receipt of this pass and the "
    "opponent's events: a shot, a new box entry, or a gain of >= 10 m beyond the pass end into a more "
    "advanced zone (reports/label_methodology.md, Label G)."
)


# --------------------------------------------------------------------------- data


def load_data():
    passes = pd.read_parquet(PROC / "pass_instances.parquet")
    passes = compute_progression_beyond_pass(
        pd.read_parquet(PROC / "events_parsed.parquet"), passes, {}
    )
    cached = PROC / "pass_labels_beyond.parquet"
    if cached.exists():  # the same label table other analyses used: it must agree exactly
        c = pd.read_parquet(cached).set_index("event_uuid")[LABEL]
        assert (
            c.reindex(passes["event_uuid"]).to_numpy() == passes[LABEL].to_numpy()
        ).all(), "label differs from data/processed/pass_labels_beyond.parquet"
    idx = passes.set_index("event_uuid")
    parts = {}
    for s in ("train", "val", "test"):
        uuids = pd.read_parquet(PROC / f"{s}.parquet")["event_uuid"]
        assert uuids.is_unique and set(uuids) <= set(idx.index), f"split {s}: unknown event_uuid"
        parts[s] = idx.loc[uuids].reset_index()
    assert sum(len(v) for v in parts.values()) == len(passes)
    return passes, parts


def build_graphs(parts) -> dict:
    cache = PROC / "pass_graphs_8f.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    frames = pd.read_parquet(PROC / "frames_360.parquet")
    graphs = {
        s: build_graph_dataset(d[d["has_360"].astype(bool)], frames) for s, d in parts.items()
    }
    cache.write_bytes(pickle.dumps(graphs))
    return graphs


# --------------------------------------------------------------------------- baselines


def xgboost_baselines(parts, y_all, cfg):
    """EXP-027 recipe (TabularClassifier: scale_pos_weight, all 500 trees) + labelled reference variants."""
    geo = pd.read_parquet(PROC / "geometry_features_v1.parquet")
    X, X360 = {}, {}
    for s, d in parts.items():
        X[s] = build_event_features(d)
        g = geo.reindex(d["event_uuid"].values).reset_index(drop=True)
        g.index = X[s].index
        X360[s] = pd.concat([X[s], g.fillna(0)], axis=1)
    ytr, yva = y_all["train"], y_all["val"]
    out = {}
    for name, Xs in (("XGB_event", X), ("XGB_event360", X360)):
        clf = TabularClassifier(model_type="xgboost", task=LABEL, config=cfg["xgboost"])
        clf.fit(Xs["train"], ytr, X_val=Xs["val"], y_val=yva)
        out[name] = pd.Series(
            clf.predict_proba(Xs["test"])[:, 1], index=parts["test"]["event_uuid"]
        )
    for name, Xs in (("XGB_event_ES", X), ("XGB_event360_ES", X360)):
        m = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="auc",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
        )
        m.fit(Xs["train"], ytr, eval_set=[(Xs["val"], yva)], verbose=False)
        out[name] = pd.Series(m.predict_proba(Xs["test"])[:, 1], index=parts["test"]["event_uuid"])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    ex = {s: d[["end_x"]].fillna(parts["train"]["end_x"].median()) for s, d in parts.items()}
    lr.fit(ex["train"], ytr)
    out["LR_end_x"] = pd.Series(
        lr.predict_proba(ex["test"])[:, 1], index=parts["test"]["event_uuid"]
    )
    return out


# --------------------------------------------------------------------------- GNN training


def train_gnn(
    graphs,
    y,
    val_graphs,
    val_y,
    *,
    seed,
    lr,
    dropout,
    weight_decay,
    max_epochs,
    patience,
    batch_size=64,
):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = PassGraphSAGE(dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    best = (-1.0, None, 0, None)
    bad = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(graphs))
        for i in range(0, len(order), batch_size):
            ix = order[i : i + batch_size]
            x, ei, b, n = collate([graphs[j] for j in ix])
            loss = bce(model(x, ei, b, n), torch.from_numpy(y[ix]).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = predict_proba(model, val_graphs)
        auc = float(roc_auc_score(val_y, pv))
        if auc > best[0] + 1e-5:
            best = (
                auc,
                {k: v.clone() for k, v in model.state_dict().items()},
                epoch,
                float(log_loss(val_y, np.clip(pv, 1e-6, 1 - 1e-6))),
            )
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best[1])
    model.eval()
    return model, {
        "val_auc": best[0],
        "best_epoch": best[2],
        "val_logloss": best[3],
        "epochs_run": epoch,
    }


# --------------------------------------------------------------------------- calibration


def calibration(y, p):
    """Logistic recalibration (outcome ~ logit(p)): intercept and slope (ideal 0 and 1), and the
    calibration-in-the-large intercept with the slope fixed at 1."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    fit = LogisticRegression(C=1e8, max_iter=1000).fit(lg.reshape(-1, 1), y)
    citl = brentq(lambda a: float((y - expit(a + lg)).sum()), -10.0, 10.0)
    return {
        "intercept": float(fit.intercept_[0]),
        "slope": float(fit.coef_[0, 0]),
        "calibration_in_the_large": float(citl),
        "mean_pred": float(p.mean()),
        "prevalence": float(y.mean()),
    }


def git_info():
    def run(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "working_tree_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--grid-epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    if Path("models/pass_gnn_progression.pt").exists():
        raise SystemExit(
            "models/pass_gnn_progression.pt already exists; refusing to overwrite (move it first)"
        )
    cfg = yaml.safe_load(open("configs/model_baseline.yaml"))
    passes, parts = load_data()
    y_all = {s: d[LABEL].astype(int).to_numpy() for s, d in parts.items()}
    graphs = build_graphs(parts)
    lab = passes.set_index("event_uuid")[LABEL]
    gy = {s: lab.loc[[g.event_uuid for g in graphs[s]]].astype(int).to_numpy() for s in graphs}
    for s, n in EXPECTED_360.items():
        assert len(graphs[s]) == n, f"{s}: {len(graphs[s])} graphs, expected {n}"
        expected = parts[s].loc[parts[s]["has_360"].astype(bool), "event_uuid"].tolist()
        assert [g.event_uuid for g in graphs[s]] == expected, f"{s}: graph / pass alignment"
    print({s: (len(graphs[s]), float(gy[s].mean())) for s in graphs}, flush=True)

    # ---- XGBoost / baselines on the identical test subset
    xgb_all = xgboost_baselines(parts, y_all, cfg)
    full_auc = {k: roc_auc_score(y_all["test"], v.to_numpy()) for k, v in xgb_all.items()}
    print(
        "overall test AUC (all passes):", {k: round(v, 4) for k, v in full_auc.items()}, flush=True
    )
    assert (
        abs(full_auc["XGB_event"] - 0.7279) < 5e-4 and abs(full_auc["XGB_event360"] - 0.7294) < 5e-4
    ), "EXP-027 XGBoost results are not reproduced; stopping"

    # ---- GNN: validation-only grid (seed 0), then 5 seeds
    grid_rows = []
    for lr, do, wd in itertools.product(GRID["lr"], GRID["dropout"], GRID["weight_decay"]):
        _, info = train_gnn(
            graphs["train"],
            gy["train"],
            graphs["val"],
            gy["val"],
            seed=0,
            lr=lr,
            dropout=do,
            weight_decay=wd,
            max_epochs=args.grid_epochs,
            patience=4,
        )
        grid_rows.append({"lr": lr, "dropout": do, "weight_decay": wd, **info})
        print("grid", grid_rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    best_cfg = max(grid_rows, key=lambda r: r["val_auc"])
    chosen = {k: best_cfg[k] for k in ("lr", "dropout", "weight_decay")}
    models, seed_rows = {}, []
    for sd in SEEDS:
        m, info = train_gnn(
            graphs["train"],
            gy["train"],
            graphs["val"],
            gy["val"],
            seed=sd,
            max_epochs=args.max_epochs,
            patience=args.patience,
            **chosen,
        )
        models[sd] = m
        seed_rows.append({"seed": sd, **info})
        print("seed", seed_rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    sel = max(seed_rows, key=lambda r: r["val_auc"])["seed"]  # chosen by VALIDATION
    print("selected seed", sel, flush=True)

    # ---- test (read once, after selection)
    test_uuids = [g.event_uuid for g in graphs["test"]]
    y = gy["test"]
    mid = parts["test"].set_index("event_uuid").loc[test_uuids, "match_id"].to_numpy()
    preds = {"GNN": predict_proba(models[sel], graphs["test"])}
    for k, v in xgb_all.items():
        preds[k] = v.loc[test_uuids].to_numpy()
    for r in seed_rows:
        r["test_auc"] = float(roc_auc_score(y, predict_proba(models[r["seed"]], graphs["test"])))
        r["test_ap"] = float(
            metrics(y, predict_proba(models[r["seed"]], graphs["test"]))["avg_precision"]
        )
        r["test_brier"] = float(
            metrics(y, predict_proba(models[r["seed"]], graphs["test"]))["brier"]
        )
    rows = pd.DataFrame({"match_id": mid, "y": y})
    names = ["GNN", "XGB_event360", "XGB_event", "LR_end_x", "XGB_event360_ES", "XGB_event_ES"]
    point = {n: metrics(y, preds[n]) for n in names}
    calib = {n: calibration(y, preds[n]) for n in names}
    rel = {n: reliability_table(y, preds[n]) for n in names}
    deltas = [
        ("GNN", "XGB_event360"),
        ("GNN", "XGB_event"),
        ("GNN", "LR_end_x"),
        ("GNN", "XGB_event360_ES"),
        ("XGB_event360", "XGB_event"),
    ]
    boot = bootstrap_by_match(
        rows, {n: preds[n] for n in names}, deltas=deltas, n_boot=args.n_boot, seed=42
    )

    def ci(model, stat):
        r = boot[(boot.model == model) & (boot.stat == stat)].iloc[0]
        return float(r.estimate), float(r.ci_low), float(r.ci_high)

    seeds_df = pd.DataFrame(seed_rows)
    seed_summary = {
        c: [float(seeds_df[c].mean()), float(seeds_df[c].std(ddof=1))]
        for c in ("val_auc", "test_auc", "test_ap", "test_brier")
    }

    # ---- write results
    results = {
        "n_test_graphs": len(y),
        "test_matches": int(len(set(mid))),
        "test_prevalence": float(y.mean()),
        "grid": grid_rows,
        "chosen": chosen,
        "seeds": seed_rows,
        "selected_seed": sel,
        "seed_summary_mean_sd": seed_summary,
        "point": point,
        "calibration": calib,
        "bootstrap": boot.to_dict("records"),
        "overall_test_auc_all_passes": full_auc,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gnn_comparison.json").write_text(
        json.dumps(results, indent=1, default=float), encoding="utf-8"
    )

    def md(df, fmt=".3f"):
        cols = list(df.columns)
        out = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
        for r in df.itertuples(index=False):
            out += (
                "| "
                + " | ".join(
                    format(v, fmt) if isinstance(v, (float, np.floating)) else str(v) for v in r
                )
                + " |\n"
            )
        return out

    label_of = {
        "GNN": "PassFrameGNN (graph-only; reported seed)",
        "XGB_event360": "XGBoost event+360 (EXP-027 recipe)",
        "XGB_event": "XGBoost event-only (EXP-027 recipe)",
        "LR_end_x": "`end_x`-only logistic regression",
        "XGB_event360_ES": "*Reference:* XGBoost event+360, early-stopped, unweighted",
        "XGB_event_ES": "*Reference:* XGBoost event-only, early-stopped, unweighted",
    }
    tab = []
    for n in names:
        a, b, c = ci(n, "roc_auc")
        d, e, f = ci(n, "avg_precision")
        g, h, i = ci(n, "brier")
        tab.append(
            {
                "model": label_of[n],
                "ROC AUC [95% CI]": f"{a:.3f} [{b:.3f}, {c:.3f}]",
                "AP [95% CI]": f"{d:.3f} [{e:.3f}, {f:.3f}]",
                "Brier [95% CI]": f"{g:.3f} [{h:.3f}, {i:.3f}]",
                "cal. intercept": f"{calib[n]['intercept']:+.2f}",
                "cal. slope": f"{calib[n]['slope']:.2f}",
                "mean pred": f"{calib[n]['mean_pred']:.3f}",
            }
        )
    dtab = []
    for a_, b_ in deltas:
        r = {"difference": f"{a_} - {b_}"}
        for st, nm in (("roc_auc", "ROC AUC"), ("avg_precision", "AP"), ("brier", "Brier")):
            e_, l_, h_ = ci(f"{a_} - {b_}", st)
            r[nm] = f"{e_:+.4f} [{l_:+.4f}, {h_:+.4f}]"
        dtab.append(r)
    verdict = ci("GNN - XGB_event360", "roc_auc")
    beats = verdict[1] > 0
    worse = verdict[2] < 0
    sentence = (
        "The GNN is **better** than XGBoost event+360 on ROC AUC (interval above zero)."
        if beats
        else (
            "The GNN is **worse** than XGBoost event+360 on ROC AUC (interval below zero)."
            if worse
            else "The GNN is **not distinguishable** from XGBoost event+360 on ROC AUC (the paired interval includes zero)."
        )
    )
    L = [
        "# PassFrameGNN vs XGBoost on `progression_beyond_pass_k` (360 test subset)",
        "",
        "Generated by `scripts/train_pass_gnn.py` (EXP-032).  Unchanged match-level split; test read once, after the "
        "model and seed were chosen on validation.",
        "",
        f"Test subset: {len(y):,} passes with 360 data from {len(set(mid))} World Cup 2022 matches; positive rate {y.mean():.3f}.  "
        f"Graphs: train {len(graphs['train']):,} / val {len(graphs['val']):,} / test {len(graphs['test']):,}.",
        "",
        "## Reading",
        "",
        "Player positions alone (graph-only, no pass destination) are compared with `end_x` alone and with the event-based models; the paired differences are reported for completeness and are not a like-for-like contest between feature sets.",
        "",
        "## Test metrics (identical 360 test subset; 1,000 match-level bootstrap resamples)",
        "",
        md(pd.DataFrame(tab)),
        "",
        "Paired differences (same resamples):",
        "",
        md(pd.DataFrame(dtab)),
        "",
        "## GNN seeds (5; reported model chosen by validation ROC AUC)",
        "",
        md(seeds_df[["seed", "val_auc", "best_epoch", "test_auc", "test_ap", "test_brier"]], ".4f"),
        "",
        f"Mean (sd) across seeds: validation AUC {seed_summary['val_auc'][0]:.4f} ({seed_summary['val_auc'][1]:.4f}); "
        f"test AUC {seed_summary['test_auc'][0]:.4f} ({seed_summary['test_auc'][1]:.4f}); test AP {seed_summary['test_ap'][0]:.4f} "
        f"({seed_summary['test_ap'][1]:.4f}); test Brier {seed_summary['test_brier'][0]:.4f} ({seed_summary['test_brier'][1]:.4f}).  "
        f"Selected seed: {sel}.",
        "",
        f"## Validation grid (seed 0, {args.grid_epochs} epochs max)",
        "",
        md(
            pd.DataFrame(grid_rows)[
                ["lr", "dropout", "weight_decay", "val_auc", "best_epoch", "val_logloss"]
            ],
            ".4f",
        ),
        f"Chosen on validation: {chosen}.",
        "",
        "## Calibration",
        "",
        "Intercept and slope come from a logistic recalibration of the outcome on logit(p) (ideal: 0 and 1); "
        "`mean pred` against the prevalence shows the overall shift.  The EXP-027 XGBoost recipe uses `scale_pos_weight`, so its "
        "probabilities are inflated by design; the reference rows are unweighted.",
        "",
    ]
    for n in ("GNN", "XGB_event360", "XGB_event"):
        L += [f"### Reliability, {label_of[n]}", "", md(rel[n]), ""]
    L += [
        "## Reproduction guard",
        "",
        f"Overall test ROC AUC over all passes (not only the 360 subset) of the XGBoost EXP-027 recipe: event-only {full_auc['XGB_event']:.4f}, event+360 {full_auc['XGB_event360']:.4f} "
        "(expected 0.7279 / 0.7294; the script stops if they differ).",
    ]
    (OUT / "gnn_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- save the model (new names; nothing is overwritten)
    meta = {
        "name": "pass_gnn_progression",
        "label": LABEL,
        "label_definition": LABEL_DEFINITION,
        "architecture": "graph-only PassFrameGNN: node projection 8->64 + ReLU, 3 x GraphSAGE (mean aggregation, plain torch), "
        "mean pooling, fusion Linear(64,64)+ReLU+dropout, one Linear(64,1) head; BCE, no class weighting",
        "node_features": NODE_FEATURES,
        "graph": "nodes = visible players; edges = 5 nearest neighbours + same-team links (both directions); no edge features; "
        "pass-START location used for dist_to_passer; no pass-destination information",
        "deviations_from_original": [
            "node feature is_receiver (derived from the pass END location) dropped: 8 node features instead of 9",
            "graph-only: the original 27-feature event branch and the 5 multitask heads are not used",
            "GraphSAGE implemented in plain torch instead of torch_geometric",
            "node features scaled to ~[0, 1]",
        ],
        "hyperparameters": {
            **chosen,
            "hidden": 64,
            "layers": 3,
            "batch_size": 64,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
        },
        "seeds": SEEDS,
        "selected_seed": sel,
        "seed_results": seed_rows,
        "seed_summary_mean_sd": seed_summary,
        "split": {
            "source": "data/processed/{train,val,test}.parquet (saved match-level split, seed 42)",
            "graphs": {s: len(graphs[s]) for s in graphs},
            "matches_with_graphs": {
                s: int(parts[s][parts[s]["has_360"].astype(bool)]["match_id"].nunique())
                for s in parts
            },
        },
        "test_subset": {"n": len(y), "matches": int(len(set(mid))), "prevalence": float(y.mean())},
        "test_metrics_with_95ci_match_bootstrap": {
            n: {
                st: dict(zip(("estimate", "ci_low", "ci_high"), ci(n, st)))
                for st in ("roc_auc", "avg_precision", "brier")
            }
            for n in names
        },
        "calibration_test": calib["GNN"],
        "git": git_info(),
        "torch": torch.__version__,
        "trained_utc": pd.Timestamp.now("UTC").isoformat(),
    }
    save_model("models/pass_gnn_progression.pt", models[sel], meta)
    print("\n".join(L[:30]))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
