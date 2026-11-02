"""EXP-021/022: event-time (in-play) danger forecasting.

Trains Baseline A (LR on ball x/y), Baseline B (LR on events-so-far), a start-context-only
XGBoost reference, XGBoost on the snapshot features, and a causal GRU with a learned
event-type embedding -- all on the unchanged match-level train split, with every
hyper-parameter chosen on validation.  Evaluates on the test split and writes
``reports/event_time/`` plus ``models/eventtime_*``.

Usage:  python scripts/run_event_time.py [--n-boot 500] [--rebuild]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.event_types import label as type_label  # noqa: E402
from src.evaluation.event_time_eval import (  # noqa: E402
    EVENT_BUCKETS,
    SECOND_BUCKETS,
    bootstrap_by_match,
    bucket_metrics,
    metrics,
    one_snapshot_per_possession,
    reliability_table,
    threshold_matching_prevalence,
)
from src.features.event_time import (  # noqa: E402
    FEATURE_NAMES,
    N_SEQ_NUMERIC,
    ORIGIN_COLS,
    build_possession_arrays,
    build_snapshot_table,
    danger_flags,
    sequence_inputs,
    snapshot_labels,
)
from src.models.event_time import SeqSample, predict_steps, train_event_time_gru  # noqa: E402

warnings.filterwarnings("ignore")
PROC = Path("data/processed")
OUT = Path("reports/event_time")
SEED = 42
THREADS = 2
START_COLS = ["start_x", "start_y", "period", "game_minute", *ORIGIN_COLS]
MODELS = ["LR_xy", "LR_len", "XGB_start", "XGB", "GRU"]
DESCR = {
    "LR_xy": "Baseline A: logistic regression, current ball x/y only",
    "LR_len": "Baseline B: logistic regression, events-so-far only",
    "XGB_start": "Reference: XGBoost, start context only (no events)",
    "XGB": "XGBoost, full snapshot features",
    "GRU": "Causal GRU (1 layer, hidden 64, learned type embedding)",
}


def load_split_ids() -> dict[str, np.ndarray]:
    ids = {
        s: pd.read_parquet(PROC / f"{s}.parquet")["match_id"].unique()
        for s in ("train", "val", "test")
    }
    manifest = pd.read_csv(PROC / "split_manifest.csv")
    if manifest.match_id.duplicated().any() or not manifest.split.isin(ids).all():
        raise ValueError("split manifest must assign each match exactly once to train/val/test")
    for s, v in ids.items():
        assert set(manifest.loc[manifest.split == s, "match_id"]) == set(v), f"split mismatch: {s}"
    return ids


def get_arrays(rebuild: bool):
    cache = PROC / "event_time_arrays.pkl"
    if cache.exists() and not rebuild:
        return pickle.loads(cache.read_bytes())
    ev = pd.read_parquet(PROC / "events_parsed.parquet")
    keys = pd.read_parquet(
        PROC / "possession_sequences.parquet", columns=["match_id", "possession_id"]
    )
    arrs = build_possession_arrays(ev, keys)
    cache.write_bytes(pickle.dumps(arrs))
    return arrs


def fit_lr(Xtr, ytr, Xva, yva, cols):
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        m.fit(Xtr[cols], ytr)
        v = log_loss(yva, m.predict_proba(Xva[cols])[:, 1])
        if best is None or v < best[0]:
            best = (v, C, m)
    return best[2], {"C": best[1], "val_logloss": best[0]}


def fit_xgb(Xtr, ytr, Xva, yva, cols):
    best = None
    for depth in (3, 5):
        for mcw in (1, 20):
            m = XGBClassifier(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=depth,
                min_child_weight=mcw,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                early_stopping_rounds=30,
                random_state=SEED,
                n_jobs=THREADS,
            )
            m.fit(Xtr[cols], ytr, eval_set=[(Xva[cols], yva)], verbose=False)
            v = m.best_score
            if best is None or v < best[0]:
                best = (
                    v,
                    dict(max_depth=depth, min_child_weight=mcw, n_estimators=m.best_iteration + 1),
                    m,
                )
    return best[2], {**best[1], "val_logloss": float(best[0])}


def seq_samples(arrs, idxs):
    out = []
    for i in idxs:
        p = arrs[i]
        elig, lab, _ = snapshot_labels(p)
        tid, num = sequence_inputs(p)
        out.append(SeqSample(tid, num, elig, int(lab[0])))
    return out


def step_probs_to_rows(step_probs, arr_pos, rows):
    """Pick GRU probabilities at (possession, t) of each snapshot row."""
    return np.array([step_probs[arr_pos[a]][t] for a, t in zip(rows["arr_idx"], rows["t"])])


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for r in df.itertuples(index=False):
        body += (
            "| "
            + " | ".join(
                (format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)) for v in r
            )
            + " |\n"
        )
    return head + body


def tipping(arrs, steps, thr, label):
    """First eligible t where P >= thr; event-type shares with rate normalisation.

    ``steps[i]`` holds per-event probabilities for ``arrs[i]`` (only eligible steps are read).
    """
    recs, base, tip_types, jump_types = [], {}, {}, {}
    for i, p in enumerate(arrs):
        elig, lab, _ = snapshot_labels(p)
        ts = np.nonzero(elig)[0]
        if len(ts) == 0:
            continue
        pr = steps[i][ts]
        for t in ts:
            base[int(p.type_id[t])] = base.get(int(p.type_id[t]), 0) + 1
        hit = np.nonzero(pr >= thr)[0]
        tip_t = int(ts[hit[0]]) if len(hit) else -1
        rec = {"y": int(lab[0]), "tip_t": tip_t, "n_elig": len(ts)}
        if tip_t >= 0:
            tip_types[int(p.type_id[tip_t])] = tip_types.get(int(p.type_id[tip_t]), 0) + 1
        if len(ts) >= 2:
            j = int(np.argmax(np.diff(pr))) + 1
            jt = int(p.type_id[ts[j]])
            jump_types[jt] = jump_types.get(jt, 0) + 1
        if lab[0] == 1 and tip_t >= 0:
            d = int(np.argmax(danger_flags(p)))
            rec["lead_events"] = d - tip_t
            rec["lead_seconds"] = float(p.abs_s[d] - p.abs_s[tip_t])
        recs.append(rec)
    df = pd.DataFrame(recs)
    btot = sum(base.values())

    def dist(counts):
        tot = sum(counts.values())
        rows = [
            {
                "event type": type_label(k),
                "n": v,
                "share": v / tot,
                "base freq": base[k] / btot,
                "rate": (v / tot) / (base[k] / btot),
            }
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        ]
        return pd.DataFrame(rows)

    summary = {
        "model": label,
        "threshold": thr,
        "dangerous_possessions_tipped": float((df[df.y == 1].tip_t >= 0).mean()),
        "safe_possessions_tipped (false alarm)": float((df[df.y == 0].tip_t >= 0).mean()),
        "median_lead_events": float(df["lead_events"].median()),
        "median_lead_seconds": float(df["lead_seconds"].median()),
        "median_tip_event_index": float(df[df.tip_t >= 0].tip_t.median()),
    }
    return summary, dist(tip_types), dist(jump_types)


def main() -> None:
    global PROC, OUT, THREADS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=PROC)
    ap.add_argument("--output-dir", type=Path, default=Path("."), help="Isolated run root")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    THREADS = args.threads
    PROC = args.data_dir.resolve()
    OUT = args.output_dir / "reports/event_time"
    model_dir = args.output_dir / "models"
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    ids = load_split_ids()
    # Always rebuild from source: unversioned pickle caches can silently retain old features.
    arrs = get_arrays(True)
    snap = build_snapshot_table(arrs)
    elig = snap[~snap.terminal].reset_index(drop=True)
    term = snap[snap.terminal].reset_index(drop=True)
    split_of = {m: s for s, v in ids.items() for m in v}
    elig["split"] = elig.match_id.map(split_of)
    term["split"] = term.match_id.map(split_of)
    assert elig.split.notna().all()
    tr, va, te = (elig[elig.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    te_term = term[term.split == "test"].reset_index(drop=True)
    info = {
        "rows": {s: int(len(d)) for s, d in zip(("train", "val", "test"), (tr, va, te))},
        "possessions_with_rows": {
            s: int(d.groupby(["match_id", "possession_id"]).ngroups)
            for s, d in zip(("train", "val", "test"), (tr, va, te))
        },
        "prevalence_rows": {
            s: float(d.y.mean()) for s, d in zip(("train", "val", "test"), (tr, va, te))
        },
        "test_matches": int(te.match_id.nunique()),
    }
    print(info)

    # ---- tabular models ----------------------------------------------------------
    for d in (tr, va, te, te_term):
        d["log_events"] = np.log1p(d["events_so_far"])
    tuned: dict = {}
    lr_xy, tuned["LR_xy"] = fit_lr(tr, tr.y, va, va.y, ["cur_x", "cur_y"])
    lr_len, tuned["LR_len"] = fit_lr(tr, tr.y, va, va.y, ["events_so_far", "log_events"])
    xgb_start, tuned["XGB_start"] = fit_xgb(tr, tr.y, va, va.y, START_COLS)
    xgb, tuned["XGB"] = fit_xgb(tr, tr.y, va, va.y, FEATURE_NAMES)
    print("tuned:", tuned)

    def tab_preds(d):
        return {
            "LR_xy": lr_xy.predict_proba(d[["cur_x", "cur_y"]])[:, 1],
            "LR_len": lr_len.predict_proba(d[["events_so_far", "log_events"]])[:, 1],
            "XGB_start": xgb_start.predict_proba(d[START_COLS])[:, 1],
            "XGB": xgb.predict_proba(d[FEATURE_NAMES])[:, 1],
        }

    # ---- GRU ----------------------------------------------------------------------
    split_idx = {s: [] for s in ids}
    for i, a in enumerate(arrs):
        split_idx[split_of[a.match_id]].append(i)
    samples = {s: seq_samples(arrs, split_idx[s]) for s in ids}
    keep = {s: [k for k, smp in enumerate(samples[s]) if smp.eligible.any()] for s in ids}
    train_s = [samples["train"][k] for k in keep["train"]]
    val_s = [samples["val"][k] for k in keep["val"]]
    gru, best_cfg = None, None
    for lr in (3e-3, 1e-3):
        m, cfg = train_event_time_gru(train_s, val_s, N_SEQ_NUMERIC, lr=lr, seed=SEED)
        print(
            f"GRU lr={lr}: best val logloss {cfg['best_val_logloss']:.4f} "
            f"({len(cfg['history'])} epochs)"
        )
        if best_cfg is None or cfg["best_val_logloss"] < best_cfg["best_val_logloss"]:
            gru, best_cfg = m, cfg
    tuned["GRU"] = {k: v for k, v in best_cfg.items() if k != "history"}

    def gru_row_preds(d, split):
        steps = predict_steps(gru, samples[split])
        by_arr = {split_idx[split][k]: steps[k] for k in range(len(steps))}
        return np.array([by_arr[a][t] for a, t in zip(d["arr_idx"], d["t"])]), by_arr

    p_va = tab_preds(va)
    p_va["GRU"], steps_val = gru_row_preds(va, "val")
    p_te = tab_preds(te)
    p_te["GRU"], steps_test = gru_row_preds(te, "test")
    p_te_term = tab_preds(te_term)
    p_te_term["GRU"], _ = gru_row_preds(te_term, "test")

    # ---- evaluation ---------------------------------------------------------------
    y = te.y.to_numpy()
    sel = one_snapshot_per_possession(te, SEED)
    te_one = te.iloc[sel].reset_index(drop=True)
    p_one = {m: np.asarray(p)[sel] for m, p in p_te.items()}

    def pooled(rows, preds):
        return pd.DataFrame([{"model": m, **metrics(rows.y.to_numpy(), preds[m])} for m in MODELS])

    pooled_all = pooled(te, p_te)
    pooled_one = pooled(te_one, p_one)
    # robustness: mean over 20 independent one-per-possession draws
    draw_auc = {m: [] for m in MODELS}
    for s in range(20):
        ix = one_snapshot_per_possession(te, 1000 + s)
        for m in MODELS:
            draw_auc[m].append(metrics(te.y.to_numpy()[ix], np.asarray(p_te[m])[ix])["roc_auc"])
    both = pd.concat([te, te_term], ignore_index=True)
    p_both = {m: np.concatenate([p_te[m], p_te_term[m]]) for m in MODELS}
    pooled_term = pooled(both, p_both)

    deltas = [
        ("XGB", "LR_xy"),
        ("XGB", "LR_len"),
        ("GRU", "LR_xy"),
        ("GRU", "LR_len"),
        ("GRU", "XGB"),
        ("XGB", "XGB_start"),
    ]
    boot_all = bootstrap_by_match(te, p_te, deltas=deltas, n_boot=args.n_boot, seed=SEED)
    boot_one = bootstrap_by_match(te_one, p_one, deltas=deltas, n_boot=args.n_boot, seed=SEED)

    ev_buckets = pd.concat(
        [bucket_metrics(te, {m: p_te[m] for m in MODELS}, "events_so_far", EVENT_BUCKETS)]
    )
    sec_buckets = bucket_metrics(te, {m: p_te[m] for m in MODELS}, "elapsed_s", SECOND_BUCKETS)
    rel = {m: reliability_table(y, p_te[m]) for m in ("XGB", "GRU")}

    # ---- tipping points (threshold chosen on validation) --------------------------
    steps_xgb_test = {}
    for a, grp in te.groupby("arr_idx"):
        arr_steps = np.full(arrs[a].n, np.nan)
        arr_steps[grp.t.to_numpy()] = np.asarray(p_te["XGB"])[grp.index.to_numpy()]
        steps_xgb_test[a] = arr_steps
    test_arr_ids = sorted(steps_xgb_test)
    arrs_test = [arrs[a] for a in test_arr_ids]
    thr = {
        "XGB": threshold_matching_prevalence(va.y, p_va["XGB"]),
        "GRU": threshold_matching_prevalence(va.y, p_va["GRU"]),
    }
    tip_out = {}
    for m, stepmap in (("XGB", steps_xgb_test), ("GRU", {a: steps_test[a] for a in test_arr_ids})):
        tip_out[m] = tipping(arrs_test, [stepmap[a] for a in test_arr_ids], thr[m], m)

    # ---- persist models -----------------------------------------------------------
    torch.save(
        {
            "state_dict": gru.state_dict(),
            "n_numeric": N_SEQ_NUMERIC,
            "hidden": 64,
            "embed_dim": 8,
            "threshold": thr["GRU"],
        },
        model_dir / "eventtime_gru.pt",
    )
    joblib.dump(
        {"model": xgb, "features": FEATURE_NAMES, "threshold": thr["XGB"]},
        model_dir / "eventtime_xgboost.joblib",
    )
    # Native model IO preserves probability semantics across supported XGBoost versions.
    xgb.save_model(model_dir / "eventtime_xgboost.ubj")
    (model_dir / "eventtime_meta.json").write_text(
        json.dumps(
            {
                "formulation": "event-time hazard: P(shot or box entry in events t+1..n-1 | events 0..t, none yet)",
                "schema_version": 2,
                "ownership": "provider possession_team on first event; fallback first actor (never whole-possession majority)",
                "features": FEATURE_NAMES,
                "n_seq_numeric": N_SEQ_NUMERIC,
                "seed": SEED,
                "versions": {
                    p: importlib.metadata.version(p)
                    for p in ("numpy", "pandas", "torch", "xgboost", "scikit-learn")
                },
                "input_sha256": {
                    p: hashlib.sha256((PROC / p).read_bytes()).hexdigest()
                    for p in (
                        "events_parsed.parquet",
                        "possession_sequences.parquet",
                        "split_manifest.csv",
                    )
                },
                "gru_threshold": thr["GRU"],
                "xgb_threshold": thr["XGB"],
                "threshold_rule": "score quantile on VALIDATION snapshots whose predicted-positive rate equals validation prevalence",
                "median_inter_event_seconds": float(
                    np.median(
                        np.concatenate(
                            [
                                np.diff(a.abs_s)
                                for a in arrs
                                if a.n > 1 and a.match_id in ids["train"]
                            ]
                        )
                    )
                ),
                "tuned": tuned,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- write results ------------------------------------------------------------
    res = {
        "info": info,
        "tuned": tuned,
        "thresholds": thr,
        "pooled_all": pooled_all.to_dict("records"),
        "pooled_one_per_possession": pooled_one.to_dict("records"),
        "pooled_incl_terminal": pooled_term.to_dict("records"),
        "mean_auc_20_one_per_possession_draws": {
            m: [float(np.mean(v)), float(np.std(v))] for m, v in draw_auc.items()
        },
        "bootstrap_all": boot_all.to_dict("records"),
        "bootstrap_one": boot_one.to_dict("records"),
        "tipping": {m: tip_out[m][0] for m in tip_out},
    }
    (OUT / "event_time_results.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8"
    )
    ev_buckets.to_csv(OUT / "buckets_events.csv", index=False)
    sec_buckets.to_csv(OUT / "buckets_seconds.csv", index=False)
    for m in rel:
        rel[m].to_csv(OUT / f"reliability_{m}.csv", index=False)

    L = [
        "# Event-time (in-play) danger forecasting: results",
        "",
        "Generated by `scripts/run_event_time.py`. Test split, unchanged match-level split (seed 42).",
        "",
        f"Rows (eligible snapshots) train/val/test: {info['rows']}; possessions with >=1 row: "
        f"{info['possessions_with_rows']}; row prevalence: "
        + ", ".join(f"{k} {v:.3f}" for k, v in info["prevalence_rows"].items())
        + f"; test matches: {info['test_matches']}.",
        "",
        "## Models",
        "",
    ] + [f"- **{m}**: {DESCR[m]}" for m in MODELS]
    L += [
        "",
        "Hyper-parameters (chosen on validation log loss only):",
        "",
        "```",
        json.dumps(tuned, indent=1, default=float),
        "```",
        "",
    ]
    for title, tbl in (
        ("All eligible test snapshots (long possessions contribute more rows)", pooled_all),
        ("One random eligible snapshot per test possession (seed 42)", pooled_one),
        (
            "Sensitivity: eligible + terminal snapshots (last event of non-dangerous possessions, label 0)",
            pooled_term,
        ),
    ):
        L += [f"## {title}", "", md_table(tbl), ""]
    L += [
        "Mean (sd) ROC AUC over 20 independent one-per-possession draws: "
        + "; ".join(f"{m} {np.mean(v):.3f} ({np.std(v):.3f})" for m, v in draw_auc.items()),
        "",
    ]
    for title, b in (
        ("All eligible snapshots", boot_all),
        ("One snapshot per possession", boot_one),
    ):
        L += [f"## Match-level bootstrap 95% intervals ({args.n_boot} resamples): {title}", ""]
        t = b.copy()
        t["95% CI"] = [f"[{a:.3f}, {c:.3f}]" for a, c in zip(t.ci_low, t.ci_high)]
        L += [md_table(t[["model", "stat", "estimate", "95% CI"]]), ""]
    L += [
        "## ROC AUC / AP / Brier by events-so-far bucket",
        "",
        md_table(ev_buckets),
        "",
        "## By elapsed-seconds bucket",
        "",
        md_table(sec_buckets),
        "",
    ]
    for m in rel:
        L += [f"## Reliability table: {m} (all eligible test snapshots)", "", md_table(rel[m]), ""]
    L += [
        f"## Tipping points (threshold = validation score quantile with predicted-positive rate = validation prevalence {va.y.mean():.3f})",
        "",
    ]
    for m in tip_out:
        s, tdist, jdist = tip_out[m]
        L += [
            f"### {m} (threshold {thr[m]:.3f})",
            "",
            "```",
            json.dumps(s, indent=1, default=float),
            "```",
            "",
            "First eligible event where P >= threshold; rate = share / base frequency among eligible events:",
            "",
            md_table(tdist),
            "",
            "Largest single-step increase in P:",
            "",
            md_table(jdist),
            "",
        ]
    (OUT / "event_time_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
