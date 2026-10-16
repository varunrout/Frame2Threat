"""EXP-033: final pass models on ``progression_beyond_pass_k`` (the pre-specified recipe).

Recipe: XGBoost with the ``configs/model_baseline.yaml`` settings, early stopping on validation, **no**
``scale_pos_weight``; a calibrator is fitted on validation only if validation reliability shows one is
needed (rule below).  Unchanged match-level split; test is read once, after training.

* ``models/pass_progression_event_only.joblib``  (27 event features)
* ``models/pass_progression_event_360.joblib``   (27 + 14 geometry features)

Also re-runs the 360 comparison with this recipe on all test passes and on the 360-only subset
(paired match-level bootstrap, 1,000 resamples).  Writes ``reports/pass_label/pass_model_final.{md,json}``.

Calibrator rule (validation): fit Platt scaling on validation if |calibration intercept| > 0.15 or the
calibration slope is outside [0.85, 1.15]; otherwise save no calibrator.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.event_time_eval import (
    bootstrap_by_match,
    metrics,
    reliability_table,
)  # noqa: E402
from src.models.pass_progression import (  # noqa: E402
    EVENT_360,
    EVENT_ONLY,
    LABEL,
    LABEL_DEFINITION,
    PassProgressionModel,
    build_feature_frame,
    model_path,
)
from train_pass_gnn import calibration, git_info, load_data  # noqa: E402

warnings.filterwarnings("ignore")
PROC = Path("data/processed")
OUT = Path("reports/pass_label")
STATS = ("roc_auc", "avg_precision", "brier", "log_loss")


class Platt:
    """Platt scaling on the logit of the raw probability (fitted on validation only)."""

    def __init__(self, p_val, y_val):
        lg = self._logit(p_val)
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg.reshape(-1, 1), y_val)

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict(self, p):
        return self.lr.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]


def train_one(kind, Xs, y, cfg):
    m = XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric=cfg["eval_metric"],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        random_state=cfg["random_state"],
        n_jobs=-1,
    )  # no scale_pos_weight
    m.fit(Xs["train"], y["train"], eval_set=[(Xs["val"], y["val"])], verbose=False)
    raw_val = m.predict_proba(Xs["val"])[:, 1]
    cal_val = calibration(y["val"], raw_val)
    needs = abs(cal_val["intercept"]) > 0.15 or not (0.85 <= cal_val["slope"] <= 1.15)
    calibrator = Platt(raw_val, y["val"]) if needs else None
    art = PassProgressionModel(m, list(Xs["train"].columns), calibrator, LABEL, kind=kind)
    return art, {
        "validation_calibration_raw": cal_val,
        "calibrator_fitted": bool(needs),
        "best_iteration": int(m.best_iteration),
        "n_trees": int(m.best_iteration) + 1,
    }


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


def main() -> None:
    for name in (EVENT_ONLY, EVENT_360):
        if model_path(name).exists():
            raise SystemExit(
                f"{model_path(name)} already exists; refusing to overwrite (move it first)"
            )
    cfg = yaml.safe_load(open("configs/model_baseline.yaml"))["xgboost"]
    passes, parts = load_data()
    geo = pd.read_parquet(PROC / "geometry_features_v1.parquet")
    y = {s: d[LABEL].astype(int).to_numpy() for s, d in parts.items()}
    X = {s: build_feature_frame(d) for s, d in parts.items()}
    X360 = {s: build_feature_frame(d, geo, with_360=True) for s, d in parts.items()}
    assert (
        list(X360["train"].columns)[:27] == list(X["train"].columns)
        and X360["train"].shape[1] == 41
    )

    arts, info = {}, {}
    for kind, Xs, name in (("event_only", X, EVENT_ONLY), ("event_360", X360, EVENT_360)):
        arts[name], info[name] = train_one(kind, Xs, y, cfg)
        print(name, info[name], flush=True)

    te = parts["test"]
    has360 = te["has_360"].astype(bool).to_numpy()
    preds = {
        "event_only": arts[EVENT_ONLY].predict_proba_frame(X["test"]),
        "event_360": arts[EVENT_360].predict_proba_frame(X360["test"]),
    }
    yt, mid = y["test"], te["match_id"].to_numpy()
    sub = {"all test passes": np.ones(len(yt), bool), "360-only subset": has360}
    results = {"views": {}}
    lines = [
        "# Final pass models (EXP-033): XGBoost, early stopping on validation, no `scale_pos_weight`",
        "",
        "Unchanged match-level split; test read once.  Supersedes the EXP-027 recipe (`scale_pos_weight`, all 500 trees), "
        "whose probabilities were inflated (mean prediction 0.41 vs prevalence 0.224).",
        "",
        f"Trees: event-only {info[EVENT_ONLY]['n_trees']}, event+360 {info[EVENT_360]['n_trees']}.  Calibrator fitted on validation: "
        f"event-only {info[EVENT_ONLY]['calibrator_fitted']}, event+360 {info[EVENT_360]['calibrator_fitted']} (rule in the script docstring).  "
        f"Validation calibration (raw): event-only intercept {info[EVENT_ONLY]['validation_calibration_raw']['intercept']:+.3f} / slope "
        f"{info[EVENT_ONLY]['validation_calibration_raw']['slope']:.3f}; event+360 {info[EVENT_360]['validation_calibration_raw']['intercept']:+.3f} / "
        f"{info[EVENT_360]['validation_calibration_raw']['slope']:.3f}.",
        "",
    ]
    excl = []
    for view, mask in sub.items():
        rows = pd.DataFrame({"match_id": mid[mask], "y": yt[mask]})
        P = {k: v[mask] for k, v in preds.items()}
        boot = bootstrap_by_match(
            rows, P, deltas=[("event_360", "event_only")], stats=STATS, n_boot=1000, seed=42
        )

        def ci(model, stat, sign=False, nd=4):
            r = boot[(boot.model == model) & (boot.stat == stat)].iloc[0]
            f = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
            return f"{f.format(r.estimate)} [{f.format(r.ci_low)}, {f.format(r.ci_high)}]", r

        point = {k: metrics(yt[mask], P[k]) for k in P}
        calib = {k: calibration(yt[mask], P[k]) for k in P}
        results["views"][view] = {
            "n": int(mask.sum()),
            "matches": int(len(set(mid[mask]))),
            "prevalence": float(yt[mask].mean()),
            "point": point,
            "calibration": calib,
            "bootstrap": boot.to_dict("records"),
        }
        tab = pd.DataFrame(
            [
                {
                    "model": k,
                    "n": int(mask.sum()),
                    **{s: ci(k, s)[0] for s in STATS},
                    "cal. intercept": f"{calib[k]['intercept']:+.2f}",
                    "cal. slope": f"{calib[k]['slope']:.2f}",
                    "mean pred": f"{calib[k]['mean_pred']:.3f}",
                }
                for k in P
            ]
        )
        dtab = pd.DataFrame(
            [
                {
                    "statistic": s,
                    "event+360 - event-only [95% CI]": ci("event_360 - event_only", s, True)[0],
                    "interval excludes zero": (
                        "yes"
                        if (lambda r: r.ci_low > 0 or r.ci_high < 0)(
                            ci("event_360 - event_only", s)[1]
                        )
                        else "no"
                    ),
                }
                for s in STATS
            ]
        )
        for s in STATS:
            r = ci("event_360 - event_only", s)[1]
            if r.ci_low > 0 or r.ci_high < 0:
                excl.append((view, s, float(r.estimate), float(r.ci_low), float(r.ci_high)))
        lines += [
            f"## {view}: {int(mask.sum()):,} passes, {len(set(mid[mask]))} matches, positive rate {yt[mask].mean():.3f}",
            "",
            md(tab),
            "",
            "Paired match-level bootstrap (1,000 resamples), event+360 minus event-only:",
            "",
            md(dtab),
            "",
        ]
    results["intervals_excluding_zero"] = excl
    lines += ["## Conclusion on the 360 gain", ""]
    if excl:
        lines += [
            "**At least one interval excludes zero** (statistic: view, estimate [95% CI]): "
            + "; ".join(f"{s}: {v}, {e:+.4f} [{lo:+.4f}, {hi:+.4f}]" for v, s, e, lo, hi in excl)
            + ".  "
            '"360 adds nothing measurable" no longer holds for those statistics; see the documentation for the updated reading.',
            "",
        ]
    else:
        lines += [
            'No interval excludes zero on either view: "360 adds nothing measurable" still holds.',
            "",
        ]
    for k in ("event_only", "event_360"):
        lines += [
            f"### Reliability, {k} (all test passes, 10 equal-width bins)",
            "",
            md(reliability_table(yt, preds[k])),
            "",
        ]
    (OUT / "pass_model_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "pass_model_final.json").write_text(
        json.dumps(results, indent=1, default=float), encoding="utf-8"
    )
    for k in ("event_only", "event_360"):
        reliability_table(yt, preds[k]).to_csv(OUT / f"pass_model_reliability_{k}.csv", index=False)

    # ---- save artefacts + metadata
    git = git_info()
    for kind, name in (("event_only", EVENT_ONLY), ("event_360", EVENT_360)):
        art = arts[name]
        full = results["views"]["all test passes"]
        boot_all = pd.DataFrame(full["bootstrap"])
        meta = {
            "name": name,
            "label": LABEL,
            "label_definition": LABEL_DEFINITION,
            "recipe": "XGBoost, early stopping on validation, no scale_pos_weight; "
            + (
                "Platt calibrator fitted on validation"
                if info[name]["calibrator_fitted"]
                else "no calibrator (validation reliability did not call for one)"
            ),
            "features": art.features,
            "n_features": len(art.features),
            "hyperparameters": {
                **{
                    k: cfg[k]
                    for k in (
                        "n_estimators",
                        "max_depth",
                        "learning_rate",
                        "subsample",
                        "colsample_bytree",
                        "early_stopping_rounds",
                        "eval_metric",
                        "random_state",
                    )
                },
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "n_trees": info[name]["n_trees"],
            },
            "training_data": "64 World Cup 2022 matches (360 on 87% of passes) + 35 La Liga 2020/21 Barcelona matches (no 360); geometry features are 0 where there is no freeze frame",
            "matches": {s: int(parts[s]["match_id"].nunique()) for s in parts},
            "passes": {s: int(len(parts[s])) for s in parts},
            "git": git,
            "validation": info[name],
            "test_all_passes": {
                "n": full["n"],
                "matches": full["matches"],
                "metrics_with_95ci_match_bootstrap": {
                    st: dict(
                        zip(
                            ("estimate", "ci_low", "ci_high"),
                            (lambda r: (float(r.estimate), float(r.ci_low), float(r.ci_high)))(
                                boot_all[(boot_all.model == kind) & (boot_all.stat == st)].iloc[0]
                            ),
                        )
                    )
                    for st in STATS
                },
                "calibration": full["calibration"][kind],
                "reliability_table": reliability_table(yt, preds[kind]).to_dict("records"),
            },
            "test_360_subset": {
                "n": results["views"]["360-only subset"]["n"],
                "metrics": results["views"]["360-only subset"]["point"][kind],
                "calibration": results["views"]["360-only subset"]["calibration"][kind],
            },
            "comparison_360": {
                "intervals_excluding_zero": excl,
                "paired_delta_event360_minus_event_only": {
                    view: [r for r in v["bootstrap"] if r["model"] == "event_360 - event_only"]
                    for view, v in results["views"].items()
                },
            },
        }
        joblib.dump(art, model_path(name))
        model_path(name).with_suffix(".json").write_text(
            json.dumps(meta, indent=2, default=float), encoding="utf-8"
        )
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
