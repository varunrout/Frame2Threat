"""Before/after tables for the possession_length fix in the v1 pass features (EXP-027)."""

from __future__ import annotations

import json
from pathlib import Path

OLD = Path("reports/pass_label/superseded_pre_possession_length_fix/pass_label_results.json")
NEW = Path("reports/pass_label/pass_label_results.json")
MODELS = ["LR_event", "XGB_event", "XGB_event360", "LR_end_x", "XGB_end_xy"]


def ci(rows, model, stat):
    r = next(b for b in rows if b["model"] == model and b["stat"] == stat)
    return f"{r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"


def main() -> None:
    o, n = json.loads(OLD.read_text()), json.loads(NEW.read_text())
    L = [
        "# possession_length fix: before / after (EXP-027)",
        "",
        "*Before* = `superseded_pre_possession_length_fix/` (`groupby('possession_id').cumcount()`, mixing matches); "
        "*after* = grouped by `(match_id, possession_id)` in event order.  Same split, configuration and seeds.",
        "",
    ]
    for lab in ("dangerous_progression_k", "progression_beyond_pass_k"):
        L += [
            f"## `{lab}`",
            "",
            "| model | ROC AUC before | after | AP before | after | Brier before | after |",
            "|---|---|---|---|---|---|---|",
        ]
        po = {r["model"]: r for r in o[lab]["point"]}
        pn = {r["model"]: r for r in n[lab]["point"]}
        for m in MODELS:
            a, b = po[m], pn[m]
            L.append(
                f"| {m} | {a['roc_auc']:.4f} | {b['roc_auc']:.4f} | {a['avg_precision']:.4f} | {b['avg_precision']:.4f} | {a['brier']:.4f} | {b['brier']:.4f} |"
            )
        L += [
            "",
            "360 delta (event+360 minus event-only XGBoost), 1000 match-level bootstrap resamples:",
            "",
            "| subset | statistic | before | after |",
            "|---|---|---|---|",
        ]
        for key, name in (
            ("bootstrap", "all test passes"),
            ("bootstrap_360_subset", "passes with 360 only"),
        ):
            for s, sn in (
                ("roc_auc", "ROC AUC"),
                ("avg_precision", "average precision"),
                ("brier", "Brier"),
            ):
                L.append(
                    f"| {name} | {sn} | {ci(o[lab][key], 'XGB_event360 - XGB_event', s)} | {ci(n[lab][key], 'XGB_event360 - XGB_event', s)} |"
                )
        L += [""]
    Path("reports/pass_label/possession_length_fix_before_after.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8"
    )
    print("\n".join(L))


if __name__ == "__main__":
    main()
