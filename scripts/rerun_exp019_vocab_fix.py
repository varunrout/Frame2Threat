"""EXP-020: re-run EXP-019 (tipping points) and the LOO high-impact event-type summary
with the corrected event-type vocabulary, using the existing trained GRU and the
unchanged match-level test split.  Writes reports/audit/exp020_vocab_fix.md.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.event_types import label as type_label
from src.data.parse_possessions import load_possession_sequences
from src.evaluation.early_evaluation import (
    tipping_event_type_distribution,
    tipping_point_analysis,
)
from src.evaluation.possession_attribution import (
    MAX_SEQ_LEN,
    _parse_event_seq,
    leave_one_out_attribution,
    load_gru_model,
    unlock_event_index,
)

warnings.filterwarnings("ignore")
OLD_LABELS = {
    0: "Pass",
    1: "Carry",
    2: "Shot",
    3: "Dribble",
    4: "Pressure",
    5: "Ball Receipt",
    6: "Duel",
    7: "Clearance",
    8: "Interception",
    9: "Other",
}  # superseded decoder


def _dist_table(counts: Counter, base: Counter, title: str) -> list[str]:
    tot, btot = sum(counts.values()), sum(base.values())
    rows = []
    for tid, c in counts.most_common():
        b = base.get(tid, 0) / btot
        rows.append((type_label(tid), c, c / tot, b, (c / tot) / b if b else float("nan")))
    out = [
        f"### {title}",
        "",
        "| event type | n | share | base frequency | rate (share / base) |",
        "|---|---:|---:|---:|---:|",
    ]
    out += [f"| {n} | {c:,} | {s:.1%} | {b:.1%} | {r:.2f} |" for n, c, s, b, r in rows]
    return out + [""]


def main() -> None:
    poss = load_possession_sequences("data/processed/possession_sequences.parquet")
    test_ids = pd.read_parquet("data/processed/test.parquet")["match_id"].unique()
    test = poss[poss["match_id"].isin(test_ids)].reset_index(drop=True)
    gru, _ = load_gru_model("models/gru_poss_dangerous.pt")
    gru.eval()

    tip = tipping_point_analysis(test, gru, threshold=0.5, min_events=3, verbose=False)
    # event-type ids at tipping / max-delta (so we can compute rates)
    tip_ids, delta_ids, base_all, base_nofirst = [], [], Counter(), Counter()
    for _, row in test.iterrows():
        ev = _parse_event_seq(row)
        if len(ev) < 3:
            continue
        ids = [int(e["type_id"]) for e in ev[:MAX_SEQ_LEN]]
        base_all.update(ids)
        base_nofirst.update(ids[1:])
    keep = [
        (row, _parse_event_seq(row))
        for _, row in test.iterrows()
        if len(_parse_event_seq(row)) >= 3
    ]
    assert len(keep) == len(tip)
    for (row, ev), (_, r) in zip(keep, tip.iterrows()):
        if r["tipping_step"] >= 0:
            tip_ids.append(int(ev[int(r["tipping_step"])]["type_id"]))
        delta_ids.append(int(ev[int(r["max_delta_step"])]["type_id"]))

    cross = (tip.tipping_step >= 0).mean()
    cross_d = (tip.loc[tip.poss_dangerous == 1, "tipping_step"] >= 0).mean()
    med = tip.loc[tip.tipping_step >= 0, "tipping_frac"].median()
    new_dist = tipping_event_type_distribution(tip)
    old_view = Counter(OLD_LABELS.get(i, "Other") for i in tip_ids)
    otot = sum(old_view.values())

    lines = [
        "# EXP-020 — EXP-019 and LOO summary re-run with the corrected event-type vocabulary",
        "",
        f"Test split (unchanged, {len(test):,} possessions, {len(keep):,} with >=3 events); "
        "existing GRU `models/gru_poss_dangerous.pt`; threshold 0.50 (as in EXP-019).",
        "",
        "## Tipping point (first event where P(dangerous) >= 0.50)",
        "",
        f"- All possessions crossing 0.50: **{cross:.1%}**",
        f"- Dangerous possessions crossing 0.50: **{cross_d:.1%}**",
        f"- Median tipping fraction (fraction-of-n; superseded formulation, see EXP-021): **{med:.3f}**",
        "",
        "Old decoder applied to the same tipping events (reproduces the published figure):",
        "",
        "| old (wrong) label | share |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v / otot:.1%} |" for k, v in old_view.most_common(5)]
    lines += [
        "",
        "Id 3 is *Ball Receipt* in the encoder, which the old decoder printed as 'Dribble'. "
        "The published '64.4% dribbles' is therefore the ball-receipt share.",
        "",
    ]
    lines += _dist_table(
        Counter(tip_ids), base_all, "Corrected tipping-event distribution (with rate normalisation)"
    )
    lines += _dist_table(
        Counter(delta_ids), base_nofirst, "Corrected largest-single-step-jump event distribution"
    )

    # LOO summary
    unlock_ids, loo_by_type, n_by_type = [], {}, Counter()
    for row, ev in keep:
        ev = ev[:MAX_SEQ_LEN]
        a = leave_one_out_attribution(ev, gru)
        u = unlock_event_index(a)
        if u >= 0:
            unlock_ids.append(int(ev[u]["type_id"]))
        for e, v in zip(ev, a):
            loo_by_type.setdefault(int(e["type_id"]), []).append(float(v))
            n_by_type[int(e["type_id"])] += 1
    lines += _dist_table(
        Counter(unlock_ids), base_all, "LOO 'unlock' event (argmax attribution) by type"
    )
    lines += [
        "### LOO mean attribution by event type",
        "",
        "| event type | n events | mean LOO attribution | mean |LOO| |",
        "|---|---:|---:|---:|",
    ]
    for tid, v in sorted(loo_by_type.items(), key=lambda kv: -np.mean(kv[1])):
        lines.append(
            f"| {type_label(tid)} | {len(v):,} | {np.mean(v):+.4f} | {np.mean(np.abs(v)):.4f} |"
        )
    lines += [
        "",
        "## Known modelling weakness (not changed in this phase)",
        "",
        "`type_id` is fed to the GRU as one raw numeric feature, which imposes a false ordering "
        "on a categorical variable (e.g. Pass=1 < Carry=2 < Ball Receipt=3). Phase 2 replaces it "
        "with a learned embedding in the new event-time model. The v2 GRU itself is unchanged.",
    ]
    Path("reports/audit/exp020_vocab_fix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    new_dist.to_csv("reports/audit/exp020_tipping_event_dist.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
