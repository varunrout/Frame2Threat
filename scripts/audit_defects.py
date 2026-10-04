"""Reproduce the four defects found in the self-audit (Phase 0 audit).

Usage:  python scripts/audit_defects.py [--out reports/audit/defect_audit_before.md]

Loads the cached parquets only; changes nothing.  See reports/audit/ for the
committed outputs.  The "BEFORE" measurements describe the fraction-based
formulation and the pass label as they were prior to the fixes.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROC = Path("data/processed")
FRACS = (0.25, 0.5, 0.75)
SHOT_ID = 5  # TYPE_VOCAB id for "shot" (parse_possessions.py; unchanged by the fix)


def _events(seq) -> list[dict]:
    return json.loads(seq) if isinstance(seq, str) else list(seq)


def _in_box(x: float, y: float) -> bool:
    return x >= 102.0 and 18.0 <= y <= 62.0


def _is_danger_event(ev: dict, include_ends: bool = False) -> bool:
    """Shot or box entry.  ``include_ends=False`` mirrors the stored label definition
    (event location only); ``True`` also counts pass/carry end locations."""
    if float(ev.get("own_team", 1.0)) < 0.5:  # defending-team event: opposite frame, never danger
        return False
    x, y = float(ev["loc_x_norm"]) * 120.0, float(ev["loc_y_norm"]) * 80.0
    hit = int(ev["type_id"]) == SHOT_ID or _in_box(x, y)
    if include_ends and not hit:
        hit = _in_box(float(ev["end_x_norm"]) * 120.0, float(ev["end_y_norm"]) * 80.0)
    return hit


def audit_vocab(poss: pd.DataFrame) -> list[str]:
    counts: dict[int, int] = {}
    for seq in poss["event_sequence"]:
        for ev in seq:
            counts[int(ev["type_id"])] = counts.get(int(ev["type_id"]), 0) + 1
    out = ["## 1. Vocabulary counts (`type_id` over all `event_sequence` entries)", ""]
    out += ["| type_id | count |", "|---:|---:|"]
    for k in sorted(counts):
        out.append(f"| {k} | {counts[k]:,} |")
    out += [
        "",
        f"Total events: {sum(counts.values()):,}. Id 3 (ball receipt) = {counts.get(3, 0):,}; "
        f"id 4 (dribble) = {counts.get(4, 0):,}.",
        "",
    ]
    return out


def audit_length_leak(poss: pd.DataFrame) -> list[str]:
    n = poss["event_sequence"].map(len).to_numpy()
    y = poss["poss_dangerous"].astype(int).to_numpy()
    out = ["## 2. Possession-length leak", ""]
    out.append(f"- ROC AUC of total length n for `poss_dangerous`: **{roc_auc_score(y, n):.3f}**")
    out.append(
        f"- Median length: dangerous = {np.median(n[y == 1]):.0f}, safe = {np.median(n[y == 0]):.0f}"
    )
    for f in FRACS:
        k = np.ceil(n * f).astype(int)
        out.append(f"- AUC of prefix length ceil(n*{f}): {roc_auc_score(y, k):.3f}")
    out.append("")
    return out


def audit_realised(poss: pd.DataFrame) -> list[str]:
    seqs = [_events(s) for s in poss["event_sequence"]]
    y = poss["poss_dangerous"].astype(int).to_numpy()
    n = np.array([len(s) for s in seqs])
    out = ["## 3. Already-realised outcome in the prefix", ""]
    out += [
        "Two box definitions: *event location* (what `poss_dangerous` uses; safe = 0% by "
        "construction) and *+ pass/carry end location* (the initial audit's figures; the label ignores "
        "ends, so some 'safe' possessions already put the ball in the box via a pass/carry end).",
        "",
        "| definition | f | dangerous already realised | safe already realised |",
        "|---|---:|---:|---:|",
    ]
    for name, inc in (("event location", False), ("+ end location", True)):
        fd = np.array(
            [next((i for i, e in enumerate(s) if _is_danger_event(e, inc)), 10**9) for s in seqs]
        )
        for f in FRACS:
            k = np.ceil(n * f).astype(int)
            r = fd < k
            out.append(f"| {name} | {f} | {r[y == 1].mean():.1%} | {r[y == 0].mean():.1%} |")
    out.append("")
    return out


def audit_pass_label(passes: pd.DataFrame) -> list[str]:
    d = passes.dropna(subset=["end_x", "dangerous_progression_k"])
    lab = d["dangerous_progression_k"].astype(int)
    hi = d["end_x"] >= 80
    out = ["## 4. Pass-label self-determination (`dangerous_progression_k`)", ""]
    out.append(f"- P(label | end_x >= 80): **{lab[hi].mean():.3f}**")
    out.append(f"- P(label | end_x < 80): **{lab[~hi].mean():.3f}**")
    out.append(f"- Share of positives with end_x >= 80: **{hi[lab == 1].mean():.3f}**")
    out.append(f"- ROC AUC of end_x alone: **{roc_auc_score(lab, d['end_x']):.3f}**")
    out.append(f"- n passes = {len(d):,}; prevalence = {lab.mean():.3f}")
    out.append("")
    return out


def audit_vocab_after(poss: pd.DataFrame) -> list[str]:
    """Same counts as section 1, decoded with the shared enum (src/data/event_types.py)."""
    from src.data.event_types import label

    counts: dict[int, int] = {}
    for seq in poss["event_sequence"]:
        for ev in seq:
            counts[int(ev["type_id"])] = counts.get(int(ev["type_id"]), 0) + 1
    out = ["## 1. Vocabulary: one encoder/decoder (`src/data/event_types.py`)", ""]
    out += ["| type_id | event type (shared decoder) | count |", "|---:|---|---:|"]
    out += [f"| {k} | {label(k)} | {counts[k]:,} |" for k in sorted(counts)]
    out += [
        "",
        "id 3 is Ball Receipt and id 4 is Dribble in every module (tests/test_event_types.py).",
        "",
    ]
    return out


def audit_event_time(poss_keys: pd.DataFrame) -> list[str]:
    """Length leak and realised-outcome share of the event-time snapshot formulation."""
    import pickle

    from src.features.event_time import build_possession_arrays, danger_flags, snapshot_labels

    cache = PROC / "event_time_arrays.pkl"
    if cache.exists():
        arrs = pickle.loads(cache.read_bytes())
    else:
        ev = pd.read_parquet(PROC / "events_parsed.parquet")
        arrs = build_possession_arrays(ev, poss_keys)
    ys, ts, ns, realised, rows, pids = [], [], [], 0, 0, []
    for i, p in enumerate(arrs):
        elig, lab, _ = snapshot_labels(p)
        d = danger_flags(p)
        for t in np.nonzero(elig)[0]:
            ys.append(int(lab[t]))
            ts.append(int(t) + 1)
            pids.append(i)
            rows += 1
            realised += int(d[: t + 1].any())
    y, ev_so_far, pid = np.array(ys), np.array(ts), np.array(pids)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(pid))
    _, first = np.unique(pid[order], return_index=True)
    one = np.sort(order[first])
    out = ["## 2-3. Event-time (in-play) formulation", ""]
    out.append(
        f"- Eligible snapshot rows: {rows:,} from {len(set(pids)):,} possessions; prevalence {y.mean():.3f}"
    )
    out.append(
        f"- Share of snapshots whose prefix 0..t already contains a shot or box entry: "
        f"**{realised / rows:.1%}** (0% by construction)"
    )
    out.append(
        f"- ROC AUC of events-so-far alone (the only length information a snapshot has), all rows: "
        f"**{roc_auc_score(y, ev_so_far):.3f}**; one random snapshot per possession: "
        f"**{roc_auc_score(y[one], ev_so_far[one]):.3f}**"
    )
    out.append(
        "- The total possession length n is not available to any feature or model "
        "(tests/test_event_time.py appends future events and checks the row for t is unchanged)."
    )
    out.append("")
    return out


def audit_pass_label_after(passes: pd.DataFrame) -> list[str]:
    from src.labels.progression_beyond_pass import compute_progression_beyond_pass

    ev = pd.read_parquet(PROC / "events_parsed.parquet")
    new = compute_progression_beyond_pass(ev, passes)
    out = ["## 4. Pass labels: original vs `progression_beyond_pass_k`", ""]
    out += [
        "| label | prevalence | P(label given end_x >= 80) | P(label given end_x < 80) | share of positives with end_x >= 80 | AUC of end_x alone |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    d = new.dropna(subset=["end_x"])
    hi = d["end_x"] >= 80
    for col in ("dangerous_progression_k", "progression_beyond_pass_k"):
        lab = d[col].astype(int)
        out.append(
            f"| {col} | {lab.mean():.3f} | {lab[hi].mean():.3f} | {lab[~hi].mean():.3f} | "
            f"{hi[lab == 1].mean():.3f} | {roc_auc_score(lab, d['end_x']):.3f} |"
        )
    out.append("")
    return out


def audit_frame_fix(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    out = ["## Possession labels changed by the coordinate-frame fix", ""]
    out += [
        "| label | old rate | corrected rate | true -> false | false -> true |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in ("poss_dangerous", "poss_entered_box", "poss_has_shot", "poss_entered_final_third"):
        o, n = old[c].astype(bool), new[c].astype(bool)
        out.append(
            f"| {c} | {o.mean():.4f} | {n.mean():.4f} | {int((o & ~n).sum())} | {int((~o & n).sum())} |"
        )
    out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--mode",
        choices=["before", "after"],
        default="before",
        help="before: reproduce the defects on the pre-fix possession table; after: report the "
        "equivalents for the corrected code and data",
    )
    args = ap.parse_args()
    backup = PROC / "possession_sequences_pre_frame_fix.parquet"
    passes = pd.read_parquet(PROC / "pass_instances.parquet")
    if args.mode == "before":
        poss = pd.read_parquet(backup if backup.exists() else PROC / "possession_sequences.parquet")
        poss["event_sequence"] = poss["event_sequence"].map(_events)
        lines = [
            "# Defect audit (before fixes)",
            "",
            f"Possessions: {len(poss):,}; passes: {len(passes):,}.",
            "",
        ]
        lines += audit_vocab(poss) + audit_length_leak(poss) + audit_realised(poss)
        lines += audit_pass_label(passes)
    else:
        poss = pd.read_parquet(PROC / "possession_sequences.parquet")
        poss["event_sequence"] = poss["event_sequence"].map(_events)
        lines = [
            "# Defect audit (after fixes)",
            "",
            f"Possessions: {len(poss):,}; passes: {len(passes):,}.",
            "",
        ]
        lines += audit_vocab_after(poss)
        lines += audit_event_time(poss[["match_id", "possession_id"]])
        lines += audit_pass_label_after(passes)
        if backup.exists():
            old = pd.read_parquet(backup)
            lines += audit_frame_fix(old, poss)
        lines += [
            "## For reference: the superseded fraction-based measurements on the corrected labels",
            "",
        ]
        lines += audit_length_leak(poss)[2:] + [""] + audit_realised(poss)[2:]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
