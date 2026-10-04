"""Reproduce the four defects found in the external review (Phase 0 audit).

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
        "construction) and *+ pass/carry end location* (the review's figures; the label ignores "
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    poss = pd.read_parquet(PROC / "possession_sequences.parquet")
    poss["event_sequence"] = poss["event_sequence"].map(_events)
    passes = pd.read_parquet(PROC / "pass_instances.parquet")
    lines = [
        "# Defect audit (before fixes)",
        "",
        f"Possessions: {len(poss):,}; passes: {len(passes):,}.",
        "",
    ]
    lines += audit_vocab(poss) + audit_length_leak(poss) + audit_realised(poss)
    lines += audit_pass_label(passes)
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
