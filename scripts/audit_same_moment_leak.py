"""Self-audit: the same-moment leak in the event-time model.

At a possessing-team pass whose END location is in the box, the snapshot features
(``max_x_so_far`` and the GRU's end-x / end-y inputs) already contain the box entry that the
next event (the receipt, at the pass end) will record, while danger was judged on event
location only.  This script measures how many snapshots are affected and how strongly the
label is tied to them.  Run before and after the fix:

    python scripts/audit_same_moment_leak.py --out reports/audit/same_moment_before.md
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.event_types import EventType  # noqa: E402
from src.data.possession_frame import in_box  # noqa: E402
from src.features.event_time import danger_flags, snapshot_labels  # noqa: E402

PROC = Path("data/processed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="Same-moment leak audit")
    args = ap.parse_args()
    arrs = pickle.loads((PROC / "event_time_arrays.pkl").read_bytes())
    n_rows = n_end_box = pos_end_box = pos_other = n_pos_poss = last_is_end_box = 0
    n_other = 0
    poss = pd.read_parquet(
        PROC / "possession_sequences.parquet",
        columns=["match_id", "possession_id", "poss_dangerous"],
    )
    label_of = {(a, b): bool(c) for a, b, c in poss.itertuples(index=False)}
    disagree = event_time_only = 0
    for p in arrs:
        d_new = bool(danger_flags(p).any())
        d_poss = label_of.get((p.match_id, p.possession_id))
        if d_poss is not None and d_new != d_poss:
            disagree += 1
            event_time_only += int(d_new and not d_poss)
        elig, lab, _ = snapshot_labels(p)
        ends_in_box = p.own & (p.type_id == int(EventType.PASS)) & in_box(p.end_x, p.end_y)
        idx = np.nonzero(elig)[0]
        if len(idx) == 0:
            continue
        flag = ends_in_box[idx]
        y = lab[idx]
        n_rows += len(idx)
        n_end_box += int(flag.sum())
        pos_end_box += int(y[flag].sum())
        n_other += int((~flag).sum())
        pos_other += int(y[~flag].sum())
        if y[0] == 1:
            n_pos_poss += 1
            last_is_end_box += int(flag[-1])
    lines = [
        f"# {args.title}",
        "",
        f"- Eligible snapshots: {n_rows:,}",
        f"- Snapshots at a possessing-team pass whose END is in the box: {n_end_box:,} "
        f"({n_end_box / n_rows:.2%})",
        f"- Label rate at those snapshots: {pos_end_box / max(n_end_box, 1):.1%}; elsewhere: "
        f"{pos_other / max(n_other, 1):.1%}",
        f"- Dangerous possessions whose last eligible snapshot is such a pass: "
        f"{last_is_end_box:,} / {n_pos_poss:,} ({last_is_end_box / max(n_pos_poss, 1):.1%})",
        f"- Possessions where the event-time danger definition disagrees with the stored "
        f"`poss_dangerous` (which was NOT changed): {disagree} ({event_time_only} dangerous only in event time, "
        f"i.e. a pass ending in the box with no receipt in the box)",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
