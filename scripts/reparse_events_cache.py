"""Re-parse the cached raw StatsBomb events (data/raw/events_*.parquet, no network) and add
the shot columns (``shot_statsbomb_xg``, ``shot_outcome_name``) to ``events_parsed.parquet``.

Safety: every column that already exists must be identical (matched on ``event_uuid``,
NaN-aware) or the script aborts without writing.  The previous file is kept as
``events_parsed_pre_shot_columns.parquet``.
"""

from __future__ import annotations

import glob
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parse_events import parse_events  # noqa: E402

PROC = Path("data/processed")


def main() -> None:
    files = sorted(glob.glob("data/raw/events_*.parquet"))
    new = pd.concat([parse_events(pd.read_parquet(f)) for f in files], ignore_index=True)
    old = pd.read_parquet(PROC / "events_parsed.parquet")
    assert len(new) == len(old) and set(new.event_uuid) == set(old.event_uuid), "row sets differ"
    n = new.set_index("event_uuid").loc[old["event_uuid"]].reset_index()
    for c in old.columns:
        a, b = old[c], n[c]
        same = (
            a.astype(object).where(a.notna(), None).to_numpy()
            == b.astype(object).where(b.notna(), None).to_numpy()
        )
        if not same.all():
            raise SystemExit(f"column {c} differs in {(~same).sum()} rows; not writing")
    print(
        "existing columns identical;",
        "new columns:",
        [c for c in n.columns if c not in old.columns],
    )
    shutil.copy(PROC / "events_parsed.parquet", PROC / "events_parsed_pre_shot_columns.parquet")
    n[list(old.columns) + [c for c in n.columns if c not in old.columns]].to_parquet(
        PROC / "events_parsed.parquet", index=False
    )
    shots = n[n.type_name == "Shot"]
    print(
        f"shots {len(shots):,}; xG total {shots.shot_statsbomb_xg.sum():.1f}; "
        f"goals {int((shots.shot_outcome_name == 'Goal').sum())}"
    )


if __name__ == "__main__":
    main()
