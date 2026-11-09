"""Apply the sealed-holdout selection rule to StatsBomb Open Data and record the result.

Rule (fixed before any holdout scoring, see ``reports/repro/sealed_holdout_protocol.md``): among men's
competition-seasons, pick the one with the most matches that have 360 data and are outside the 99-match
research cohort; break ties by the lowest competition id, then by the latest season. A match counts as
having 360 data when its ``match_status_360`` is ``available`` and its ``three-sixty`` file exists.

Reads only public match lists (no events). Writes ``reports/repro/holdout_rule_check.json``.

Usage:  python scripts/check_holdout_rule.py [--revision SHA]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/statsbomb/open-data/{rev}/data"


def season_start(name: str) -> int:
    return int(str(name)[:4])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default=None, help="open-data commit (default: pinned revision)")
    ap.add_argument("--output", type=Path, default=ROOT / "reports/repro/holdout_rule_check.json")
    args = ap.parse_args()
    pinned = json.loads((ROOT / "configs/research_data.json").read_text(encoding="utf-8"))
    rev = args.revision or pinned["revision"]
    base = RAW.format(rev=rev)
    cohort = set(pd.read_csv(ROOT / "configs/research_split_manifest.csv").match_id)
    comps = requests.get(f"{base}/competitions.json", timeout=60).json()
    rows = []
    for c in comps:
        matches = requests.get(
            f"{base}/matches/{c['competition_id']}/{c['season_id']}.json", timeout=60
        ).json()
        flagged = [
            m["match_id"]
            for m in matches
            if m.get("match_status_360") == "available" and m["match_id"] not in cohort
        ]
        with cf.ThreadPoolExecutor(16) as ex:
            ok = list(
                ex.map(
                    lambda mid: requests.head(f"{base}/three-sixty/{mid}.json", timeout=60).ok,
                    flagged,
                )
            )
        rows.append(
            {
                "competition_id": c["competition_id"],
                "season_id": c["season_id"],
                "competition": c["competition_name"],
                "season": c["season_name"],
                "gender": c["competition_gender"],
                "matches": len(matches),
                "matches_in_cohort": sum(m["match_id"] in cohort for m in matches),
                "matches_360_outside_cohort": int(sum(ok)),
            }
        )
    eligible = [r for r in rows if r["gender"] == "male" and r["matches_360_outside_cohort"] > 0]
    eligible.sort(
        key=lambda r: (
            -r["matches_360_outside_cohort"],
            r["competition_id"],
            -season_start(r["season"]),
        )
    )
    record = {
        "open_data_revision": rev,
        "rule": "most men's-competition-season matches with 360 data outside the 99-match cohort; "
        "ties: lowest competition id, then latest season",
        "selected": eligible[0] if eligible else None,
        "ranking": eligible,
        "excluded_womens": [
            r for r in rows if r["gender"] != "male" and r["matches_360_outside_cohort"] > 0
        ],
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for r in eligible[:6]:
        print(r)
    print("selected:", record["selected"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
