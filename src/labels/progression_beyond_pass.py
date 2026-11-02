"""
src/labels/progression_beyond_pass.py
=====================================
Pass-level label that does not decide itself from the pass's own destination.

Why a new label
---------------
``dangerous_progression_k`` (src/labels/dangerous_progression.py) scans the next ``k`` events
for a final-third entry, box entry or shot.  The event straight after a completed pass is
normally the *receipt of that pass*, recorded at the pass end location, so any pass ending at
``x >= 80`` is positive about 94% of the time and ``end_x`` alone gives ROC AUC 0.786.
(The old label also reads defending-team events in the opposite coordinate frame.)

Definition: ``progression_beyond_pass_k`` (k = 5 by default)
------------------------------------------------------------
Look at the next ``k`` events of the same possession (the window is the same as for
``dangerous_progression_k``), and

1. drop every event not performed by the passing team (their coordinates are in the
   opposite frame and they are not ball progression);
2. drop **the receipt of this pass**: the first ``Ball Receipt*`` by the passing team in the
   window, provided no passing-team Pass / Carry / Dribble / Shot precedes it (the receipt
   still occupies one of the ``k`` slots);
3. the label is positive if any remaining event satisfies one of

   * **shot** -- it is a Shot;
   * **new box entry** -- its ball position is in the box (x >= 102, 18 <= y <= 62) while the
     pass end was *not* in the box;
   * **further upfield** -- its ball position is at least ``min_gain_m`` (10) metres beyond
     the pass end in x **and** in a more advanced zone than the pass end, with zones ordered
     own third (x < 40) < middle third (40 <= x < 80) < final third outside the box <
     penalty box.

Ball position of an event: ``(pass_end_x, pass_end_y)`` for a Pass, otherwise
``(location_x, location_y)`` (carries contribute their start only, as in the original label).
Passes at the end of a possession have no window and are negative.

See ``reports/label_methodology.md`` for worked examples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.possession_frame import in_box

_RECEIPT_NAMES = {"ball receipt*", "ball receipt"}
_BALL_ACTIONS = {"pass", "carry", "dribble", "shot"}
_DEFAULT_K = 5
_DEFAULT_MIN_GAIN_M = 10.0


def zone_index(x: float, y: float) -> int:
    """0 own third, 1 middle third, 2 final third (outside box), 3 penalty box."""
    if in_box(x, y):
        return 3
    if x >= 80.0:
        return 2
    if x >= 40.0:
        return 1
    return 0


def label_window(
    pass_team: str,
    pass_end: tuple[float, float],
    window: list[dict],
    min_gain_m: float = _DEFAULT_MIN_GAIN_M,
) -> int:
    """Label one pass from the ``k`` events that follow it (pure function, used by tests).

    Each window event is a dict with keys ``type`` (lower-case), ``team``, ``x``, ``y``,
    ``end_x``, ``end_y``.
    """
    ex, ey = pass_end
    end_known = not (np.isnan(ex) or np.isnan(ey))
    end_in_box = bool(in_box(ex, ey)) if end_known else False
    end_zone = zone_index(ex, ey) if end_known else None
    own_action_seen = False
    receipt_dropped = False
    for ev in window:
        if ev["team"] != pass_team:
            continue
        t = ev["type"]
        if t in _RECEIPT_NAMES and not own_action_seen and not receipt_dropped:
            receipt_dropped = True
            continue
        if t in _BALL_ACTIONS:
            own_action_seen = True
        if t == "shot":
            return 1
        if t == "pass" and not (np.isnan(ev["end_x"]) or np.isnan(ev["end_y"])):
            bx, by = ev["end_x"], ev["end_y"]
        else:
            bx, by = ev["x"], ev["y"]
        if np.isnan(bx) or np.isnan(by):
            continue
        if bool(in_box(bx, by)) and not end_in_box:
            return 1
        if end_zone is not None and bx >= ex + min_gain_m and zone_index(bx, by) > end_zone:
            return 1
    return 0


def compute_progression_beyond_pass(
    events_df: pd.DataFrame,
    pass_instances_df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Add ``progression_beyond_pass_k`` (bool) to a copy of ``pass_instances_df``.

    ``config`` may contain ``k`` and ``min_gain_m`` (section ``progression_beyond_pass`` of
    configs/labels.yaml).  Required event columns: event_uuid, match_id, possession_id,
    index, type_name, team_name, location_x, location_y, pass_end_x, pass_end_y.
    """
    cfg = config or {}
    k = int(cfg.get("k", _DEFAULT_K))
    gain = float(cfg.get("min_gain_m", _DEFAULT_MIN_GAIN_M))

    ev = events_df.sort_values(
        ["match_id", "possession_id", "index"], kind="mergesort"
    ).reset_index(drop=True)
    gid = ev.groupby(["match_id", "possession_id"], sort=False).ngroup().to_numpy()
    n = len(ev)
    last_of_group = np.zeros(n, dtype=int)
    # exclusive end offset of each event's possession
    boundaries = np.flatnonzero(np.diff(gid, append=gid[-1] + 1) != 0) + 1
    start = 0
    for b in boundaries:
        last_of_group[start:b] = b
        start = int(b)

    typ = ev["type_name"].str.lower().to_numpy()
    team = ev["team_name"].to_numpy()
    x = ev["location_x"].to_numpy(dtype=float)
    y = ev["location_y"].to_numpy(dtype=float)
    pex = ev["pass_end_x"].to_numpy(dtype=float)
    pey = ev["pass_end_y"].to_numpy(dtype=float)
    pos_of_uuid = pd.Series(np.arange(n), index=ev["event_uuid"].to_numpy())

    out = pass_instances_df.copy()
    positions = pos_of_uuid.reindex(out["event_uuid"].to_numpy()).to_numpy()
    labels = np.zeros(len(out), dtype=bool)
    for r, p in enumerate(positions):
        if np.isnan(p):
            continue
        p = int(p)
        hi = min(p + 1 + k, last_of_group[p])
        window = [
            {
                "type": typ[j],
                "team": team[j],
                "x": x[j],
                "y": y[j],
                "end_x": pex[j],
                "end_y": pey[j],
            }
            for j in range(p + 1, hi)
        ]
        labels[r] = bool(label_window(team[p], (pex[p], pey[p]), window, gain))
    out["progression_beyond_pass_k"] = labels
    return out
