"""
possession_frame.py
===================
Coordinate-frame handling for possession event sequences.

StatsBomb records every event's ``location`` from the *acting* team's attacking
perspective (own goal at x=0, opposition goal at x=120).  A possession's event stream
also contains the defending team's events (pressure, duel, clearance, block, ...), whose
coordinates are therefore in the *opposite* frame.  Using them unflipped as ball
progression (e.g. ``max_x_reached`` or box entry) credits the possessing team with
positions that are really near its own goal.

This module is the single place that decides:
  * which team is "possessing" (``possessing_team``),
  * which events are the possessing team's (``own_team_flags``),
  * how to express any event in the possessing team's frame (``to_possession_frame``),
  * what counts as a danger event (``is_danger_event``): a shot or a box entry **by the
    possessing team**, judged on the event location.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.event_types import EventType

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
BOX_X = 102.0
BOX_Y_LO = 18.0
BOX_Y_HI = 62.0

#: Event types that only the team in possession performs; their majority identifies it.
_ATTACKING_NAMES = frozenset({"pass", "carry", "shot", "dribble"})


def possessing_team(grp: pd.DataFrame) -> str:
    """Team that performs the majority of the possession's pass/carry/shot/dribble events.

    Ties are broken by first appearance; falls back to the first event's team.
    """
    att = grp.loc[grp["type_name"].str.lower().isin(_ATTACKING_NAMES), "team_name"].dropna()
    if len(att):
        counts = att.value_counts()
        top = {k for k, v in counts.items() if v == counts.max()}
        return next(a for a in att if a in top)
    return grp["team_name"].iloc[0]


def own_team_flags(grp: pd.DataFrame) -> np.ndarray:
    """Boolean array: True where the event was performed by the possessing team."""
    return (grp["team_name"] == possessing_team(grp)).to_numpy()


def possessing_team_at_start(grp: pd.DataFrame) -> str:
    """Causal owner: provider possession team on the first event, else first actor.

    Unlike the retrospective majority heuristic, appending future events cannot
    change this choice. A live feed must supply its possession-team assignment.
    """
    first = grp.iloc[0]
    owner = first.get("possession_team_name")
    if owner is not None and pd.notna(owner):
        return str(owner)
    actor = first["team_name"]
    if pd.isna(actor):
        raise ValueError("first event must identify a possession team or acting team")
    return str(actor)


def to_possession_frame(x, y, own):
    """Express coordinates in the possessing team's frame (flip opponent events)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    own = np.asarray(own, dtype=bool)
    return np.where(own, x, PITCH_LENGTH - x), np.where(own, y, PITCH_WIDTH - y)


def in_box(x, y):
    """True where (x, y) (pitch metres, attacking frame) lies in the opposition penalty area."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (x >= BOX_X) & (y >= BOX_Y_LO) & (y <= BOX_Y_HI)


def is_danger_event(type_id, x, y, own, end_x=None, end_y=None):
    """Shot or box entry by the possessing team (``own`` False -> never a danger event).

    A possessing-team **pass whose end location is in the box** is itself a danger event: the
    ball has entered the box at that pass, and the receipt that follows only records it.
    Judging danger on the event location alone would let a snapshot at such a pass see the
    entry in its own features (``max_x_so_far``, the GRU's end-x/end-y) before the label
    registers it.  Pass ``end_x``/``end_y`` to apply this rule; without them only event
    locations count (the possession-label definition, which is unchanged).
    """
    type_id = np.asarray(type_id)
    own = np.asarray(own, dtype=bool)
    hit = (type_id == int(EventType.SHOT)) | in_box(x, y)
    if end_x is not None and end_y is not None:
        hit = hit | ((type_id == int(EventType.PASS)) & in_box(end_x, end_y))
    return own & hit
