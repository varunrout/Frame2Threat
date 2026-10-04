"""
event_types.py
==============
Single source of truth for the integer encoding of StatsBomb event types used in
possession ``event_sequence`` entries (the ``type_id`` field).

The integer values are **frozen**: the trained GRU (``models/gru_poss_dangerous.pt``)
and the stored ``possession_sequences.parquet`` were built with them.  Do not renumber.

Every encoder / decoder in ``src/`` must import from this module rather than defining
its own mapping (enforced by ``tests/test_event_types.py``).
"""

from __future__ import annotations

from enum import IntEnum


class EventType(IntEnum):
    UNKNOWN = 0
    PASS = 1
    CARRY = 2
    BALL_RECEIPT = 3
    DRIBBLE = 4
    SHOT = 5
    PRESSURE = 6
    DUEL = 7
    CLEARANCE = 8
    INTERCEPTION = 9
    BLOCK = 10
    FOUL_COMMITTED = 11
    FOUL_WON = 12
    GOALKEEPER = 13


#: Canonical (lower-case) StatsBomb-style name for each id.
_CANONICAL_NAME: dict[EventType, str] = {
    EventType.UNKNOWN: "unknown",
    EventType.PASS: "pass",
    EventType.CARRY: "carry",
    EventType.BALL_RECEIPT: "ball receipt",
    EventType.DRIBBLE: "dribble",
    EventType.SHOT: "shot",
    EventType.PRESSURE: "pressure",
    EventType.DUEL: "duel",
    EventType.CLEARANCE: "clearance",
    EventType.INTERCEPTION: "interception",
    EventType.BLOCK: "block",
    EventType.FOUL_COMMITTED: "foul committed",
    EventType.FOUL_WON: "foul won",
    EventType.GOALKEEPER: "goalkeeper",
}

#: Alternative spellings that StatsBomb exports use for the same event type.
_ALIASES: dict[str, EventType] = {
    "ball receipt*": EventType.BALL_RECEIPT,
    "goal keeper": EventType.GOALKEEPER,
}

#: Every StatsBomb type name (lower-case) that is encoded into sequences.
SEQUENCE_TYPES: frozenset[str] = frozenset(
    {name for t, name in _CANONICAL_NAME.items() if t is not EventType.UNKNOWN} | set(_ALIASES)
)

_BY_NAME: dict[str, EventType] = {name: t for t, name in _CANONICAL_NAME.items()} | _ALIASES

N_EVENT_TYPES = len(EventType)  # number of distinct ids, incl. UNKNOWN (for embeddings)


def encode(type_name: str | None) -> int:
    """StatsBomb type name -> integer id (``UNKNOWN`` = 0 for anything unrecognised)."""
    if type_name is None:
        return int(EventType.UNKNOWN)
    return int(_BY_NAME.get(str(type_name).strip().lower(), EventType.UNKNOWN))


def decode(type_id: int) -> str:
    """Integer id -> canonical lower-case name (``"unknown"`` for out-of-range ids)."""
    try:
        return _CANONICAL_NAME[EventType(int(type_id))]
    except ValueError:
        return "unknown"


def label(type_id: int) -> str:
    """Human-readable title-case label for display (CLI / app / reports)."""
    return decode(type_id).title()
