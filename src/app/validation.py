"""
validation.py
=============
Input and artefact validation for the ``frame2threat`` CLI.

Every failure raises :class:`InputError` with a message that names the problem and how to fix it,
so the CLI can print it and exit with status 2 instead of showing a traceback.
"""

from __future__ import annotations

import math
from pathlib import Path

from src.data.event_types import EventType

REQUIRED_EVENT_KEYS = ("type_id", "loc_x_norm", "loc_y_norm")
NORMALISED_KEYS = ("loc_x_norm", "loc_y_norm", "end_x_norm", "end_y_norm", "pass_length_norm")
MAX_EVENTS = 500

#: Columns :func:`src.features.event_time.build_possession_arrays` needs from a flat events table.
REQUIRED_EVENT_COLUMNS = (
    "match_id",
    "possession_id",
    "index",
    "period",
    "minute",
    "second",
    "type_name",
    "team_name",
    "location_x",
    "location_y",
)

EVENT_TIME_ARTEFACTS = ("eventtime_meta.json", "eventtime_xgboost.joblib", "eventtime_gru.pt")


class InputError(ValueError):
    """Raised for invalid user input or missing artefacts; the CLI prints it and exits 2."""


def require_file(path: Path, what: str) -> Path:
    if not path.exists():
        raise InputError(f"{what} not found: {path}")
    if not path.is_file():
        raise InputError(f"{what} is not a file: {path}")
    return path


def require_artefacts(root: Path, model: str = "both") -> None:
    """Fail clearly when the trained event-time models are missing.

    Model files are gitignored; ``python scripts/run_event_time.py`` regenerates them.
    """
    needed = ["eventtime_meta.json"]
    if model in ("xgb", "both"):
        needed.append("eventtime_xgboost.joblib")
    if model in ("gru", "both"):
        needed.append("eventtime_gru.pt")
    missing = [n for n in needed if not (root / "models" / n).exists()]
    if missing:
        raise InputError(
            "missing model artefact(s) in models/: "
            + ", ".join(missing)
            + ". Train them with `python scripts/run_event_time.py` (needs data/processed/)."
        )


def validate_event_list(events: object) -> list[dict]:
    """Validate the ``event_sequence`` schema accepted by ``score-live``."""
    if not isinstance(events, list) or not events:
        raise InputError("the input must contain a non-empty list of event objects")
    if len(events) > MAX_EVENTS:
        raise InputError(f"too many events ({len(events)}); the limit is {MAX_EVENTS}")
    known = {int(v) for v in EventType}
    has_time = ["elapsed_seconds" in e for e in events if isinstance(e, dict)]
    if any(has_time) and not all(has_time):
        raise InputError("'elapsed_seconds' must be given for every event or for none")
    last_t = -math.inf
    for i, e in enumerate(events):
        where = f"event {i}"
        if not isinstance(e, dict):
            raise InputError(f"{where}: expected an object, got {type(e).__name__}")
        for key in REQUIRED_EVENT_KEYS:
            if key not in e:
                raise InputError(f"{where}: missing required key '{key}'")
        try:
            tid = int(e["type_id"])
        except (TypeError, ValueError):
            raise InputError(f"{where}: 'type_id' must be an integer") from None
        if tid not in known:
            raise InputError(f"{where}: unknown type_id {tid}; valid ids are {sorted(known)}")
        for key in NORMALISED_KEYS:
            if key in e:
                try:
                    v = float(e[key])
                except (TypeError, ValueError):
                    raise InputError(f"{where}: '{key}' must be a number") from None
                if not (0.0 <= v <= 1.0) or math.isnan(v):
                    raise InputError(f"{where}: '{key}'={e[key]} outside the normalised range [0, 1]")
        if "elapsed_seconds" in e:
            try:
                t = float(e["elapsed_seconds"])
            except (TypeError, ValueError):
                raise InputError(f"{where}: 'elapsed_seconds' must be a number") from None
            if math.isnan(t) or t < last_t:
                raise InputError(f"{where}: 'elapsed_seconds' must be non-decreasing")
            last_t = t
    return events


def validate_events_table(df) -> None:
    """Validate the flat events table accepted by ``score-batch``."""
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise InputError(
            "events table is missing column(s): "
            + ", ".join(missing)
            + " (expected the events_parsed.parquet schema)"
        )
    if len(df) == 0:
        raise InputError("events table is empty")
