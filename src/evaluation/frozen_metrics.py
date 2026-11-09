"""Frozen definitions of the alert and calibration metrics for event-time models.

Fixed by ``reports/repro/sealed_holdout_protocol.md`` (2026-11-09). Every report that quotes a false-alert
rate or an ECE for the event-time models uses these functions, so the numbers share one definition.

* Eligible snapshots are the non-terminal rows of ``build_snapshot_table`` (columns ``match_id``,
  ``possession_id``, ``t``, ``y``).
* **False-alert rate**: share of safe possessions (``y == 0``) with at least one eligible snapshot whose
  score is at or above the threshold.
* **ECE**: ten equal-width bins on [0, 1], weighted by snapshot count, in absolute probability units.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.evaluation.event_time_eval import reliability_table

FALSE_ALERT_DEFINITION = (
    "share of safe possessions (no later shot or box entry) with at least one eligible snapshot "
    "at or above the validation-chosen threshold"
)
ECE_DEFINITION = (
    "10 equal-width bins on [0, 1], weighted by snapshot count, over all eligible snapshots; "
    "absolute probability units"
)


def ece_weighted(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Snapshot-weighted expected calibration error over equal-width bins."""
    y = np.asarray(y)
    rel = reliability_table(y, p, n_bins).dropna()
    return float((rel.n * (rel.mean_pred - rel.observed).abs()).sum() / len(y))


def alert_metrics(
    rows: pd.DataFrame,
    p: np.ndarray,
    threshold: float,
    danger_index: Mapping[tuple[int, int], int] | None = None,
) -> dict[str, float | int | None]:
    """Snapshot precision/recall and possession-level alert rates at ``threshold``.

    ``rows`` are eligible snapshots only. ``danger_index`` maps (match_id, possession_id) to the event
    index of the first shot or box entry, for lead times; lead time is that index minus the first alert.
    """
    y = rows["y"].to_numpy().astype(int)
    alert = np.asarray(p) >= threshold
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum())
    frame = pd.DataFrame(
        {
            "match_id": rows["match_id"].to_numpy(),
            "possession_id": rows["possession_id"].to_numpy(),
            "t": rows["t"].to_numpy(),
            "y": y,
            "alert": alert,
        }
    )
    frame["alert_t"] = np.where(frame["alert"], frame["t"], np.iinfo(np.int64).max)
    poss = frame.groupby(["match_id", "possession_id"]).agg(
        y=("y", "max"), alerted=("alert", "any"), first_alert=("alert_t", "min")
    )
    safe = poss[poss.y == 0]
    dangerous = poss[poss.y == 1]
    leads: list[int] = []
    if danger_index is not None:
        for key, r in dangerous[dangerous.alerted].iterrows():
            leads.append(int(danger_index[key]) - int(r.first_alert))
    return {
        "threshold": float(threshold),
        "snapshots": int(len(y)),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "possessions": int(len(poss)),
        "safe_possessions": int(len(safe)),
        "dangerous_possessions": int(len(dangerous)),
        "false_alert_rate": float(safe.alerted.mean()) if len(safe) else float("nan"),
        "dangerous_alert_rate": float(dangerous.alerted.mean()) if len(dangerous) else float("nan"),
        "median_lead_events": float(np.median(leads)) if leads else None,
    }
