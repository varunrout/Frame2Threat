"""Pass option ranking model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Pitch centre y
_GOAL_Y: float = 40.0
_GOAL_X: float = 120.0


class _Classifier(Protocol):
    """Minimal duck-type interface expected from any classifier."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class PassOptionRanker:
    """Rank hypothetical pass recipients for a given passer position.

    For each visible teammate of the passer, a synthetic pass row is
    constructed and scored by *classifier*.  The resulting candidates are
    ranked by predicted threat probability.

    This is useful both for post-hoc analysis ("was the best option
    chosen?") and for interactive visualisation.
    """

    def rank_options(
        self,
        event_uuid: str,
        pass_instances_df: pd.DataFrame,
        frames_df: pd.DataFrame,
        classifier: _Classifier,
        n_candidates: int | None = None,
    ) -> pd.DataFrame:
        """Rank all visible teammate positions as potential pass targets.

        Parameters
        ----------
        event_uuid:
            UUID of the actual pass event to analyse.
        pass_instances_df:
            Full pass instances table.
        frames_df:
            Full freeze-frame table.
        classifier:
            Fitted classifier with a ``predict_proba(X)`` method that
            accepts the same feature columns as the event feature matrix.
        n_candidates:
            If given, return only the top-*n* ranked candidates.

        Returns
        -------
        pd.DataFrame
            Columns: candidate_idx, x, y, predicted_prob, rank.
            Sorted by predicted_prob descending (rank 1 = best).
            Returns an empty DataFrame if the event is not found or there
            are no visible teammates.
        """
        actual_row = _find_row(pass_instances_df, event_uuid)
        if actual_row is None:
            logger.warning("rank_options: event_uuid %s not found", event_uuid)
            return pd.DataFrame()

        event_frame = frames_df[frames_df["event_uuid"] == event_uuid].copy()
        if event_frame.empty:
            logger.warning("rank_options: no 360 frame for %s", event_uuid)
            return pd.DataFrame()

        # Teammates excluding the actor (passer)
        candidates = event_frame[
            event_frame["teammate"].fillna(False).astype(bool)
            & ~event_frame["actor"].fillna(False).astype(bool)
        ].reset_index(drop=True)

        if candidates.empty:
            logger.info("rank_options: no visible teammates for %s", event_uuid)
            return pd.DataFrame()

        rows: list[dict] = []
        for idx, player in candidates.iterrows():
            cx = float(player["x"])
            cy = float(player["y"])

            synth_row = _build_synthetic_pass_row(actual_row, cx, cy)
            synth_df = pd.DataFrame([synth_row])

            try:
                proba = classifier.predict_proba(synth_df)
                # Take probability of positive class (index 1)
                prob = float(proba[0, 1]) if proba.ndim == 2 else float(proba[0])
            except Exception as exc:
                logger.debug("predict_proba failed for candidate %d: %s", idx, exc)
                prob = np.nan

            rows.append(
                {
                    "candidate_idx": int(idx),
                    "x": cx,
                    "y": cy,
                    "predicted_prob": prob,
                }
            )

        result = pd.DataFrame(rows)
        result = result.sort_values("predicted_prob", ascending=False, na_position="last")
        result["rank"] = range(1, len(result) + 1)
        result = result.reset_index(drop=True)

        if n_candidates is not None:
            result = result.head(n_candidates)

        return result

    def compare_actual_to_alternatives(
        self,
        event_uuid: str,
        pass_instances_df: pd.DataFrame,
        frames_df: pd.DataFrame,
        classifier: _Classifier,
    ) -> dict:
        """Compare the actual pass choice to all alternative recipients.

        Parameters
        ----------
        event_uuid:
            UUID of the pass event to analyse.
        pass_instances_df, frames_df, classifier:
            As in :meth:`rank_options`.

        Returns
        -------
        dict
            Keys:
            - ``event_uuid`` (str)
            - ``actual_prob`` (float): predicted probability for the actual pass
            - ``actual_rank`` (int): rank of the actual pass among all options
            - ``n_candidates`` (int): total number of pass options scored
            - ``alternatives`` (pd.DataFrame): full ranked option table
            - ``best_option`` (dict): row of the highest-ranked alternative
        """
        actual_row = _find_row(pass_instances_df, event_uuid)
        if actual_row is None:
            logger.warning("compare_actual_to_alternatives: %s not found", event_uuid)
            return {}

        # Score the actual pass endpoint
        actual_ex = float(actual_row.get("end_x", 0.0))
        actual_ey = float(actual_row.get("end_y", 0.0))

        actual_synth = pd.DataFrame([_build_synthetic_pass_row(actual_row, actual_ex, actual_ey)])
        try:
            actual_proba = classifier.predict_proba(actual_synth)
            actual_prob = (
                float(actual_proba[0, 1]) if actual_proba.ndim == 2 else float(actual_proba[0])
            )
        except Exception:
            actual_prob = np.nan

        alternatives = self.rank_options(event_uuid, pass_instances_df, frames_df, classifier)

        if alternatives.empty:
            return {
                "event_uuid": event_uuid,
                "actual_prob": actual_prob,
                "actual_rank": 1,
                "n_candidates": 0,
                "alternatives": alternatives,
                "best_option": {},
            }

        # Determine where the actual endpoint would rank
        all_probs = alternatives["predicted_prob"].values
        actual_rank = int((all_probs > actual_prob).sum()) + 1

        best_option = alternatives.iloc[0].to_dict() if len(alternatives) > 0 else {}

        return {
            "event_uuid": event_uuid,
            "actual_prob": actual_prob,
            "actual_rank": actual_rank,
            "n_candidates": len(alternatives),
            "alternatives": alternatives,
            "best_option": best_option,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_row(
    pass_instances_df: pd.DataFrame,
    event_uuid: str,
) -> "pd.Series | None":
    """Return the pass row for *event_uuid*, or None."""
    mask = pass_instances_df["event_uuid"] == event_uuid
    if not mask.any():
        return None
    return pass_instances_df.loc[mask].iloc[0]


def _build_synthetic_pass_row(
    actual_row: "pd.Series",
    end_x: float,
    end_y: float,
) -> dict:
    """Construct a synthetic pass row with a different end position.

    All fields are copied from *actual_row* except end coordinates and
    derived features (pass_length, pass_angle, x_gain, dist_to_goal_end,
    goal_dist_gain) which are recomputed.

    Parameters
    ----------
    actual_row:
        The real pass event row.
    end_x, end_y:
        Hypothetical pass destination (a teammate's position).

    Returns
    -------
    dict
        Synthetic pass row with updated spatial features.
    """
    row = actual_row.to_dict() if isinstance(actual_row, pd.Series) else dict(actual_row)

    sx = float(row.get("start_x", 0.0))
    sy = float(row.get("start_y", 0.0))

    dx = end_x - sx
    dy = end_y - sy

    pass_length = float(np.sqrt(dx**2 + dy**2))
    pass_angle = float(np.arctan2(dy, dx))

    dist_to_goal_start = float(np.sqrt((_GOAL_X - sx) ** 2 + (_GOAL_Y - sy) ** 2))
    dist_to_goal_end = float(np.sqrt((_GOAL_X - end_x) ** 2 + (_GOAL_Y - end_y) ** 2))
    goal_dist_gain = dist_to_goal_start - dist_to_goal_end

    row.update(
        {
            "end_x": end_x,
            "end_y": end_y,
            "pass_length": pass_length,
            "pass_angle": pass_angle,
            "pass_angle_rad": pass_angle,
            "pass_angle_sin": float(np.sin(pass_angle)),
            "pass_angle_cos": float(np.cos(pass_angle)),
            "x_gain": dx,
            "is_forward": float(end_x > sx),
            "dist_to_goal_end": dist_to_goal_end,
            "goal_dist_gain": goal_dist_gain,
        }
    )
    return row
