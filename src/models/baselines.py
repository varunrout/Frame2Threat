"""Rule-based benchmark models for line_break and dangerous_progression."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds for rule-based line break
_LINE_BREAK_MIN_LENGTH: float = 15.0
_LINE_BREAK_MIN_END_X: float = 75.0

# Thresholds for rule-based dangerous progression
_DANGER_PROG_MIN_X_GAIN: float = 10.0
_DANGER_PROG_MIN_END_X: float = 70.0


class RuleBasedLineBreak:
    """Rule-based sanity-check baseline for line-break prediction.

    A pass is classified as a line break when:
    - ``pass_length > 15`` metres, AND
    - ``end_x > 75`` (ball is played into the final quarter of the pitch)

    This provides a strong heuristic baseline that exploits the fact that
    long forward passes into advanced positions often break defensive lines.
    """

    def predict(
        self,
        pass_instances_df: pd.DataFrame,
        frames_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Predict line-break labels using the rule heuristic.

        Parameters
        ----------
        pass_instances_df:
            Pass instances table with at least pass_length and end_x columns.
        frames_df:
            Ignored.  Present for API compatibility with other models.

        Returns
        -------
        pd.Series
            Boolean prediction indexed by the DataFrame's index.
        """
        if pass_instances_df is None or pass_instances_df.empty:
            logger.warning("RuleBasedLineBreak.predict: empty input")
            return pd.Series(dtype=bool)

        pass_length = pass_instances_df.get(
            "pass_length", pd.Series(0.0, index=pass_instances_df.index)
        ).fillna(0.0).astype(float)

        end_x = pass_instances_df.get(
            "end_x", pd.Series(0.0, index=pass_instances_df.index)
        ).fillna(0.0).astype(float)

        prediction = (
            (pass_length > _LINE_BREAK_MIN_LENGTH)
            & (end_x > _LINE_BREAK_MIN_END_X)
        )

        logger.debug(
            "RuleBasedLineBreak: %d / %d predicted positive",
            prediction.sum(),
            len(prediction),
        )
        return prediction


class RuleBasedDangerousProgression:
    """Rule-based sanity-check baseline for dangerous-progression prediction.

    A pass is classified as dangerous progression when:
    - ``x_gain > 10`` metres (significant forward movement), AND
    - ``end_x > 70`` (ball reaches the final third)
    """

    def predict(
        self,
        pass_instances_df: pd.DataFrame,
    ) -> pd.Series:
        """Predict dangerous-progression labels using the rule heuristic.

        Parameters
        ----------
        pass_instances_df:
            Pass instances table.  Must have start_x / end_x (to compute
            x_gain) or a pre-computed x_gain column.

        Returns
        -------
        pd.Series
            Boolean prediction indexed by the DataFrame's index.
        """
        if pass_instances_df is None or pass_instances_df.empty:
            logger.warning("RuleBasedDangerousProgression.predict: empty input")
            return pd.Series(dtype=bool)

        if "x_gain" in pass_instances_df.columns:
            x_gain = pass_instances_df["x_gain"].fillna(0.0).astype(float)
        else:
            start_x = pass_instances_df.get(
                "start_x", pd.Series(0.0, index=pass_instances_df.index)
            ).fillna(0.0).astype(float)
            end_x_series = pass_instances_df.get(
                "end_x", pd.Series(0.0, index=pass_instances_df.index)
            ).fillna(0.0).astype(float)
            x_gain = end_x_series - start_x

        end_x = pass_instances_df.get(
            "end_x", pd.Series(0.0, index=pass_instances_df.index)
        ).fillna(0.0).astype(float)

        prediction = (
            (x_gain > _DANGER_PROG_MIN_X_GAIN)
            & (end_x > _DANGER_PROG_MIN_END_X)
        )

        logger.debug(
            "RuleBasedDangerousProgression: %d / %d predicted positive",
            prediction.sum(),
            len(prediction),
        )
        return prediction
