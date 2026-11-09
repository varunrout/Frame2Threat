"""Frozen alert and calibration definitions (reports/repro/sealed_holdout_protocol.md)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.frozen_metrics import alert_metrics, ece_weighted


def _rows():
    # possession (1,1) dangerous, danger at event 4; (1,2) safe; (1,3) safe
    return pd.DataFrame(
        {
            "match_id": [1, 1, 1, 1, 1, 1, 1],
            "possession_id": [1, 1, 1, 2, 2, 3, 3],
            "t": [0, 1, 2, 0, 1, 0, 1],
            "y": [1, 1, 1, 0, 0, 0, 0],
        }
    )


def test_false_alert_rate_counts_safe_possessions_with_any_alert():
    p = np.array([0.1, 0.6, 0.7, 0.2, 0.9, 0.1, 0.2])
    out = alert_metrics(_rows(), p, 0.5, {(1, 1): 4})
    assert out["safe_possessions"] == 2 and out["dangerous_possessions"] == 1
    assert out["false_alert_rate"] == pytest.approx(0.5)
    assert out["dangerous_alert_rate"] == pytest.approx(1.0)
    assert out["median_lead_events"] == pytest.approx(3.0)  # first alert t=1, danger at 4
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)


def test_threshold_is_inclusive():
    p = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert alert_metrics(_rows(), p, 0.5)["recall"] == pytest.approx(1 / 3)


def test_ece_is_snapshot_weighted():
    # nine snapshots predicted 0.0 and observed 0.0, one predicted 0.95 and observed 0.0
    y = np.array([0] * 9 + [0])
    p = np.array([0.0] * 9 + [0.95])
    assert ece_weighted(y, p) == pytest.approx(0.095)
    # an unweighted bin average would give 0.475
