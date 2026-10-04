"""Tests for the shared event-type vocabulary (src/data/event_types.py)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import event_types as et
from src.data.event_types import SEQUENCE_TYPES, EventType, decode, encode

SRC = Path(__file__).resolve().parents[1] / "src"
_ALIAS_CANONICAL = {"ball receipt*": "ball receipt", "goal keeper": "goalkeeper"}

# Ids frozen by the trained GRU / stored parquet.
_FROZEN = {
    "pass": 1,
    "carry": 2,
    "ball receipt": 3,
    "dribble": 4,
    "shot": 5,
    "pressure": 6,
    "duel": 7,
    "clearance": 8,
    "interception": 9,
    "block": 10,
    "foul committed": 11,
    "foul won": 12,
    "goalkeeper": 13,
}


def test_frozen_ids_unchanged():
    for name, i in _FROZEN.items():
        assert encode(name) == i
    assert encode("something else") == 0 == int(EventType.UNKNOWN)
    assert encode(None) == 0


@pytest.mark.parametrize("name", sorted(SEQUENCE_TYPES))
def test_round_trip(name):
    """encode -> decode returns the original name (aliases return their canonical form)."""
    assert decode(encode(name)) == _ALIAS_CANONICAL.get(name, name)


def test_encode_is_case_insensitive_and_decode_total():
    assert encode("Ball Receipt*") == EventType.BALL_RECEIPT
    assert decode(999) == "unknown"
    assert et.label(int(EventType.BALL_RECEIPT)) == "Ball Receipt"


def _py_files():
    return [p for p in SRC.rglob("*.py") if p.name != "event_types.py"]


def _is_type_id_expr(node: ast.AST) -> bool:
    """True for ``x["type_id"]`` / ``x.get("type_id", ...)`` expressions."""
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == "type_id"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        return bool(node.args) and isinstance(node.args[0], ast.Constant) and node.args[0].value == "type_id"
    return False


def test_no_module_defines_its_own_type_mapping():
    offenders: list[str] = []
    labels = {"pass", "carry", "shot", "dribble", "pressure", "duel"}
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and len(node.keys) >= 3:
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                vals = [v.value for v in node.values if isinstance(v, ast.Constant)]
                if sum(isinstance(k, str) and k.lower() in SEQUENCE_TYPES for k in keys) >= 3:
                    offenders.append(f"{path.name}:{node.lineno} name->id dict")
                if sum(isinstance(v, str) and v.lower() in labels for v in vals) >= 3 and any(
                    isinstance(k, int) for k in keys
                ):
                    offenders.append(f"{path.name}:{node.lineno} id->name dict")
            if isinstance(node, ast.Compare) and (
                _is_type_id_expr(node.left) or any(_is_type_id_expr(c) for c in node.comparators)
            ):
                others = [node.left, *node.comparators]
                if any(isinstance(o, ast.Constant) and isinstance(o.value, int) for o in others):
                    offenders.append(f"{path.name}:{node.lineno} integer comparison on type_id")
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.upper().replace("_", "") in {
                        "TYPEVOCAB",
                        "EVENTTYPELABELS",
                        "EVENTTYPEPASS",
                        "EVENTTYPECARRY",
                    }:
                        offenders.append(f"{path.name}:{node.lineno} assigns {t.id}")
    assert not offenders, offenders


def _toy_events() -> pd.DataFrame:
    rows = []
    idx = 0
    for poss, spec in enumerate(
        [
            ["Pass", "Ball Receipt*", "Carry", "Pass", "Pressure", "Ball Receipt*", "Dribble"],
            ["Pass", "Pass", "Carry", "Carry", "Shot"],
            ["Carry", "Pass", "Ball Recovery", "Pass", "Ball Receipt*", "Duel"],
        ],
        start=1,
    ):
        for k, name in enumerate(spec):
            idx += 1
            rows.append(
                {
                    "match_id": 1,
                    "possession_id": poss,
                    "index": idx,
                    "period": 1,
                    "minute": 1,
                    "second": k,
                    "type_name": name,
                    "team_name": "A",
                    "play_pattern_name": "Regular Play",
                    "location_x": 20.0 + 6 * k,
                    "location_y": 40.0,
                    "under_pressure": False,
                    "pass_end_x": 26.0 + 6 * k if name == "Pass" else np.nan,
                    "pass_end_y": 40.0 if name == "Pass" else np.nan,
                    "pass_length": 12.0 if name == "Pass" else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_prefix_features_at_full_fraction_reproduce_stored_counts():
    from src.data.parse_possessions import build_possession_sequences
    from src.features.early_features import _recompute_aggregates_from_prefix

    poss = build_possession_sequences(_toy_events(), min_events=2)
    out = _recompute_aggregates_from_prefix(poss, 1.0)
    assert out["n_passes"].tolist() == poss["n_passes"].tolist()
    assert out["n_carries"].tolist() == poss["n_carries"].tolist()
    assert out["n_events"].tolist() == poss["n_events"].tolist()
    np.testing.assert_allclose(out["mean_pass_length"], poss["mean_pass_length"], atol=0.01)
    assert poss["n_passes"].tolist() == [2, 2, 2]  # sanity: independent of the vocabulary code
