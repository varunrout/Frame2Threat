"""PassFrameGNN on the corrected label: leak-free graphs, GraphSAGE maths, save/load."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.features import pass_graph
from src.features.pass_graph import (
    FRAME_COLUMNS,
    NODE_FEATURES,
    build_edges,
    build_graph_dataset,
    build_pass_graph,
)


def _frame(seed: int = 0, n: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(
        {
            "x": rng.uniform(20, 110, n),
            "y": rng.uniform(5, 75, n),
            "teammate": [i % 2 == 0 for i in range(n)],
            "keeper": [i < 2 for i in range(n)],
            "actor": [i == 2 for i in range(n)],
        }
    )
    return f


def _same(a, b):
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.edge_index, b.edge_index)


# ------------------------------------------------------------------ graph definition


def test_graph_shape_and_features():
    g = build_pass_graph("u", _frame(), 50.0, 40.0)
    assert g.x.shape == (14, len(NODE_FEATURES)) == (14, 8)
    assert "is_receiver" not in NODE_FEATURES
    assert g.edge_index.shape[0] == 2 and g.edge_index.dtype == np.int64
    assert g.edge_index.max() < 14 and (g.edge_index[0] != g.edge_index[1]).all()
    assert np.isfinite(g.x).all()


def test_edges_are_knn_plus_same_team_links():
    px, py = np.array([0.0, 1.0, 2.0, 50.0]), np.zeros(4)
    team = np.array([1.0, 0.0, 1.0, 0.0])
    ei = build_edges(px, py, team, k=1)
    pairs = set(map(tuple, ei.T.tolist()))
    assert (0, 1) in pairs and (3, 2) in pairs  # nearest neighbours
    assert (0, 2) in pairs and (2, 0) in pairs  # same-team link, both directions
    assert (1, 3) in pairs and (3, 1) in pairs
    assert build_edges(px[:1], py[:1], team[:1]).shape == (2, 0)


def test_graph_builder_is_deterministic_and_row_order_invariant():
    f = _frame(3)
    _same(build_pass_graph("u", f, 50, 40), build_pass_graph("u", f.copy(), 50, 40))
    _same(
        build_pass_graph("u", f, 50, 40),
        build_pass_graph("u", f.sample(frac=1, random_state=1), 50, 40),
    )


# ------------------------------------------------------------------ leakage contract


def test_builder_signature_has_no_route_for_end_location_label_or_later_events():
    params = set(inspect.signature(build_pass_graph).parameters)
    assert params == {"event_uuid", "frame", "start_x", "start_y", "k"}
    assert not any("end" in p or "label" in p for p in params)
    assert set(FRAME_COLUMNS) == {"x", "y", "teammate", "keeper", "actor"}
    # no executable code (names, attributes, string keys) mentions an end location, a label or a later event
    import ast

    tree = ast.parse(inspect.getsource(pass_graph))
    doc_nodes = {
        id(n.body[0])
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
    }
    tokens = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.arg):
            tokens.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            tokens.add(node.value)
    tokens -= {
        n.value.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Expr) and id(n) in doc_nodes and isinstance(n.value, ast.Constant)
    }
    banned = {
        "end_x",
        "end_y",
        "pass_end",
        "pass_end_x",
        "pass_end_y",
        "progression_beyond_pass_k",
        "dangerous_progression_k",
        "possession_id",
        "index",
        "is_receiver",
    }
    assert not (tokens & banned), tokens & banned


def test_graph_depends_only_on_the_freeze_frame_and_pass_start():
    """Changing end location, labels or any other pass/event column cannot change the graph."""
    frame = _frame(1)
    frames = frame.assign(event_uuid="a")
    other = _frame(2).assign(event_uuid="b")  # another event's frame
    base = pd.DataFrame(
        {
            "event_uuid": ["a"],
            "start_x": [60.0],
            "start_y": [30.0],
            "end_x": [70.0],
            "end_y": [35.0],
            "progression_beyond_pass_k": [0],
            "dangerous_progression_k": [0],
            "possession_id": [5],
        }
    )
    g0 = build_graph_dataset(base, pd.concat([frames, other]))[0]
    # move the pass destination, flip labels, change the possession and later-event columns
    changed = base.assign(
        end_x=118.0,
        end_y=2.0,
        progression_beyond_pass_k=1,
        dangerous_progression_k=1,
        possession_id=99,
    )
    g1 = build_graph_dataset(changed, pd.concat([frames, other]))[0]
    _same(g0, g1)
    # the other event's frame is irrelevant too
    g2 = build_graph_dataset(base, frames)[0]
    _same(g0, g2)
    # but the pass START does matter (dist_to_passer), as designed
    moved = base.assign(start_x=10.0)
    g3 = build_graph_dataset(moved, frames)[0]
    col = NODE_FEATURES.index("dist_to_passer")
    assert not np.allclose(g0.x[:, col], g3.x[:, col])
    keep = [i for i in range(len(NODE_FEATURES)) if i != col]
    np.testing.assert_array_equal(g0.x[:, keep], g3.x[:, keep])
    np.testing.assert_array_equal(g0.edge_index, g3.edge_index)


# ------------------------------------------------------------------ model (torch)


def _toy_graphs(n=12):
    return [build_pass_graph(f"u{i}", _frame(i, n=8 + i % 5), 50.0, 40.0) for i in range(n)]


def test_sage_mean_matches_the_formula():
    torch = pytest.importorskip("torch")
    from src.models.pass_gnn import SAGEMean

    torch.manual_seed(0)
    layer = SAGEMean(3, 4)
    h = torch.randn(4, 3)
    ei = torch.tensor(
        [[0, 1, 2, 0], [2, 2, 3, 3]]
    )  # 0->2, 1->2, 2->3, 0->3 ; node 0,1 have no incoming edges
    out = layer(h, ei)
    mean2 = (h[0] + h[1]) / 2
    mean3 = (h[2] + h[0]) / 2
    expect = layer.lin_l(torch.stack([torch.zeros(3), torch.zeros(3), mean2, mean3])) + layer.lin_r(
        h
    )
    torch.testing.assert_close(out, expect)
    assert layer.lin_r.bias is None


def test_predictions_in_unit_interval_and_batch_invariant():
    torch = pytest.importorskip("torch")
    from src.models.pass_gnn import PassGraphSAGE, predict_proba

    torch.manual_seed(0)
    model = PassGraphSAGE()
    gs = _toy_graphs()
    p = predict_proba(model, gs, batch_size=5)
    assert p.shape == (12,) and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p, predict_proba(model, gs, batch_size=256), atol=1e-6)
    assert sum(1 for m in model.modules() if m.__class__.__name__ == "SAGEMean") == 3
    assert model.cfg["hidden"] == 64


def test_saved_model_loads_and_predicts_identically(tmp_path):
    torch = pytest.importorskip("torch")
    from src.models.pass_gnn import PassGraphSAGE, load_model, predict_proba, save_model

    torch.manual_seed(1)
    model = PassGraphSAGE(dropout=0.1)
    save_model(tmp_path / "m.pt", model, {"label": "progression_beyond_pass_k"})
    loaded = load_model(tmp_path / "m.pt")
    gs = _toy_graphs()
    np.testing.assert_allclose(predict_proba(model, gs), predict_proba(loaded, gs), atol=1e-7)
    assert (tmp_path / "m.json").exists()


def test_committed_artefact_loads_and_matches_its_metadata():
    torch = pytest.importorskip("torch")
    import json
    from pathlib import Path

    from src.models.pass_gnn import load_model, predict_proba

    path = Path("models/pass_gnn_progression.pt")
    if not path.exists():
        pytest.skip("model artefact not present (models/ is not committed)")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["label"] == "progression_beyond_pass_k" and meta["node_features"] == NODE_FEATURES
    assert "is_receiver" not in meta["node_features"] and len(meta["seeds"]) == 5
    p = predict_proba(load_model(path), _toy_graphs())
    assert ((p >= 0) & (p <= 1)).all() and torch.isfinite(torch.from_numpy(p)).all()
