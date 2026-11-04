"""Determinism and provenance: the same seed must give the same model, and runs must be traceable."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.provenance import file_sha256, run_metadata
from src.features.event_time import sequence_inputs, snapshot_labels
from src.models.event_time import SeqSample, predict_steps, train_event_time_gru
from src.models.tabular import TabularClassifier
from tests.test_event_time import _random_possession


def _samples(seed: int, count: int = 24) -> list[SeqSample]:
    rng = np.random.default_rng(seed)
    out = []
    for pid in range(count):
        arr = _random_possession(rng, int(rng.integers(5, 12)), pid=pid)
        eligible, label, _ = snapshot_labels(arr)
        type_ids, numeric = sequence_inputs(arr)
        out.append(SeqSample(type_ids, numeric, eligible, int(label[0])))
    return out


def _train_gru(seed: int):
    train, val = _samples(1), _samples(2, count=10)
    model, _ = train_event_time_gru(
        train, val, n_numeric=train[0].numeric.shape[1], max_epochs=3, patience=3, seed=seed
    )
    return model, val


def test_gru_training_is_deterministic_for_a_fixed_seed():
    torch.set_num_threads(1)
    first, val = _train_gru(seed=7)
    second, _ = _train_gru(seed=7)
    a, b = predict_steps(first, val), predict_steps(second, val)
    assert len(a) == len(b)
    for steps_a, steps_b in zip(a, b):
        np.testing.assert_array_equal(np.asarray(steps_a), np.asarray(steps_b))


def test_gru_training_actually_depends_on_the_seed():
    torch.set_num_threads(1)
    first, val = _train_gru(seed=7)
    other, _ = _train_gru(seed=8)
    a = np.concatenate([np.asarray(s, dtype=float) for s in predict_steps(first, val)])
    b = np.concatenate([np.asarray(s, dtype=float) for s in predict_steps(other, val)])
    assert not np.array_equal(a, b)


def test_xgboost_training_is_deterministic_for_a_fixed_seed():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"] + rng.normal(scale=0.5, size=300) > 0).astype(int)
    config = {"random_state": 3, "n_jobs": 1}
    first = TabularClassifier(model_type="xgboost", config=config).fit(X, y).predict_proba(X)
    second = TabularClassifier(model_type="xgboost", config=config).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_run_metadata_records_code_environment_and_file_hashes(tmp_path):
    target = tmp_path / "model.bin"
    target.write_bytes(b"frame2threat")
    meta = run_metadata(tmp_path, [target])
    assert meta["files_sha256"] == {"model.bin": hashlib.sha256(b"frame2threat").hexdigest()}
    assert file_sha256(target) == meta["files_sha256"]["model.bin"]
    assert meta["python"] and meta["packages"]["numpy"]
    assert meta["git_sha"] is None and meta["git_dirty"] is None  # tmp_path is not a repository
