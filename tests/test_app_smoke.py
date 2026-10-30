"""Streamlit smoke tests (streamlit.testing): real artefacts when present, demo mode otherwise."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "src" / "app" / "app.py")

_HAVE_REAL = all(
    (ROOT / p).exists()
    for p in (
        "data/processed/pass_instances.parquet",
        "data/processed/frames_360.parquet",
        "data/processed/possession_sequences.parquet",
        "data/processed/event_time_arrays.pkl",
        "data/processed/split_manifest.csv",
        "models/pass_progression_event_only.joblib",
        "models/eventtime_gru.pt",
        "models/eventtime_xgboost.joblib",
    )
)


@pytest.fixture(autouse=True)
def _fresh_streamlit_caches():
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def test_demo_mode_is_explicit_and_never_claims_model_output(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAME2THREAT_PROCESSED_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    banners = " ".join(w.value for w in at.warning)
    assert "DEMO MODE" in banners and "random placeholder" in banners
    assert not any("loaded" in s.value for s in at.sidebar.success)


@pytest.mark.skipif(not _HAVE_REAL, reason="processed data / trained models not present")
def test_every_page_renders_on_real_data():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception and not at.error
    assert any("loaded" in s.value for s in at.sidebar.success)
    assert not any("DEMO MODE" in w.value for w in at.warning)
    pages = [r for r in at.sidebar.radio if r.label == "View"][0].options
    for page in pages:
        [r for r in at.sidebar.radio if r.label == "View"][0].set_value(page)
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])


@pytest.mark.skipif(not _HAVE_REAL, reason="processed data / trained models not present")
def test_possession_page_is_labelled_event_time_and_heldout_by_default():
    at = AppTest.from_file(APP, default_timeout=300).run()
    [r for r in at.sidebar.radio if r.label == "View"][0].set_value("🏃 Possession Inspector")
    at.run()
    assert not at.exception
    assert any("Event-time (prospective)" in i.value for i in at.info)
    split = [r for r in at.sidebar.radio if r.label == "Matches"][0]
    assert split.value.startswith("test")
    assert not any("seen in training" in w.value for w in at.warning)
    probs = [m for m in at.metric if m.label == "First alert"]
    assert probs
