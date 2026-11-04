"""
Streamlit application for Frame2Threat pass analysis.

Features
--------
- Match / event selector (sidebar)
- Freeze-frame pitch plot with ranked options
- Pass score panel with confidence bar
- Explanation panel (SHAP-based)
- Player / team profiling views

Run with:
    streamlit run src/app/app.py
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

# ``streamlit run src/app/app.py`` puts src/app (not the repo root) on sys.path, so the package
# imports below only work after an editable install; make them work from a bare checkout too.
_ROOT_STR = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

from src.models import pass_progression as pp  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import optional heavy dependencies gracefully
# ---------------------------------------------------------------------------
try:
    import streamlit as st

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False

try:
    from mplsoccer import Pitch  # noqa: F401  (availability probe)

    _MPLSOCCER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPLSOCCER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROCESSED_DIR = pathlib.Path(
    os.environ.get("FRAME2THREAT_PROCESSED_DIR", _REPO_ROOT / "data" / "processed")
)
_MODELS_DIR = _REPO_ROOT / "models"

_PASS_INSTANCES_PATH = _PROCESSED_DIR / "pass_instances.parquet"
_FRAMES_PATH = _PROCESSED_DIR / "frames_360.parquet"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading pass data…")
def _load_pass_instances() -> pd.DataFrame | None:
    """Load pass_instances.parquet from data/processed/."""
    if _PASS_INSTANCES_PATH.exists():
        df = pd.read_parquet(_PASS_INSTANCES_PATH)
        logger.info("Loaded %d pass instances from %s", len(df), _PASS_INSTANCES_PATH)
        return df
    logger.warning("pass_instances.parquet not found at %s", _PASS_INSTANCES_PATH)
    return None


@st.cache_data(show_spinner="Loading 360 frame data…")
def _load_frames() -> pd.DataFrame | None:
    """Load frames_360.parquet from data/processed/."""
    if _FRAMES_PATH.exists():
        df = pd.read_parquet(_FRAMES_PATH)
        logger.info("Loaded %d frame rows from %s", len(df), _FRAMES_PATH)
        return df
    logger.warning("frames_360.parquet not found at %s", _FRAMES_PATH)
    return None


@st.cache_data(show_spinner="Scoring 50k passes with the trained model (first load only)")
def _score_all_passes() -> np.ndarray:
    """Pass-progression probabilities for every pass; cached so page switches stay instant."""
    pm = _load_model()
    return pm.predict_proba_frame(pp.build_feature_frame(_load_pass_instances()))


@st.cache_resource(show_spinner="Loading pass model…")
def _load_model(name: str = pp.DEFAULT_MODEL) -> Any:
    """Load a saved pass-progression model (default: ``pass_progression_event_only.joblib``).

    Raises :class:`pp.ArtefactError` naming the missing artefact; never returns a stand-in.
    """
    return pp.load_model(name, _MODELS_DIR)


@st.cache_data(show_spinner=False)
def _load_geometry_cache() -> pd.DataFrame | None:
    path = _PROCESSED_DIR / "geometry_features_v1.parquet"
    return pd.read_parquet(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Synthetic demo-data generators (used when processed data is absent)
# ---------------------------------------------------------------------------


def _make_demo_pass_instances(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic pass_instances DataFrame for demo purposes."""
    rng = np.random.default_rng(seed)
    uuids = [f"demo-{i:06d}" for i in range(n)]
    match_ids = rng.integers(1001, 1006, size=n)
    df = pd.DataFrame(
        {
            "event_uuid": uuids,
            "match_id": match_ids,
            "match_name": [f"Demo Match {mid}" for mid in match_ids],
            "player_name": rng.choice(
                ["Player A", "Player B", "Player C", "Player D", "Player E"], size=n
            ),
            "team_name": rng.choice(["Team Alpha", "Team Beta"], size=n),
            "start_x": rng.uniform(30, 115, size=n),
            "start_y": rng.uniform(5, 75, size=n),
            "end_x": rng.uniform(35, 120, size=n),
            "end_y": rng.uniform(5, 75, size=n),
            "pass_length": rng.uniform(3, 40, size=n),
            "pass_angle": rng.uniform(-np.pi, np.pi, size=n),
            "minute": rng.integers(1, 90, size=n),
            "period": rng.choice([1, 2], size=n),
            "under_pressure": rng.choice([True, False], size=n),
            "zone_start": rng.integers(1, 7, size=n),
            "threat_gain": rng.uniform(-0.05, 0.15, size=n),
            "line_break": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
            "dangerous_progression_k": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        }
    )
    return df


def _make_demo_frames(pass_instances_df: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    """Return a synthetic frames_360 DataFrame for demo purposes."""
    rng = np.random.default_rng(seed)
    rows = []
    for uuid in pass_instances_df["event_uuid"].unique()[:50]:  # limit for demo
        n_players = int(rng.integers(8, 16))
        for j in range(n_players):
            rows.append(
                {
                    "event_uuid": uuid,
                    "teammate": bool(j < n_players // 2),
                    "actor": j == 0,
                    "keeper": j == n_players - 1,
                    "x": float(rng.uniform(0, 120)),
                    "y": float(rng.uniform(0, 80)),
                    "player_name": f"Player_{j}",
                }
            )
    return pd.DataFrame(rows)


def _make_demo_scores(n: int, seed: int = 7) -> np.ndarray:
    """Return random demo predicted scores in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.beta(2, 5, size=n).astype(float)


def _make_ranked_options(
    start_x: float, start_y: float, n_options: int = 8, seed: int = 42
) -> pd.DataFrame:
    """Return synthetic ranked candidate pass destinations."""
    rng = np.random.default_rng(seed)
    scores = rng.beta(2, 5, size=n_options)
    ranks = np.argsort(scores)[::-1].argsort() + 1
    return pd.DataFrame(
        {
            "end_x": rng.uniform(start_x + 5, min(start_x + 30, 118), size=n_options),
            "end_y": np.clip(rng.normal(start_y, 10, size=n_options), 2, 78),
            "score": scores,
            "rank": ranks,
        }
    )


# ---------------------------------------------------------------------------
# Plot helpers (wrapped to avoid import errors)
# ---------------------------------------------------------------------------


def _render_freeze_frame(
    event_uuid: str,
    pass_instances_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    scores: np.ndarray,
    ranked_options: pd.DataFrame | None = None,
) -> Any:
    """Return a matplotlib Figure for the freeze frame."""
    from src.visualization.freeze_frame_view import plot_freeze_frame

    fig, _ = plot_freeze_frame(
        event_uuid=event_uuid,
        pass_instances_df=pass_instances_df,
        frames_df=frames_df,
        scores=scores,
        ranked_options=ranked_options,
    )
    return fig


def _render_pass_map(
    pass_instances_df: pd.DataFrame,
    scores: np.ndarray,
    title: str = "Pass Map",
) -> Any:
    """Return a matplotlib Figure for the full pass map."""
    from src.visualization.pitch_plots import plot_pass_map

    df = pass_instances_df.copy()
    df["predicted_score"] = scores
    fig, _ = plot_pass_map(df, title=title, color_col="predicted_score")
    return fig


def _render_player_profile(
    player_name: str,
    pass_instances_df: pd.DataFrame,
    scores: np.ndarray,
) -> Any:
    """Return a matplotlib Figure for the player profile."""
    from src.visualization.pitch_plots import plot_player_profile

    return plot_player_profile(player_name, pass_instances_df, scores)


def _render_shap_summary(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
) -> Any:
    """Return a matplotlib Figure for SHAP feature importance."""
    from src.visualization.explanations import compute_shap_values, plot_shap_summary

    sv = compute_shap_values(model, X, feature_names)
    return plot_shap_summary(sv["shap_values"], feature_names, title="Feature Importance (SHAP)")


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit Frame2Threat application."""
    st.set_page_config(
        page_title="Frame2Threat | Football danger analytics",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    st.title("⚽ Frame2Threat")
    st.caption(
        "Football danger analytics on StatsBomb Open Data: in-play possession danger forecasting "
        "and pass-level progression scores.  Research demo, held-out matches by default."
    )

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    pass_df_real = _load_pass_instances()
    frames_df_real = _load_frames()
    demo_mode = pass_df_real is None or frames_df_real is None

    pm = None
    model = None
    feature_cols: list[str] = []
    actual_labels = None
    score_source = "model"

    if demo_mode:
        # Synthetic demo only when the processed data is absent, clearly labelled as demo.
        st.warning(
            "⚠️ **DEMO MODE** – processed data files not found in `data/processed/`. "
            "Everything below is synthetic and the scores are random placeholders, not model output.",
            icon="⚠️",
        )
        pass_df = _make_demo_pass_instances()
        frames_df = _make_demo_frames(pass_df)
        scores = _make_demo_scores(len(pass_df))
        score_source = "demo_random"
        st.sidebar.warning("DEMO MODE: random placeholder scores.")
    else:
        assert pass_df_real is not None and frames_df_real is not None
        pass_df = pass_df_real
        frames_df = frames_df_real
        try:
            pm = _load_model()
            scores = _score_all_passes()
        except (
            Exception
        ) as exc:  # real data is loaded: show the cause, never substitute placeholders
            st.error(f"Cannot score passes with the trained model: {exc}")
            st.stop()
        model = pm.model
        feature_cols = list(pm.features)
        try:
            actual_labels = pp.load_actual_labels(pass_df, _PROCESSED_DIR)
        except pp.ArtefactError as exc:
            st.sidebar.warning(f"Actual progression rates unavailable: {exc}")
        st.sidebar.success(
            f"✅ {pp.DEFAULT_MODEL} loaded — {len(feature_cols)} features, {len(pass_df):,} passes scored"
        )

    if score_source == "demo_random":
        st.error(
            "DEMO: scores in this view are random placeholders for UI demonstration only; they are "
            "not Frame2Threat model output."
        )

    # ----------------------------------------------------------------
    # Sidebar – navigation and selectors
    # ----------------------------------------------------------------
    st.sidebar.header("🔍 Navigation")
    page = st.sidebar.radio(
        "View",
        options=[
            "🏃 Possession Inspector",
            "🎯 Event Inspector",
            "📊 Match Overview",
            "👤 Player Profile",
            "📈 Model Diagnostics",
        ],
        index=0,
    )

    # The pass-page match selector is irrelevant to the possession page (it has its own).
    if page == "🏃 Possession Inspector":
        selected_match_label = "All Passes"
        match_df = pass_df.reset_index(drop=True)
        match_scores = scores
    # Match selector – build human-readable "Team A vs Team B (ID)" labels
    elif "match_id" in pass_df.columns and "team_name" in pass_df.columns:
        _match_label_map: dict = {}
        for _mid, _grp in pass_df.groupby("match_id"):
            _teams = sorted(_grp["team_name"].dropna().unique())
            _label = " vs ".join(str(t) for t in _teams[:2]) if _teams else f"Match {_mid}"
            _match_label_map[_mid] = _label
        _label_to_mid = {v: k for k, v in _match_label_map.items()}
        _available_match_labels = sorted(_match_label_map.values())
        selected_match_label = str(
            st.sidebar.selectbox("Select Match", _available_match_labels, index=0)
        )
        _selected_mid = _label_to_mid[selected_match_label]
        match_mask = pass_df["match_id"] == _selected_mid
        match_df = pass_df[match_mask].reset_index(drop=True)
        match_scores = scores[match_mask]
    elif "match_name" in pass_df.columns:
        available_matches = sorted(pass_df["match_name"].dropna().unique())
        selected_match_label = str(st.sidebar.selectbox("Select Match", available_matches, index=0))
        match_mask = pass_df["match_name"] == selected_match_label
        match_df = pass_df[match_mask].reset_index(drop=True)
        match_scores = scores[match_mask]
    else:
        selected_match_label = "All Passes"
        match_df = pass_df.reset_index(drop=True)
        match_scores = scores

    if page != "🏃 Possession Inspector":
        st.sidebar.markdown(f"**{len(match_df)} passes** in selected match")
        st.sidebar.markdown(
            f"**Mean score:** {np.mean(match_scores):.3f} | "
            f"**Max score:** {np.max(match_scores):.3f}"
        )

    uuid_col = "event_uuid" if "event_uuid" in match_df.columns else "id"

    # ================================================================
    # PAGE 1 – Event Inspector
    # ================================================================
    if page == "🎯 Event Inspector":
        st.header("🎯 Event Inspector")
        st.info(
            "**What the score means.** "
            + pp.LABEL_PLAIN_ENGLISH
            + " The score is the model's probability of that outcome for this pass."
        )

        # Event selection
        match_df_sorted = match_df.copy()
        match_df_sorted["_score"] = match_scores
        match_df_sorted = match_df_sorted.sort_values("_score", ascending=False)

        event_options = match_df_sorted[uuid_col].tolist()
        event_labels = []
        for _, _erow in match_df_sorted.iterrows():
            _pname = str(_erow.get("player_name", "Unknown"))[:22]
            _min = int(_erow["minute"]) if pd.notna(_erow.get("minute")) else "?"
            _sx = _erow.get("start_x", 0) or 0
            _ex = _erow.get("end_x", 0) or 0
            _sc = float(_erow["_score"])
            event_labels.append(f"{_pname} | min {_min} | {_sx:.0f}→{_ex:.0f} m | ⚡ {_sc:.3f}")
        label_to_uuid = dict(zip(event_labels, event_options))

        selected_label = st.selectbox("Select Pass Event (sorted by danger score)", event_labels)
        if selected_label is None:
            st.info("No passes available for this match.")
            return
        selected_uuid = label_to_uuid[selected_label]

        event_row = match_df[match_df[uuid_col] == selected_uuid].iloc[0]
        event_pos = match_df.index[match_df[uuid_col] == selected_uuid][0]
        # Reindex scores to match match_df positions
        match_scores_series = pd.Series(match_scores, index=match_df.index)
        event_score = float(match_scores_series.loc[event_pos])

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "🎯 Progression probability (event-only)",
            f"{event_score:.3f}",
            help=(
                "Predicted dangerous progression probability"
                if score_source == "model"
                else "Random placeholder score; not a model prediction"
            ),
        )
        col2.metric("📐 Pass Length", f"{event_row.get('pass_length', 'N/A'):.1f}m")
        col3.metric("⏱ Minute", int(event_row.get("minute", 0)))
        col4.metric("🗺 Zone", f"Zone {int(event_row.get('zone_start', 0))}")

        if not demo_mode:
            if bool(event_row.get("has_360", False)):
                try:
                    _single = match_df[match_df[uuid_col] == selected_uuid].copy()
                    _geo = pp.geometry_for(_single, frames_df, _load_geometry_cache())
                    _pm360 = _load_model(pp.EVENT_360)
                    _p360 = _pm360.predict_proba_frame(
                        pp.build_feature_frame(_single, _geo, with_360=True)
                    )[0]
                    st.metric(
                        "❄️ Event+360 probability",
                        f"{_p360:.3f}",
                        help="Same label, model with 14 freeze-frame geometry features added.",
                    )
                except Exception as exc:
                    st.error(f"Event+360 score unavailable: {exc}")
            else:
                st.caption("No freeze frame for this pass: only the event-only score is available.")

        # Score confidence bar
        st.markdown("#### Confidence Bar")
        score_text = (
            f"{event_score:.1%} model danger score"
            if score_source == "model"
            else f"{event_score:.1%} random placeholder score"
        )
        st.progress(float(np.clip(event_score, 0, 1)), text=score_text)

        # Freeze frame plot
        st.markdown("#### ❄️ Freeze Frame")

        ranked_opts = None
        if demo_mode:
            if st.checkbox("Show ranked pass options (demo only)", value=True):
                sx = float(event_row.get("start_x", 60))
                sy = float(event_row.get("start_y", 40))
                ranked_opts = _make_ranked_options(sx, sy)
                st.caption("Ranked pass options are synthetic random placeholders in demo mode.")
        else:
            st.caption("Ranked pass options are not shown: there is no option-level model.")

        event_frames = frames_df[
            frames_df.get("event_uuid", frames_df.get("id", pd.Series())) == selected_uuid
        ]
        if event_frames.empty:
            st.info("No freeze-frame data available for this event. Showing pass arrow only.")
            # Create a minimal frame with just the passer
            passer_frame = pd.DataFrame(
                [
                    {
                        "event_uuid": selected_uuid,
                        "x": event_row.get("start_x", 60),
                        "y": event_row.get("start_y", 40),
                        "teammate": True,
                        "actor": True,
                        "keeper": False,
                    }
                ]
            )
            event_frames = passer_frame

        try:
            # Build a single-event pass_df for the freeze frame function
            single_event_df = match_df[match_df[uuid_col] == selected_uuid].copy()
            single_scores = match_scores_series.loc[single_event_df.index].values

            fig_ff = _render_freeze_frame(
                event_uuid=selected_uuid,
                pass_instances_df=single_event_df,
                frames_df=frames_df,
                scores=single_scores,
                ranked_options=ranked_opts,
            )
            st.pyplot(fig_ff, use_container_width=True)
            plt.close(fig_ff)
        except Exception as exc:
            st.error(f"Could not render freeze frame: {exc}")
            logger.exception("Freeze frame render error")

        # Explanation panel
        st.markdown("#### 💡 Explanation")
        with st.expander("Show detailed explanation", expanded=True):
            if not demo_mode:
                try:
                    from src.features.event_features import build_event_features
                    from src.visualization.explanations import explain_single_event

                    # Build the engineered feature matrix for this single event
                    # so explain_single_event finds all 27 feature columns, not
                    # just the ~8 raw parquet columns that share names with them.
                    _single_raw = match_df[match_df[uuid_col] == selected_uuid].copy()
                    _X_single = build_event_features(_single_raw)
                    # Attach event_uuid so explain_single_event can look it up
                    _X_single[uuid_col] = _single_raw[uuid_col].values

                    explanation = explain_single_event(
                        event_uuid=selected_uuid,
                        pass_instances_df=_X_single,
                        frames_df=frames_df,
                        model=model,
                        feature_names=feature_cols,
                    )
                    st.markdown(f"**Narrative:**\n\n{explanation['narrative']}")
                    reasons_df = pd.DataFrame(explanation["top_reasons"])
                    if not reasons_df.empty:
                        st.dataframe(
                            reasons_df[["description", "value", "contribution"]]
                            .rename(
                                columns={
                                    "description": "Feature",
                                    "value": "Value",
                                    "contribution": "SHAP Contribution",
                                }
                            )
                            .round(4),
                            use_container_width=True,
                        )
                    if explanation.get("geometry_context"):
                        st.json(explanation["geometry_context"])
                except Exception as exc:
                    st.warning(f"Explanation unavailable: {exc}")
            else:
                _show_demo_explanation(event_score, event_row)

    # ================================================================
    # PAGE 2 – Match Overview
    # ================================================================
    elif page == "📊 Match Overview":
        st.header("📊 Match Overview")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Passes", len(match_df))
        col_b.metric("Mean Score", f"{np.mean(match_scores):.3f}")
        col_c.metric(
            "High-Danger Passes (≥0.5)",
            int((match_scores >= 0.5).sum()),
        )

        st.markdown("#### 🗺 Pass Map (coloured by danger score)")
        try:
            fig_map = _render_pass_map(
                match_df,
                match_scores,
                title=f"Pass Map – {selected_match_label}",
            )
            st.pyplot(fig_map, use_container_width=True)
            plt.close(fig_map)
        except Exception as exc:
            st.error(f"Pass map error: {exc}")

        st.markdown("#### 📋 Score Distribution")
        score_hist_fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(match_scores, bins=30, color="steelblue", edgecolor="white", alpha=0.9)
        ax.set_xlabel("Predicted Danger Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(score_hist_fig, use_container_width=True)
        plt.close(score_hist_fig)

        st.markdown("#### 🏆 Top 20 Most Dangerous Passes")
        from src.evaluation.tactical_review import get_top_scoring_passes

        top_passes = get_top_scoring_passes(match_df, match_scores, n=20)
        display_cols = [
            c
            for c in [
                "player_name",
                "pass_recipient_name",
                "team_name",
                "minute",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "pass_length",
                "predicted_score",
            ]
            if c in top_passes.columns
        ]
        st.dataframe(top_passes[display_cols].round(3), use_container_width=True)

        st.markdown("#### 📊 Breakdown by Zone")
        from src.evaluation.tactical_review import breakdown_by_zone

        label_col = "line_break" if "line_break" in match_df.columns else None
        if label_col is not None:
            zone_df = breakdown_by_zone(match_df, match_df[label_col].fillna(0), match_scores)
            st.dataframe(zone_df.round(4), use_container_width=True)
        else:
            st.info("Label columns not available for zone breakdown.")

    # ================================================================
    # PAGE 3 – Player Profile
    # ================================================================
    elif page == "👤 Player Profile":
        st.header("👤 Player Profile")

        if demo_mode:
            st.info("Demo mode: the player table needs real data and the trained model.")
        elif actual_labels is None:
            st.warning(
                "Actual progression rates are unavailable, so the player table cannot be built."
            )
        else:
            st.markdown(
                "Mean **predicted** progression probability against the **actual** progression rate "
                "(`progression_beyond_pass_k`).  *Actual minus predicted* is descriptive: it mixes "
                "finishing, team and opponent effects and is noisy for small samples."
            )
            min_passes = st.slider("Minimum passes per player", 1, 300, 50)
            table = pp.player_profile_table(pass_df, scores, actual_labels, min_passes)
            st.caption(
                f"{len(table)} players with at least {min_passes} passes (default 50 hides smaller samples)."
            )
            st.dataframe(
                table.rename(
                    columns={
                        "player": "Player",
                        "team": "Team",
                        "passes": "Passes",
                        "mean_predicted": "Mean predicted",
                        "actual_rate": "Actual rate",
                        "actual_minus_predicted": "Actual - predicted",
                    }
                ).round(3),
                use_container_width=True,
            )
            if not table.empty:
                selected_player = st.selectbox("Pass map for player", table["player"].tolist())
                if selected_player is not None:
                    try:
                        fig_player = _render_player_profile(selected_player, pass_df, scores)
                        st.pyplot(fig_player, use_container_width=True)
                        plt.close(fig_player)
                    except Exception as exc:
                        st.error(f"Player profile error: {exc}")

        # Team profiles
        st.markdown("---")
        st.markdown("#### 🏟 Team Profiles")
        from src.evaluation.tactical_review import team_progression_profile

        team_profile = team_progression_profile(pass_df, scores, min_passes=20)
        if not team_profile.empty:
            display_cols = [c for c in team_profile.columns if c != "team_id"]
            st.dataframe(team_profile[display_cols].round(4), use_container_width=True)
        else:
            st.info("Insufficient data for team profiles.")

    # ================================================================
    # PAGE 4 – Model Diagnostics
    # ================================================================
    elif page == "📈 Model Diagnostics":
        st.header("📈 Model Diagnostics")

        if demo_mode:
            st.warning(
                "Demo mode: diagnostics come from the trained model's saved metadata, which needs real data."
            )
        else:
            try:
                meta = pp.load_metadata(pp.EVENT_ONLY, _MODELS_DIR)
                meta360 = pp.load_metadata(pp.EVENT_360, _MODELS_DIR)
            except pp.ArtefactError as exc:
                st.error(f"Diagnostics unavailable: {exc}")
                st.stop()
            st.caption(
                "Shown from the metadata saved at training time (test split, match-level bootstrap 95% "
                "intervals); nothing is recomputed here.  Label: "
                + meta["label"]
                + ".  "
                + meta["recipe"]
                + "."
            )
            rows = []
            for m in (meta, meta360):
                t = m["test_all_passes"]
                r = {"Model": m["name"], "Test passes": t["n"], "Matches": t["matches"]}
                for stat, nm in (
                    ("roc_auc", "ROC AUC"),
                    ("avg_precision", "AP"),
                    ("brier", "Brier"),
                    ("log_loss", "Log loss"),
                ):
                    v = t["metrics_with_95ci_match_bootstrap"][stat]
                    r[nm] = f"{v['estimate']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]"
                r["Cal. intercept"] = round(t["calibration"]["intercept"], 3)
                r["Cal. slope"] = round(t["calibration"]["slope"], 3)
                rows.append(r)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("#### 🎛 Reliability table (test split, 10 bins)")
            rel_cols = st.columns(2)
            for col, m in zip(rel_cols, (meta, meta360)):
                col.markdown(f"**{m['name']}**")
                col.dataframe(
                    pd.DataFrame(m["test_all_passes"]["reliability_table"]).round(3),
                    use_container_width=True,
                )

            st.markdown(
                "#### ❄️ Does 360 add anything? (event+360 minus event-only, paired match-level bootstrap)"
            )
            deltas = meta360["comparison_360"]["paired_delta_event360_minus_event_only"]
            for view, rows_d in deltas.items():
                st.markdown(f"**{view}**")
                dd = pd.DataFrame(rows_d)[["stat", "estimate", "ci_low", "ci_high"]].copy()
                dd["interval excludes zero"] = (dd["ci_low"] > 0) | (dd["ci_high"] < 0)
                st.dataframe(dd.round(4), use_container_width=True)
            st.caption(
                "The 360 gain is small but measurable: intervals exclude zero for most statistics "
                "(see reports/pass_label/pass_model_final.md)."
            )

            if model is not None and feature_cols:
                st.markdown("#### 🔍 SHAP Feature Importance")
                with st.spinner("Computing SHAP values…"):
                    try:
                        from src.features.event_features import build_event_features

                        sample_df = build_event_features(pass_df.head(200))[feature_cols].fillna(
                            0.0
                        )
                        fig_shap = _render_shap_summary(model, sample_df, feature_cols)
                        st.pyplot(fig_shap, use_container_width=True)
                        plt.close(fig_shap)
                    except Exception as exc:
                        st.warning(f"SHAP summary unavailable: {exc}")

    # ================================================================
    # PAGE 5 - Possession Inspector (event-time model, held-out matches by default)
    # ================================================================
    elif page == "🏃 Possession Inspector":
        from src.app import enhanced_possession_page

        enhanced_possession_page.render(
            st, _REPO_ROOT, cache_data=st.cache_data, cache_resource=st.cache_resource
        )

    # ----------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        "🔬 **Frame2Threat** | StatsBomb 360 pass danger prediction | "
        "[GitHub](https://github.com) | Demo mode: " + ("✅ Yes" if demo_mode else "❌ No")
    )


def _show_demo_explanation(score: float, event_row: pd.Series) -> None:
    """Show a placeholder explanation panel when model is unavailable."""
    st.warning(
        "This explanation is generated from random placeholder scores and simple "
        "event attributes. It is not a model explanation."
    )
    from src.visualization.explanations import generate_explanation_narrative

    demo_reasons = [
        {
            "feature": "x_gain",
            "value": float(event_row.get("end_x", 70) - event_row.get("start_x", 60)),
            "contribution": 0.15 * score,
            "description": "Horizontal distance gained",
        },
        {
            "feature": "dist_to_goal_end",
            "value": float(event_row.get("end_x", 70)),
            "contribution": 0.12 * score,
            "description": "Receiver distance to goal",
        },
        {
            "feature": "pass_length",
            "value": float(event_row.get("pass_length", 15)),
            "contribution": -0.05 * score,
            "description": "Pass length",
        },
    ]
    narrative = generate_explanation_narrative(demo_reasons, score, "dangerous_progression")
    st.markdown(f"**Narrative (demo):**\n\n{narrative}")
    st.info("Train a model and rerun to see SHAP-based explanations.")


if __name__ == "__main__":
    main()
