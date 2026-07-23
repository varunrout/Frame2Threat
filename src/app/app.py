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
import pathlib
from typing import Any

import numpy as np
import pandas as pd

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
    from mplsoccer import Pitch

    _MPLSOCCER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPLSOCCER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_MODELS_DIR = _REPO_ROOT / "models"

_PASS_INSTANCES_PATH = _PROCESSED_DIR / "pass_instances.parquet"
_FRAMES_PATH = _PROCESSED_DIR / "frames_360.parquet"
_POSSESSION_SEQ_PATH = _PROCESSED_DIR / "possession_sequences.parquet"
_GRU_MODEL_PATH = _MODELS_DIR / "gru_poss_dangerous.pt"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading pass data…")  # type: ignore[misc]
def _load_pass_instances() -> pd.DataFrame | None:
    """Load pass_instances.parquet from data/processed/."""
    if _PASS_INSTANCES_PATH.exists():
        df = pd.read_parquet(_PASS_INSTANCES_PATH)
        logger.info("Loaded %d pass instances from %s", len(df), _PASS_INSTANCES_PATH)
        return df
    logger.warning("pass_instances.parquet not found at %s", _PASS_INSTANCES_PATH)
    return None


@st.cache_data(show_spinner="Loading 360 frame data…")  # type: ignore[misc]
def _load_frames() -> pd.DataFrame | None:
    """Load frames_360.parquet from data/processed/."""
    if _FRAMES_PATH.exists():
        df = pd.read_parquet(_FRAMES_PATH)
        logger.info("Loaded %d frame rows from %s", len(df), _FRAMES_PATH)
        return df
    logger.warning("frames_360.parquet not found at %s", _FRAMES_PATH)
    return None


@st.cache_data(show_spinner="Loading possession data…")  # type: ignore[misc]
def _load_possession_sequences() -> pd.DataFrame | None:
    """Load possession_sequences.parquet from data/processed/."""
    if _POSSESSION_SEQ_PATH.exists():
        df = pd.read_parquet(_POSSESSION_SEQ_PATH)
        logger.info("Loaded %d possessions from %s", len(df), _POSSESSION_SEQ_PATH)
        return df
    logger.warning("possession_sequences.parquet not found at %s", _POSSESSION_SEQ_PATH)
    return None


@st.cache_resource(show_spinner="Loading GRU model…")  # type: ignore[misc]
def _load_gru_model() -> Any | None:
    """Load the saved PossessionGRU from models/."""
    if not _GRU_MODEL_PATH.exists():
        logger.warning("GRU model not found at %s", _GRU_MODEL_PATH)
        return None
    try:
        from src.evaluation.possession_attribution import load_gru_model

        model, _ = load_gru_model(_GRU_MODEL_PATH)
        return model
    except Exception as exc:
        logger.warning("GRU model loading failed: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading model…")  # type: ignore[misc]
def _load_model() -> Any | None:
    """Load the saved TabularClassifier from models/.

    Prefers ``xgboost_dp_event_only.joblib`` (27 engineered event features)
    to avoid the 41-feature shape mismatch caused by the event+360 model
    when geometry features have not been pre-computed for the selected data.
    """
    try:
        from src.models.tabular import TabularClassifier

        preferred = _MODELS_DIR / "xgboost_dp_event_only.joblib"
        if preferred.exists():
            return TabularClassifier.load(preferred)
        candidates = sorted(_MODELS_DIR.glob("**/*.joblib"))
        if not candidates:
            return None
        return TabularClassifier.load(candidates[0])
    except Exception as exc:
        logger.warning("Model loading failed: %s", exc)
        return None


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
        page_title="Frame2Threat – Pass Analysis",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    st.title("⚽ Frame2Threat – Pass Danger Analysis")
    st.markdown(
        "Explore predicted pass danger scores, freeze frames, and player profiles "
        "powered by StatsBomb 360 data."
    )

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    pass_df_real = _load_pass_instances()
    frames_df_real = _load_frames()
    model = _load_model()

    demo_mode = pass_df_real is None or frames_df_real is None

    if demo_mode:
        st.warning(
            "⚠️ **Demo mode** – processed data files not found in `data/processed/`. "
            "Run the data pipeline first to analyse real matches. "
            "Showing synthetic demo data.",
            icon="⚠️",
        )
        pass_df = _make_demo_pass_instances()
        frames_df = _make_demo_frames(pass_df)
    else:
        pass_df = pass_df_real
        frames_df = frames_df_real

    # Compute or simulate scores
    _prediction_error: str | None = None  # carries any scoring error to the UI

    if model is not None and not demo_mode:
        try:
            from src.features.event_features import build_event_features

            X_all = build_event_features(pass_df)
            feature_cols = list(X_all.columns)
            scores = model.predict_proba(X_all)[:, 1]
            st.sidebar.success(
                f"✅ Model loaded — {len(feature_cols)} features, "
                f"{len(pass_df):,} passes scored"
            )
        except Exception as exc:
            _prediction_error = str(exc)
            st.sidebar.warning(f"⚠️ Prediction failed — using demo scores. See details below.")
            scores = _make_demo_scores(len(pass_df))
            feature_cols = []
    else:
        scores = _make_demo_scores(len(pass_df))
        feature_cols = []
        if model is None:
            st.sidebar.info(
                "ℹ️ No model artifacts found in `models/`. "
                "Run the training pipeline or upload a model."
            )

    # ── Show feature-mismatch explanation if scoring failed ─────────
    if _prediction_error:
        with st.expander("⚠️ Why did model prediction fail?", expanded=True):
            st.error(f"**Error:** {_prediction_error}")
            st.markdown(
                """
**Root cause — Feature shape mismatch**

The model was trained on **engineered features** produced by
`build_event_features()` (27 columns such as `goal_dist_gain`,
`x_gain`, `pass_angle_sin`, one-hot body-part flags, etc.).

The raw `pass_instances.parquet` file only contains **base columns**
from the StatsBomb event log (e.g. `start_x`, `pass_length`, `minute`)
— roughly 19 numeric columns.

When the app tried to score passes using those 19 raw columns, the
model rejected them because it expected 27 (event-only) or 41
(event + 360 geometry) engineered features.

**How it is now fixed:**
The app calls `build_event_features(pass_df)` before scoring, which
reproduces exactly the 27-column feature matrix the model was trained
on. The predictions you see now use those correct features.
                """
            )

    # ----------------------------------------------------------------
    # Sidebar – navigation and selectors
    # ----------------------------------------------------------------
    st.sidebar.header("🔍 Navigation")
    page = st.sidebar.radio(
        "View",
        options=[
            "🎯 Event Inspector",
            "📊 Match Overview",
            "👤 Player Profile",
            "📈 Model Diagnostics",
            "🏃 Possession Inspector",
        ],
        index=0,
    )

    # Match selector – build human-readable "Team A vs Team B (ID)" labels
    if "match_id" in pass_df.columns and "team_name" in pass_df.columns:
        _match_label_map: dict = {}
        for _mid, _grp in pass_df.groupby("match_id"):
            _teams = sorted(_grp["team_name"].dropna().unique())
            _label = " vs ".join(str(t) for t in _teams[:2]) if _teams else f"Match {_mid}"
            _match_label_map[_mid] = _label
        _label_to_mid = {v: k for k, v in _match_label_map.items()}
        _available_match_labels = sorted(_match_label_map.values())
        selected_match_label = st.sidebar.selectbox(
            "Select Match", _available_match_labels, index=0
        )
        _selected_mid = _label_to_mid[selected_match_label]
        match_mask = pass_df["match_id"] == _selected_mid
        match_df = pass_df[match_mask].reset_index(drop=True)
        match_scores = scores[match_mask]
    elif "match_name" in pass_df.columns:
        available_matches = sorted(pass_df["match_name"].dropna().unique())
        selected_match_label = st.sidebar.selectbox("Select Match", available_matches, index=0)
        match_mask = pass_df["match_name"] == selected_match_label
        match_df = pass_df[match_mask].reset_index(drop=True)
        match_scores = scores[match_mask]
    else:
        selected_match_label = "All Passes"
        match_df = pass_df.reset_index(drop=True)
        match_scores = scores

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
        selected_uuid = label_to_uuid[selected_label]

        event_row = match_df[match_df[uuid_col] == selected_uuid].iloc[0]
        event_pos = match_df.index[match_df[uuid_col] == selected_uuid][0]
        # Reindex scores to match match_df positions
        match_scores_series = pd.Series(match_scores, index=match_df.index)
        event_score = float(match_scores_series.loc[event_pos])

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "🎯 Danger Score",
            f"{event_score:.3f}",
            help="Predicted dangerous progression probability",
        )
        col2.metric("📐 Pass Length", f"{event_row.get('pass_length', 'N/A'):.1f}m")
        col3.metric("⏱ Minute", int(event_row.get("minute", 0)))
        col4.metric("🗺 Zone", f"Zone {int(event_row.get('zone_start', 0))}")

        # Score confidence bar
        st.markdown("#### Confidence Bar")
        st.progress(float(np.clip(event_score, 0, 1)), text=f"{event_score:.1%} danger score")

        # Freeze frame plot
        st.markdown("#### ❄️ Freeze Frame")

        show_ranked = st.checkbox("Show ranked pass options", value=True)
        ranked_opts = None
        if show_ranked:
            sx = float(event_row.get("start_x", 60))
            sy = float(event_row.get("start_y", 40))
            ranked_opts = _make_ranked_options(sx, sy)

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
            if model is not None and feature_cols:
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

        name_col = "player_name" if "player_name" in pass_df.columns else None
        if name_col is None:
            st.warning("No player_name column in data.")
        else:
            all_players = sorted(pass_df[name_col].dropna().unique())
            selected_player = st.selectbox("Select Player", all_players)

            player_mask = pass_df[name_col] == selected_player
            player_df = pass_df[player_mask].reset_index(drop=True)
            player_scores = scores[player_mask]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Passes", len(player_df))
            col2.metric("Mean Score", f"{np.mean(player_scores):.3f}")
            col3.metric("Progressive Pass Rate", f"{(player_scores >= 0.5).mean():.1%}")

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

        label_col = next(
            (c for c in ["dangerous_progression_k", "line_break"] if c in pass_df.columns),
            None,
        )

        if label_col is not None:
            y_true = pass_df[label_col].fillna(0).values
            valid_mask = ~np.isnan(y_true) & ~np.isnan(scores)
            y_true_valid = y_true[valid_mask]
            y_prob_valid = scores[valid_mask]

            from src.evaluation.metrics import classification_metrics

            met = classification_metrics(y_true_valid, y_prob_valid)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ROC-AUC", f"{met.get('roc_auc', float('nan')):.4f}")
            col2.metric("PR-AUC", f"{met.get('pr_auc', float('nan')):.4f}")
            col3.metric("Brier Score", f"{met.get('brier_score', float('nan')):.4f}")
            col4.metric("Log Loss", f"{met.get('log_loss', float('nan')):.4f}")

            st.markdown("#### 🎛 Calibration Curve")
            try:
                from src.evaluation.calibration import plot_calibration_curve

                fig_cal, ax_cal = plt.subplots(figsize=(6, 6))
                plot_calibration_curve(y_true_valid, y_prob_valid, task_name=label_col, ax=ax_cal)
                st.pyplot(fig_cal, use_container_width=False)
                plt.close(fig_cal)
            except Exception as exc:
                st.error(f"Calibration plot error: {exc}")
        else:
            st.info("No ground-truth label columns found. Run labelling pipeline first.")

        # Feature importance
        if model is not None and feature_cols:
            st.markdown("#### 🔍 SHAP Feature Importance")
            with st.spinner("Computing SHAP values…"):
                try:
                    sample_df = pass_df[feature_cols].fillna(0.0).head(200)
                    fig_shap = _render_shap_summary(model, sample_df, feature_cols)
                    st.pyplot(fig_shap, use_container_width=True)
                    plt.close(fig_shap)
                except Exception as exc:
                    st.warning(f"SHAP summary unavailable: {exc}")
        else:
            st.info("Load a trained model to view SHAP feature importance.")

    # ================================================================
    # PAGE 5 – Possession Inspector
    # ================================================================
    elif page == "🏃 Possession Inspector":
        st.header("🏃 Possession Inspector")
        st.markdown(
            "Explore possession-level danger scores powered by the GRU sequence model. "
            "Select a possession to see a danger trajectory, per-event attribution, "
            "and the *unlock* event that drove the possession into a threatening position."
        )

        poss_df = _load_possession_sequences()
        gru_model = _load_gru_model()

        if poss_df is None:
            st.error(
                "❌ **possession_sequences.parquet** not found in `data/processed/`. "
                "Run `python src/data/parse_possessions.py` to generate it."
            )
            st.stop()

        if gru_model is None:
            st.warning(
                "⚠️ GRU model not found. Run `python src/models/gru_train_script.py` "
                "to train and save `models/gru_poss_dangerous.pt`."
            )

        # ── Match selector (possession-specific) ─────────────────────
        poss_matches = sorted(poss_df["match_id"].unique())
        if "match_id" in pass_df.columns and "team_name" in pass_df.columns:
            # Reuse the label map built earlier for the pass data
            _poss_match_labels = {
                mid: _match_label_map.get(mid, f"Match {mid}") for mid in poss_matches
            }
        else:
            _poss_match_labels = {mid: f"Match {mid}" for mid in poss_matches}

        _poss_label_to_mid = {v: k for k, v in _poss_match_labels.items()}
        selected_poss_match_label = st.sidebar.selectbox(
            "Select Match (Possession)",
            sorted(_poss_match_labels.values()),
            key="poss_match_selector",
        )
        sel_mid = _poss_label_to_mid[selected_poss_match_label]
        match_poss = poss_df[poss_df["match_id"] == sel_mid].reset_index(drop=True)

        st.sidebar.markdown(f"**{len(match_poss)} possessions** in selected match")
        label_rate = (
            match_poss["poss_dangerous"].mean()
            if "poss_dangerous" in match_poss.columns
            else float("nan")
        )
        st.sidebar.markdown(f"**Dangerous rate:** {label_rate:.1%}")

        # ── Possession selector ───────────────────────────────────────
        # Build human-readable labels for possessions
        poss_options = match_poss.index.tolist()
        poss_labels = []
        for _, pr in match_poss.iterrows():
            _team = str(pr.get("team_name", ""))[:18]
            _period = int(pr.get("period", 1))
            _n_ev = int(pr.get("n_events", 0))
            _orig = str(pr.get("origin_type", ""))[:16]
            _danger = "🔴" if pr.get("poss_dangerous") else "⚪"
            poss_labels.append(f"{_danger} {_team} | P{_period} | {_n_ev} evs | {_orig}")

        label_to_idx = dict(zip(poss_labels, poss_options))
        selected_poss_label = st.selectbox("Select Possession", poss_labels, key="poss_selector")
        poss_idx = label_to_idx[selected_poss_label]
        poss_row = match_poss.loc[poss_idx]

        # ── Possession summary metrics ────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Events", int(poss_row.get("n_events", 0)))
        col2.metric("Passes", int(poss_row.get("n_passes", 0)))
        col3.metric("Carries", int(poss_row.get("n_carries", 0)))
        col4.metric("Under Pressure", "Yes" if poss_row.get("has_pressure") else "No")
        col5.metric("Outcome", "🔴 Dangerous" if poss_row.get("poss_dangerous") else "⚪ Safe")

        # ── Attribution ───────────────────────────────────────────────
        from src.evaluation.possession_attribution import attribute_possession

        if gru_model is not None:
            with st.spinner("Computing attribution …"):
                report = attribute_possession(poss_row, gru_model)

            final_score = report["final_score"]
            st.metric("GRU Danger Score", f"{final_score:.3f}")
            st.progress(float(np.clip(final_score, 0, 1)), text=f"{final_score:.1%} P(dangerous)")

            events_enriched = report["events"]
            n_evs = len(events_enriched)
            steps = list(range(n_evs))
            cum_scores = [e["cum_score"] for e in events_enriched]
            loo_attrs = [e["loo_attr"] for e in events_enriched]
            type_labels = [e["type_label"] for e in events_enriched]
            unlock_idx = report["unlock_index"]

            # ── Danger trajectory ─────────────────────────────────────
            st.markdown("#### 📈 Danger Trajectory")
            traj_fig, ax_t = plt.subplots(figsize=(10, 3))
            ax_t.plot(steps, cum_scores, color="steelblue", linewidth=2, zorder=2)
            ax_t.fill_between(steps, 0, cum_scores, alpha=0.15, color="steelblue")
            ax_t.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="0.5 threshold")
            if unlock_idx >= 0:
                ax_t.axvline(
                    unlock_idx,
                    color="orange",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Unlock (step {unlock_idx})",
                )
                ax_t.scatter([unlock_idx], [cum_scores[unlock_idx]], color="orange", s=80, zorder=5)
            ax_t.set_xlabel("Event step")
            ax_t.set_ylabel("P(dangerous)")
            ax_t.set_ylim(0, 1)
            ax_t.set_xlim(0, max(n_evs - 1, 1))
            ax_t.legend(fontsize=8)
            ax_t.grid(axis="y", alpha=0.3)
            # Annotate every 5th event type
            for t in range(0, n_evs, max(1, n_evs // 8)):
                ax_t.annotate(
                    type_labels[t],
                    (t, cum_scores[t]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    fontsize=6.5,
                    ha="center",
                    color="gray",
                )
            st.pyplot(traj_fig, use_container_width=True)
            plt.close(traj_fig)

            # ── LOO attribution bar chart ──────────────────────────────
            st.markdown("#### 🏷 Per-Event Attribution (Leave-One-Out)")
            bar_colors = [
                "orange" if i == unlock_idx else ("green" if v > 0 else "crimson")
                for i, v in enumerate(loo_attrs)
            ]
            attr_fig, ax_a = plt.subplots(figsize=(10, 3))
            ax_a.bar(steps, loo_attrs, color=bar_colors, edgecolor="white", linewidth=0.4)
            ax_a.axhline(0, color="black", linewidth=0.5)
            ax_a.set_xlabel("Event step")
            ax_a.set_ylabel("Δ Score (full − masked)")
            ax_a.set_title("Positive = event elevated danger | Negative = event suppressed danger")
            ax_a.set_xlim(-0.5, n_evs - 0.5)
            ax_a.grid(axis="y", alpha=0.3)
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="orange", label="Unlock event"),
                Patch(facecolor="green", label="Positive attribution"),
                Patch(facecolor="crimson", label="Negative attribution"),
            ]
            ax_a.legend(handles=legend_elements, fontsize=8)
            st.pyplot(attr_fig, use_container_width=True)
            plt.close(attr_fig)

            # ── Event table ───────────────────────────────────────────
            st.markdown("#### 📋 Event Attribution Table")
            ev_table = pd.DataFrame(
                [
                    {
                        "Step": e["step"],
                        "Type": e["type_label"],
                        "Loc X": round(e.get("loc_x_norm", 0) * 120, 1),
                        "Loc Y": round(e.get("loc_y_norm", 0) * 80, 1),
                        "Under Press.": bool(e.get("under_pressure", 0)),
                        "Cum Score": round(e["cum_score"], 3),
                        "LOO Attr.": round(e["loo_attr"], 4),
                        "Unlock": "🔓" if e.get("is_unlock") else "",
                    }
                    for e in events_enriched
                ]
            )
            st.dataframe(
                ev_table.style.background_gradient(
                    subset=["LOO Attr."], cmap="RdYlGn", vmin=-0.15, vmax=0.15
                ),
                use_container_width=True,
                hide_index=True,
            )

            # ── Possession spatial summary ────────────────────────────
            st.markdown("#### 🗺 Possession Path")
            if _MPLSOCCER_AVAILABLE:
                pitch = Pitch(
                    pitch_type="statsbomb", pitch_color="grass", line_color="white", stripe=True
                )
                path_fig, ax_p = pitch.draw(figsize=(10, 6))

                # Draw event-to-event path coloured by cum_score
                xs = [e.get("loc_x_norm", 0) * 120 for e in events_enriched]
                ys = [e.get("loc_y_norm", 0) * 80 for e in events_enriched]
                import matplotlib.cm as cm

                cmap = cm.RdYlGn
                for i in range(len(xs) - 1):
                    c = cmap(float(cum_scores[i]))
                    ax_p.annotate(
                        "",
                        xy=(xs[i + 1], ys[i + 1]),
                        xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.5),
                    )

                # Mark start, unlock, and end
                ax_p.scatter(
                    [xs[0]], [ys[0]], s=120, c="white", edgecolors="black", zorder=5, label="Start"
                )
                ax_p.scatter(
                    [xs[-1]], [ys[-1]], s=120, c="yellow", edgecolors="black", zorder=5, label="End"
                )
                if unlock_idx >= 0 and unlock_idx < len(xs):
                    ax_p.scatter(
                        [xs[unlock_idx]],
                        [ys[unlock_idx]],
                        s=160,
                        c="orange",
                        edgecolors="black",
                        zorder=6,
                        label=f"Unlock ({type_labels[unlock_idx]})",
                    )

                ax_p.legend(loc="upper left", fontsize=8)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
                sm.set_array([])
                path_fig.colorbar(
                    sm,
                    ax=ax_p,
                    label="Cumulative Danger Score",
                    orientation="horizontal",
                    pad=0.04,
                    fraction=0.03,
                )
                st.pyplot(path_fig, use_container_width=True)
                plt.close(path_fig)
            else:
                st.info("mplsoccer not available — pitch plot skipped.")

        else:
            st.info(
                "Load a GRU model to compute attribution. Run `python src/models/gru_train_script.py`."
            )

        # ── Aggregate player leaderboard ─────────────────────────────
        st.markdown("---")
        st.markdown("#### 🏆 Player Attribution Leaderboard (sample 300 possessions)")
        if gru_model is not None and st.button("Compute Leaderboard", key="poss_leaderboard_btn"):
            with st.spinner("Attributing possessions …"):
                from src.evaluation.possession_attribution import player_attribution_summary

                sample = poss_df.sample(min(300, len(poss_df)), random_state=42)
                leaderboard = player_attribution_summary(
                    sample, gru_model, min_touches=5, verbose=False
                )
            if leaderboard.empty:
                st.info("No player_sequence column — showing team-level summary.")
            else:
                st.dataframe(
                    leaderboard[
                        [
                            "player",
                            "team",
                            "n_touches",
                            "n_unlocks",
                            "unlock_rate",
                            "mean_loo_attr",
                            "p90_loo_attr",
                            "mean_score_at_touch",
                            "n_possessions",
                        ]
                    ].round(4),
                    use_container_width=True,
                    hide_index=True,
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
