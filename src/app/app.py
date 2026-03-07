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


@st.cache_resource(show_spinner="Loading model…")  # type: ignore[misc]
def _load_model() -> Any | None:
    """Attempt to load a saved model artifact from models/."""
    try:
        import joblib

        candidates = sorted(_MODELS_DIR.glob("**/*.joblib"))
        if not candidates:
            return None
        path = candidates[0]
        payload = joblib.load(path)
        # TabularClassifier wraps the actual sklearn model
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"]
        return payload
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
    if model is not None and not demo_mode:
        feature_cols = [
            c for c in pass_df.columns
            if c not in {
                "event_uuid", "id", "match_id", "player_id", "player_name",
                "team_id", "team_name", "match_name",
                "line_break", "strict_line_break", "loose_line_break",
                "dangerous_progression_k", "final_third_entry_k", "box_entry_k",
                "shot_within_k", "threat_gain",
            }
            and pd.api.types.is_numeric_dtype(pass_df[c])
        ]
        try:
            X_all = pass_df[feature_cols].fillna(0.0)
            scores = model.predict_proba(X_all.values)[:, 1]
            st.sidebar.success("✅ Model loaded and predictions computed")
        except Exception as exc:
            st.sidebar.warning(f"Model prediction failed: {exc}. Using demo scores.")
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

    # ----------------------------------------------------------------
    # Sidebar – navigation and selectors
    # ----------------------------------------------------------------
    st.sidebar.header("🔍 Navigation")
    page = st.sidebar.radio(
        "View",
        options=["🎯 Event Inspector", "📊 Match Overview", "👤 Player Profile", "📈 Model Diagnostics"],
        index=0,
    )

    # Match selector
    match_col = "match_name" if "match_name" in pass_df.columns else "match_id"
    available_matches = sorted(pass_df[match_col].dropna().unique())
    selected_match = st.sidebar.selectbox("Select Match", available_matches, index=0)

    match_df = pass_df[pass_df[match_col] == selected_match].reset_index(drop=True)
    match_scores = scores[pass_df[match_col] == selected_match]

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
        event_labels = [
            f"{uid[:8]}… | score={sc:.3f}"
            for uid, sc in zip(
                match_df_sorted[uuid_col], match_df_sorted["_score"]
            )
        ]
        label_to_uuid = dict(zip(event_labels, event_options))

        selected_label = st.selectbox("Select Pass Event (sorted by score)", event_labels)
        selected_uuid = label_to_uuid[selected_label]

        event_row = match_df[match_df[uuid_col] == selected_uuid].iloc[0]
        event_pos = match_df.index[match_df[uuid_col] == selected_uuid][0]
        # Reindex scores to match match_df positions
        match_scores_series = pd.Series(match_scores, index=match_df.index)
        event_score = float(match_scores_series.loc[event_pos])

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Danger Score", f"{event_score:.3f}", help="Predicted dangerous progression probability")
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

        event_frames = frames_df[frames_df.get("event_uuid", frames_df.get("id", pd.Series())) == selected_uuid]
        if event_frames.empty:
            st.info("No freeze-frame data available for this event. Showing pass arrow only.")
            # Create a minimal frame with just the passer
            passer_frame = pd.DataFrame(
                [{"event_uuid": selected_uuid, "x": event_row.get("start_x", 60),
                  "y": event_row.get("start_y", 40), "teammate": True, "actor": True, "keeper": False}]
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
                    from src.visualization.explanations import explain_single_event

                    explanation = explain_single_event(
                        event_uuid=selected_uuid,
                        pass_instances_df=match_df,
                        frames_df=frames_df,
                        model=model,
                        feature_names=feature_cols,
                    )
                    st.markdown(f"**Narrative:**\n\n{explanation['narrative']}")
                    reasons_df = pd.DataFrame(explanation["top_reasons"])
                    if not reasons_df.empty:
                        st.dataframe(
                            reasons_df[["description", "value", "contribution"]]
                            .rename(columns={
                                "description": "Feature",
                                "value": "Value",
                                "contribution": "SHAP Contribution",
                            })
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
                title=f"Pass Map – {selected_match}",
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
            c for c in [uuid_col, "player_name", "start_x", "start_y", "end_x", "end_y",
                        "minute", "pass_length", "predicted_score"]
            if c in top_passes.columns
        ]
        st.dataframe(top_passes[display_cols].round(3), use_container_width=True)

        st.markdown("#### 📊 Breakdown by Zone")
        from src.evaluation.tactical_review import breakdown_by_zone

        label_col = (
            "line_break" if "line_break" in match_df.columns else None
        )
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

    # ----------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        "🔬 **Frame2Threat** | StatsBomb 360 pass danger prediction | "
        "[GitHub](https://github.com) | Demo mode: "
        + ("✅ Yes" if demo_mode else "❌ No")
    )


def _show_demo_explanation(score: float, event_row: pd.Series) -> None:
    """Show a placeholder explanation panel when model is unavailable."""
    from src.visualization.explanations import generate_explanation_narrative

    demo_reasons = [
        {"feature": "x_gain", "value": float(event_row.get("end_x", 70) - event_row.get("start_x", 60)),
         "contribution": 0.15 * score, "description": "Horizontal distance gained"},
        {"feature": "dist_to_goal_end", "value": float(event_row.get("end_x", 70)),
         "contribution": 0.12 * score, "description": "Receiver distance to goal"},
        {"feature": "pass_length", "value": float(event_row.get("pass_length", 15)),
         "contribution": -0.05 * score, "description": "Pass length"},
    ]
    narrative = generate_explanation_narrative(demo_reasons, score, "dangerous_progression")
    st.markdown(f"**Narrative (demo):**\n\n{narrative}")
    st.info("Train a model and rerun to see SHAP-based explanations.")


if __name__ == "__main__":
    main()
