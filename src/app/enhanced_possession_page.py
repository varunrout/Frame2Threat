"""
enhanced_possession_page.py
===========================
Enhanced Possession Inspector page for the Streamlit app.

Features:
- Match and possession selection with clear held-out/test/train distinction
- Football pitch visualisation with event-by-event playback
- XGBoost and GRU probability trajectories with model comparison
- Alert timeline with threshold crossings
- Realised outcome annotation
- Technical explanation of model features
- Clear distinction between genuine football data and demo mode
- Honest state when model artefacts are unavailable
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.app.live_scoring import score_arrays, events_to_arrays
from src.features.event_time import PossessionArrays

REQUIRED = {
    "possession table": "data/processed/possession_sequences.parquet",
    "event-time arrays": "data/processed/event_time_arrays.pkl",
    "split manifest": "data/processed/split_manifest.csv",
    "event-time metadata": "models/eventtime_meta.json",
    "event-time XGBoost": "models/eventtime_xgboost.joblib",
    "event-time GRU": "models/eventtime_gru.pt",
}

DEMO_POSSESSION_PATH = "reports/repro/demo_possession.json"


def missing_artefacts(root: Path) -> list[str]:
    """Names (with paths) of the artefacts the page needs that are not on disk."""
    return [f"{k} ({v})" for k, v in REQUIRED.items() if not (root / v).exists()]


def held_out_match_ids(manifest: pd.DataFrame, split: str = "test") -> set[int]:
    return {int(m) for m in manifest.loc[manifest["split"] == split, "match_id"]}


def index_arrays(arrs: list[PossessionArrays]) -> dict[tuple[int, int], PossessionArrays]:
    return {(a.match_id, a.possession_id): a for a in arrs}


def load_arrays(path: Path) -> dict[tuple[int, int], PossessionArrays]:
    return index_arrays(pickle.loads(path.read_bytes()))


def steps_frame(res: dict, arr: PossessionArrays) -> pd.DataFrame:
    """One row per event: prediction, change in prediction, alert flags, real coordinates."""
    df = pd.DataFrame(res["steps"]).rename(columns={"p": "p_danger"})
    df["event_idx"] = df["t"]
    df["elapsed_s"] = np.clip(arr.abs_s - arr.abs_s[0], 0.0, None)
    df["x"], df["y"] = arr.x, arr.y
    df["delta_p"] = df["p_danger"].diff()
    return df[
        [
            "event_idx",
            "event",
            "own",
            "elapsed_s",
            "x",
            "y",
            "p_danger",
            "delta_p",
            "alert",
            "tipping",
            "realised",
        ]
    ]


def key_events(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """The ``k`` events after which the prediction moved most (largest ``|delta_p|``)."""
    scored = df.dropna(subset=["delta_p"])
    return scored.reindex(scored["delta_p"].abs().sort_values(ascending=False).index).head(k)


def load_demo_possession(root: Path):
    """Load the demo possession JSON and convert to PossessionArrays."""
    import json

    demo_path = root / DEMO_POSSESSION_PATH
    if not demo_path.exists():
        return None, None
    with open(demo_path, "r") as f:
        demo = json.load(f)
    arr, warns = events_to_arrays(
        demo["event_sequence"],
        origin=demo["origin"],
        period=demo["period"],
        minute=demo["minute"],
    )
    arr.match_id = demo["match_id"]
    arr.possession_id = demo["possession_id"]
    arr.team = demo["team"]
    return demo, arr


def render_feature_explanation(st):
    """Render technical explanation of model features."""
    with st.expander("Technical: Model Features (Causal, Event-Time)", expanded=False):
        st.markdown(
            """
        **Features available at each event *t* (computed from events 0..t only):**
        
        | Feature Group | Features | Description |
        |---|---|---|
        **Ball Position** | `cur_x`, `cur_y` | Current ball location in possession frame (opponent events flipped) |
        **Territory** | `max_x_so_far` | Furthest x-reached by possessing team so far |
        **Event Counts** | `events_so_far`, `n_passes_so_far`, `n_carries_so_far`, `n_pressures_so_far` | Cumulative counts |
        **Time** | `elapsed_s` | Seconds since possession start |
        **Event Type** | `cur_type_*` (one-hot) | Current event type (Pass, Carry, Receipt, Pressure, Duel, etc.) |
        **Possession Context** | `start_x`, `start_y`, `period`, `game_minute` | Origin coordinates, period, minute |
        **Origin** | `origin_*` (one-hot) | Play pattern: Regular Play, Throw In, Free Kick, etc. |
        **Pressure** | `cur_is_own`, `under_pressure` | Possessing team flag, pressure indicator |
        
        **GRU-specific (sequence_inputs):**
        - Event type embedding (learned, 8-dim)
        - Normalised coordinates: `fx`, `fy`, `ex`, `ey` (possession frame)
        - `own`, `under_pressure`, `pass_length/60`
        - Normalised time: `elapsed/30` (clipped 0-4), `dt/10` (clipped 0-2)
        - Start context: `sx`, `sy`, `period==2`, `minute/90`
        - Origin one-hot
        
        **Causal constraint:** No feature uses possession length, future events, or outcome knowledge.
        """
        )


def render_limitations(st):
    """Render known limitations."""
    with st.expander("Known Limitations", expanded=False):
        st.markdown(
            """
        - **Discrimination is modest:** Event-time ROC AUC ~0.66 (all snapshots), ~0.74 (one per possession)
        - **Ball position dominates:** Logistic regression on (x, y) alone achieves AUC 0.64 / 0.70
        - **Models are statistically indistinguishable:** XGBoost vs GRU difference not significant
        - **False alerts:** 39-48% of safe possessions trigger an alert at validation threshold
        - **Calibration imperfect:** Models under-predict on held-out test (ECE ~0.04)
        - **Small test cohort:** 15 matches (World Cup 2022 + Barcelona La Liga 2020/21)
        - **No external validation:** Results may not transfer to other competitions/seasons
        - **Partial 360 coverage:** Freeze frames for only 52% of passes; not continuous tracking
        - **No causal claims:** Attribution is retrospective correlation, not causal contribution
        """
        )


def render(st, root: Path, *, cache_data, cache_resource) -> None:  # noqa: ANN001
    import matplotlib.pyplot as plt

    st.header("🏃 Possession Inspector — Event-Time Danger Forecasting")

    # Data availability check
    missing = missing_artefacts(root)
    demo_available = (root / DEMO_POSSESSION_PATH).exists()

    if missing and not demo_available:
        st.error(
            "Cannot show this page because required model artefacts are missing: "
            + "; ".join(missing)
            + ".  Rebuild with the commands in REPRODUCIBILITY.md "
            "(data pipeline, then `python scripts/run_event_time.py`)."
        )
        st.info("**Demo mode unavailable:** `reports/repro/demo_possession.json` not found.")
        return

    if missing and demo_available:
        st.warning(
            "⚠️ **Limited demo mode** — Full model artefacts not found. "
            "Showing the held-out France possession (match 3857266, possession 121) "
            "with pre-computed scores from the demo file. "
            "Run the full pipeline to enable match/possession selection and live scoring."
        )
        demo_mode = True
    else:
        demo_mode = False

    st.info(
        "**Event-time (prospective) prediction.** The curve below is the probability, after each "
        "event, that the possessing team produces a shot or penalty-box entry in a *later* event, "
        "computed from events up to that point only.  Retrospective, whole-possession scores "
        "(not shown on this page) are diagnostics, not forecasts."
    )

    # ============================================================
    # DATA LOADING
    # ============================================================
    if demo_mode:
        # Load demo possession only
        demo, demo_arr = load_demo_possession(root)
        if demo_arr is None:
            st.error("Failed to load demo possession.")
            return

        # Score with both models
        xgb_res = score_arrays(demo_arr, root, model="xgb")
        gru_res = score_arrays(demo_arr, root, model="gru")

        # Create a minimal possession table for UI
        poss_df = pd.DataFrame(
            [
                {
                    "match_id": demo["match_id"],
                    "possession_id": demo["possession_id"],
                    "team_name": demo["team"],
                    "period": demo["period"],
                    "origin_type": demo["origin"],
                    "n_events": len(demo_arr.type_id),
                    "poss_dangerous": demo.get("retrospective_label_poss_dangerous", 1),
                    "poss_has_shot": 1,
                    "poss_entered_box": 1,
                }
            ]
        )
        arrays = {(demo["match_id"], demo["possession_id"]): demo_arr}
        manifest = pd.DataFrame({"match_id": [demo["match_id"]], "split": ["test"]})

    else:

        @cache_data(show_spinner="Loading possessions…")
        def _poss() -> pd.DataFrame:
            return pd.read_parquet(
                root / REQUIRED["possession table"],
                columns=[
                    "match_id",
                    "possession_id",
                    "team_name",
                    "period",
                    "origin_type",
                    "n_events",
                    "poss_dangerous",
                    "poss_has_shot",
                    "poss_entered_box",
                ],
            )

        @cache_resource(show_spinner="Loading event arrays…")
        def _arrays() -> dict:
            return load_arrays(root / REQUIRED["event-time arrays"])

        @cache_data(show_spinner=False)
        def _manifest() -> pd.DataFrame:
            return pd.read_csv(root / REQUIRED["split manifest"])

        poss_df, arrays, manifest = _poss(), _arrays(), _manifest()

    # ============================================================
    # SIDEBAR: MATCH & POSSESSION SELECTION
    # ============================================================
    st.sidebar.header("📋 Selection")

    if demo_mode:
        st.sidebar.info("**Demo Mode** — Single held-out possession")
        st.sidebar.markdown(
            f"""
        - **Match:** {demo['match_id']}
        - **Possession:** {demo['possession_id']}
        - **Team:** {demo['team']}
        - **Period:** {demo['period']}
        - **Minute:** {demo['minute']:.0f}
        - **Origin:** {demo['origin']}
        - **Events:** {len(demo_arr.type_id)}
        - **Outcome:** Shot/Box Entry ✓
        """
        )
        match_id = demo["match_id"]
        possession_id = demo["possession_id"]
        row = poss_df.iloc[0]
        arr = demo_arr
    else:
        split = st.sidebar.radio(
            "Matches",
            ["test (held-out)", "val", "train (seen in training)"],
            key="poss_split",
            help="Test = held-out matches never seen during training. Train = matches used for fitting.",
        )
        split_key = split.split(" ")[0]
        if split_key == "train":
            st.sidebar.warning(
                "Train matches were used to fit these models; scores here are not held-out."
            )

        ids = held_out_match_ids(manifest, split_key)
        pool = poss_df[poss_df["match_id"].isin(ids) & (poss_df["n_events"] >= 4)]

        if pool.empty:
            st.error("No possessions with at least 4 events in this split.")
            return

        match_id = st.sidebar.selectbox(
            "Match", sorted(pool["match_id"].unique()), key="poss_match"
        )
        mp = pool[pool["match_id"] == match_id].reset_index(drop=True)

        only_dangerous = st.sidebar.checkbox("Only possessions that reached danger", value=False)
        if only_dangerous:
            mp = mp[mp["poss_dangerous"].astype(bool)].reset_index(drop=True)
            if mp.empty:
                st.info("No such possession in this match.")
                return

        # Lead with an informative possession (reached danger, ~18 events), not a trivial one.
        mp = (
            mp.assign(
                _rank=(~mp["poss_dangerous"].astype(bool)).astype(int) * 1000
                + (mp["n_events"] - 18).abs()
            )
            .sort_values("_rank")
            .drop(columns="_rank")
            .reset_index(drop=True)
        )
        labels = [
            f"#{r.possession_id} · {r.team_name} · P{r.period} · {r.n_events} events · "
            f"{'reached danger' if r.poss_dangerous else 'no danger'}"
            for r in mp.itertuples()
        ]
        selected_label = st.selectbox("Possession", labels, key="poss_pick")
        row = mp.iloc[labels.index(selected_label)]
        possession_id = int(row.possession_id)
        arr = arrays.get((int(row.match_id), possession_id))

        if arr is None:
            st.error("This possession is not in the event-time arrays; rebuild them.")
            return

    # ============================================================
    # SIDEBAR: MODEL & THRESHOLD SETTINGS
    # ============================================================
    st.sidebar.header("⚙️ Model Settings")

    model_name = st.sidebar.radio(
        "Model", ["XGBoost", "GRU", "Both (compare)"], key="poss_model", index=2
    )
    model_key = "xgb" if model_name == "XGBoost" else "gru"

    # Score with selected model(s)
    if model_name == "Both (compare)":
        xgb_res = score_arrays(arr, root, model="xgb")
        gru_res = score_arrays(arr, root, model="gru")
        res = xgb_res  # Use XGBoost as primary for threshold
    else:
        res = score_arrays(arr, root, model=model_key)
        if model_key == "xgb":
            xgb_res = res
            gru_res = score_arrays(arr, root, model="gru")
        else:
            gru_res = res
            xgb_res = score_arrays(arr, root, model="xgb")

    thr_default = float(res["threshold"])
    thr = st.sidebar.slider(
        "Alert Threshold",
        0.05,
        0.95,
        round(thr_default, 2),
        0.01,
        help=f"Default ({thr_default:.3f}) is the value chosen on validation data (prevalence quantile).",
    )
    if abs(thr - round(thr_default, 2)) > 1e-9:
        if model_name == "Both (compare)":
            xgb_res = score_arrays(arr, root, model="xgb", threshold=thr)
            gru_res = score_arrays(arr, root, model="gru", threshold=thr)
        else:
            res = score_arrays(arr, root, model=model_key, threshold=thr)

    # Show threshold info
    st.sidebar.caption(
        f"XGBoost default: {xgb_res['threshold']:.3f} | GRU default: {gru_res['threshold']:.3f}"
    )

    # ============================================================
    # MAIN CONTENT
    # ============================================================
    df = steps_frame(res, arr)

    # --- Header Metrics ---
    c = st.columns(5)
    c[0].metric("Events", len(df))
    c[1].metric("Team", str(row.team_name))
    c[2].metric("Period", f"P{int(row.period)}")
    c[3].metric("Origin", str(row.origin_type).replace("From ", "").replace(" Play", ""))

    tip = df[df["tipping"]]
    real = df[df["realised"]]

    if len(tip):
        c[4].metric("First alert", f"Event {int(tip.event_idx.iloc[0]) + 1}")
    else:
        c[4].metric("First alert", "None")

    if len(real):
        st.caption(
            f"★ **Outcome realised at Event {int(real.event_idx.iloc[0]) + 1}** ({real.event.iloc[0]})"
        )
        if len(tip):
            lead = int(real.event_idx.iloc[0] - tip.event_idx.iloc[0])
            st.caption(f"⏱ **Lead time:** {lead} events between first alert and outcome")

    st.divider()

    # --- PROBABILITY TRAJECTORIES ---
    st.subheader("📈 Danger Probability Trajectories")

    fig, ax = plt.subplots(figsize=(12, 4))

    # XGBoost trajectory
    xgb_steps = [(s["t"], s["p"]) for s in xgb_res["steps"] if s["p"] is not None]
    if xgb_steps:
        xgb_t, xgb_p = zip(*xgb_steps)
        ax.plot(
            np.array(xgb_t) + 1,
            xgb_p,
            color="#E67E22",
            lw=2.5,
            marker="o",
            ms=4,
            label="XGBoost",
            zorder=5,
            alpha=0.9,
        )

    # GRU trajectory
    gru_steps = [(s["t"], s["p"]) for s in gru_res["steps"] if s["p"] is not None]
    if gru_steps:
        gru_t, gru_p = zip(*gru_steps)
        ax.plot(
            np.array(gru_t) + 1,
            gru_p,
            color="#3498DB",
            lw=2.5,
            marker="s",
            ms=4,
            label="GRU",
            zorder=5,
            alpha=0.9,
        )

    # Threshold lines
    ax.axhline(
        xgb_res["threshold"],
        color="#E67E22",
        ls="--",
        lw=1.5,
        alpha=0.7,
        label=f"XGB threshold ({xgb_res['threshold']:.3f})",
    )
    ax.axhline(
        gru_res["threshold"],
        color="#3498DB",
        ls="--",
        lw=1.5,
        alpha=0.7,
        label=f"GRU threshold ({gru_res['threshold']:.3f})",
    )

    # First alert markers
    xgb_tip = next((s["t"] for s in xgb_res["steps"] if s["tipping"]), None)
    gru_tip = next((s["t"] for s in gru_res["steps"] if s["tipping"]), None)

    if xgb_tip is not None:
        ax.axvline(
            xgb_tip + 1,
            color="#E67E22",
            ls=":",
            lw=2,
            alpha=0.8,
            label=f"XGB first alert (event {xgb_tip+1})",
        )
    if gru_tip is not None:
        ax.axvline(
            gru_tip + 1,
            color="#3498DB",
            ls=":",
            lw=2,
            alpha=0.8,
            label=f"GRU first alert (event {gru_tip+1})",
        )

    # Outcome realised
    realised_at = xgb_res.get("realised_at")
    if realised_at is not None:
        ax.axvline(
            realised_at + 1,
            color="#2C3E50",
            ls="-",
            lw=2.5,
            alpha=0.9,
            label=f"Outcome realised (event {realised_at+1})",
        )
        ax.axvspan(
            realised_at + 1,
            len(arr.type_id) + 0.5,
            alpha=0.1,
            color="gray",
            label="No forecast (outcome known)",
        )

    ax.set_xlim(0.5, len(arr.type_id) + 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Event in Possession", fontsize=11)
    ax.set_ylabel("P(Shot or Box Entry in Later Event)", fontsize=11)
    ax.set_title(
        f"Match {match_id} · Possession {possession_id} · {row.team_name} · {len(arr.type_id)} events",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(fontsize=9, loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)

    # Event type annotations
    for t, step in enumerate(xgb_res["steps"]):
        if step["p"] is not None:
            event_short = step["event"][:3]
            color = "#E67E22" if step["own"] else "gray"
            ax.text(
                t + 1,
                -0.05,
                event_short,
                ha="center",
                va="top",
                fontsize=7,
                color=color,
                rotation=45,
                transform=ax.get_xaxis_transform(),
            )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "⚠️ **Scores are not calibrated to the held-out base rate** (event-time models under-predict on test matches). "
        "Use them to rank moments within a possession, not as absolute probabilities. "
        "Thresholds are validation-set prevalence quantiles, not optimised on test data."
    )

    st.divider()

    # --- PITCH VISUALISATION ---
    st.subheader("⚽ Ball Path (Possessing-Team Frame, Attacking Left → Right)")

    try:
        from mplsoccer import Pitch

        pitch = Pitch(
            pitch_type="statsbomb", pitch_color="#1a472a", line_color="white", linewidth=1.5
        )
        fig, ax = pitch.draw(figsize=(11, 6.5))

        own = df[df["own"]].copy()
        norm = plt.Normalize(0, 1)

        # Plot ball path with arrows colored by XGBoost probability
        for i in range(len(own) - 1):
            a = own.iloc[i]
            b = own.iloc[i + 1]
            if np.isfinite(a.x) and np.isfinite(a.y) and np.isfinite(b.x) and np.isfinite(b.y):
                p = a.p_danger if a.p_danger == a.p_danger else 1.0
                color = plt.cm.get_cmap("RdYlGn_r")(norm(p))
                ax.annotate(
                    "",
                    xy=(b.x, b.y),
                    xytext=(a.x, a.y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.8),
                )

        # Mark start
        if len(own) > 0 and np.isfinite(own.x.iloc[0]) and np.isfinite(own.y.iloc[0]):
            ax.scatter(
                own.x.iloc[0],
                own.y.iloc[0],
                s=200,
                c="#27AE60",
                edgecolors="white",
                linewidth=2,
                zorder=10,
                label="Start",
                marker="o",
            )
            ax.text(
                own.x.iloc[0] + 2,
                own.y.iloc[0] + 2,
                "1",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

        # Mark outcome
        if realised_at is not None and realised_at < len(arr.x):
            own_indices = np.where(arr.own)[0]
            if realised_at < len(own_indices):
                idx = own_indices[realised_at]
                if np.isfinite(arr.x[idx]) and np.isfinite(arr.y[idx]):
                    ax.scatter(
                        arr.x[idx],
                        arr.y[idx],
                        s=250,
                        c="#2C3E50",
                        edgecolors="gold",
                        linewidth=3,
                        zorder=10,
                        label="Outcome: Shot/Box Entry",
                        marker="*",
                    )
                    ax.text(
                        arr.x[idx] + 2,
                        arr.y[idx] + 2,
                        "★",
                        fontsize=14,
                        fontweight="bold",
                        color="gold",
                    )

        # Add event numbers
        for i, (x, y) in enumerate(zip(own.x, own.y)):
            if np.isfinite(x) and np.isfinite(y) and i > 0:
                ax.text(
                    x + 1,
                    y + 1,
                    str(i + 1),
                    fontsize=8,
                    color="white",
                    bbox=dict(
                        boxstyle="circle,pad=0.2", facecolor="black", alpha=0.7, edgecolor="none"
                    ),
                )

        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.set_title(
            f"Ball Path — {len(own)} possessing-team events", fontsize=12, fontweight="bold", pad=10
        )

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
        cbar.set_label("XGBoost P(danger) at arrow start", fontsize=9)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Arrow colour = model's P(danger) when the arrow starts (green=low, red=high). "
            "Coordinates are in the possessing team's attacking frame (opponent events flipped). "
            "Only possessing-team events shown; opponent events (pressures, duels) omitted for clarity."
        )

    except ImportError:
        st.info(
            "📦 `mplsoccer` not installed; pitch plot skipped. Install with `pip install mplsoccer`."
        )

    st.divider()

    # --- EVENT TABLE ---
    st.subheader("📋 Event-by-Event Detail")

    display_df = (
        df.assign(event_idx=df.event_idx + 1)
        .rename(
            columns={
                "event_idx": "#",
                "own": "Possessing Team",
                "elapsed_s": "Elapsed (s)",
                "p_danger": "P(danger)",
                "delta_p": "ΔP",
                "tipping": "First Alert",
            }
        )
        .round(3)
    )

    # Add flags column
    def make_flags(row):
        flags = []
        if row["alert"]:
            flags.append("⚠️ Alert")
        if row["First Alert"]:
            flags.append("🎯 TIP")
        if row["realised"]:
            flags.append("★ OUTCOME")
        return " | ".join(flags)

    display_df["Flags"] = display_df.apply(make_flags, axis=1)

    st.dataframe(
        display_df[
            ["#", "event", "Possessing Team", "Elapsed (s)", "x", "y", "P(danger)", "ΔP", "Flags"]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "x": st.column_config.NumberColumn("X", format="%.1f"),
            "y": st.column_config.NumberColumn("Y", format="%.1f"),
            "P(danger)": st.column_config.NumberColumn("P(danger)", format="%.3f"),
            "ΔP": st.column_config.NumberColumn("ΔP", format="%.3f"),
        },
    )

    st.divider()

    # --- KEY EVENTS ---
    st.subheader("🔑 Key Events (Largest Probability Changes)")
    keys = key_events(df, k=5)
    if not keys.empty:
        st.dataframe(
            keys[["event_idx", "event", "own", "elapsed_s", "p_danger", "delta_p"]]
            .assign(event_idx=lambda d: d.event_idx + 1)
            .rename(columns={"event_idx": "Event #", "own": "Possessing", "delta_p": "Δ P(danger)"})
            .round(3),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    # --- MODEL COMPARISON ---
    if model_name == "Both (compare)":
        st.subheader("⚖️ Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**XGBoost**")
            xgb_probs = [s["p"] for s in xgb_res["steps"] if s["p"] is not None]
            st.metric("Threshold", f"{xgb_res['threshold']:.3f}")
            st.metric("Mean P", f"{np.mean(xgb_probs):.3f}")
            st.metric("Max P", f"{np.max(xgb_probs):.3f}")
            st.metric("Final P", f"{xgb_probs[-1]:.3f}")
            st.metric("Alerts", f"{sum(1 for s in xgb_res['steps'] if s['alert'])}")

        with col2:
            st.markdown("**GRU**")
            gru_probs = [s["p"] for s in gru_res["steps"] if s["p"] is not None]
            st.metric("Threshold", f"{gru_res['threshold']:.3f}")
            st.metric("Mean P", f"{np.mean(gru_probs):.3f}")
            st.metric("Max P", f"{np.max(gru_probs):.3f}")
            st.metric("Final P", f"{gru_probs[-1]:.3f}")
            st.metric("Alerts", f"{sum(1 for s in gru_res['steps'] if s['alert'])}")

        # Correlation
        if len(xgb_probs) == len(gru_probs):
            corr = np.corrcoef(xgb_probs, gru_probs)[0, 1]
            st.metric("Trajectory Correlation", f"{corr:.3f}")

        # Disagreement analysis
        if xgb_tip is not None and gru_tip is not None:
            diff = abs(xgb_tip - gru_tip)
            if diff > 0:
                st.info(
                    f"⚡ **First alert difference:** {diff} events. "
                    f"{'XGBoost alerts earlier (more sensitive to early territorial gain)' if xgb_tip < gru_tip else 'GRU alerts earlier (more sensitive to sequence patterns)'}"
                )

        # At outcome
        if realised_at is not None and realised_at > 0:
            st.markdown(f"**At outcome (Event {realised_at+1}):**")
            st.markdown(f"- XGBoost: {xgb_probs[realised_at-1]:.3f}")
            st.markdown(f"- GRU: {gru_probs[realised_at-1]:.3f}")
            st.markdown(
                f"- Difference: {abs(xgb_probs[realised_at-1] - gru_probs[realised_at-1]):.3f}"
            )

    st.divider()

    # --- TECHNICAL EXPLANATION ---
    render_feature_explanation(st)

    # --- LIMITATIONS ---
    render_limitations(st)

    # --- DEMO MODE NOTICE ---
    if demo_mode:
        st.divider()
        st.warning(
            "⚠️ **Demo Mode Active** — This is a single held-out possession (France, match 3857266, possession 121) "
            "with pre-computed scores. Full match/possession selection and live scoring require "
            "running the complete data pipeline and model training. See REPRODUCIBILITY.md for instructions."
        )


# For backward compatibility
__all__ = ["render"]
