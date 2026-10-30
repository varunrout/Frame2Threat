"""
possession_page.py
==================
Possession Inspector page of the Streamlit app.

* The danger trajectory comes from the **event-time** models (prospective: step ``t`` uses
  events ``0..t`` only), scored on real possessions, held-out test matches by default.
* The per-event attribution shown below it is **retrospective** (cross-fitted leave-one-out
  drops, ``reports/attribution``) and is labelled as such.
* Nothing here is synthetic: if an artefact is missing the page says which one and stops.

The pure helpers (no Streamlit) are unit-tested in ``tests/test_possession_page.py``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.live_scoring import score_arrays
from src.data.event_types import label as type_label
from src.features.event_time import PossessionArrays

REQUIRED = {
    "possession table": "data/processed/possession_sequences.parquet",
    "event-time arrays": "data/processed/event_time_arrays.pkl",
    "split manifest": "data/processed/split_manifest.csv",
    "event-time metadata": "models/eventtime_meta.json",
    "event-time XGBoost": "models/eventtime_xgboost.joblib",
    "event-time GRU": "models/eventtime_gru.pt",
}


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
        ["event_idx", "event", "own", "elapsed_s", "x", "y", "p_danger", "delta_p", "alert",
         "tipping", "realised"]
    ]


def key_events(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """The ``k`` events after which the prediction moved most (largest ``|delta_p|``)."""
    scored = df.dropna(subset=["delta_p"])
    return scored.reindex(scored["delta_p"].abs().sort_values(ascending=False).index).head(k)


def possession_attribution(att: pd.DataFrame, match_id: int, possession_id: int) -> pd.DataFrame:
    """Cross-fitted retrospective LOO drops for one possession (seed 42 run)."""
    sel = att[(att["match_id"] == match_id) & (att["possession_id"] == possession_id)]
    if "seed" in sel.columns:
        sel = sel[sel["seed"] == sel["seed"].min()]
    return sel.sort_values("event_idx")


def leaderboard(players: pd.DataFrame, min_events: int) -> pd.DataFrame:
    """Players with at least ``min_events`` possessing-team events, best first."""
    out = players[players["events"] >= min_events].copy()
    out["interval_excludes_zero"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    return out.sort_values("mean_loo", ascending=False)


# ----------------------------------------------------------------------------------------
# Streamlit rendering
# ----------------------------------------------------------------------------------------


def render(st, root: Path, *, cache_data, cache_resource) -> None:  # noqa: ANN001
    import matplotlib.pyplot as plt

    st.header("🏃 Possession Inspector")
    st.info(
        "**Event-time (prospective) prediction.** The curve below is the probability, after each "
        "event, that the possessing team produces a shot or penalty-box entry in a *later* event, "
        "computed from events up to that point only.  The attribution further down is "
        "**retrospective** (it uses the whole possession) and is not a forecast."
    )
    missing = missing_artefacts(root)
    if missing:
        st.error(
            "Cannot show this page because these files are missing: "
            + "; ".join(missing)
            + ".  Rebuild with the commands in REPRODUCIBILITY.md (data pipeline, then "
            "`python scripts/run_event_time.py`)."
        )
        return

    @cache_data(show_spinner="Loading possessions…")
    def _poss() -> pd.DataFrame:
        return pd.read_parquet(
            root / REQUIRED["possession table"],
            columns=["match_id", "possession_id", "team_name", "period", "origin_type",
                     "n_events", "poss_dangerous", "poss_has_shot", "poss_entered_box"],
        )

    @cache_resource(show_spinner="Loading event arrays…")
    def _arrays() -> dict:
        return load_arrays(root / REQUIRED["event-time arrays"])

    @cache_data(show_spinner=False)
    def _manifest() -> pd.DataFrame:
        return pd.read_csv(root / REQUIRED["split manifest"])

    poss, arrays, manifest = _poss(), _arrays(), _manifest()
    split = st.sidebar.radio(
        "Matches", ["test (held-out)", "val", "train (seen in training)"], key="poss_split"
    )
    split_key = split.split(" ")[0]
    if split_key == "train":
        st.warning("Train matches were used to fit these models; scores here are not held-out.")
    ids = held_out_match_ids(manifest, split_key)
    pool = poss[poss["match_id"].isin(ids) & (poss["n_events"] >= 4)]
    if pool.empty:
        st.error("No possessions with at least 4 events in this split.")
        return
    match_id = st.sidebar.selectbox("Match", sorted(pool["match_id"].unique()), key="poss_match")
    mp = pool[pool["match_id"] == match_id].reset_index(drop=True)
    only_dangerous = st.sidebar.checkbox("Only possessions that reached danger", value=False)
    if only_dangerous:
        mp = mp[mp["poss_dangerous"].astype(bool)].reset_index(drop=True)
        if mp.empty:
            st.info("No such possession in this match.")
            return
    labels = [
        f"#{r.possession_id} · {r.team_name} · P{r.period} · {r.n_events} events · "
        f"{'reached danger' if r.poss_dangerous else 'no danger'}"
        for r in mp.itertuples()
    ]
    row = mp.iloc[labels.index(st.selectbox("Possession", labels, key="poss_pick"))]
    arr = arrays.get((int(row.match_id), int(row.possession_id)))
    if arr is None:
        st.error("This possession is not in the event-time arrays; rebuild them.")
        return

    model_name = st.sidebar.radio("Model", ["GRU", "XGBoost"], key="poss_model").lower()
    model_key = "gru" if model_name == "gru" else "xgb"
    res = score_arrays(arr, root, model=model_key)
    thr_default = float(res["threshold"])
    thr = st.sidebar.slider("Alert threshold", 0.05, 0.95, round(thr_default, 2), 0.01,
                            help="Default is the value chosen on validation data.")
    if abs(thr - round(thr_default, 2)) > 1e-9:
        res = score_arrays(arr, root, model=model_key, threshold=thr)
    df = steps_frame(res, arr)

    c = st.columns(4)
    c[0].metric("Events", len(df))
    c[1].metric("Team", str(row.team_name))
    tip = df[df["tipping"]]
    c[2].metric("First alert", f"event {int(tip.event_idx.iloc[0]) + 1}" if len(tip) else "none")
    real = df[df["realised"]]
    c[3].metric(
        "Outcome",
        f"shot / box entry at event {int(real.event_idx.iloc[0]) + 1}" if len(real) else "none",
    )
    if len(tip) and len(real):
        st.caption(
            f"Lead time: the first alert came {int(real.event_idx.iloc[0] - tip.event_idx.iloc[0])} "
            "events before the outcome."
        )

    st.subheader("Predicted danger over time (event-time)")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    scored = df.dropna(subset=["p_danger"])
    ax.plot(scored.event_idx + 1, scored.p_danger, color="steelblue", lw=2, marker="o", ms=3)
    ax.axhline(thr, color="crimson", ls="--", lw=1, label=f"alert threshold {thr:.2f}")
    if len(tip):
        ax.axvline(tip.event_idx.iloc[0] + 1, color="orange", ls=":", label="first alert")
    if len(real):
        ax.axvline(real.event_idx.iloc[0] + 1, color="black", ls="-", lw=1, label="outcome realised")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Event in possession")
    ax.set_ylabel("P(shot or box entry later)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "Scores are not calibrated to the held-out base rate (the event-time models under-predict on "
        "the test matches).  Use them to rank moments within a possession, not as probabilities."
    )

    st.subheader("Key events: where the prediction moved most")
    keys = key_events(df)
    st.dataframe(
        keys[["event_idx", "event", "own", "elapsed_s", "p_danger", "delta_p"]]
        .assign(event_idx=lambda d: d.event_idx + 1)
        .rename(columns={"event_idx": "event #", "own": "possessing team", "delta_p": "Δ P(danger)"})
        .round(3),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Ball path (possessing-team frame, attacking left → right)")
    try:
        from mplsoccer import Pitch

        pitch = Pitch(pitch_type="statsbomb", pitch_color="grass", line_color="white")
        fig, ax = pitch.draw(figsize=(9, 5.5))
        own = df[df["own"]]
        norm = plt.Normalize(0, 1)
        for a, b in zip(own.itertuples(), own.iloc[1:].itertuples()):
            p = a.p_danger if a.p_danger == a.p_danger else 1.0
            ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y),
                        arrowprops=dict(arrowstyle="->", color=plt.cm.RdYlGn_r(norm(p)), lw=1.6))
        ax.scatter(own.x.iloc[0], own.y.iloc[0], s=110, c="white", edgecolors="black", zorder=5,
                   label="start")
        if len(real):
            r = df.iloc[int(real.event_idx.iloc[0])]
            ax.scatter(r.x, r.y, s=170, c="gold", edgecolors="black", zorder=6, label="outcome")
        ax.legend(loc="upper left", fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Arrow colour = the model's P(danger) when the arrow starts (green low, red high).")
    except ImportError:
        st.info("mplsoccer is not installed; pitch plot skipped.")

    st.subheader("Event table")
    st.dataframe(
        df.assign(event_idx=df.event_idx + 1)
        .rename(columns={"event_idx": "#", "own": "possessing team", "elapsed_s": "elapsed (s)"})
        .round(3),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Retrospective attribution (not a forecast)")
    att_path = root / "data/processed/attribution_events.parquet"
    if not att_path.exists():
        st.info("Cross-fitted attribution file not found (`data/processed/attribution_events.parquet`).")
    else:
        @cache_data(show_spinner=False)
        def _att() -> pd.DataFrame:
            return pd.read_parquet(att_path)

        a = possession_attribution(_att(), int(row.match_id), int(row.possession_id))
        if a.empty:
            st.info("No attribution for this possession (it is outside the first 40 events or too short).")
        else:
            fig, ax = plt.subplots(figsize=(10, 2.8))
            colors = np.where(a.possessing_team_event, "steelblue", "lightgray")
            ax.bar(a.event_idx + 1, a.loo_drop, color=colors)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xlabel("Event in possession")
            ax.set_ylabel("score drop when event removed")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "Cross-fitted leave-one-out drops from the retrospective possession GRU (a model that never "
                "saw this match).  Blue = possessing team, grey = defending team (evidence of danger, not credit).  "
                "It shows what the model leans on, not causal contribution."
            )

    st.subheader("Player and team tables (retrospective, cross-fitted)")
    players_path = root / "reports/attribution/player_leaderboard.csv"
    teams_path = root / "reports/attribution/team_leaderboard.csv"
    if players_path.exists():
        n_min = st.slider("Minimum possessing-team events per player", 100, 600, 100, 50)
        board = leaderboard(pd.read_csv(players_path), n_min)
        st.caption(
            f"{len(board)} players qualify.  Intervals are player-level cluster bootstrap 95% intervals; "
            "the ranking is dominated by playing role (attackers high, defenders low) and split-half "
            "rank stability is 0.62.  Use it to build a shortlist to inspect, not to rank players."
        )
        st.dataframe(board.round(4), hide_index=True, use_container_width=True)
    if teams_path.exists():
        st.dataframe(pd.read_csv(teams_path).round(4), hide_index=True, use_container_width=True)
    with st.expander("Limitations of this view"):
        st.markdown(
            "- Event-time ROC AUC is about 0.66 over all snapshots and 0.74 for one random snapshot "
            "per possession; ball position alone gives 0.64 / 0.70.\n"
            "- XGBoost and the GRU are statistically indistinguishable.\n"
            "- Alerts fire on 39–48% of possessions that never reach danger.\n"
            "- The test split is 15 matches; the data are StatsBomb Open Data (World Cup 2022 and La Liga "
            "2020/21 Barcelona), so results may not transfer to other competitions.\n"
            "- `type_label` names and coordinates are in the possessing team's attacking frame."
        )


__all__ = ["render", "type_label"]
