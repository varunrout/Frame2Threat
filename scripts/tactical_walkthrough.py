#!/usr/bin/env python
"""
Tactical Match Walkthrough Generator
=====================================
Creates a detailed visual walkthrough of the genuine held-out France possession
(match 3857266, possession 121) showing how XGBoost and GRU danger probabilities
evolve event-by-event.

Outputs:
- reports/figures/tactical_walkthrough.png (main visual)
- reports/repro/tactical_walkthrough.json (detailed data)
- reports/repro/tactical_walkthrough.md (markdown narrative)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.app.live_scoring import score_arrays, events_to_arrays
from src.data.event_types import label as type_label


def load_possession_data(root: Path):
    """Load the demo possession and score it with both models."""
    # Load the demo possession JSON
    demo_path = root / "reports" / "repro" / "demo_possession.json"
    with open(demo_path, "r") as f:
        demo = json.load(f)

    # Convert to PossessionArrays
    arr, warns = events_to_arrays(
        demo["event_sequence"],
        origin=demo["origin"],
        period=demo["period"],
        minute=demo["minute"],
    )
    arr.match_id = demo["match_id"]
    arr.possession_id = demo["possession_id"]
    arr.team = demo["team"]

    # Score with both models
    xgb_res = score_arrays(arr, root, model="xgb")
    gru_res = score_arrays(arr, root, model="gru")

    return demo, arr, xgb_res, gru_res


def create_walkthrough_figure(demo, arr, xgb_res, gru_res, root):
    """Create the main tactical walkthrough figure."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1, 1], width_ratios=[1.5, 1, 1])

    # Color scheme
    XGB_COLOR = "#E67E22"  # Dark orange
    GRU_COLOR = "#3498DB"  # Steel blue
    OUTCOME_COLOR = "#2C3E50"  # Dark blue/black
    START_COLOR = "#27AE60"  # Green

    # ============================================================
    # PANEL 1: Pitch diagram with ball path (top-left, spans 2 cols)
    # ============================================================
    ax_pitch = fig.add_subplot(gs[0, :2])

    try:
        from mplsoccer import Pitch

        pitch = Pitch(
            pitch_type="statsbomb", pitch_color="#1a472a", line_color="white", linewidth=1.5
        )
        pitch.draw(ax=ax_pitch)

        # Get own-team events with coordinates
        own_mask = arr.own
        x_own = arr.x[own_mask]
        y_own = arr.y[own_mask]

        # Plot ball path with event numbers
        for i in range(len(x_own) - 1):
            if (
                np.isfinite(x_own[i])
                and np.isfinite(y_own[i])
                and np.isfinite(x_own[i + 1])
                and np.isfinite(y_own[i + 1])
            ):
                # Color by XGBoost probability at this step
                p_xgb = xgb_res["steps"][np.where(own_mask)[0][i]]["p"]
                if p_xgb is not None:
                    color = plt.cm.RdYlGn_r(p_xgb)
                else:
                    color = "gray"
                ax_pitch.annotate(
                    "",
                    xy=(x_own[i + 1], y_own[i + 1]),
                    xytext=(x_own[i], y_own[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.8),
                )

        # Mark start
        if len(x_own) > 0 and np.isfinite(x_own[0]) and np.isfinite(y_own[0]):
            ax_pitch.scatter(
                x_own[0],
                y_own[0],
                s=200,
                c=START_COLOR,
                edgecolors="white",
                linewidth=2,
                zorder=10,
                label="Start (Throw-in)",
                marker="o",
            )
            ax_pitch.text(
                x_own[0] + 2, y_own[0] + 2, "1", fontsize=10, fontweight="bold", color="white"
            )

        # Mark outcome event
        realised_idx = xgb_res.get("realised_at")
        if realised_idx is not None and realised_idx < len(arr.x):
            # Find the corresponding own-team event index
            own_indices = np.where(own_mask)[0]
            if realised_idx < len(own_indices):
                idx = own_indices[realised_idx]
                if np.isfinite(arr.x[idx]) and np.isfinite(arr.y[idx]):
                    ax_pitch.scatter(
                        arr.x[idx],
                        arr.y[idx],
                        s=250,
                        c=OUTCOME_COLOR,
                        edgecolors="gold",
                        linewidth=3,
                        zorder=10,
                        label="Outcome: Shot/Box Entry",
                        marker="*",
                    )
                    ax_pitch.text(
                        arr.x[idx] + 2,
                        arr.y[idx] + 2,
                        "*",
                        fontsize=14,
                        fontweight="bold",
                        color="gold",
                    )

        # Add event numbers along the path
        for i, (x, y) in enumerate(zip(x_own, y_own)):
            if np.isfinite(x) and np.isfinite(y) and i > 0:
                ax_pitch.text(
                    x + 1,
                    y + 1,
                    str(i + 1),
                    fontsize=8,
                    color="white",
                    bbox=dict(
                        boxstyle="circle,pad=0.2", facecolor="black", alpha=0.7, edgecolor="none"
                    ),
                )

        ax_pitch.set_title(
            f"France vs Opponent -- Match {demo['match_id']}, Possession {demo['possession_id']}\n"
            f"Period {demo['period']}, Minute {demo['minute']:.0f} -- {demo['origin']} -- {len(arr.type_id)} events",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        ax_pitch.legend(loc="lower left", fontsize=9, framealpha=0.9)

        # Add colorbar for probability
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_pitch, shrink=0.6, aspect=20, pad=0.02)
        cbar.set_label("XGBoost P(danger)", fontsize=9)

    except ImportError:
        ax_pitch.text(
            0.5,
            0.5,
            "mplsoccer not installed",
            ha="center",
            va="center",
            transform=ax_pitch.transAxes,
        )
        ax_pitch.set_title("Pitch Diagram (mplsoccer required)")

    # ============================================================
    # PANEL 2: Event sequence table (top-right)
    # ============================================================
    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis("off")

    # Build event table
    events_data = []
    for t, step in enumerate(xgb_res["steps"]):
        event_type = step["event"]
        team = "France" if step["own"] else "Opponent"
        p_xgb = step["p"]
        p_gru = gru_res["steps"][t]["p"]
        alert_xgb = "[XGB]" if step["alert"] else ""
        alert_gru = "[GRU]" if gru_res["steps"][t]["alert"] else ""
        realised = "[OUTCOME]" if step["realised"] else ""
        tipping = "[TIP]" if step["tipping"] else ""

        events_data.append(
            {
                "#": t + 1,
                "Event": event_type[:12],
                "Team": team[:3],
                "XGB": f"{p_xgb:.3f}" if p_xgb is not None else "--",
                "GRU": f"{p_gru:.3f}" if p_gru is not None else "--",
                "Flags": f"{alert_xgb}{alert_gru}{tipping}{realised}".strip(),
            }
        )

    df_events = pd.DataFrame(events_data)
    table = ax_table.table(
        cellText=df_events.values,
        colLabels=df_events.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.08, 0.25, 0.12, 0.15, 0.15, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Color code rows
    for i, row in enumerate(events_data):
        if row["Flags"] and "[OUTCOME]" in row["Flags"]:
            for j in range(len(df_events.columns)):
                table[(i + 1, j)].set_facecolor("#FFF3CD")
        elif row["Flags"] and "[TIP]" in row["Flags"]:
            for j in range(len(df_events.columns)):
                table[(i + 1, j)].set_facecolor("#F8D7DA")
        elif row["Flags"] and ("[XGB]" in row["Flags"] or "[GRU]" in row["Flags"]):
            for j in range(len(df_events.columns)):
                table[(i + 1, j)].set_facecolor("#FFF8E1")

    ax_table.set_title("Event Sequence with Probabilities", fontsize=11, fontweight="bold", pad=10)

    # ============================================================
    # PANEL 3: Probability trajectories (middle row, spans all)
    # ============================================================
    ax_prob = fig.add_subplot(gs[1, :])

    # Get scored steps (where p is not None)
    xgb_steps = [(s["t"], s["p"]) for s in xgb_res["steps"] if s["p"] is not None]
    gru_steps = [(s["t"], s["p"]) for s in gru_res["steps"] if s["p"] is not None]

    if xgb_steps:
        xgb_t, xgb_p = zip(*xgb_steps)
        ax_prob.plot(
            np.array(xgb_t) + 1,
            xgb_p,
            color=XGB_COLOR,
            lw=2.5,
            marker="o",
            ms=4,
            label="XGBoost",
            zorder=5,
        )

    if gru_steps:
        gru_t, gru_p = zip(*gru_steps)
        ax_prob.plot(
            np.array(gru_t) + 1,
            gru_p,
            color=GRU_COLOR,
            lw=2.5,
            marker="s",
            ms=4,
            label="GRU",
            zorder=5,
        )

    # Threshold lines
    ax_prob.axhline(
        xgb_res["threshold"],
        color=XGB_COLOR,
        ls="--",
        lw=1.5,
        alpha=0.7,
        label=f"XGB threshold ({xgb_res['threshold']:.3f})",
    )
    ax_prob.axhline(
        gru_res["threshold"],
        color=GRU_COLOR,
        ls="--",
        lw=1.5,
        alpha=0.7,
        label=f"GRU threshold ({gru_res['threshold']:.3f})",
    )

    # First alert markers
    xgb_tip = next((s["t"] for s in xgb_res["steps"] if s["tipping"]), None)
    gru_tip = next((s["t"] for s in gru_res["steps"] if s["tipping"]), None)

    if xgb_tip is not None:
        ax_prob.axvline(
            xgb_tip + 1,
            color=XGB_COLOR,
            ls=":",
            lw=2,
            alpha=0.8,
            label=f"XGB first alert (event {xgb_tip+1})",
        )
    if gru_tip is not None:
        ax_prob.axvline(
            gru_tip + 1,
            color=GRU_COLOR,
            ls=":",
            lw=2,
            alpha=0.8,
            label=f"GRU first alert (event {gru_tip+1})",
        )

    # Outcome realised
    realised_at = xgb_res.get("realised_at")
    if realised_at is not None:
        ax_prob.axvline(
            realised_at + 1,
            color=OUTCOME_COLOR,
            ls="-",
            lw=2.5,
            alpha=0.9,
            label=f"Outcome realised (event {realised_at+1})",
        )
        # Shade post-outcome region
        ax_prob.axvspan(
            realised_at + 1,
            len(arr.type_id) + 0.5,
            alpha=0.1,
            color="gray",
            label="No forecast (outcome known)",
        )

    ax_prob.set_xlim(0.5, len(arr.type_id) + 0.5)
    ax_prob.set_ylim(0, 1.02)
    ax_prob.set_xlabel("Event in Possession", fontsize=11)
    ax_prob.set_ylabel("P(Shot or Box Entry in Later Event)", fontsize=11)
    ax_prob.set_title(
        "Event-Time Danger Probability Trajectories", fontsize=13, fontweight="bold", pad=10
    )
    ax_prob.legend(fontsize=9, loc="upper left", ncol=2, framealpha=0.9)
    ax_prob.grid(alpha=0.3)

    # Add event type annotations at bottom
    for t, step in enumerate(xgb_res["steps"]):
        if step["p"] is not None:
            event_short = step["event"][:3]
            color = XGB_COLOR if step["own"] else "gray"
            ax_prob.text(
                t + 1,
                -0.05,
                event_short,
                ha="center",
                va="top",
                fontsize=7,
                color=color,
                rotation=45,
                transform=ax_prob.get_xaxis_transform(),
            )

    # ============================================================
    # PANEL 4: Key moments detail (bottom-left)
    # ============================================================
    ax_moments = fig.add_subplot(gs[2, 0])
    ax_moments.axis("off")

    moments_text = "KEY MOMENTS\n" + "=" * 30 + "\n\n"

    # First alerts
    if xgb_tip is not None:
        moments_text += (
            f"[XGB] XGBoost first alert: Event {xgb_tip+1} ({xgb_res['steps'][xgb_tip]['event']})\n"
        )
        moments_text += (
            f"   P(danger) = {xgb_res['steps'][xgb_tip]['p']:.3f} >= {xgb_res['threshold']:.3f}\n\n"
        )

    if gru_tip is not None:
        moments_text += (
            f"[GRU] GRU first alert: Event {gru_tip+1} ({gru_res['steps'][gru_tip]['event']})\n"
        )
        moments_text += (
            f"   P(danger) = {gru_res['steps'][gru_tip]['p']:.3f} >= {gru_res['threshold']:.3f}\n\n"
        )

    # Lead time
    if xgb_tip is not None and realised_at is not None:
        lead_events = realised_at - xgb_tip
        moments_text += f"[TIME] XGBoost lead time: {lead_events} events before outcome\n\n"
    if gru_tip is not None and realised_at is not None:
        lead_events = realised_at - gru_tip
        moments_text += f"[TIME] GRU lead time: {lead_events} events before outcome\n\n"

    # Outcome
    if realised_at is not None:
        moments_text += f"[OUTCOME] Outcome realised: Event {realised_at+1} ({xgb_res['steps'][realised_at]['event']})\n"
        moments_text += f"   Final XGBoost P = {xgb_res['steps'][realised_at-1]['p']:.3f}\n"
        moments_text += f"   Final GRU P = {gru_res['steps'][realised_at-1]['p']:.3f}\n\n"

    # Model disagreement
    if xgb_tip is not None and gru_tip is not None:
        diff = abs(xgb_tip - gru_tip)
        if diff > 0:
            moments_text += f"[DIFF] Model disagreement: {diff} events difference in first alert\n"
            if xgb_tip < gru_tip:
                moments_text += "   XGBoost alerts earlier (more sensitive)\n"
            else:
                moments_text += "   GRU alerts earlier (more sensitive)\n"

    ax_moments.text(
        0.02,
        0.98,
        moments_text,
        transform=ax_moments.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#DEE2E6"),
    )

    # ============================================================
    # PANEL 5: What was observable (bottom-middle)
    # ============================================================
    ax_observable = fig.add_subplot(gs[2, 1])
    ax_observable.axis("off")

    observable_text = "OBSERVABLE AT EACH STEP\n" + "=" * 30 + "\n\n"
    observable_text += "Features available to models at step t:\n\n"
    observable_text += "- Ball location (x, y) in possession frame\n"
    observable_text += "- Pass end location (if pass completed)\n"
    observable_text += "- Event type (Pass, Carry, Receipt, etc.)\n"
    observable_text += "- Under pressure (binary)\n"
    observable_text += "- Elapsed time since possession start\n"
    observable_text += "- Cumulative counts: passes, carries, pressures\n"
    observable_text += "- Max x-reached so far (territory gained)\n"
    observable_text += "- Possession origin & period\n"
    observable_text += "- Starting minute & coordinates\n\n"
    observable_text += "NOT available (causal constraint):\n"
    observable_text += "- Future events\n"
    observable_text += "- Possession length\n"
    observable_text += "- Whether outcome will occur\n"
    observable_text += "- Opponent positions (except 360 frames)\n"
    observable_text += "- Continuous tracking data\n"

    ax_observable.text(
        0.02,
        0.98,
        observable_text,
        transform=ax_observable.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#DEE2E6"),
    )

    # ============================================================
    # PANEL 6: Model comparison summary (bottom-right)
    # ============================================================
    ax_summary = fig.add_subplot(gs[2, 2])
    ax_summary.axis("off")

    # Calculate summary stats
    xgb_probs = [s["p"] for s in xgb_res["steps"] if s["p"] is not None]
    gru_probs = [s["p"] for s in gru_res["steps"] if s["p"] is not None]

    summary_text = "MODEL COMPARISON\n" + "=" * 30 + "\n\n"
    summary_text += f"Events scored: {len(xgb_probs)}\n\n"
    summary_text += "XGBoost:\n"
    summary_text += f"  Threshold: {xgb_res['threshold']:.3f}\n"
    summary_text += f"  Mean P: {np.mean(xgb_probs):.3f}\n"
    summary_text += f"  Max P: {np.max(xgb_probs):.3f}\n"
    summary_text += f"  Final P: {xgb_probs[-1]:.3f}\n"
    summary_text += f"  Alerts: {sum(1 for s in xgb_res['steps'] if s['alert'])}\n\n"
    summary_text += "GRU:\n"
    summary_text += f"  Threshold: {gru_res['threshold']:.3f}\n"
    summary_text += f"  Mean P: {np.mean(gru_probs):.3f}\n"
    summary_text += f"  Max P: {np.max(gru_probs):.3f}\n"
    summary_text += f"  Final P: {gru_probs[-1]:.3f}\n"
    summary_text += f"  Alerts: {sum(1 for s in gru_res['steps'] if s['alert'])}\n\n"

    # Correlation
    if len(xgb_probs) == len(gru_probs):
        corr = np.corrcoef(xgb_probs, gru_probs)[0, 1]
        summary_text += f"Correlation: {corr:.3f}\n"

    # Disagreement at outcome
    if realised_at is not None and realised_at > 0:
        summary_text += f"\nAt outcome (event {realised_at}):\n"
        summary_text += f"  XGB: {xgb_probs[realised_at-1]:.3f}\n"
        summary_text += f"  GRU: {gru_probs[realised_at-1]:.3f}\n"
        summary_text += f"  Diff: {abs(xgb_probs[realised_at-1] - gru_probs[realised_at-1]):.3f}\n"

    ax_summary.text(
        0.02,
        0.98,
        summary_text,
        transform=ax_summary.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#DEE2E6"),
    )

    plt.tight_layout()
    return fig


def create_walkthrough_json(demo, arr, xgb_res, gru_res):
    """Create detailed JSON output for the walkthrough."""
    steps = []
    for t in range(len(arr.type_id)):
        xgb_step = xgb_res["steps"][t]
        gru_step = gru_res["steps"][t]

        step_data = {
            "event_index": t + 1,
            "event_type": type_label(int(arr.type_id[t])),
            "team": "possessing" if arr.own[t] else "opponent",
            "coordinates": {
                "x": float(arr.x[t]) if np.isfinite(arr.x[t]) else None,
                "y": float(arr.y[t]) if np.isfinite(arr.y[t]) else None,
                "end_x": float(arr.end_x[t]) if np.isfinite(arr.end_x[t]) else None,
                "end_y": float(arr.end_y[t]) if np.isfinite(arr.end_y[t]) else None,
            },
            "elapsed_seconds": float(arr.abs_s[t] - arr.abs_s[0]),
            "under_pressure": bool(arr.under_pressure[t]),
            "xgboost": {
                "p_danger": xgb_step["p"],
                "alert": xgb_step["alert"],
                "tipping": xgb_step["tipping"],
                "realised": xgb_step["realised"],
            },
            "gru": {
                "p_danger": gru_step["p"],
                "alert": gru_step["alert"],
                "tipping": gru_step["tipping"],
                "realised": gru_step["realised"],
            },
        }
        steps.append(step_data)

    return {
        "source": demo["source"],
        "match_id": demo["match_id"],
        "possession_id": demo["possession_id"],
        "team": demo["team"],
        "origin": demo["origin"],
        "period": demo["period"],
        "minute": demo["minute"],
        "total_events": len(arr.type_id),
        "xgb_threshold": xgb_res["threshold"],
        "gru_threshold": gru_res["threshold"],
        "realised_at": xgb_res.get("realised_at"),
        "steps": steps,
        "summary": {
            "xgb_first_alert": next((s["t"] + 1 for s in xgb_res["steps"] if s["tipping"]), None),
            "gru_first_alert": next((s["t"] + 1 for s in gru_res["steps"] if s["tipping"]), None),
            "xgb_alerts": sum(1 for s in xgb_res["steps"] if s["alert"]),
            "gru_alerts": sum(1 for s in gru_res["steps"] if s["alert"]),
            "lead_time_events_xgb": (
                (
                    xgb_res.get("realised_at")
                    - next((s["t"] for s in xgb_res["steps"] if s["tipping"]), 0)
                )
                if xgb_res.get("realised_at") and any(s["tipping"] for s in xgb_res["steps"])
                else None
            ),
            "lead_time_events_gru": (
                (
                    xgb_res.get("realised_at")
                    - next((s["t"] for s in gru_res["steps"] if s["tipping"]), 0)
                )
                if xgb_res.get("realised_at") and any(s["tipping"] for s in gru_res["steps"])
                else None
            ),
        },
    }


def create_walkthrough_markdown(demo, arr, xgb_res, gru_res, walkthrough_json):
    """Create markdown narrative for the walkthrough."""
    md = f"""# Tactical Walkthrough: France Possession 121 (Match 3857266)

**Source:** {demo['source']}  
**Match:** {demo['match_id']} | **Possession:** {demo['possession_id']} | **Team:** {demo['team']}  
**Period:** {demo['period']} | **Minute:** {demo['minute']:.0f} | **Origin:** {demo['origin']}  
**Total Events:** {len(arr.type_id)} | **Outcome:** Shot/Box Entry Realised

---

## Event-by-Event Probability Evolution

| Event | Type | Team | XGBoost P | GRU P | Flags |
|-------|------|------|-----------|-------|-------|
"""

    for step in walkthrough_json["steps"]:
        flags = []
        if step["xgboost"]["tipping"]:
            flags.append("[XGB TIP]")
        elif step["xgboost"]["alert"]:
            flags.append("[XGB]")
        if step["gru"]["tipping"]:
            flags.append("[GRU TIP]")
        elif step["gru"]["alert"]:
            flags.append("[GRU]")
        if step["xgboost"]["realised"]:
            flags.append("[OUTCOME]")

        xgb_p = (
            f"{step['xgboost']['p_danger']:.3f}"
            if step["xgboost"]["p_danger"] is not None
            else "--"
        )
        gru_p = f"{step['gru']['p_danger']:.3f}" if step["gru"]["p_danger"] is not None else "--"

        md += f"| {step['event_index']} | {step['event_type']} | {step['team'][:3]} | {xgb_p} | {gru_p} | {' '.join(flags)} |\n"

    md += """

---

## Key Moments

### First Alerts
"""

    xgb_tip = walkthrough_json["summary"]["xgb_first_alert"]
    gru_tip = walkthrough_json["summary"]["gru_first_alert"]
    realised = walkthrough_json["realised_at"]

    if xgb_tip:
        md += f"- **XGBoost** first crosses threshold at **Event {xgb_tip}** (P = {xgb_res['steps'][xgb_tip-1]['p']:.3f} >= {xgb_res['threshold']:.3f})\n"
    if gru_tip:
        md += f"- **GRU** first crosses threshold at **Event {gru_tip}** (P = {gru_res['steps'][gru_tip-1]['p']:.3f} >= {gru_res['threshold']:.3f})\n"

    md += "\n### Lead Time Before Outcome\n"
    if xgb_tip and realised:
        lead = realised - xgb_tip + 1
        md += f"- **XGBoost:** {lead} events lead time\n"
    if gru_tip and realised:
        lead = realised - gru_tip + 1
        md += f"- **GRU:** {lead} events lead time\n"

    md += "\n### Outcome Realised\n"
    if realised is not None:
        md += f"- **Event {realised+1}** ({xgb_res['steps'][realised]['event']}) -- Shot or box entry by possessing team\n"
        md += f"- Final XGBoost P(danger) = {xgb_res['steps'][realised-1]['p']:.3f}\n"
        md += f"- Final GRU P(danger) = {gru_res['steps'][realised-1]['p']:.3f}\n"

    md += f"""

---

## Model Disagreement Analysis

- **XGBoost threshold:** {xgb_res['threshold']:.3f} (validation prevalence quantile)
- **GRU threshold:** {gru_res['threshold']:.3f} (validation prevalence quantile)
- **XGBoost alerts:** {walkthrough_json['summary']['xgb_alerts']} events
- **GRU alerts:** {walkthrough_json['summary']['gru_alerts']} events

"""

    if xgb_tip and gru_tip:
        diff = abs(xgb_tip - gru_tip)
        md += f"- **First alert difference:** {diff} events\n"
        if xgb_tip < gru_tip:
            md += "- XGBoost alerts earlier -- more sensitive to early territorial gain\n"
        else:
            md += "- GRU alerts earlier -- more sensitive to sequence patterns\n"

    # Correlation
    xgb_probs = [s["p"] for s in xgb_res["steps"] if s["p"] is not None]
    gru_probs = [s["p"] for s in gru_res["steps"] if s["p"] is not None]
    if len(xgb_probs) == len(gru_probs):
        corr = np.corrcoef(xgb_probs, gru_probs)[0, 1]
        md += f"- **Trajectory correlation:** {corr:.3f}\n"

    md += """

---

## What the Models Observe (Causal Features Only)

At each event *t*, models see only events 0..t:

1. **Ball position** (x, y) in possession frame -- opponent events flipped
2. **Pass end position** -- only after pass completes
3. **Event type** -- Pass, Carry, Receipt, Pressure, Duel, etc.
4. **Pressure indicator** -- binary, from StatsBomb
5. **Elapsed time** -- seconds since possession start
6. **Cumulative counts** -- passes, carries, pressures so far
7. **Max x-reached** -- furthest territorial advance
8. **Possession context** -- origin, period, starting minute, start coordinates

**Not observed (by design):**
- Future events
- Total possession length
- Whether danger will occur
- Opponent player positions (except partial 360 freeze frames)
- Continuous tracking data

---

## Interpretation for Analysts

This possession shows:
- **Early XGBoost alert** (Event 1): XGBoost responds immediately to the first pass moving toward goal
- **Delayed GRU alert** (Event 8): GRU requires sequence context before raising probability
- **Both models agree** on rising danger as possession progresses into final third
- **Outcome realised** at Event 19 (Dribble into box) -- both models show elevated probability
- **Lead time:** ~10-18 events before outcome depending on model

**Caveats:**
- Probabilities are not calibrated to held-out base rate (models under-predict on test)
- Use for ranking moments within possession, not as absolute probabilities
- No causal claim: correlation with ball position dominates
- 360 freeze frames cover only 52% of passes -- not continuous tracking

---

*Generated by Frame2Threat tactical walkthrough script*
"""
    return md


def main():
    parser = argparse.ArgumentParser(
        description="Generate tactical walkthrough for France possession"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "figures" / "tactical"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading possession data...")
    demo, arr, xgb_res, gru_res = load_possession_data(ROOT)

    print("Creating walkthrough figure...")
    fig = create_walkthrough_figure(demo, arr, xgb_res, gru_res, ROOT)
    fig_path = output_dir / "tactical_walkthrough.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {fig_path}")

    print("Creating JSON output...")
    walkthrough_json = create_walkthrough_json(demo, arr, xgb_res, gru_res)
    json_path = output_dir / "tactical_walkthrough.json"
    with open(json_path, "w") as f:
        json.dump(walkthrough_json, f, indent=2)
    print(f"Saved JSON to {json_path}")

    print("Creating Markdown narrative...")
    md = create_walkthrough_markdown(demo, arr, xgb_res, gru_res, walkthrough_json)
    md_path = output_dir / "tactical_walkthrough.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved Markdown to {md_path}")

    print("\nDone! Tactical walkthrough generated.")


if __name__ == "__main__":
    main()
