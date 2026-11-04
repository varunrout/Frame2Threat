# Audit trail

Frame2Threat was audited in October and November 2026. The audits found real defects; the fixes, the
before/after evidence and the results that had to be **withdrawn** are kept here rather than deleted.
Start with the first two documents; the rest are evidence.

| Read | What it is |
|---|---|
| [REPAIR_REPORT.md](REPAIR_REPORT.md) | The November repair: what was changed, retrained and re-verified. Current. |
| [final_audit_2026-10.md](final_audit_2026-10.md) | The 30 October audit: what was verified by execution and what was not. |

## Defects found and where the evidence is

| Defect | Effect | Before / after evidence |
|---|---|---|
| Event-type vocabulary mismatch between encoder and decoder | Tipping-point and attribution analyses read the wrong event names | [defect_audit_before.md](defect_audit_before.md), [defect_audit_after.md](defect_audit_after.md), [exp020_vocab_fix.md](exp020_vocab_fix.md) |
| Fraction-based early forecasting leaked possession length | The reported AUC of 0.82 to 0.85 was inflated; **withdrawn** | defect audit above; replaced by the event-time formulation |
| Pass label `dangerous_progression_k` was self-determining (it depended on the pass's own receipt) | Scores on that label overstated predictability; replaced by `progression_beyond_pass_k` | [defect_audit_before.md](defect_audit_before.md) section 4; `reports/pass_label/` |
| Same-moment leak: a pass ending in the box counted as known before it happened | 1.30% of snapshots had a label rate of 80.4% (31.9% elsewhere); the earlier event-time 0.668/0.670 were **superseded** | [same_moment_before.md](same_moment_before.md), [same_moment_after.md](same_moment_after.md) |
| Shot xG and outcome columns were not parsed | xG sum was 0.0 and no goals were recorded; now 295.1 and 305 | [shot_columns_after.md](shot_columns_after.md) |
| Opponent coordinates were not expressed in the possessing team's frame | Possession-level labels and models rebuilt on the frame-corrected labels | [exp023_v2_corrected_labels.md](exp023_v2_corrected_labels.md) |

The audit scripts that reproduce these numbers are listed in [scripts/README.md](../../scripts/README.md) (section 4).

## Withdrawn or superseded (do not reuse)

- Fraction-prefix early-forecasting results, including AUC 0.82 to 0.85.
- The first event-time results (0.668 / 0.670), before the same-moment fix:
  [reports/event_time/superseded_pre_same_moment_fix](../event_time/superseded_pre_same_moment_fix/).
- Pass-model results before the possession-length fix:
  [reports/pass_label/superseded_pre_possession_length_fix](../pass_label/superseded_pre_possession_length_fix/).
- The in-sample v2 attribution leaderboard; the cross-fitted replacement is in [reports/attribution/](../attribution/),
  with a partial comparison in [ranking_change_audit.md](../attribution/ranking_change_audit.md).
- Near-parity of the GNN and XGBoost (0.841 vs 0.845), measured on the old label.

Every number elsewhere in the repository should be read against these notes. Earlier documents that mention
withdrawn numbers carry a banner saying so.
