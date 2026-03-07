"""
src.labels – Binary and multi-label target construction for pass instances.

Labelling strategies implemented:
- line_break          : did the pass break a defensive line?
- dangerous_progression: did possession move into a high-value zone (VAEP/EPV)?
- final_third_entry   : did the ball enter the final third?
- box_entry           : did the ball enter the penalty area?
- shot_within         : did a shot occur within k subsequent events?
- threat_gain         : continuous EPV delta label.
"""

from src.labels.dangerous_progression import compute_downstream_labels
from src.labels.downstream_outcomes import compute_threat_gain
from src.labels.line_break import compute_line_break_labels
from src.labels.validation_sampling import (
    label_by_pass_type,
    label_by_zone,
    label_prevalence_table,
    label_sanity_checks,
    sample_negatives,
    sample_positives,
)

__all__ = [
    "compute_line_break_labels",
    "compute_downstream_labels",
    "compute_threat_gain",
    "label_prevalence_table",
    "label_by_zone",
    "label_by_pass_type",
    "label_sanity_checks",
    "sample_positives",
    "sample_negatives",
]
