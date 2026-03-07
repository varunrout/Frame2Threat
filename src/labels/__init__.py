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
