# Shot columns audit (after the fix)

## Before / after in `possession_sequences.parquet`

- Before: poss_xg_generated sum = 0.0; poss_has_goal true = 0; max outcome tier = 3
- After: poss_xg_generated sum = 295.1; poss_has_goal true = 305; max outcome tier = 4

Columns that changed (everything else is identical to the pre-fix table):

- poss_xg_generated, poss_has_goal, poss_outcome_tier

## Sanity checks

- Shots: 2,333; goals: 306 (26 in penalty shootouts); matches: 99
- Goals per match: **3.09** including shootouts, **2.83** excluding them
  - competition 11 season 90: 35 matches, 3.17 goals per match (excl. shootouts), xG 3.07 per match
  - competition 43 season 106: 64 matches, 2.64 goals per match (excl. shootouts), xG 2.94 per match
- Total xG 295.4 vs 306 goals (ratio 0.97); possession table xG sum 295.1
- Share of possessions with a goal: 1.79% (305 of 17,065); share with a shot: 12.19%
- Outcome tier distribution: tier 0: 6562, tier 1: 5674, tier 2: 2743, tier 3: 1781, tier 4: 305
- Goals not attached to a possession row (possessions under 2 events dropped): 1
