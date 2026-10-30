# Ranking-change audit: superseded v2 leaderboard vs corrected cross-fitted leaderboard

**Limitation (BLOCKED for a full comparison).**  The v2 player table was never committed as data, only as a figure
(`reports/figures/superseded_attribution/v2_player_attribution_leaderboard.png`, top 11 players).  A full
players-changed / ranks-moved count therefore cannot be computed.  What can be checked is the 11 names the old figure showed,
against the current table (623 players with >= 100 possessing-team events; `player_leaderboard.csv`).

| v2 top-11 player (in-sample, pre-fix) | Current status |
|---|---|
| Claudio Bravo | not qualifying (< 100 possessing-team events) |
| Jeremias Ledesma | not qualifying |
| Mohammed Al Owais | rank 396 / 623 |
| Roberto Soldado | rank 5 |
| Kasper Dolberg | rank 78 |
| Andreas Cornelius | rank 24 |
| Daizen Maeda | not qualifying |
| Sergio Guardiola | rank 556 |
| Jordi Masip | not qualifying |
| Thibaut Courtois | rank 237 |
| Youssef En-Nesyri | rank 53 |

Result: **1 of 11** old top-11 players is in the current top 10 (Soldado); 4 of 11 do not meet the minimum-sample rule at all
(goalkeepers and low-minute players that the old 20-touch rule admitted).  Goalkeepers scored high before because
defending-team events (goalkeeper actions follow shots) were credited to players; that credit is removed.

Reliability of the *current* ranking (see `ATTRIBUTION_REPORT.md` section 7): split-half Spearman 0.62, cross-seed
Spearman 0.89.  It is a shortlist tool, not a ranking, and not causal.
