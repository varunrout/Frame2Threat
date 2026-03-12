# Data Dictionary — Frame2Threat

**Project:** Frame2Threat — Predicting and Explaining Dangerous Progression from StatsBomb Event Data and 360 Freeze Frames  
**Data source:** StatsBomb Open Data (https://github.com/statsbomb/open-data)  
**Pitch coordinate system:** x ∈ [0, 120] (own-goal end → opponent-goal end), y ∈ [0, 80] (left touchline → right touchline). All coordinates are in metres.

---

## Table: `matches`

Built by `src/data/ingest.py :: get_matches()`.  One row per match.

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | int | StatsBomb unique match identifier |
| `competition_id` | int | Competition identifier |
| `competition_name` | str | Human-readable competition name |
| `season_id` | int | Season identifier |
| `season_name` | str | Human-readable season name |
| `match_date` | date | Date of the match |
| `home_team` | str | Home team name |
| `away_team` | str | Away team name |
| `home_score` | int | Full-time home goals |
| `away_score` | int | Full-time away goals |
| `has_360` | bool | Whether 360 freeze-frame data is available for this match |

---

## Table: `events`

Built by `src/data/parse_events.py :: parse_events()`.  One row per event.

| Column | Type | Description |
|--------|------|-------------|
| `event_uuid` | str | StatsBomb unique event identifier (UUID) |
| `match_id` | int | Foreign key → `matches.match_id` |
| `index` | int | Event order within the match |
| `period` | int | Match period (1=first half, 2=second half, 3/4=extra time) |
| `timestamp` | str | Absolute timestamp within the period (HH:MM:SS.mmm) |
| `minute` | int | Minute of the match |
| `second` | int | Second within the minute |
| `type_name` | str | Event type (e.g. "Pass", "Carry", "Shot", "Pressure") |
| `team_name` | str | Team performing the event |
| `player_name` | str | Player performing the event |
| `possession_id` | int | Possession sequence identifier (resets each time possession changes) |
| `play_pattern_name` | str | How possession was won (e.g. "Regular Play", "From Corner") |
| `location_x` | float | Ball x-coordinate at event start |
| `location_y` | float | Ball y-coordinate at event start |
| `pass_recipient_name` | str | (Pass only) Intended recipient player name |
| `pass_length` | float | (Pass only) Euclidean pass distance in metres |
| `pass_angle` | float | (Pass only) Angle of the pass in radians (0=right, π/2=up) |
| `pass_end_x` | float | (Pass only) Ball x-coordinate at pass end |
| `pass_end_y` | float | (Pass only) Ball y-coordinate at pass end |
| `pass_body_part` | str | (Pass only) Body part used (e.g. "Right Foot", "Head") |
| `pass_height` | str | (Pass only) Pass height category ("Ground Pass", "Low Pass", "High Pass") |
| `pass_type` | str | (Pass only) Set-piece type if applicable ("Corner", "Free Kick", etc.) |
| `pass_outcome_name` | str | (Pass only) Outcome ("Incomplete", "Out" etc.; NULL = completed) |
| `under_pressure` | bool | Whether player was under pressure when performing the event |
| `pass_switch` | bool | (Pass only) Whether this is a cross-field switch |
| `pass_cross` | bool | (Pass only) Whether this is a cross into the box |
| `pass_through_ball` | bool | (Pass only) Whether this is a through ball |

---

## Table: `lineups`

Built by `src/data/parse_lineups.py :: parse_lineups()`.  One row per player per match.

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | int | Foreign key → `matches.match_id` |
| `team_id` | int | Team identifier |
| `team_name` | str | Team name |
| `player_id` | int | Player identifier |
| `player_name` | str | Player name |
| `jersey_number` | int | Jersey number |
| `position_name` | str | Starting position (e.g. "Center Back", "Left Wing") |

---

## Table: `frames_360`

Built by `src/data/parse_360.py :: parse_360_frames()`.  One row per visible player per event.  Only available for matches with 360 data.

| Column | Type | Description |
|--------|------|-------------|
| `event_uuid` | str | Foreign key → `events.event_uuid` |
| `player_id` | int | Player identifier (may be NULL for partially tracked players) |
| `player_name` | str | Player name |
| `teammate` | bool | True if this player is on the same team as the event performer |
| `actor` | bool | True if this is the player performing the event |
| `keeper` | bool | True if this is a goalkeeper |
| `x` | float | Player x-coordinate (StatsBomb pitch, metres) |
| `y` | float | Player y-coordinate (StatsBomb pitch, metres) |

**Limitations:**  
- Not all players on the pitch are visible in every frame — only those captured by the broadcast cameras.  
- Coordinates are positional snapshots at the moment of the event, NOT continuous tracking trajectories.  
- Velocities and movement directions are NOT available.

---

## Table: `pass_instances` (modelling table)

Built by `src/data/join_pass_frames.py :: build_pass_instances()`.  One row per open-play pass event.  This is the primary modelling table.

### Identity columns

| Column | Type | Description |
|--------|------|-------------|
| `event_uuid` | str | Unique event identifier (primary key) |
| `match_id` | int | Match identifier |
| `competition_id` | int | Competition identifier |
| `season_id` | int | Season identifier |
| `possession_id` | int | Possession sequence identifier |
| `period` | int | Match period |
| `minute` | int | Match minute |
| `second` | int | Second within minute |

### Team / player columns

| Column | Type | Description |
|--------|------|-------------|
| `team_name` | str | Team performing the pass |
| `player_name` | str | Passer name |
| `pass_recipient_name` | str | Intended recipient name (may be NULL) |

### Spatial columns

| Column | Type | Description |
|--------|------|-------------|
| `start_x` | float | Pass start x-coordinate |
| `start_y` | float | Pass start y-coordinate |
| `end_x` | float | Pass end x-coordinate |
| `end_y` | float | Pass end y-coordinate |

### Pass attribute columns

| Column | Type | Description |
|--------|------|-------------|
| `pass_length` | float | Euclidean distance (metres) |
| `pass_angle` | float | Angle in radians |
| `pass_body_part` | str | Body part category |
| `pass_height` | str | Height category |
| `pass_type` | str | Set-piece type (NULL = open play) |
| `pass_outcome_name` | str | Outcome (NULL = completed) |
| `under_pressure` | bool | Under pressure flag |
| `pass_switch` | bool | Cross-field switch |
| `pass_cross` | bool | Cross into box |
| `pass_through_ball` | bool | Through ball |
| `play_pattern_name` | str | How possession was won |

### 360 linkage columns

| Column | Type | Description |
|--------|------|-------------|
| `has_360` | bool | Whether 360 data is available for this event |
| `n_visible_players` | int | Total visible players in freeze frame (NULL if no 360) |
| `n_visible_teammates` | int | Visible teammates (NULL if no 360) |
| `n_visible_opponents` | int | Visible opponents (NULL if no 360) |

### Label columns (populated by `src/labels/`)

| Column | Type | Description |
|--------|------|-------------|
| `strict_line_break` | bool/NaN | Pass crosses ≥2 defenders between start_x and end_x AND end_x > start_x + 5m (requires 360) |
| `loose_line_break` | bool/NaN | Pass crosses ≥1 defender between start_x and end_x AND end_x > start_x + 5m (requires 360) |
| `dangerous_progression_k` | bool | Within next 5 same-possession events: final-third entry OR box entry OR shot |
| `final_third_entry_k` | bool | Within next 5 same-possession events: ball reaches x ≥ 80 |
| `box_entry_k` | bool | Within next 5 same-possession events: ball reaches x ≥ 102 AND 18 ≤ y ≤ 62 |
| `shot_within_k` | bool | Within next 5 same-possession events: a shot event occurs |
| `threat_gain` | float | Zone-value delta: value(end_zone) − value(start_zone); range approximately [−1, 1] |

---

## Table: `possession_sequences` (v2 modelling table)

Built by `src/data/parse_possessions.py :: build_possession_sequences()`.  One row per possession (minimum 2 events).  Labels attached automatically by `src/labels/possession_labels.py :: attach_possession_labels()`.

![Possession data structure](figures/poss_eda_structure.png)
*Figure: Possession-level data structure overview — from flat event stream to one-row-per-possession.*

![Possession label prevalence](figures/poss_eda_label_prevalence.png)
*Figure: Prevalence rates for all 13 possession-level labels.*

### Identity columns

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | int32 | Foreign key → `matches.match_id` |
| `possession_id` | int32 | Possession sequence id (resets when possession changes) |
| `team_name` | str | Team in possession |
| `period` | int32 | Match period (1–4) |
| `origin_type` | str | How the possession was won (`Regular Play`, `From Corner`, etc.) |

### Spatial columns

| Column | Type | Description |
|--------|------|-------------|
| `start_x` | float32 | x-coordinate of the first event in the possession |
| `start_y` | float32 | y-coordinate of the first event |
| `end_x` | float32 | x-coordinate of the last event |
| `end_y` | float32 | y-coordinate of the last event |
| `max_x_reached` | float32 | Maximum x-coordinate of any event in the possession |
| `territory_gained` | float32 | `max_x_reached − start_x` (metres; NaN if either is missing) |

### Temporal / counting columns

| Column | Type | Description |
|--------|------|-------------|
| `n_events` | int32 | Total events in the possession |
| `n_passes` | int32 | Number of pass events |
| `n_carries` | int32 | Number of carry events |
| `n_pressures_faced` | int32 | Events where the acting player was under pressure |
| `duration_seconds` | int32 | `(last_minute×60 + last_second) − (first_minute×60 + first_second)` |

### Meta columns

| Column | Type | Description |
|--------|------|-------------|
| `mean_pass_length` | float32 | Mean Euclidean pass length (metres) across all passes in the possession; NaN if no passes |
| `has_pressure` | bool | `n_pressures_faced > 0` |

### Sequence columns

| Column | Type | Description |
|--------|------|-------------|
| `event_sequence` | list[dict] (JSON-serialised in parquet) | Per-event feature vectors for GRU/sequence model input.  Each dict has 8 keys: `type_id` (int, TYPE_VOCAB-encoded), `loc_x_norm` (x/120), `loc_y_norm` (y/80), `end_x_norm`, `end_y_norm`, `under_pressure` (0/1), `pass_length_norm` (length/60), `minute_norm` (minute/90). |
| `player_sequence` | list[str] (JSON-serialised in parquet) | Player name for each event step, aligned 1-to-1 with `event_sequence`.  `"Unknown"` when the event has no player actor. |

**TYPE_VOCAB** (event type → integer encoding for `type_id`):

| Type | ID |
|------|----|
| pass | 1 |
| carry | 2 |
| ball receipt / ball receipt* | 3 |
| dribble | 4 |
| shot | 5 |
| pressure | 6 |
| duel | 7 |
| clearance | 8 |
| interception | 9 |
| block | 10 |
| foul committed | 11 |
| foul won | 12 |
| goalkeeper / goal keeper | 13 |
| (other) | 0 |

### Label columns — Group A (core outcome)

| Column | Type | Description |
|--------|------|-------------|
| `poss_has_shot` | bool | At least one shot event exists in the possession's event sequence |
| `poss_entered_final_third` | bool | `max_x_reached ≥ 80` |
| `poss_entered_box` | bool | At least one event has `x ≥ 102` AND `18 ≤ y ≤ 62` (penalty-box entry) |
| `poss_dangerous` | bool | `poss_has_shot OR poss_entered_box` — **primary v2 prediction target** |

### Label columns — Group B (richer outcome)

| Column | Type | Description |
|--------|------|-------------|
| `poss_xg_generated` | float | Sum of shot xG within the possession (NaN if events_df was not provided at build time) |
| `poss_has_goal` | bool | A goal was scored during the possession |
| `poss_outcome_tier` | int8 | 0 = nothing, 1 = final-third entry only, 2 = box entry, 3 = shot, 4 = goal |

### Label columns — Group C (tempo / structural)

| Column | Type | Description |
|--------|------|-------------|
| `poss_tempo` | float32 | `n_events / max(duration_seconds, 1)` — events per second |
| `poss_verticality` | float32 | `territory_gained / (n_events × mean_pass_length + ε)` — how directly the possession advanced |
| `poss_recycled` | bool | x fell ≥ 15 m during the possession and then recovered ≥ 15 m (ball recycled through the back) |
| `poss_phase` | str | Possession phase classification: `counter`, `build_up`, `progression`, or `final_third` (based on tempo and start_x thresholds) |

### Label columns — Group D (defensive disruption)

| Column | Type | Description |
|--------|------|-------------|
| `poss_broke_pressure` | bool | The team survived ≥ 1 pressure event and continued for ≥ 3 more events afterwards |
| `poss_bypassed_lines` | bool | Reached `max_x ≥ 80` within the first 4 events AND `start_x ≤ 50` — rapid line bypass |

### Label columns — derived (used for possession-level XGBoost features)

| Column | Type | Description |
|--------|------|-------------|
| `poss_pressure_index` | float | `n_pressures_faced / n_events` — normalised defensive pressure faced |
| `poss_built_up` | bool | `poss_phase == "build_up"` — binary indicator for build-up possessions |

---

## Early-warning feature sets (v3)

Built by `src/features/early_features.py`.  These are *derived feature matrices* (not stored columns) used by v3 models and the CLI pipeline.

### Start-only features

Built by `build_start_features(poss_df)`.  ~19 features available at possession start (before any within-possession events are observed).

| Feature | Type | Description |
|---------|------|-------------|
| `start_x` | float | x-coordinate of the first event |
| `start_y` | float | y-coordinate of the first event |
| `start_x_norm` | float | `start_x / 120` |
| `start_y_norm` | float | `start_y / 80` |
| `started_final_third` | int | `start_x ≥ 80` |
| `started_own_half` | int | `start_x < 40` |
| `started_mid_third` | int | `40 ≤ start_x < 80` |
| `dist_to_box_start` | float | Euclidean distance from start to centre of penalty box |
| `start_zone` | int | 0=left flank, 1=central, 2=right flank |
| `period` | int | Match period (1–4) |
| `origin_*` | int | One-hot columns for `origin_type` (regular play, counter, goal kick, etc.) |

### Cumulative prefix features

Built by `build_cumulative_tabular_features(poss_df, frac)`.  Returns the same ~41 columns as `build_tabular_features()`, but counting/spatial/temporal aggregates are recomputed from only the first `frac` of each possession's event sequence.  Label-derived columns (`poss_tempo`, `poss_verticality`, `poss_recycled`, `poss_broke_pressure`, `poss_bypassed_lines`) are zeroed for `frac < 1.0` to prevent retrospective leakage.

---

## Model artefacts

| Artefact | Path | Version | Description |
|----------|------|---------|-------------|
| v1 XGBoost (event-only) | `models/xgboost_dp_event_only.joblib` | v1 | 27 event features, EXP-003 |
| v1 XGBoost (event+360) | `models/xgboost_dp_event_360.joblib` | v1 | 41 features (event+geometry), EXP-004 |
| v2 XGBoost (possession) | `models/xgboost_poss_dangerous.joblib` | v2 | 41 possession features, EXP-009 |
| v2 PossessionGRU | `models/gru_poss_dangerous.pt` | v2 | GRU checkpoint (arch+weights), EXP-010 |
| v3 XGBoost start-only | `models/xgboost_start_only.joblib` | v3 | ~19 start features, EXP-015 |
| v3 XGBoost cumulative @25% | `models/xgboost_cumulative_25pct.joblib` | v3 | Prefix-built tabular, EXP-017 |
| v3 XGBoost cumulative @50% | `models/xgboost_cumulative_50pct.joblib` | v3 | Prefix-built tabular, EXP-017 |
| v3 XGBoost cumulative @75% | `models/xgboost_cumulative_75pct.joblib` | v3 | Prefix-built tabular, EXP-017 |
| v1 results | `models/v1_results_summary.json` | v1 | All v1 metrics |
| v2+v3 results | `models/results_summary.json` | v2+v3 | All v2/v3 metrics + ensemble |

---

## Provenance

| Item | Detail |
|------|--------|
| Data source | StatsBomb Open Data GitHub repository |
| Library | `statsbombpy` ≥ 1.13 |
| Competitions covered | FIFA World Cup (id=43), Premier League (id=2), La Liga (id=11) — configurable in `configs/data.yaml` |
| 360 availability | Subset of matches only; see `data/interim/inventory.parquet` for per-match coverage |
| Pitch coordinate system | StatsBomb standard: 120×80 metres, origin at own-goal bottom-left |

---

## Known limitations

1. **360 coverage is partial.** Not all competitions or seasons have 360 data. Geometry-dependent labels and features are NaN for events without 360 context.
2. **Positional snapshots, not trajectories.** The 360 data captures player positions at the moment of the event only. No velocity, acceleration, or movement direction is available.
3. **Partial pitch visibility.** Not all 22 players are visible in every freeze frame. Pass option ranking is restricted to *visible* teammates only.
4. **Recipient visibility.** The nominal pass recipient may not be visible in the freeze frame.
5. **Open-play filter.** The `pass_instances` table contains only open-play passes. Set-piece deliveries are excluded from modelling.
6. **Possession boundary inherited from StatsBomb.** The `possession_id` definition is StatsBomb's own segmentation — it resets on turnovers and certain dead-ball events. Short possessions (< 2 events) are dropped during `build_possession_sequences()`.
7. **`player_sequence` relies on `player_name` in events.** Some event types (e.g. ball receipt, goalkeeper events) may have missing or generic player names, returned as `"Unknown"`.
8. **`poss_xg_generated` / `poss_has_goal` require full events table.** If `events_df` is not passed to `attach_possession_labels()`, xG is NaN and goals default to False.
9. **Sequence length varies.** Possessions range from 2 events to 50+. GRU input is padded/truncated to a fixed length at training time; very long possessions lose tail information.
