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
