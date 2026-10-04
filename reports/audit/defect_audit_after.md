# Defect audit (after fixes)

Possessions: 17,065; passes: 50,688.

## 1. Vocabulary: one encoder/decoder (`src/data/event_types.py`)

| type_id | event type (shared decoder) | count |
|---:|---|---:|
| 0 | Unknown | 18,559 |
| 1 | Pass | 108,617 |
| 2 | Carry | 86,664 |
| 3 | Ball Receipt | 102,912 |
| 4 | Dribble | 2,999 |
| 5 | Shot | 2,324 |
| 6 | Pressure | 27,385 |
| 7 | Duel | 6,151 |
| 8 | Clearance | 3,721 |
| 9 | Interception | 2,162 |
| 10 | Block | 3,662 |
| 11 | Foul Committed | 2,728 |
| 12 | Foul Won | 2,602 |
| 13 | Goalkeeper | 2,617 |

id 3 is Ball Receipt and id 4 is Dribble in every module (tests/test_event_types.py).

## 2-3. Event-time (in-play) formulation

- Eligible snapshot rows: 305,399 from 16,881 possessions; prevalence 0.325
- Share of snapshots whose prefix 0..t already contains a shot or box entry: **0.0%** (0% by construction)
- ROC AUC of events-so-far alone (the only length information a snapshot has), all rows: **0.557**; one random snapshot per possession: **0.544**
- The total possession length n is not available to any feature or model (tests/test_event_time.py appends future events and checks the row for t is unchanged).

## 4. Pass labels: original vs `progression_beyond_pass_k`

| label | prevalence | P(label given end_x >= 80) | P(label given end_x < 80) | share of positives with end_x >= 80 | AUC of end_x alone |
|---|---:|---:|---:|---:|---:|
| dangerous_progression_k | 0.433 | 0.943 | 0.248 | 0.580 | 0.786 |
| progression_beyond_pass_k | 0.216 | 0.174 | 0.231 | 0.214 | 0.447 |

## Possession labels changed by the coordinate-frame fix

| label | old rate | corrected rate | true -> false | false -> true |
|---|---:|---:|---:|---:|
| poss_dangerous | 0.3275 | 0.2827 | 765 | 0 |
| poss_entered_box | 0.2998 | 0.2551 | 762 | 0 |
| poss_has_shot | 0.1236 | 0.1219 | 30 | 0 |
| poss_entered_final_third | 0.7544 | 0.6149 | 2381 | 0 |

## For reference: the superseded fraction-based measurements on the corrected labels

- ROC AUC of total length n for `poss_dangerous`: **0.677**
- Median length: dangerous = 24, safe = 12
- AUC of prefix length ceil(n*0.25): 0.676
- AUC of prefix length ceil(n*0.5): 0.676
- AUC of prefix length ceil(n*0.75): 0.676


Two box definitions: *event location* (what `poss_dangerous` uses; safe = 0% by construction) and *+ pass/carry end location* (the initial audit's figures; the label ignores ends, so some 'safe' possessions already put the ball in the box via a pass/carry end).

| definition | f | dangerous already realised | safe already realised |
|---|---:|---:|---:|
| event location | 0.25 | 18.4% | 0.0% |
| event location | 0.5 | 32.8% | 0.0% |
| event location | 0.75 | 53.9% | 0.0% |
| + end location | 0.25 | 25.9% | 2.3% |
| + end location | 0.5 | 38.4% | 2.7% |
| + end location | 0.75 | 59.9% | 3.5% |
