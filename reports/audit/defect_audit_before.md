# Defect audit (before fixes)

Possessions: 17,065; passes: 50,688.

## 1. Vocabulary counts (`type_id` over all `event_sequence` entries)

| type_id | count |
|---:|---:|
| 0 | 18,559 |
| 1 | 108,617 |
| 2 | 86,664 |
| 3 | 102,912 |
| 4 | 2,999 |
| 5 | 2,324 |
| 6 | 27,385 |
| 7 | 6,151 |
| 8 | 3,721 |
| 9 | 2,162 |
| 10 | 3,662 |
| 11 | 2,728 |
| 12 | 2,602 |
| 13 | 2,617 |

Total events: 373,103. Id 3 (ball receipt) = 102,912; id 4 (dribble) = 2,999.

## 2. Possession-length leak

- ROC AUC of total length n for `poss_dangerous`: **0.683**
- Median length: dangerous = 23, safe = 12
- AUC of prefix length ceil(n*0.25): 0.683
- AUC of prefix length ceil(n*0.5): 0.682
- AUC of prefix length ceil(n*0.75): 0.682

## 3. Already-realised outcome in the prefix

Two box definitions: *event location* (what `poss_dangerous` uses; safe = 0% by construction) and *+ pass/carry end location* (the initial audit's figures; the label ignores ends, so some 'safe' possessions already put the ball in the box via a pass/carry end).

| definition | f | dangerous already realised | safe already realised |
|---|---:|---:|---:|
| event location | 0.25 | 20.8% | 0.0% |
| event location | 0.5 | 36.9% | 0.0% |
| event location | 0.75 | 58.7% | 0.0% |
| + end location | 0.25 | 27.4% | 2.6% |
| + end location | 0.5 | 42.0% | 3.0% |
| + end location | 0.75 | 64.0% | 3.9% |

## 4. Pass-label self-determination (`dangerous_progression_k`)

- P(label | end_x >= 80): **0.943**
- P(label | end_x < 80): **0.248**
- Share of positives with end_x >= 80: **0.580**
- ROC AUC of end_x alone: **0.786**
- n passes = 50,688; prevalence = 0.433

