# Dataset card

## Source
[StatsBomb Open Data](https://github.com/statsbomb/open-data), pinned to revision
`3bfbffe1de5750ebd47d770be0bb924a10cde54f` (a snapshot preceding the May 2026 event-data refresh;
[configs/research_data.json](../configs/research_data.json)). Follow StatsBomb's licence and attribution terms when
redistributing data or derived examples. Raw and processed data are **not** committed.

## Cohort

| | |
|---|---|
| Matches | 99 |
| Competitions | FIFA World Cup 2022 (64 matches, with 360 freeze frames); La Liga 2020/21, Barcelona matches only (35 matches, no 360) |
| Possessions | 17,065 (stubs shorter than two events dropped) |
| Passes | 50,688; 360 freeze frames cover 26,515 of them (52.3%), concentrated in the World Cup matches |
| Split | Match-level, seed 42: 69 train, 15 validation, 15 test ([manifest](../configs/research_split_manifest.csv)) |
| Test composition | 9 World Cup and 6 La Liga matches (competition derived from match-id range; the manifest records only `match_id` and `split`) |
| Event-time test set | 43,541 non-terminal snapshots; training prevalence of the danger label 0.362 |

## Known biases and limits

- **One club dominates the La Liga half.** All 35 La Liga matches feature Barcelona, so style and opposition are narrow.
- **360 availability is confounded with competition.** Geometry features exist only for the World Cup matches; a lift or
  lack of lift from 360 data cannot be separated from the competition effect.
- **Small and selective.** 99 matches; observations within a possession and a match overlap; confidence intervals use a
  match bootstrap and do not account for the research selection or for the test split having been inspected repeatedly.
- **Random match holdout.** No unseen competition or future season has been tested.
- **Label is an attempted-entry proxy.** A possessing-team shot, an event in the box or a pass ending in the box
  realises the outcome, including incomplete passes; see [label_methodology.md](label_methodology.md).
- **Possessing team.** The causal rule uses the first actor when the provider's possession-team column is absent; a
  live feed must supply the possession team (see open item 4 in [CLAIMS.md](CLAIMS.md)).

## Tables

Column-level documentation: [data_dictionary.md](data_dictionary.md). The maintained route builds
`events_parsed.parquet`, `possession_sequences.parquet`, the split files and `event_time_arrays.pkl` with
`scripts/prepare_event_time.py`.
