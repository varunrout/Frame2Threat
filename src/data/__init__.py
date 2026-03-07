"""
src.data – StatsBomb data download, parsing, and train/val/test splitting.

Key responsibilities:
- Fetch events and 360 freeze-frame data via statsbombpy.
- Filter pass events and attach freeze-frame snapshots.
- Produce deterministic match-level train / val / test splits.
"""
