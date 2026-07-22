"""Single raw-to-scores pipeline command for Frame2Threat.

This entry point delegates to the v1 event-only reproduction path, which is
currently the maintained end-to-end pipeline from StatsBomb Open Data or
synthetic smoke data to scored pass outputs.
"""

from __future__ import annotations

from reproduce_v1 import main


if __name__ == "__main__":
    main()
