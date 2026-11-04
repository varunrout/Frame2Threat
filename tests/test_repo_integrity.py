"""Guards that keep the repository's documentation and committed artefacts consistent.

These need no data: they read only committed files. They fail when a document points at a file that
no longer exists, when a model file changes without its documented hash, or when the fixed research
split is edited.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".tmp",
    ".venv",
    "venv",
    "build",
    "dist",
    "data",
    "node_modules",
    "__pycache__",
}


def _markdown_files() -> list[Path]:
    return [
        p
        for p in ROOT.rglob("*.md")
        if not SKIP_DIRS & set(p.relative_to(ROOT).parts)
        and ".egg-info" not in str(p)
        and "site-packages" not in str(p)
    ]


def test_committed_split_manifest_is_the_fixed_research_split():
    manifest = pd.read_csv(ROOT / "configs" / "research_split_manifest.csv")
    assert list(manifest.columns) == ["match_id", "split"]
    assert not manifest.match_id.duplicated().any(), "a match appears in two splits"
    assert manifest.match_id.notna().all()
    assert manifest.split.value_counts().to_dict() == {"train": 69, "val": 15, "test": 15}


def test_relative_markdown_links_resolve():
    broken = []
    for md in _markdown_files():
        for match in re.finditer(
            r"\]\(([^)#\s]+)", md.read_text(encoding="utf-8", errors="ignore")
        ):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (md.parent / target).resolve().exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")
    assert not broken, "broken relative links:\n" + "\n".join(broken)


def test_models_readme_hashes_match_the_committed_files():
    text = (ROOT / "models" / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([\w.]+\.(?:pt|joblib))` .*?`([0-9a-f]{16})`", text, re.MULTILINE)
    assert len(rows) >= 7, "models/README.md should list every current model with its hash"
    for name, short_hash in rows:
        path = ROOT / "models" / name
        assert path.exists(), f"models/README.md lists a missing file: {name}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest.startswith(short_hash), f"{name} changed; update models/README.md"


def test_models_readme_quotes_the_verified_aucs():
    check = (ROOT / "reports" / "repro" / "artefact_check.md").read_text(encoding="utf-8")
    verified = re.findall(
        r"^\|[^|]+\|\s*([0-9.]{6})\s*\|\s*[0-9.]{6}\s*\|\s*PASS", check, re.MULTILINE
    )
    assert len(verified) == 4, "artefact_check.md should hold four verified AUCs"
    readme = (ROOT / "models" / "README.md").read_text(encoding="utf-8")
    for value in verified:
        assert value in readme, f"verified AUC {value} is missing from models/README.md"
