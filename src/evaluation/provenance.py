"""Run provenance: which code, packages and files produced a result.

Result files that record these fields can be traced back to an exact commit and to the exact
model binaries they scored, which a bare metrics table cannot do.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "torch")


def file_sha256(path: Path) -> str:
    """Full SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_metadata(root: Path, files: list[Path] | None = None) -> dict:
    """Code revision, environment and file hashes for a run started from ``root``.

    ``git_sha`` and ``git_dirty`` are ``None`` outside a git checkout. ``files`` are hashed and
    reported relative to ``root``.
    """
    status = _git(root, "status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git(root, "rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "files_sha256": {
            str(path.resolve().relative_to(root.resolve())).replace("\\", "/"): file_sha256(path)
            for path in (files or [])
        },
    }
