#!/usr/bin/env python3
"""Check the pinned source and patch metadata without building or running native code."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import read_manifest  # noqa: E402


def digest(path: Path) -> str:
    """Return the file SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Check each prepared source file and reviewed patch."""
    directory = Path(__file__).resolve().parent
    manifest = read_manifest(directory / "manifest.json")
    source = Path(cast(str, manifest["source_directory"])).resolve()
    patch = Path(cast(str, manifest["patch"]))
    if digest(patch) != manifest["patch_sha256"]:
        raise SystemExit("The combined patch hash differs from the manifest.")
    archive = cast(dict[str, Any], manifest["source_archive"])
    archive_path = Path(cast(str, archive["path"]))
    if archive_path.is_file() and digest(archive_path) != archive["sha256"]:
        raise SystemExit("The cached base archive hash differs from the manifest.")
    entries = cast(list[dict[str, Any]], manifest["upstream_patches"])
    for entry in entries:
        if digest(Path(cast(str, entry["path"]))) != entry["sha256"]:
            raise SystemExit(f"The upstream patch hash differs: {entry['number']}")
    count = 0
    hash_file = Path(cast(str, manifest["source_files_sha256"]))
    for line in hash_file.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise SystemExit(f"Source path is outside the prepared directory: {relative}")
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"The prepared source hash differs: {relative}")
        count += 1
    if count != manifest["source_files_count"]:
        raise SystemExit("The prepared source count differs from the manifest.")
    print(f"Verified {count} prepared source files and {len(entries)} upstream patches.")
    print("No configure, build, model start, or benchmark ran.")


if __name__ == "__main__":
    main()
