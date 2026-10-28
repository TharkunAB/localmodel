#!/usr/bin/env python3
"""Verify staged Dory source. Use --fetch to restore missing pinned source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
from typing import Any, cast
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVISION = "43e865ec059a35d57f1167eaea76ca187077f063"
ARCHIVE_SHA256 = "bf8aed7048a226cf22b27b5c8871ed746f4d78c84d7e50daf36184862af0fdca"
ARCHIVE_SIZE = 38191065
ARCHIVE_URL = f"https://codeload.github.com/brianchmielNV/llama.cpp/tar.gz/{REVISION}"
STAGING = ROOT / "state/sources/dory-native"
SOURCE = STAGING / REVISION
ARCHIVE = STAGING / f"{REVISION}.tar.gz"


def digest(path: Path) -> str:
    """Return the file SHA256 without loading the complete file into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_archive() -> None:
    """Refuse an archive with different bytes or a symbolic link."""
    if ARCHIVE.is_symlink() or ARCHIVE.stat().st_size != ARCHIVE_SIZE or digest(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("The cached source archive differs from the pinned archive. It was left unchanged.")


def check_source(source: Path, hashes: dict[str, str]) -> None:
    """Check all source files and refuse extra files or symbolic links."""
    if source.is_symlink():
        raise ValueError("The source directory must not be a symbolic link.")
    paths = list(source.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("The source contains a symbolic link. It was left unchanged.")
    if any(not path.is_file() and not path.is_dir() for path in paths):
        raise ValueError("The source contains an unsupported file type. It was left unchanged.")
    actual = {path.relative_to(source).as_posix() for path in paths if path.is_file()}
    if actual != set(hashes):
        raise ValueError("The source contains missing or extra files. It was left unchanged.")
    for relative, expected in hashes.items():
        if digest(source / relative) != expected:
            raise ValueError(f"The source hash differs: {relative}. It was left unchanged.")


def restore(hashes: dict[str, str]) -> None:
    """Restore pinned source without executing downloaded code."""
    STAGING.mkdir(parents=True, exist_ok=True)
    if not STAGING.resolve().is_relative_to((ROOT / "state/sources").resolve()):
        raise ValueError("The staging directory is outside state/sources.")
    if not ARCHIVE.exists():
        with tempfile.NamedTemporaryFile(dir=STAGING, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > ARCHIVE_SIZE:
                            raise ValueError("The source download exceeds the pinned size.")
                        stream.write(chunk)
                stream.flush()
                if size != ARCHIVE_SIZE or digest(temporary) != ARCHIVE_SHA256:
                    raise ValueError("The source download differs from the pinned archive.")
                if ARCHIVE.exists():
                    raise ValueError("An archive appeared during download. It was left unchanged.")
                temporary.rename(ARCHIVE)
            finally:
                temporary.unlink(missing_ok=True)
    check_archive()
    with tempfile.TemporaryDirectory(dir=STAGING, prefix="extract-") as temporary:
        destination = Path(temporary) / REVISION
        destination.mkdir()
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            members: list[tarfile.TarInfo] = []
            for member in archive.getmembers():
                parts = PurePosixPath(member.name).parts
                if not parts or parts[0] != f"llama.cpp-{REVISION}" or ".." in parts:
                    raise ValueError("The archive contains an unexpected path.")
                if not member.isdir() and not member.isfile():
                    raise ValueError("The archive contains an unsupported file type.")
                member.name = "/".join(parts[1:])
                if member.name:
                    members.append(member)
            archive.extractall(destination, members=members, filter="data")
        check_source(destination, hashes)
        if SOURCE.exists() or SOURCE.is_symlink():
            raise ValueError("A source directory already exists. It was left unchanged.")
        destination.rename(SOURCE)


def main() -> None:
    """Verify existing source offline, or restore it when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Restore missing source; download only a missing pinned source archive. Never build or fetch weights.")
    args = parser.parse_args()
    manifest = cast(dict[str, Any], json.loads((HERE / "manifest.json").read_text()))
    archive = cast(dict[str, Any], manifest["source_archive"])
    if (manifest["source_revision"] != REVISION or Path(str(manifest["source_directory"])) != SOURCE
            or archive["url"] != ARCHIVE_URL or archive["sha256"] != ARCHIVE_SHA256
            or archive["size"] != ARCHIVE_SIZE or Path(str(archive["path"])) != ARCHIVE):
        raise ValueError("The manifest differs from the source pins in this helper.")
    hashes: dict[str, str] = {}
    for line in (HERE / "source-files.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or not path.parts or relative in hashes
                or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("The source hash list contains an invalid entry.")
        hashes[relative] = expected
    if not hashes or len(hashes) != manifest["source_files_count"]:
        raise ValueError("The source hash count differs from the manifest.")
    if ARCHIVE.exists() or ARCHIVE.is_symlink():
        check_archive()
    if not SOURCE.exists() and not SOURCE.is_symlink():
        if not args.fetch:
            raise ValueError("Source is missing. Run this helper with --fetch to restore it.")
        restore(hashes)
    check_source(SOURCE, hashes)
    print(f"Verified {len(hashes)} pinned source files. No build, weights, inference, or benchmarks ran.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        sys.exit(str(error))
