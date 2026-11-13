"""Verify and locate the isolated Qwen cache-reuse backport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from project_paths import read_manifest

ROOT = Path(__file__).resolve().parent
HERE = ROOT / "experiments/strata-cache-reuse-20261113"
SOURCE = ROOT / "state/sources/strata-cache-reuse-20261113"
BUILD = ROOT / "state/builds/strata-cache-reuse-20261113/native"
OFFICIAL = ROOT / "state/sources/strata-official-0.2.13"
RUNTIME = ROOT / "state/runtimes/strata-qwen-0.1.59/engine"
BASE = "d2f096475817d4666c364c820b0ca830f751cfc5"
FIX = "13eedf552417a6a791a7d13c6597e24300ee3a09"
SOURCE_RECEIPT = "731a327b1779d142dcae124d750452b2bfc4f483c777a5abf54ca301a3c9d83b"


def binary_path() -> Path:
    """Return the deterministic candidate path without requiring a build."""
    return BUILD / "strata"


def lib_dirs() -> list[str]:
    """Use the candidate kernel library and the signed release CUDA libraries."""
    return [str(BUILD), str(RUNTIME), "/usr/lib/wsl/lib"]


def digest(path: Path) -> str:
    """Hash a regular file whose path contains no symbolic links."""
    if not path.is_file() or path.resolve() != path.absolute():
        raise RuntimeError(f"The cache-reuse artifact is missing or linked: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify() -> dict[str, Any]:
    """Check reviewed source, executable, dependencies, and receipt before startup."""
    try:
        official = read_manifest(ROOT / "experiments/qwen-strata/manifest.json")
        if official["source_revision"] != BASE or official["runtime"]["binary"] != str(RUNTIME / "strata"):
            raise RuntimeError("The official Qwen source or runtime changed. Review and rebuild the cache-reuse backport.")
        pins = cast(dict[str, str], json.loads((HERE / "runtime-pins.json").read_text()))
        receipt = HERE / "build-receipt.json"
        if digest(receipt) != pins["build_receipt_sha256"]:
            raise RuntimeError("The cache-reuse build receipt changed.")
        record = read_manifest(receipt)
        if record["base_revision"] != BASE or record["fix_revision"] != FIX:
            raise RuntimeError("The cache-reuse source revisions changed.")
        if record["binary_path"] != str(binary_path()) or record["lib_dirs"] != lib_dirs():
            raise RuntimeError("The cache-reuse artifact paths changed.")
        checks = {HERE / "upstream.patch": record["patch_sha256"],
                  HERE / "source-files.json": record["source_files_sha256"],
                  HERE / "dependency-receipt.json": record["dependency_receipt_sha256"],
                  HERE / "build.py": record["build_helper_sha256"],
                  HERE / "build-plan.json": record["build_plan_sha256"],
                  HERE / "cpu-test.json": record["cpu_test_sha256"],
                  OFFICIAL / "source-files.json": SOURCE_RECEIPT,
                  binary_path(): record["binary_sha256"]}
        expected_libraries = {BUILD / "libstrata_kq.so", *(RUNTIME / name for name in
                              ("libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"))}
        libraries = {Path(item["path"]): str(item["sha256"]) for item in record["libraries"]}
        if set(libraries) != expected_libraries:
            raise RuntimeError("The cache-reuse runtime library set changed.")
        checks.update(libraries)
        for path, expected in checks.items():
            if digest(path) != expected:
                raise RuntimeError(f"The cache-reuse artifact changed: {path}")
        files = cast(dict[str, str], json.loads((OFFICIAL / "source-files.json").read_text()))
        changed = cast(dict[str, str], json.loads((HERE / "source-files.json").read_text()))
        files.update(changed)
        if len(files) != record["source_files_count"]:
            raise RuntimeError("The cache-reuse source file count changed.")
        names = {str(path.relative_to(SOURCE)) for path in SOURCE.rglob("*") if path.is_file()}
        if names != set(files):
            raise RuntimeError("The cache-reuse source file set changed.")
        for name, expected in files.items():
            if digest(SOURCE / name) != expected:
                raise RuntimeError(f"The cache-reuse source changed: {name}")
        return record
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError("The Qwen cache-reuse build is absent or incomplete. Run "
                           "experiments/strata-cache-reuse-20261113/build.py under its documented cap.") from error
