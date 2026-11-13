"""Fetch pinned Edge0 artifacts without starting an engine or executing model files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TARGET = ROOT / "models/edge0-145b-q3k"
MANIFEST = HERE / "manifest.json"
REPOSITORY = "Edge0/Edge0-145B-q3k"
REVISION = "00f91ca2dc8290d74c79360f0d9941704e0f5c69"
HEADROOM = 20 * 2**30
SMALL_LIMIT = 16 * 2**20

SPEC = importlib.util.spec_from_file_location("qwen_artifact_helpers", ROOT / "experiments/qwen-strata/runtime.py")
assert SPEC and SPEC.loader
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def safe_path(name: str) -> Path:
    """Require a relative artifact path with ordinary parent directories."""
    relative = Path(name)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise RuntimeError(f"Invalid artifact path: {name!r}")
    path = TARGET / relative
    for parent in path.parents:
        if parent == ROOT:
            break
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise RuntimeError(f"Artifact parent is not an ordinary directory: {parent}")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"Artifact is not an ordinary file: {path}")
    return path


def pins() -> list[dict[str, Any]]:
    """Validate the fixed repository revision and complete artifact inventory."""
    info: dict[str, Any] = json.loads(MANIFEST.read_text())
    if (info.get("schema") != 1 or info.get("repository") != REPOSITORY or info.get("revision") != REVISION
            or info.get("directory") != "models/edge0-145b-q3k"):
        raise RuntimeError("The Edge0 manifest does not match its pinned repository revision.")
    files = cast(list[dict[str, Any]], info["files"])
    names: set[str] = set()
    for item in files:
        name = str(item["name"])
        safe_path(name)
        if (name in names or name.startswith("assets/") or name in ("start.sh", ".gitattributes", "verification.json", "verification.json.tmp")
                or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", name)
                or not isinstance(item["size"], int) or item["size"] < 1
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise RuntimeError(f"Invalid or duplicate Edge0 pin: {name}")
        if "git_blob_sha1" in item and (item["size"] > SMALL_LIMIT or not re.fullmatch(r"[0-9a-f]{40}", item["git_blob_sha1"])):
            raise RuntimeError(f"Invalid small-file Git pin: {name}")
        names.add(name)
    allowed = names | {name + ".part" for name in names} | {"verification.json", "verification.json.tmp"}
    parents = {str(parent) for name in names for parent in Path(name).parents if str(parent) != "."}
    if TARGET.exists():
        for path in TARGET.rglob("*"):
            name = str(path.relative_to(TARGET))
            if path.is_symlink() or (path.is_dir() and name not in parents) or (not path.is_dir() and (not path.is_file() or name not in allowed)):
                raise RuntimeError(f"The Edge0 directory contains an unknown file or link: {path}")
    return files


def verified(item: dict[str, Any], record: dict[str, Any]) -> bool:
    """Check the recorded identity without rereading large verified files."""
    path = safe_path(item["name"])
    if not path.exists() or record.get("sha256") != item["sha256"]:
        return False
    identity = HELPERS.identity(path)
    return (identity["size"] == item["size"] and record.get("file") == identity
            and record.get("git_blob_sha1") == item.get("git_blob_sha1"))


def fetch_one(item: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Resume one artifact, verify its pins, and return its receipt entry."""
    if verified(item, record):
        return record
    path = safe_path(item["name"])
    partial = safe_path(item["name"] + ".part")
    pin = {**item, "url": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{item['name']}"}
    if not path.exists() and partial.exists() and partial.stat().st_size == item["size"]:
        if HELPERS.digest(partial) != item["sha256"]:
            raise RuntimeError(f"Completed partial differs from its pin: {partial}")
        partial.rename(path)
    else:
        HELPERS.download(pin, path)
    if "git_blob_sha1" in item:
        data = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != item["git_blob_sha1"]:
            raise RuntimeError(f"Small artifact differs from its Git blob pin: {path}")
    return {"sha256": item["sha256"], "file": HELPERS.identity(path),
            **({"git_blob_sha1": item["git_blob_sha1"]} if "git_blob_sha1" in item else {})}


def execute(files: list[dict[str, Any]], check: bool) -> None:
    """Fetch with at most two workers and save each completed verification."""
    receipt = safe_path("verification.json")
    proof: dict[str, Any] = {"schema": 1, "manifest_sha256": HELPERS.digest(MANIFEST), "files": {}}
    if receipt.exists():
        previous: dict[str, Any] = json.loads(receipt.read_text())
        if previous.get("schema") != 1 or previous.get("manifest_sha256") != proof["manifest_sha256"]:
            raise RuntimeError("The Edge0 receipt uses a different artifact manifest.")
        proof = previous
    if check:
        for item in files:
            if not verified(item, proof["files"].get(item["name"], {})):
                raise RuntimeError(f"Artifact is absent or changed: {item['name']}")
    else:
        remaining = 0
        for item in files:
            path = safe_path(item["name"])
            partial = safe_path(item["name"] + ".part")
            have = path.stat().st_size if path.exists() else partial.stat().st_size if partial.exists() else 0
            if have > item["size"]:
                raise RuntimeError(f"Artifact or partial exceeds its pinned size: {item['name']}")
            remaining += max(0, item["size"] - have)
        if shutil.disk_usage(ROOT).free < remaining + HEADROOM:
            raise RuntimeError(f"Edge0 needs {remaining / 2**30:.2f} GiB free plus 20 GiB headroom.")
        TARGET.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(fetch_one, item, proof["files"].get(item["name"], {})): item
                       for item in sorted(files, key=lambda item: item["size"], reverse=True)}
            try:
                for future in as_completed(pending):
                    item = pending[future]
                    proof["files"][item["name"]] = future.result()
                    temporary = safe_path("verification.json.tmp")
                    temporary.write_text(json.dumps(proof, indent=2) + "\n")
                    temporary.replace(receipt)
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    print(json.dumps({"verified_files": len(files), "bytes": sum(item["size"] for item in files),
                      "directory": str(TARGET), "receipt": str(receipt)}))


def main() -> None:
    """Preview by default; require an explicit flag for downloads or verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Fetch and verify selected artifacts. Never run inference.")
    mode.add_argument("--check", action="store_true", help="Check existing receipt identities without downloading.")
    parser.add_argument("--small-only", action="store_true", help="Select only artifacts of at most 16 MiB.")
    args = parser.parse_args()
    files = pins()
    if args.small_only:
        files = [item for item in files if item["size"] <= SMALL_LIMIT]
    if not args.run and not args.check:
        print(json.dumps({"repository": REPOSITORY, "revision": REVISION, "directory": str(TARGET),
                          "files": len(files), "bytes": sum(item["size"] for item in files),
                          "headroom_bytes": HEADROOM, "workers": 2}, indent=2))
        return
    state = ROOT / "state"
    if state.is_symlink() or not state.is_dir():
        raise RuntimeError("The state path must be an ordinary directory.")
    descriptor = os.open(state / "models-edge0.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Edge0 artifact operation is running.") from None
        execute(pins() if not args.small_only else files, args.check)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(str(error))
