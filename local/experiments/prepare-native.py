"""Install the pinned Bonsai source and apply the reviewed patch without compiling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_paths import read_manifest  # noqa: E402


def digest(path: Path) -> str:
    """Return the complete file digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(source: Path, info: dict[str, Any]) -> None:
    """Check every source file against the reviewed list."""
    entries = Path(info["source_files_sha256"]).read_text().splitlines()
    if len(entries) != info["source_files_count"]:
        raise RuntimeError("The source hash count differs from the manifest.")
    for line in entries:
        expected, name = line.split("  ", 1)
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()) or digest(path) != expected:
            raise RuntimeError(f"The prepared source differs from its pin: {name}")


def main() -> None:
    """Print the preparation plan or install the verified source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=("bonsai-next", "bonsai-native"))
    parser.add_argument("--run", action="store_true", help="Download and prepare source. Omit to preview.")
    args = parser.parse_args()
    info = read_manifest(ROOT / "experiments" / args.runtime / "manifest.json")
    source = Path(info["source_directory"])
    archive_info = info["source_archive"]
    archive = ROOT / "state/downloads" / f"{archive_info['sha256']}.tar.gz"
    patch = Path(info["patch"])
    print(json.dumps({"source": str(source), "archive": str(archive), "url": archive_info["url"],
                      "patch": str(patch), "compile": False, "run": args.run}, indent=2), flush=True)
    if not args.run:
        return
    if not source.resolve().is_relative_to(ROOT / "state/sources"):
        raise RuntimeError("The source directory must be under state/sources.")
    if digest(patch) != info["patch_sha256"]:
        raise RuntimeError("The reviewed patch changed.")
    if source.exists():
        verify(source, info)
        print("The prepared source matches all pinned hashes.")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--proto", "=https",
                        "--proto-redir", "=https", "--output", str(partial), archive_info["url"]], check=True)
        if digest(partial) != archive_info["sha256"]:
            raise RuntimeError("The source archive differs from its pin.")
        partial.replace(archive)
    if archive.stat().st_size != archive_info["size"] or digest(archive) != archive_info["sha256"]:
        raise RuntimeError("The source archive differs from its size or digest pin.")
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{args.runtime}-", dir=source.parent) as folder:
        stage = Path(folder)
        with tarfile.open(archive) as contents:
            contents.extractall(stage, filter="data")
        roots = list(stage.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise RuntimeError("Expected one source directory in the archive.")
        tree = roots[0]
        subprocess.run(["git", "init", "--quiet", str(tree)], check=True)
        subprocess.run(["git", "-C", str(tree), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(tree), "apply", str(patch)], check=True)
        verify(tree, info)
        tree.rename(source)
    print(f"Prepared {info['source_files_count']} verified source files. No compilation or inference ran.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
