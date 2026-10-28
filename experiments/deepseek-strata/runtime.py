"""Install or verify the pinned official Strata runtime. Preview changes by default."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, cast
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from project_paths import read_manifest  # noqa: E402


def digest(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_download(url: str, path: Path, expected: str) -> None:
    """Download an HTTPS artifact and check its pinned digest before use."""
    if path.is_file():
        if digest(path) != expected:
            raise RuntimeError(f"The existing artifact differs from its pin: {path}.")
        return
    if not url.startswith("https://"):
        raise RuntimeError("The artifact URL must use HTTPS.")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, 1024**2)
    if digest(partial) != expected:
        partial.unlink()
        raise RuntimeError(f"The downloaded artifact differs from its pin: {path}.")
    partial.replace(path)


def install_source(source: Path, info: dict[str, Any], cache: Path) -> None:
    """Extract the pinned source archive without links or path traversal."""
    if source.exists():
        return
    archive = cache / "source.tar.gz"
    checked_download(str(info["source_archive_url"]), archive, str(info["source_archive_sha256"]))
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="strata-source-", dir=source.parent) as directory:
        stage = Path(directory) / "source"
        stage.mkdir()
        files: dict[str, str] = {}
        unpacked = 0
        with tarfile.open(archive) as contents:
            for entry in contents:
                parts = Path(entry.name).parts
                if entry.name.startswith("/") or ".." in parts:
                    raise RuntimeError("The source archive contains an unsafe path.")
                if len(parts) < 2 or entry.isdir():
                    continue
                if not entry.isfile():
                    raise RuntimeError("The source archive contains a link or special file.")
                unpacked += entry.size
                if unpacked > 1024**3:
                    raise RuntimeError("The source archive exceeds the 1 GiB extraction limit.")
                name = str(Path(*parts[1:]))
                if name in files:
                    raise RuntimeError("The source archive contains a duplicate file.")
                entry.name = name
                contents.extract(entry, stage, filter="data")
                files[name] = digest(stage / name)
        receipt = stage / "source-files.json"
        receipt.write_text(json.dumps(files, indent=2, sort_keys=True) + "\n")
        if digest(receipt) != info["source_receipt_sha256"]:
            raise RuntimeError("The source receipt differs from its pin.")
        (stage / "SOURCE_COMMIT").write_text(str(info["source_revision"]) + "\n")
        stage.rename(source)


def verify_source(source: Path, info: dict[str, Any]) -> None:
    """Check every upstream source file against the pinned receipt."""
    receipt = source / "source-files.json"
    if not receipt.is_file() or digest(receipt) != info["source_receipt_sha256"]:
        raise RuntimeError("The pinned Strata source receipt is missing or changed.")
    files = cast(dict[str, str], json.loads(receipt.read_text()))
    for name, expected in files.items():
        path = source / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"The upstream Strata source file is missing or changed: {path}.")
    extras = [str(path.relative_to(source)) for path in source.rglob("*.py")
              if str(path.relative_to(source)) not in files]
    if extras:
        raise RuntimeError("The Strata source contains unpinned Python files: " + ", ".join(extras[:8]))


def verify_native(source: Path, runtime: Path, settings: dict[str, Any], install: bool) -> Path:
    """Use the stock verifier for the signed feed and every engine file."""
    feed_path = runtime / "strata-feed.json"
    signature_path = runtime / "strata-feed.json.sig"
    url = str(settings["feed_url"])
    if install:
        checked_download(url, feed_path, str(settings["feed_sha256"]))
        checked_download(url + ".sig", signature_path, str(settings["feed_signature_sha256"]))
    for path, key in ((feed_path, "feed_sha256"), (signature_path, "feed_signature_sha256")):
        if not path.is_file() or digest(path) != settings[key]:
            raise RuntimeError(f"The pinned Strata feed is missing or changed: {path}.")
    sys.path.insert(0, str(source))
    signing = importlib.import_module("strata_control.signing")
    feed = importlib.import_module("strata_control.feed")
    component = importlib.import_module("strata_control.engine_component")
    # The manifest pins this historical feed. Verify its signature and exact bytes,
    # without using its expiry date to force an unrequested runtime update.
    keys = signing.load_trusted_keys({})
    key_id = signing.verify_bytes(keys, feed_path.read_bytes(), json.loads(signature_path.read_text()))
    index = feed.parse(feed_path.read_bytes(), source=url, key_id=key_id)
    package = next((entry for entry in index.dsv41
                    if entry.version == settings["version"] and entry.platform == "linux-x86_64"
                    and entry.backend == "cuda" and entry.variant == "cuda13"), None)
    if package is None or package.sha256 != settings["archive_sha256"]:
        raise RuntimeError("The signed feed does not contain the pinned CUDA engine.")
    binary = runtime / "engine-dsv41" / f"{package.version}-linux-x86_64-cuda-cuda13" / "strata-dsv41"
    if install and not binary.exists():
        archive = runtime / Path(str(settings["archive_url"])).name
        checked_download(str(settings["archive_url"]), archive, str(settings["archive_sha256"]))
        if archive.stat().st_size != int(settings["archive_size"]):
            raise RuntimeError("The engine archive has the wrong size.")
        installer = importlib.import_module("strata_control.dsv41_engine")
        if installer.install_package(runtime / "engine-dsv41", archive) != binary:
            raise RuntimeError("The official Strata package installer refused the archive.")
    manifest = binary.parent / component.MANIFEST
    if not manifest.is_file() or digest(manifest) != package.manifest_sha256:
        raise RuntimeError("The engine manifest differs from the signed feed.")
    item = component.read_manifest(binary.parent)
    signing.install_component_verifier(keys)
    component.verify_signature(item, {component.ENV_REQUIRE_SIGNED: "1"})
    component.verify(item, full=True)
    if digest(binary) != settings["binary_sha256"]:
        raise RuntimeError("The Strata engine binary differs from its pin.")
    return binary


def ensure_venv(venv: Path, packages: list[str], install: bool) -> None:
    """Install or check only the pinned serving and tokenizer dependencies."""
    python = venv / "bin/python"
    if install and not python.exists():
        subprocess.run(["uv", "venv", str(venv), "--python", "python3"], check=True)
    if install:
        subprocess.run(["uv", "pip", "install", "--python", str(python), *packages], check=True)
    code = "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name'].lower():d.version for d in m.distributions()}))"
    actual = json.loads(subprocess.check_output([str(python), "-c", code], text=True))
    expected = dict(package.lower().split("==", 1) for package in packages)
    if actual != expected:
        raise RuntimeError(f"The isolated Strata Python packages differ from the pins: {actual}.")


def main() -> int:
    """Preview, install, or verify the stock Strata runtime without a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Install missing pinned official runtime files. Do not load a model.")
    mode.add_argument("--check", action="store_true", help="Verify installed files and dependencies. Do not install or download anything.")
    args = parser.parse_args()
    info = read_manifest(HERE / "manifest.json")
    settings = cast(dict[str, Any], info["runtime"])
    source = Path(str(info["source_directory"]))
    runtime = Path(str(settings["directory"]))
    venv = Path(str(settings["venv"]))
    print(json.dumps({"source_revision": info["source_revision"], "source_directory": str(source),
                      "runtime_directory": str(runtime), "runtime_version": settings["version"],
                      "venv": str(venv), "python_packages": settings["python_packages"],
                      "mode": "install" if args.run else "check" if args.check else "preview",
                      "loads_model": False}, indent=2), flush=True)
    if not args.run and not args.check:
        return 0
    if args.run:
        runtime.mkdir(parents=True, exist_ok=True)
        install_source(source, info, runtime / str(info["source_revision"]))
    verify_source(source, info)
    if info.get("pack_source_directory"):
        pack_source = Path(info["pack_source_directory"])
        pack_info = {key: info["pack_" + key] for key in
                     ("source_revision", "source_archive_url", "source_archive_sha256", "source_receipt_sha256")}
        if args.run:
            install_source(pack_source, pack_info, runtime / str(pack_info["source_revision"]))
        verify_source(pack_source, pack_info)
    binary = verify_native(source, runtime, settings, args.run)
    ensure_venv(venv, cast(list[str], settings["python_packages"]), args.run)
    record = {"source_revision": info["source_revision"], "binary": str(binary),
              "binary_sha256": settings["binary_sha256"], "venv": str(venv), "verified": True}
    if args.run:
        (runtime / "install-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
