"""Prepare and serve Magnitude's DeepSeek 1017 IQ2_XXS and DSpark on one RTX PRO 6000."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import basalt

PROFILE = "deepseek-magnitude-pro6000"
MODEL = "deepseek-v4.1-flash:gguf:iq2-xxs"
FAMILY = "deepseek-v4.1-flash"
VERSION = "0.3.1"
SOURCE_COMMIT = "a9276e5f970f150c1294dd5541a62d76916adf68"
BUILD = "0.1.0+src.cf1c986c0ca780792f1c17c4"
ARCHIVE = "magnitude-icn-base-linux-x64-gnu.tar.gz"
URL = f"https://github.com/magnitudedev/magnitude/releases/download/%40magnitudedev/cli%40{VERSION}/{ARCHIVE}"
ARCHIVE_SIZE = 476814890
ARCHIVE_SHA = "70b321ecc41492ae1c75d15c39ccb78240c8ac5d89c66edca3340098887ae0fc"
NATIVE_FILES = {
    "bin/magnitude-inference": (1486368144, "3d7e510b1c80cce7bb3d8eae05c954942698e1cb98876c395120a5734d5c9c26"),
    "catalog/model-planner-inputs.bundle": (28319837, "4d7968169fa86726f34beca956115621374ce139b5b11cf6ed7252fa2b1840b6"),
    "runtime/libnvrtc.so.12": (106237008, "c3430e8b6b0d8d3a74f5f839118a85e13e3ad6e4d6203ea1b6a27a0c0b4fe10a"),
    "runtime/libnvrtc-builtins.so.12.9": (7202760, "6507cf33450cfb22970c2c1e7aa8e7c0a33e39d864720277b05f61a92791e2e8"),
    "runtime/NVRTC-LICENSE.txt": (67876, "72c22161fc1ebf242443d45158a7dd382d68597b00eeb7a7f539be5c6114be33"),
}
TARGET = ("ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF", "65d53bc8be65123e255ff02f3ded958b6dbf54fd", 249806466016)
DRAFT = ("magnitudedev/DeepSeek-V4.1-Flash-1017-GGUF", "689503584971b7b96545821d712a9a71d8f7ac31", 7938244608)
DECLARATION = {"schemaVersion": 1, "nativeBuild": BUILD}


def install(base: Path, runtime: Path, prepare: bool, reverify: bool) -> None:
    archive = base / ARCHIVE
    if runtime.is_symlink():
        raise RuntimeError(f"Invalid Magnitude installation: {runtime}")
    if not runtime.is_dir():
        if not prepare:
            raise RuntimeError("Magnitude is missing. Run remotemodel fetch first.")
        partial = archive.with_name(archive.name + ".part")
        archive_missing = 0 if archive.is_file() else max(0, ARCHIVE_SIZE - (partial.stat().st_size if partial.is_file() else 0))
        if shutil.disk_usage(base).free < sum(n for n, _ in NATIVE_FILES.values()) + archive_missing:
            raise RuntimeError("Insufficient disk space for the Magnitude release archive and extracted runtime.")
        basalt.artifact(archive, URL, ARCHIVE_SIZE, ARCHIVE_SHA, prepare=True, reverify=reverify, reserve_bytes=0)
        with tempfile.TemporaryDirectory(dir=base) as tmp, tarfile.open(archive) as members:
            staged = Path(tmp) / "release"
            seen: set[str] = set()
            for member in members:
                name = str(Path(member.name))
                if member.isdir() and name in (".", "bin", "catalog", "runtime"):
                    continue
                if name not in NATIVE_FILES or name in seen or not member.isfile() or member.size != NATIVE_FILES[name][0]:
                    raise RuntimeError(f"Unexpected Magnitude archive entry: {name}")
                out = staged / name
                out.parent.mkdir(parents=True, exist_ok=True)
                data = members.extractfile(member)
                if data is None:
                    raise RuntimeError(f"Missing archive data: {name}")
                with out.open("xb") as stream:
                    shutil.copyfileobj(data, stream, 1 << 20)
                out.chmod(0o755 if name.startswith("bin/") else 0o644)
                seen.add(name)
            if seen != NATIVE_FILES.keys():
                raise RuntimeError("The ICN archive is incomplete.")
            (staged / "installation.json").write_text(json.dumps(DECLARATION))
            staged.rename(runtime)
    elif reverify and archive.is_file():
        basalt.artifact(archive, URL, ARCHIVE_SIZE, ARCHIVE_SHA, prepare=False, reverify=True)
    for name, (size, sha) in NATIVE_FILES.items():
        path = runtime / name
        if path.resolve() != runtime.resolve() / name:
            raise RuntimeError(f"Unexpected runtime symlink: {path}")
        basalt.artifact(path, "", size, sha, prepare=False, reverify=reverify)
    declaration = runtime / "installation.json"
    if declaration.is_symlink() or json.loads(declaration.read_text()) != DECLARATION:
        raise RuntimeError("Unexpected ICN installation declaration.")


def weight_files(runtime: Path) -> list[tuple[str, str, str, int, str]]:
    """Read individual file pins from the publisher's checksum-verified planner bundle."""
    with (runtime / "catalog/model-planner-inputs.bundle").open("rb") as f:
        if f.read(8) != b"MAGPLAN5":
            raise RuntimeError("Unexpected Magnitude planner bundle format.")
        length, = struct.unpack("<Q", f.read(8))
        if not 0 < length < 1024**2:
            raise RuntimeError("Invalid planner index length.")
        records = json.loads(f.read(length))["plannerInputs"]
    plan, = [p for p in records.values() if p["modelId"] == FAMILY and p["variantId"] == "gguf:iq2-xxs"]
    result: list[tuple[str, str, str, int, str]] = []
    for role, (repo, revision, total) in (("target", TARGET), ("draft", DRAFT)):
        package = plan[role]["package"]
        if (package["source"]["repository"], package["source"]["revision"]) != (repo, revision):
            raise RuntimeError(f"Unexpected {role} repository or revision.")
        if sum(x["sizeBytes"] for x in package["files"]) != total:
            raise RuntimeError(f"Unexpected {role} package size.")
        for entry in package["files"]:
            name, size, sha = entry["path"], entry["sizeBytes"], entry["sha256"]
            if Path(name).name != name or not name.endswith(".gguf") or not re.fullmatch("[a-f0-9]{64}", sha):
                raise RuntimeError(f"Invalid {role} file pin.")
            result.append((repo, revision, name, size, sha))
    return result


def weights(store: Path, runtime: Path, prepare: bool, reverify: bool, hf_venv: Path | None) -> None:
    files = weight_files(runtime)
    missing_bytes = 0
    for repo, _, _, size, sha in files:
        blob = store / "hub" / ("models--" + repo.replace("/", "--")) / "blobs" / sha
        if not blob.exists():
            part = blob.with_name(sha + ".part")
            missing_bytes += max(0, size - (part.stat().st_size if part.is_file() else 0))
    if prepare and missing_bytes and shutil.disk_usage(store).free < missing_bytes:
        raise RuntimeError(f"The IQ2 target and DSpark need {missing_bytes / 1024**3:.1f} GiB more.")
    for repo, revision, name, size, sha in files:
        hub = store / "hub" / ("models--" + repo.replace("/", "--"))
        blob = hub / "blobs" / sha
        snapshot = hub / "snapshots" / revision / name
        for path in (blob, snapshot):
            if path.parent.resolve() != store.resolve() / path.parent.relative_to(store):
                raise RuntimeError(f"Unexpected cache-directory symlink: {path.parent}")
            if prepare:
                path.parent.mkdir(parents=True, exist_ok=True)
        basalt.artifact(blob, f"https://huggingface.co/{repo}/resolve/{revision}/{name}", size, sha,
                        prepare=prepare, reverify=reverify, hf_venv=hf_venv, hf_source=(repo, revision, name),
                        reserve_bytes=0)
        if prepare and not snapshot.exists() and not snapshot.is_symlink():
            snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
        if not snapshot.is_file() or not snapshot.samefile(blob):
            raise RuntimeError(f"The ICN snapshot is missing or points to a different blob: {snapshot}")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def serve(command: list[str], port: int, context: int, env: dict[str, str]) -> None:
    """Load the verified catalog variant before announcing CUDA readiness."""
    identity = uuid.uuid4().hex
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(route: str, payload: dict[str, str] | None = None) -> dict[str, Any]:
        req = urllib.request.Request(f"http://127.0.0.1:{port}{route}",
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=15) as response:
            return json.load(response)

    child = subprocess.Popen([sys.executable, "-I", str(basalt.GUARD), *command, "--instance-id", identity], env=env)
    try:
        load_requested = False
        while True:
            if child.poll() is not None:
                raise RuntimeError(f"ICN exited before loading {MODEL}: {child.returncode}")
            try:
                health = request("/health")
            except urllib.error.URLError:
                time.sleep(1)
                continue
            if health.get("instanceId") != identity or health.get("nativeBuild") != BUILD:
                raise RuntimeError("The listener does not belong to this Magnitude launch.")
            if not load_requested:
                catalog = request("/api/v1/catalog/models")
                if not catalog.get("reconciliationComplete"):
                    time.sleep(1)
                    continue
                rows: list[dict[str, Any]] = catalog["models"]
                row: dict[str, Any] = next((m for m in rows if m["id"] == MODEL), dict[str, Any]())
                state = row.get("localState", {})
                effective = state.get("effective", {})
                if (state.get("_tag") != "Installed" or effective.get("_tag") != "Ready"
                        or effective.get("model", {}).get("speculativeMethod", {}).get("_tag") != "DSpark"):
                    raise RuntimeError(f"ICN did not resolve the pinned IQ2 target with DSpark: {state}")
                if request("/api/v1/hardware").get("selected_executor") == "host-cpu":
                    raise RuntimeError("ICN selected CPU. This profile requires the CUDA executor.")
                print(f"Loading {MODEL} with DSpark and {context} context tokens.", flush=True)
                instance = request("/api/v1/instances", {"modelId": MODEL})
                load_requested = True
            else:
                instances: list[dict[str, Any]] = request("/api/v1/instances")["instances"]
                instance = next((s for s in instances if s["modelId"] == MODEL), dict[str, Any]())
            state = instance.get("lifecycle", {})
            if state.get("_tag") == "Ready":
                if (str(state["device"]["backend"]).lower() != "cuda"
                        or state["allocation"]["contextWindowTokens"] != context):
                    raise RuntimeError(f"ICN loaded with different device or context: {state}")
                print(f"Magnitude ready on 127.0.0.1:{port}.", flush=True)
                break
            if state.get("_tag") in ("Failed", "Stopped"):
                raise RuntimeError(f"Magnitude load failed: {state}")
            time.sleep(2)
        if code := child.wait():
            raise RuntimeError(f"ICN exited with status {code}.")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/remote-models"))
    parser.add_argument("--profile", choices=[PROFILE], required=True)
    parser.add_argument("--context", "--ctxt", type=basalt.context_size, default=1048576)
    parser.add_argument("--port", type=int, default=8897)
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reverify", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--download-method", choices=("hf", "curl"), default="hf")
    parser.add_argument("--no-ssh", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("The API port must be between 1024 and 65535.")
    if args.install_deps and not args.prepare_only and not args.show_config:
        parser.error("--install-deps requires preparation.")
    root = args.root.expanduser().absolute()
    store = root / "models/magnitude-1017-iq2"
    basalt.require_container_weights(store)
    base = Path(os.environ.get("REMOTEMODEL_RUNTIME_DIR", str(Path.home() / ".cache/remotemodel/runtimes"))).expanduser().absolute() / f"magnitude-{VERSION}"
    runtime = base / "release"
    gpu = basalt.pro6000(PROFILE)
    command = [str(runtime / "bin/magnitude-inference"), "serve", "--bind", f"127.0.0.1:{args.port}",
               "--model-store", str(store), "--cache-root", str(root / "cache/magnitude"),
               "--installation", str(runtime / "installation.json"), "--context-tokens", str(args.context)]
    if args.show_config:
        print(json.dumps({"plan": {"profile": PROFILE, "model_id": MODEL, "engine": "magnitude", "version": VERSION,
              "root": str(root), "port": args.port, "context": args.context, "quant": "GSQ-RCO IQ2_XXS",
              "model_store": str(store), "target_bytes": TARGET[2], "drafter_bytes": DRAFT[2],
              "pack_bytes": TARGET[2] + DRAFT[2], "model_revision": TARGET[1], "draft_revision": DRAFT[1],
              "spec": "dspark", "gpus": [gpu], "release_sha256": ARCHIVE_SHA, "source_commit": SOURCE_COMMIT,
              "outgoing_network": "blocked during serving", "command": command}}, indent=2))
        return
    if args.install_deps:
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", "ca-certificates", "curl",
                        "libseccomp2", "python3-venv"], check=True)
    ctypes.CDLL("libseccomp.so.2")
    if not basalt.GUARD.is_file():
        raise RuntimeError(f"Missing network guard: {basalt.GUARD}")
    if args.prepare_only:
        base.mkdir(parents=True, exist_ok=True)
        store.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not store.is_dir() or not base.is_dir():
        raise RuntimeError("Magnitude is missing. Run remotemodel fetch first.")
    with (base / "install.lock").open("a") as lock, (store / "prepare.lock").open("a") as data_lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fcntl.flock(data_lock, fcntl.LOCK_EX)
        install(base, runtime, args.prepare_only, args.reverify)
        env = basalt.environment()
        for key in list(env):
            if key.startswith(("MAGNITUDE_", "OTEL_")):
                del env[key]
        env.update(CUDA_VISIBLE_DEVICES=str(gpu["uuid"]), MAGNITUDE_OTEL="0",
                   LD_LIBRARY_PATH=str(runtime / "runtime") + ":" + env.get("LD_LIBRARY_PATH", ""))
        info = json.loads(subprocess.check_output([sys.executable, "-I", str(basalt.GUARD),
                         str(runtime / "bin/magnitude-inference"), "version", "--json"], env=env, text=True))
        if info.get("native_build") != BUILD or info.get("api_version") != 2:
            raise RuntimeError("The native ICN build does not match this profile.")
        venv = base / f"hf-{basalt.HUB_VERSION}-xet-{basalt.XET_VERSION}" if args.download_method == "hf" else None
        weights(store, runtime, args.prepare_only, args.reverify, venv)
    if not args.prepare_only:
        serve(command, args.port, args.context, env)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"remotemodel magnitude: {error}") from error
