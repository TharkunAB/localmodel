"""Prepare and serve the one-card DeepSeek 1017 Basalt comparison on a rental GPU."""
from __future__ import annotations

import argparse
from collections.abc import Iterator
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid


PROFILE = "deepseek-basalt-pro6000"
MODEL = "deepseek-v4.1-flash-1017"
VERSION = "v0.5.0"
RELEASE = f"basalt-{VERSION}-linux-x64"
ARCHIVE_URL = f"https://github.com/jesdga95/basalt/releases/download/{VERSION}/{RELEASE}.tar.gz"
ARCHIVE_SIZE = 67_579_415
ARCHIVE_SHA256 = "dbf744069b72c4b15e32480ff32c49caf906556f52b64b5ff766924e27e28b49"
REVISION = "621661e913cb76e363bf14235667d3b5c0f787d4"
PACK = "DeepSeek-V4.1-Flash-1017-UD-IQ1_M.basalt"
PACK_REPO = "jesdga/DeepSeek-V4.1-Flash-1017-Basalt"
PACK_URL = f"https://huggingface.co/{PACK_REPO}/resolve/{REVISION}/{PACK}"
HUB_VERSION = "2.4.0"
XET_VERSION = "1.8.0"
PACK_SIZE = 259_973_824_987
PACK_SHA256 = "1cd706a3de0dca2623e7461c9eb2f4c916aa26a3ad9e3d90ee10e5730f8ae942"
GUARD = Path(__file__).resolve().parents[1] / "local/experiments/deepseek-strata/network_guard.py"
GIB = 1024**3
VERIFY_PYTHON = """\
from importlib.metadata import version
from pathlib import Path
import sys
from pip._vendor.packaging.requirements import Requirement

for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    requirement = Requirement(line)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    actual = version(requirement.name)
    if not requirement.specifier.contains(actual):
        sys.exit(f"Basalt requires {requirement}; installed {actual}")
"""


def context_size(value: str) -> int:
    match = re.fullmatch(r"(\d+)([km]?)", value.lower())
    if match:
        count = int(match[1]) * {"": 1, "k": 1024, "m": 1024**2}[match[2]]
        if 1024 <= count <= 1048576:
            return count
    raise argparse.ArgumentTypeError("DeepSeek context must be between 1024 and 1M tokens.")


def require_container_weights(pack: Path, workspace: Path = Path("/workspace")) -> None:
    """Reject the persistent volume and aliases into it before preparing a large pack."""
    volume = workspace.resolve()
    if pack.is_relative_to(workspace) or pack.resolve().is_relative_to(volume):
        raise RuntimeError("DeepSeek trial weights must stay on container disk. "
                           "Use --root /opt/remote-models and a model directory outside /workspace.")


def pro6000() -> dict[str, int | str]:
    """Use the CUDA driver's visible ordinals, including container and environment masks."""
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("The pinned Basalt archive requires Linux x86_64.")
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise RuntimeError("The NVIDIA driver is unavailable. Use a CUDA-enabled RunPod image.") from error

    def checked(symbol: str, *args: object) -> None:
        result = getattr(cuda, symbol)(*args)
        if result:
            raise RuntimeError(f"{symbol} failed with CUDA status {result}. Check the Pod's GPU allocation and driver.")

    checked("cuInit", 0)
    count = ctypes.c_int()
    checked("cuDeviceGetCount", ctypes.byref(count))
    if count.value != 1:
        raise RuntimeError(f"{PROFILE} requires one visible RTX PRO 6000 Blackwell 96 GB; CUDA sees {count.value}.")
    device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    checked("cuDeviceGet", ctypes.byref(device), 0)
    name = ctypes.create_string_buffer(256)
    checked("cuDeviceGetName", name, len(name), device)
    checked("cuDeviceGetAttribute", ctypes.byref(major), 75, device)
    checked("cuDeviceGetAttribute", ctypes.byref(minor), 76, device)
    total = ctypes.c_size_t()
    checked("cuDeviceTotalMem_v2", ctypes.byref(total), device)
    label = name.value.decode()
    arch = major.value * 10 + minor.value
    if (not re.search(r"\bRTX PRO 6000 Blackwell\b", label, re.I) or "MIG" in label.upper() or arch != 120
            or not 90 * GIB <= total.value <= 97 * GIB):
        raise RuntimeError(f"{PROFILE} requires one RTX PRO 6000 Blackwell 96 GB (sm_120); "
                           f"CUDA found {label}, sm_{arch}, {total.value/GIB:.1f} GiB.")
    raw = (ctypes.c_ubyte * 16)()
    checked("cuDeviceGetUuid_v2", ctypes.byref(raw), device)
    return {"name": label, "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(raw))),
            "compute_capability": arch, "memory_mib": total.value // 2**20}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    size = path.stat().st_size
    if size >= GIB:
        print(f"Verifying SHA-256 of {path.name} ({size/GIB:.2f} GiB)...", file=sys.stderr, flush=True)
    next_update = time.monotonic() + 10
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 2**20):
            sha.update(chunk)
            if size >= GIB and time.monotonic() >= next_update:
                print(f"SHA-256: {stream.tell()/size:.1%} ({stream.tell()/GIB:.1f}/{size/GIB:.1f} GiB)",
                      file=sys.stderr, flush=True)
                next_update = time.monotonic() + 10
    return sha.hexdigest()


def identity(path: Path, sha: str) -> list[int | str]:
    stat = path.stat()
    return [sha, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino]


def fetch_hf_pack(partial: Path, download_venv: Path, size: int, sha: str) -> None:
    """Run the ordered Xet transfer in a preparation-only environment."""
    python = download_venv / "bin/python"
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(download_venv)], check=True)
    subprocess.run([str(python), "-Im", "pip", "--isolated", "install", "--only-binary=:all:",
                    f"huggingface-hub=={HUB_VERSION}", f"hf-xet=={XET_VERSION}"], check=True)
    cache = partial.parent / ".hf-xet"
    require_container_weights(cache)
    env = dict(os.environ, HF_ENDPOINT="https://huggingface.co", HF_HOME=str(cache),
               HF_XET_CACHE=str(cache / "xet"), HF_HUB_OFFLINE="0", HF_HUB_DISABLE_XET="0",
               HF_XET_HIGH_PERFORMANCE="1", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    print("Downloading with Hugging Face Xet and verifying SHA-256 as bytes arrive.", flush=True)
    subprocess.run([str(python), "-I", "-B", str(Path(__file__).with_name("basalt_xet.py")),
                    PACK_REPO, REVISION, PACK, str(partial), str(size), sha], env=env, check=True)


def artifact(path: Path, url: str, size: int, sha: str, *, prepare: bool, reverify: bool,
             hf_venv: Path | None = None) -> None:
    """Resume interrupted HTTPS downloads. Publish the asset only after a full checksum."""
    partial = path.with_name(path.name + ".part")
    stamp = path.with_name(path.name + ".verified.json")
    for item in (path, partial, stamp):
        if item.is_symlink() or (item.exists() and not item.is_file()):
            raise RuntimeError(f"Expected a regular file at {item}.")
    verified_download = False
    if not path.exists():
        if not prepare:
            raise RuntimeError(f"Missing {path}. Run remotemodel fetch first.")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise RuntimeError(f"Partial download exceeds its pin: {partial}")
        if offset < size and shutil.disk_usage(path.parent).free < size - offset + 20 * GIB:
            raise RuntimeError(f"{path.parent} needs at least {(size-offset)/GIB:.1f} GiB for the remaining "
                               "download, plus 20 GiB free after preparation.")
        transfer_venv = hf_venv if offset == 0 else None
        if transfer_venv is not None:
            fetch_hf_pack(partial, transfer_venv, size, sha)
        elif offset < size:
            if offset:
                print(f"Resuming {path.name} with curl at {offset/GIB:.2f} GiB.", flush=True)
            subprocess.run(["curl", "--fail", "--show-error", "--location", "--proto", "=https",
                            "--proto-redir", "=https", "--retry", "3", "--continue-at", "-",
                            "--output", str(partial), url], check=True)
        if partial.stat().st_size != size or ((transfer_venv is None or reverify) and digest(partial) != sha):
            partial.unlink()
            raise RuntimeError(f"Discarded the failed download of {path.name}; size or SHA-256 differs from its pin.")
        partial.replace(path)
        if size >= GIB:
            print(f"SHA-256 verified: {path.name}", flush=True)
        verified_download = True
    if path.stat().st_size != size:
        raise RuntimeError(f"Installed file has the wrong size: {path}")
    fingerprint = identity(path, sha)
    if (not reverify and stamp.exists() and stamp.read_text() == json.dumps(fingerprint)):
        return
    if not verified_download and (digest(path) != sha or identity(path, sha) != fingerprint):
        raise RuntimeError(f"Installed file differs from its pin: {path}")
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(fingerprint, out)
    os.replace(out.name, stamp)


def release_files(archive: tarfile.TarFile) -> Iterator[tuple[Path, tarfile.TarInfo]]:
    for member in archive:
        path = Path(member.name)
        if (not member.isfile() or len(path.parts) < 2 or path.parts[0] != RELEASE or ".." in path.parts):
            raise RuntimeError(f"Unexpected entry in the release archive: {member.name}")
        yield Path(*path.parts[1:]), member


def install_release(archive: Path, runtime: Path, *, prepare: bool) -> None:
    """Use only regular archive files. Recheck installed code against the verified archive."""
    if runtime.is_symlink() or (runtime.exists() and not runtime.is_dir()):
        raise RuntimeError(f"Invalid runtime directory: {runtime}")
    if not runtime.exists():
        if not prepare:
            raise RuntimeError("Basalt is not prepared. Run remotemodel fetch first.")
        with tempfile.TemporaryDirectory(dir=runtime.parent) as tmp:
            stage = Path(tmp) / RELEASE
            with tarfile.open(archive, "r:gz") as tar:
                for name, member in release_files(tar):
                    target = stage / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Missing archive member: {name}")
                    with source, target.open("xb") as dst:
                        shutil.copyfileobj(source, dst, length=1024**2)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
            stage.rename(runtime)
        return
    all_files = set()
    for path in runtime.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise RuntimeError(f"Unexpected object in prepared runtime: {path}")
        if path.is_file():
            all_files.add(path.relative_to(runtime))
    with tarfile.open(archive, "r:gz") as tar:
        for name, member in release_files(tar):
            target = runtime / name
            if name not in all_files or target.stat().st_size != member.size:
                raise RuntimeError(f"The prepared runtime differs from its release: {name}")
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"Missing archive payload: {name}")
            with source, target.open("rb") as installed:
                while chunk := source.read(1024**2):
                    if chunk != installed.read(len(chunk)):
                        raise RuntimeError(f"The prepared runtime differs from its release: {name}")
            all_files.remove(name)
    if all_files:
        raise RuntimeError(f"The prepared runtime contains unexpected files: {sorted(all_files)[:3]}")


def environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_AUDIT"):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8897)
    parser.add_argument("--profile", choices=[PROFILE], required=True)
    parser.add_argument("--context", "--ctxt", type=context_size, default=1048576)
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--reverify", action="store_true", help="Hash the complete archive and pack even with a verified receipt.")
    parser.add_argument("--download-method", choices=("hf", "curl"), default="hf",
                        help="Fresh pack transport (default: Hugging Face Xet). Existing curl partials resume with curl.")
    parser.add_argument("--no-ssh", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or (args.install_deps and not args.prepare_only):
        parser.error("Port must be 1024..65535. --install-deps requires preparation.")
    root = args.root.expanduser().absolute()
    artifacts = root / "runtimes" / RELEASE
    archive = artifacts / (RELEASE + ".tar.gz")
    runtime_root = Path(os.environ.get("REMOTEMODEL_RUNTIME_DIR",
                                       str(Path.home() / ".cache/remotemodel/runtimes"))).expanduser().absolute()
    base = runtime_root / RELEASE
    runtime = base / "release"
    python = base / "venv" / "bin" / "python"
    pack = root / "models" / "basalt" / REVISION / PACK
    require_container_weights(pack)
    gpu = pro6000()
    command = [str(runtime / "bin/basalt"), "serve", str(pack), "--python", str(python),
               "--alias", MODEL, "--host", "127.0.0.1", "--port", str(args.port), "--gpu", "0",
               "--parallel", "1", "--max-context", str(args.context), "--spec", "4", "--draft-experts", "32"]
    if args.show_config:
        print(json.dumps({"plan": {"profile": PROFILE, "model_id": MODEL, "engine": "basalt", "version": VERSION,
                         "root": str(root), "port": args.port, "context": args.context, "quant": "UD-IQ1_M",
                         "spec": "dspark", "draft_tokens": 3, "parallel": 1, "gpus": [gpu],
                         "model_revision": REVISION, "pack": str(pack), "pack_bytes": PACK_SIZE,
                         "fresh_pack_download_method": args.download_method,
                         "release_archive": str(archive), "runtime_directory": str(runtime),
                         "serving_python": str(python),
                         "pack_sha256": PACK_SHA256, "release_sha256": ARCHIVE_SHA256,
                         "outgoing_network": "blocked during serving", "command": command}},
                         indent=2))
        return
    if args.install_deps:
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", "ca-certificates", "curl",
                        "libseccomp2", "python3-venv"], check=True)
    for library in ("libseccomp.so.2", "libcudart.so.13", "libcublas.so.13"):
        try:
            ctypes.CDLL(library)
        except OSError as error:
            raise RuntimeError(f"{library} unavailable. Use a CUDA 13 runtime image and install libseccomp2.") from error
    if not GUARD.is_file():
        raise RuntimeError(f"Missing serving network guard: {GUARD}")
    if args.prepare_only:
        artifacts.mkdir(parents=True, exist_ok=True)
        base.mkdir(parents=True, exist_ok=True)
        pack.parent.mkdir(parents=True, exist_ok=True)
    if not base.is_dir() or not artifacts.is_dir():
        raise RuntimeError(f"Basalt needs prepared assets under {root} and an executable runtime under {base}. "
                           "Run remotemodel fetch to reuse the persisted files and prepare this container.")
    if os.statvfs(base).f_flag & os.ST_NOEXEC:
        raise RuntimeError(f"The runtime mount is noexec: {base}. Set REMOTEMODEL_RUNTIME_DIR to an executable "
                           "container filesystem. Keep the weights on the container disk.")
    env = environment()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu["uuid"])
    with (artifacts / "prepare.lock").open("a") as data_lock, (base / "install.lock").open("a") as runtime_lock:
        fcntl.flock(data_lock, fcntl.LOCK_EX)
        fcntl.flock(runtime_lock, fcntl.LOCK_EX)
        artifact(archive, ARCHIVE_URL, ARCHIVE_SIZE, ARCHIVE_SHA256, prepare=args.prepare_only, reverify=args.reverify)
        install_release(archive, runtime, prepare=args.prepare_only)
        # Check the Pod's glibc and libstdc++ compatibility before any weight download.
        if args.prepare_only:
            subprocess.run([str(runtime / "bin/basalt"), "--help"], env=env, check=True, stdout=subprocess.DEVNULL)
        if args.prepare_only and not python.is_file():
            subprocess.run([sys.executable, "-m", "venv", str(base / "venv")], check=True)
        if not python.is_file():
            raise RuntimeError("The Basalt Python environment is missing. Run remotemodel fetch first.")
        requirements = runtime / "requirements.txt"
        if args.prepare_only:
            subprocess.run([str(python), "-Im", "pip", "--isolated", "install", "--only-binary=:all:",
                            "--no-deps", "-r", str(requirements)], env=env, check=True)
        verify_command = [str(python), "-I", "-c", VERIFY_PYTHON, str(requirements)]
        if not args.prepare_only:
            verify_command = [sys.executable, "-I", "-B", str(GUARD), *verify_command]
        subprocess.run(verify_command, env=env, check=True)
        download_venv = base / f"hf-{HUB_VERSION}-xet-{XET_VERSION}" if args.download_method == "hf" else None
        artifact(pack, PACK_URL, PACK_SIZE, PACK_SHA256, prepare=args.prepare_only, reverify=args.reverify,
                 hf_venv=download_venv)
    if not args.prepare_only:
        os.chdir(runtime)
        os.execve(sys.executable, [sys.executable, "-I", "-B", str(GUARD), *command], env)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Basalt: {exc}")
