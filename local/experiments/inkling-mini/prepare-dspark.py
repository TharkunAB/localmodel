"""Install or check Unsloth's Linux x86-64 CUDA 13 Inkling server."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TAG = "b12588-mix-c0c3de8"
RUNTIME = ROOT / "state/runtimes" / f"inkling-unsloth-{TAG}"
RECEIPT = {
    "tag": TAG,
    "url": f"https://github.com/unslothai/llama.cpp/releases/download/{TAG}/app-{TAG}-linux-x64-cuda13-portable.tar.gz",
    "size": 419623585,
    "sha256": "6c1985d2d75c9cc6b299891d557aca70843ba5ccb68f14863a50687048cfdd48",
    "toolkit": "13.3",
    "supported_sms": [75, 80, 86, 89, 90, 100, 103, 120],
}
# Digest of the names, link targets and SHA-256 values in the verified release.
BINARIES_SHA256 = "2cef3b7643d3f64ba6cd2fe0117729615fc0493ca4c5d60fd6c445fd205bcb46"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root: Path) -> None:
    if root.is_symlink() or json.loads((root / "asset.json").read_bytes()) != RECEIPT:
        raise ValueError(f"Unexpected Unsloth runtime receipt at {root}")
    if not os.access(root / "llama-server", os.X_OK):
        raise ValueError(f"Missing executable at {root}/llama-server")
    manifest = hashlib.sha256()
    for path in sorted([root / "llama-server", *root.glob("*.so*")]):
        if not path.is_file() or path.resolve().parent != root.resolve():
            raise ValueError(f"Unexpected runtime file: {path}")
        value = f"link:{path.readlink()}" if path.is_symlink() else digest(path)
        manifest.update(f"{value}  {path.name}\n".encode())
    if manifest.hexdigest() != BINARIES_SHA256:
        raise ValueError(f"The server or shared libraries differ from the pinned release at {root}")


def prepare(root: Path) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if root.exists() or root.is_symlink():
            verify(root)
            return
        with tempfile.TemporaryDirectory(prefix=".inkling-dspark-", dir=root.parent) as stage:
            archive = Path(stage) / "runtime.tar.gz"
            subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--proto", "=https",
                            "--proto-redir", "=https", "--output", str(archive), str(RECEIPT["url"])], check=True)
            if archive.stat().st_size != RECEIPT["size"] or digest(archive) != RECEIPT["sha256"]:
                raise ValueError("The Unsloth CUDA release archive failed its size or SHA-256 check")
            install = Path(stage) / "install"
            install.mkdir()
            with tarfile.open(archive, mode="r:gz") as bundle:
                bundle.extractall(install, filter="data")
            (install / "asset.json").write_text(json.dumps(RECEIPT, indent=2) + "\n")
            verify(install)
            install.rename(root)


def library_path(root: Path) -> str:
    paths = [str(root)]
    paths.extend(p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p)
    toolkits = [Path(s) for var in ("CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH") if (s := os.environ.get(var))]
    toolkits.extend([Path("/usr/local/cuda"), ROOT / "state/builds/strata-cache-route-20261109/cuda-site/nvidia/cu13"])
    for venv in (ROOT / ".venv", ROOT / "dory/.venv"):
        toolkits.extend(venv.glob("lib/python*/site-packages/nvidia/cu13"))
    for toolkit in toolkits:
        for suffix in ("lib64", "lib", "targets/x86_64-linux/lib", "lib/x86_64-linux-gnu"):
            folder = toolkit / suffix
            if any((folder / name).is_file() for name in ("libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13")):
                paths.append(str(folder.resolve()))
    if Path("/usr/lib/wsl/lib/libcuda.so.1").is_file():
        paths.append("/usr/lib/wsl/lib")
    value = os.pathsep.join(dict.fromkeys(paths))
    report = subprocess.run(["ldd", "-r", *(str(root / name) for name in
                            ("llama-server", "libllama-server-impl.so", "libggml-cuda.so"))],
                            env=os.environ | {"LD_LIBRARY_PATH": value, "LC_ALL": "C"},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    failures = [s.strip() for s in report.stdout.splitlines() if "not found" in s or "undefined symbol" in s]
    if report.returncode or failures:
        raise ValueError("Install a compatible NVIDIA driver and CUDA 13 runtime with cuBLAS. "
                         "Use CUDAToolkit_ROOT or LD_LIBRARY_PATH for a custom location.\n" +
                         "\n".join(failures or [report.stdout.strip()]))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog="Fetch requires curl. Serving requires "
                                     "a compatible NVIDIA driver, CUDA 13 runtime and cuBLAS.")
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--verify", action="store_true", help="Verify the installed runtime offline.")
    flags.add_argument("--library-path", action="store_true", help="Verify the runtime and print its LD_LIBRARY_PATH.")
    args = parser.parse_args()
    if args.verify or args.library_path:
        verify(RUNTIME)
    else:
        prepare(RUNTIME)
    print(library_path(RUNTIME) if args.library_path else f"Verified {RUNTIME}/llama-server")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, tarfile.TarError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
