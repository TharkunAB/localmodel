"""Build the reviewed Edge0 engine with the existing CUDA 13 toolchain."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CHECKOUT = ROOT / "state/sources/edge0-da9cc526"
SOURCE = CHECKOUT / "edge0-strata"
BUILD = ROOT / "state/builds/edge0-da9cc526"
GGML_SOURCE = ROOT / "state/sources/maya-ggml-3cf03257"
GGML = BUILD / "llama.cpp"
REVISION = "da9cc526747759227714ddde0728763d95306c04"
GGML_REVISION = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
CUDA = ROOT / "state/builds/strata-cache-route-20261109/cuda-site/nvidia/cu13"
RUNTIME = ROOT / "state/runtimes/strata-qwen-0.1.59/engine"


def save(name: str, value: object) -> None:
    """Save build evidence beside the isolated binaries."""
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / name).write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    """Print the plan, or build under the required memory and priority limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    cmake = str(ROOT / ".venv/bin/cmake")
    configure = [cmake, "-S", str(SOURCE), "-B", str(BUILD / "native"), "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_NATIVE_EXPERTS=ON",
                 "-DSTRATA_PORTABLE=ON", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_BUILD_DSV41=OFF",
                 "-DCMAKE_CUDA_ARCHITECTURES=89-real", f"-DCMAKE_CUDA_COMPILER={CUDA / 'bin/nvcc'}",
                 f"-DCUDAToolkit_ROOT={CUDA}", f"-DCMAKE_MAKE_PROGRAM={ROOT / '.venv/bin/ninja'}",
                 f"-DSTRATA_GGML_DIR={GGML}", "-DGGML_OPENMP_FETCH=OFF", "-DGGML_CPU_KLEIDIAI=OFF",
                 "-DGGML_CCACHE=OFF", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
                 f"-DCMAKE_INSTALL_RPATH={BUILD / 'native'};{RUNTIME}",
                 "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_SKIP_BUILD_RPATH=OFF"]
    compile_command = [cmake, "--build", str(BUILD / "native"), "--target", "strata", "--parallel", "2"]
    plan = {"source_revision": REVISION, "ggml_revision": GGML_REVISION,
            "configure": configure, "build": compile_command, "memory_cap_gib": 16,
            "swap_max": 0, "nice": 19, "inference": False}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        return
    group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
    control = Path("/sys/fs/cgroup") / group.lstrip("/")
    limits = {name: (control / name).read_text().strip()
              for name in ("memory.max", "memory.high", "memory.swap.max")}
    if limits != {"memory.max": str(16 * 2**30), "memory.high": str(16 * 2**30), "memory.swap.max": "0"}:
        raise RuntimeError("Run this build under a 16 GiB capped scope with no swap.")
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("Run this build with nice -n 19.")
    for folder, revision in ((CHECKOUT, REVISION), (GGML_SOURCE, GGML_REVISION)):
        actual = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(folder), "status", "--porcelain", "--untracked-files=no"], text=True)
        if actual != revision or dirty:
            raise RuntimeError(f"Source differs from its clean pin: {folder}")
    spec = importlib.util.spec_from_file_location("reviewed_cuda_build", ROOT / "experiments/strata-cache-reuse-20261113/build.py")
    assert spec and spec.loader
    dependencies = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dependencies)
    save("dependencies.json", dependencies.dependencies())
    if not GGML.exists():
        GGML.mkdir(parents=True)
        with subprocess.Popen(["git", "-C", str(GGML_SOURCE), "archive", GGML_REVISION], stdout=subprocess.PIPE) as archive:
            assert archive.stdout
            with tarfile.open(fileobj=archive.stdout, mode="r|") as stream:
                stream.extractall(GGML, filter="data")
            if archive.wait():
                raise RuntimeError("Cannot copy the pinned ggml dependency.")
        (BUILD / "ggml-revision").write_text(GGML_REVISION + "\n")
    if not (BUILD / "ggml-revision").exists() or (BUILD / "ggml-revision").read_text().strip() != GGML_REVISION:
        raise RuntimeError("The isolated ggml dependency is incomplete or uses a different pin.")
    save("plan.json", {**plan, "effective_limits": limits})
    environment = os.environ.copy()
    environment["PATH"] = str(CUDA / "bin") + os.pathsep + environment.get("PATH", "")
    subprocess.run(configure, env=environment, check=True)
    subprocess.run(compile_command, env=environment, check=True)
    binary = BUILD / "native/strata"
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    shared_libraries: list[dict[str, str | int]] = []
    for library in sorted((BUILD / "native").glob("*.so")):
        if library.is_symlink() or not library.is_file():
            raise RuntimeError(f"Expected an ordinary built library: {library}")
        with library.open("rb") as stream:
            shared_libraries.append({"path": str(library), "size": library.stat().st_size,
                                     "sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
    save("build-receipt.json", {**plan, "binary": str(binary), "binary_sha256": digest,
                           "size": binary.stat().st_size,
                           "shared_libraries": shared_libraries,
                           "libraries": subprocess.check_output(["ldd", str(binary)], text=True)})
    print(f"Built and recorded {binary}", flush=True)


if __name__ == "__main__":
    main()
