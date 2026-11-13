"""Prepare and build the isolated Qwen cache-reuse backport."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "state/sources/strata-cache-reuse-20261113"
BUILD = ROOT / "state/builds/strata-cache-reuse-20261113"
OFFICIAL = ROOT / "state/sources/strata-official-0.2.13"
CUDA_SITE = ROOT / "state/builds/strata-cache-route-20261109/cuda-site"
CUDA = CUDA_SITE / "nvidia/cu13"
RUNTIME = ROOT / "state/runtimes/strata-qwen-0.1.59/engine"
BASE = "d2f096475817d4666c364c820b0ca830f751cfc5"
FIX = "13eedf552417a6a791a7d13c6597e24300ee3a09"
SOURCE_RECEIPT = "731a327b1779d142dcae124d750452b2bfc4f483c777a5abf54ca301a3c9d83b"
CHANGED = ("include/strata/core/serve_stop_bound.hpp", "src/program/generate.cpp",
           "tests/core/serve_stop_bound_test.cpp")


def digest(path: Path) -> str:
    """Hash a regular file."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Expected a regular file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(name: str, record: object) -> None:
    """Save reproducible build evidence."""
    (HERE / name).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def prepare() -> dict[str, str]:
    """Copy verified source files and apply only the selected upstream hunks."""
    receipt = OFFICIAL / "source-files.json"
    if digest(receipt) != SOURCE_RECEIPT or (OFFICIAL / "SOURCE_COMMIT").read_text().strip() != BASE:
        raise RuntimeError("The official source pin differs from the reviewed base.")
    files = cast(dict[str, str], json.loads(receipt.read_text()))
    for name, expected in files.items():
        if digest(OFFICIAL / name) != expected:
            raise RuntimeError(f"Official source changed: {name}")
    if not SOURCE.exists():
        SOURCE.mkdir(parents=True)
        for name in files:
            destination = SOURCE / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(OFFICIAL / name, destination)
    if all(digest(SOURCE / name) == files[name] for name in CHANGED):
        subprocess.run(["patch", "-p1", "--batch", "--fuzz=0", "--forward", "--no-backup-if-mismatch",
                        "-i", str(HERE / "upstream.patch")], cwd=SOURCE, check=True)
    subprocess.run(["patch", "-p1", "--batch", "--fuzz=0", "--reverse", "--dry-run",
                    "-i", str(HERE / "upstream.patch")], cwd=SOURCE, check=True)
    names = {str(path.relative_to(SOURCE)) for path in SOURCE.rglob("*") if path.is_file()}
    if names != set(files):
        raise RuntimeError("The candidate source file set differs from the official receipt.")
    actual = {name: digest(SOURCE / name) for name in files}
    if set(name for name in files if actual[name] != files[name]) != set(CHANGED):
        raise RuntimeError("The candidate differs outside the reviewed three files.")
    changed = {name: actual[name] for name in CHANGED}
    if (HERE / "source-files.json").exists():
        if changed != json.loads((HERE / "source-files.json").read_text()):
            raise RuntimeError("Prepared source changed after its initial review.")
    else:
        save("source-files.json", changed)
    return actual


def dependencies() -> dict[str, Any]:
    """Recheck the installed CUDA wheel records and record the reused toolchain."""
    pins_path = ROOT / "experiments/strata-candidates-20261109/cache-route/dependency-pins.json"
    pins = cast(dict[str, Any], json.loads(pins_path.read_text()))
    verified = 0
    for package in pins["packages"]:
        metadata = CUDA_SITE / (package["name"].replace("-", "_") + "-" + package["version"] + ".dist-info")
        with (metadata / "RECORD").open(newline="") as stream:
            for name, encoded, _ in csv.reader(stream):
                if not encoded:
                    continue
                algorithm, expected = encoded.split("=", 1)
                if algorithm != "sha256":
                    raise RuntimeError("Unexpected CUDA wheel hash algorithm.")
                observed = base64.urlsafe_b64encode(bytes.fromhex(digest(CUDA_SITE / name))).decode().rstrip("=")
                if observed != expected:
                    raise RuntimeError(f"CUDA dependency changed: {name}")
                verified += 1
    native_manifest = RUNTIME / "engine-component.json"
    native_manifest_sha256 = "076b8a35d5d16d78b55c485f50a9cdf7dc0e7904e15b5bb2d91ac47bccff8609"
    if digest(native_manifest) != native_manifest_sha256:
        raise RuntimeError("The official Qwen runtime manifest changed.")
    native_files = cast(dict[str, Any], json.loads(native_manifest.read_text()))["files"]
    for item in native_files:
        if item["path"] in ("libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"):
            if digest(RUNTIME / item["path"]) != item["sha256"]:
                raise RuntimeError(f"The official Qwen runtime library changed: {item['path']}")
    return {"dependency_pins_sha256": digest(pins_path), "cuda_record_files_verified": verified,
            "official_runtime_manifest_sha256": native_manifest_sha256,
            "packages": pins["packages"], "ggml": "0.2.13 vendored third_party/strata-ggml",
            "cxx_version": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
            "nvcc_version": subprocess.check_output([str(CUDA / "bin/nvcc"), "--version"], text=True),
            "cmake_version": subprocess.check_output([str(ROOT / ".venv/bin/cmake"), "--version"], text=True).splitlines()[0],
            "ninja_version": subprocess.check_output([str(ROOT / ".venv/bin/ninja"), "--version"], text=True).strip(),
            "cmake_sha256": digest(ROOT / ".venv/bin/cmake"),
            "ninja_sha256": digest(ROOT / ".venv/bin/ninja")}


def main() -> int:
    """Print the plan, or prepare and compile under the fixed resource cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    cmake = str(ROOT / ".venv/bin/cmake")
    configure = [cmake, "-G", "Ninja", "-S", str(SOURCE), "-B", str(BUILD / "native"),
                 "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
                 "-DSTRATA_BUILD_DSV41=OFF", "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_PORTABLE=ON",
                 "-DCMAKE_CUDA_ARCHITECTURES=89-real", f"-DCMAKE_CUDA_COMPILER={CUDA / 'bin/nvcc'}",
                 f"-DCUDAToolkit_ROOT={CUDA}", f"-DCMAKE_MAKE_PROGRAM={ROOT / '.venv/bin/ninja'}",
                 f"-DCMAKE_INSTALL_RPATH={BUILD / 'native'};{RUNTIME}",
                 "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_SKIP_BUILD_RPATH=OFF"]
    compile_command = [cmake, "--build", str(BUILD / "native"), "--target", "strata", "--parallel", "2"]
    plan: dict[str, Any] = {"base_revision": BASE, "fix_revision": FIX, "patch_sha256": digest(HERE / "upstream.patch"),
            "official_source_receipt_sha256": SOURCE_RECEIPT, "configure_argv": configure,
            "build_argv": compile_command, "memory_cap_bytes": 16 * 1024**3, "swap_max_bytes": 0,
            "parallel_jobs": 2, "nice": 19, "time_cap": "90m", "gpu_tests": False}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        return 0
    group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
    control = Path("/sys/fs/cgroup") / group.lstrip("/")
    limits = {name: (control / name).read_text().strip()
              for name in ("memory.max", "memory.high", "memory.swap.max")}
    if limits != {"memory.max": str(16 * 1024**3), "memory.high": str(16 * 1024**3), "memory.swap.max": "0"}:
        raise RuntimeError("Run this build under the documented 16 GiB capped scope with zero swap.")
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("Run this build with nice -n 19.")
    plan["effective_cgroup_limits"] = limits
    files = prepare()
    save("dependency-receipt.json", dependencies())
    save("build-plan.json", plan)
    BUILD.mkdir(parents=True, exist_ok=True)
    test = BUILD / "serve_stop_bound_test"
    subprocess.run(["c++", "-std=c++20", "-O2", "-DNDEBUG", "-I", str(SOURCE / "include"),
                    str(SOURCE / CHANGED[2]), "-o", str(test)], check=True)
    result = subprocess.check_output([str(test)], text=True)
    save("cpu-test.json", {"command": str(test), "stdout": result, "exit_code": 0})
    for command in (configure, compile_command):
        subprocess.run(command, cwd=ROOT, check=True)
    native = BUILD / "native"
    libraries = [native / "libstrata_kq.so", *(RUNTIME / name for name in
                  ("libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"))]
    environment = os.environ | {"LD_LIBRARY_PATH": f"{native}:{RUNTIME}:/usr/lib/wsl/lib"}
    linked = "".join(str(path) + "\n" + subprocess.check_output(["ldd", "-r", str(path)],
                     env=environment, stderr=subprocess.STDOUT, text=True)
                     for path in (native / "strata", native / "libstrata_kq.so"))
    (HERE / "link-check.txt").write_text(linked)
    if "not found" in linked or "undefined symbol" in linked:
        raise RuntimeError("The built engine has unresolved runtime libraries or symbols.")
    save("build-receipt.json", {**plan, "status": "built",
         "official_binary_build": json.loads((RUNTIME / "BUILD.json").read_text()),
         "comparison_limit": "The signed binary uses f-0213-rc 712fec6b4297; this source build uses d2f09647 plus 13eedf55. Compare this candidate with STRATA_SERVE_STOP_BOUND=0 and =1 to isolate the fix.",
         "build_helper_sha256": digest(Path(__file__)),
         "build_plan_sha256": digest(HERE / "build-plan.json"),
         "cpu_test_sha256": digest(HERE / "cpu-test.json"),
         "source_directory": str(SOURCE), "source_files_count": len(files),
         "source_files_sha256": digest(HERE / "source-files.json"),
         "dependency_receipt_sha256": digest(HERE / "dependency-receipt.json"),
         "binary_path": str(native / "strata"), "binary_sha256": digest(native / "strata"),
         "libraries": [{"path": str(path), "sha256": digest(path)} for path in libraries],
         "lib_dirs": [str(native), str(RUNTIME), "/usr/lib/wsl/lib"]})
    save("runtime-pins.json", {"build_receipt_sha256": digest(HERE / "build-receipt.json")})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
