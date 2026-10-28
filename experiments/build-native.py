"""Print a native runtime build plan. Use --run to compile the prepared source."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_paths import read_manifest
from localmodel import CAPPED, STATE, check_free  # noqa: E402


def main() -> int:
    """Validate a prepared candidate before an explicitly requested build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=("bonsai-native", "bonsai-next", "dory-native"))
    parser.add_argument("--jobs", type=int, default=2, help="Compiler jobs (default: 2).")
    parser.add_argument("--run", action="store_true", help="Run the build. Omit this flag to print the plan.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Compiler jobs must be positive.")
    manifest_path = ROOT / "experiments" / args.runtime / "manifest.json"
    info = read_manifest(manifest_path)
    source = (ROOT / str(info["source_directory"])).resolve()
    build = (ROOT / str(info["build_directory"])).resolve()
    binary = (ROOT / str(info["binary"])).resolve()
    if not source.is_relative_to(STATE / "sources") or not build.is_relative_to(STATE / "builds"):
        raise ValueError("Use isolated source and build directories under state/.")
    if not binary.is_relative_to(build):
        raise ValueError("The binary must be inside the candidate build directory.")
    commands = [
        ["cmake", "-S", str(source), "-B", str(build), *cast(list[str], info["cmake_args"])],
        ["cmake", "--build", str(build), "--target", "llama-server", "--parallel", str(args.jobs)],
    ]
    memory = str(info.get("build_memory_cap", "16G"))
    duration = str(info.get("build_time_cap", "90m"))
    plan = {"runtime": args.runtime, "status": "build-not-run", "manifest": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source": str(source), "binary": str(binary), "jobs": args.jobs,
            "commands": [shlex.join(command) for command in commands],
            "memory_limit": memory, "swap_limit": "0", "time_limit": duration}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    if not (source / "CMakeLists.txt").is_file():
        raise RuntimeError(f"Source is missing. See experiments/{args.runtime}/README.md for source preparation.")
    hash_file = ROOT / str(info["source_files_sha256"])
    hashes = [line.split("  ", 1) for line in hash_file.read_text().splitlines() if line]
    if not hashes or any(len(pair) != 2 for pair in hashes):
        raise RuntimeError("The manifest has no source hashes.")
    if len(hashes) != info["source_files_count"]:
        raise RuntimeError("The source hash count differs from the manifest.")
    for expected, name in hashes:
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Source file is outside its candidate directory: {name}.")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise RuntimeError(f"Prepared source changed: {name}. Prepare and review it again.")
    environment = os.environ.copy()
    environment["PATH"] = str(ROOT / ".venv/bin") + ":" + environment.get("PATH", "")
    environment.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    environment.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={environment['XDG_RUNTIME_DIR']}/bus")
    environment["HEAVY_PEAKS"] = str(STATE / f"{args.runtime}-build-peaks")
    for variable in ("PATH", "LD_LIBRARY_PATH"):
        prefix = info.get("environment", {}).get(variable + "_prepend")
        if prefix:
            environment[variable] = str(prefix) + (":" + environment[variable] if environment.get(variable) else "")
    STATE.mkdir(exist_ok=True)
    with (STATE / "server.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A managed GPU job is running. Build after it stops.") from None
        check_free(8898)
        for command in commands:
            subprocess.run([str(CAPPED), "--mem", memory, "--time", duration, *command],
                           env=environment, cwd=ROOT, check=True)
    if not binary.is_file():
        raise RuntimeError(f"The build did not produce {binary}.")
    plan["status"] = "built-not-inference-tested"
    plan["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (build / "localmodel-build.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
