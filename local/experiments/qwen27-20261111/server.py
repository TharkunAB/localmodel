"""Own a capped dense-Qwen benchmark server and stop it on memory pressure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import localmodel

FLOOR = 8 * 2**30
CAP = 48 * 2**30
stop_requested = False


def memory() -> int:
    """Read available Linux RAM without checking Windows."""
    lines = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(lines["MemAvailable"].split()[0]) * 1024


def guard_reason(available: int, swap: int, failures: int) -> str:
    """Return the first memory condition that requires a stop."""
    if available < FLOOR:
        return "Available Linux RAM is below 8 GiB."
    if swap:
        return "The benchmark scope used swap."
    if failures >= 3:
        return "Three consecutive resource reads failed."
    return ""


def save(path: Path, value: Any) -> None:
    """Replace one complete status file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def self_test() -> None:
    """Check stop conditions and terminate an owned CPU-only child."""
    assert guard_reason(FLOOR, 0, 0) == ""
    assert guard_reason(FLOOR - 1, 0, 0)
    assert guard_reason(FLOOR, 1, 0)
    assert guard_reason(FLOOR, 0, 3)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        if guard_reason(0, 0, 0):
            child.terminate()
        child.wait(timeout=5)
        assert child.returncode is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    print("Memory guard and owned-child stop checks passed.")


def main() -> None:
    """Check resources, launch one server, and clean up on every exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("none", "mtp"), default="none")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=int, default=2400)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None or args.seconds < 1:
        parser.error("Use --output and a positive deadline.")
    if localmodel.STATE.joinpath("active.json").exists():
        raise RuntimeError("A model already owns active.json; leave it running.")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 8899))
    if memory() < CAP + FLOOR or os.getloadavg()[0] > 8:
        raise RuntimeError("Wait for at least 56 GiB available RAM and CPU load at most 8.")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5)
    if any(float(row.split(",")[0]) > 2048 or float(row.split(",")[1]) > 5 for row in gpu.splitlines()):
        raise RuntimeError("The GPU is busy; leave its work running.")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=5)
    if processes.strip():
        raise RuntimeError("A GPU compute process is already running.")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    unit = f"qwen27-benchmark-{os.getpid()}.scope"
    environment = dict(os.environ, XDG_RUNTIME_DIR="/run/user/1000",
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus", CAPPED_UNIT=unit,
        HEAVY_PEAKS=str(output / "model-peaks"))
    command = [str(ROOT / ".venv/bin/python"), str(ROOT / "localmodel.py"), "start", "qwen3.8-27b",
        "--speculative", args.arm, "--draft-tokens", "2", "--mem", "48G"]
    report: dict[str, Any] = {"arm": args.arm, "command": command, "unit": unit,
        "context_capacity": 196608, "memory_cap_bytes": CAP, "linux_floor_bytes": FLOOR,
        "started_at": time.time(), "status": "starting"}
    save(output / "status.json", report)
    failures, ready = 0, False
    scope: Path | None = None
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        report["controller_pid"] = process.pid
        try:
            while not stop_requested:
                if process.poll() is not None:
                    raise RuntimeError(f"The model exited with status {process.returncode}; inspect server.log.")
                if time.time() - report["started_at"] > (args.seconds if ready else 300):
                    raise TimeoutError("The benchmark server reached its deadline.")
                sample: dict[str, Any] = {"at": time.time()}
                try:
                    sample["linux_available_bytes"] = memory()
                    if localmodel.STATE.joinpath("active.json").exists():
                        active = localmodel.read_json(localmodel.STATE / "active.json")
                        if active.get("controller_pid") != process.pid:
                            raise RuntimeError("Another controller replaced the benchmark's active state.")
                        if scope is None:
                            group = next(line[3:] for line in Path("/proc", str(active["pid"]), "cgroup").read_text().splitlines() if line.startswith("0::"))
                            candidate = Path("/sys/fs/cgroup" + group)
                            if candidate.name == unit:
                                scope = candidate
                        if scope is not None:
                            sample["scope"] = {name: (scope / name).read_text().strip() for name in
                                ("memory.current", "memory.peak", "memory.max", "memory.swap.current", "memory.swap.max", "memory.events", "cpu.stat")}
                            sample["model_io"] = (scope / "io.stat").read_text().strip() if (scope / "io.stat").exists() else None
                            if sample["scope"]["memory.max"] != str(CAP) or sample["scope"]["memory.swap.max"] != "0":
                                raise RuntimeError("The server memory limits differ from the benchmark limits.")
                    failures = 0
                except (OSError, ValueError, StopIteration) as error:
                    failures += 1
                    sample["error"] = str(error)
                reason = guard_reason(sample.get("linux_available_bytes", FLOOR), int(sample.get("scope", {}).get("memory.swap.current", "0")), failures)
                if reason:
                    raise RuntimeError(reason)
                sample["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
                sample["cpu_load"] = os.getloadavg()[0]
                with (output / "resources.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample) + "\n")
                if not ready and scope is not None:
                    try:
                        live = localmodel.current(8899)
                        if live["model"] != "qwen3.8-27b" or live["context"] != 196608:
                            raise RuntimeError("The live model or context differs from the benchmark profile.")
                        ready = True
                        report.update(status="ready", startup_seconds=time.time() - report["started_at"], live=live)
                        save(output / "ready.json", report)
                        save(output / "status.json", report)
                    except (OSError, ValueError, KeyError):
                        pass
                    except RuntimeError as error:
                        if str(error) != "No local model is ready. Run localmodel start first.":
                            raise
                time.sleep(2)
            report["status"] = "stopped"
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            save(output / "aborted.json", {"reason": report["error"], "at": time.time()})
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    subprocess.run(["systemctl", "--user", "stop", unit], env=environment, timeout=20, check=False)
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            subprocess.run(["systemctl", "--user", "stop", unit], env=environment, timeout=20, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            report.update(ended_at=time.time(), process_exit=process.returncode, server_stopped=process.poll() is not None)
            save(output / "status.json", report)


if __name__ == "__main__":
    def stop(_number: int, _frame: Any) -> None:
        global stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    main()
