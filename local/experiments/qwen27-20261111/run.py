"""Run the existing filled OMP repair/resume fixture against an owned Qwen server."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "experiments/deepseek-strata/filled_omp.py"
spec = importlib.util.spec_from_file_location("filled_omp_qwen27", FIXTURE)
assert spec and spec.loader
fixture: Any = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
fixture.PYTHON = ROOT / ".venv/bin/python"
MODEL = "qwen3.8-27b"
CONTEXT = 196608
Record = dict[str, Any]


def request(path: str, body: Record | None = None) -> Any:
    """Read native loopback metadata or render a prompt without inference."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request("http://127.0.0.1:8899" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=30) as response:
        raw = response.read().decode()
    return raw if path == "/metrics" else json.loads(raw)


def metrics() -> dict[str, float]:
    """Read native counters, including any available draft counters."""
    return {line.rsplit(None, 1)[0]: float(line.rsplit(None, 1)[1])
            for line in request("/metrics").splitlines() if line.startswith("llamacpp:")}


def check_identity(active: Record, controller: int, speculative: str) -> None:
    """Reject a different server before submitting a request."""
    expected = {"controller_pid": controller, "model": MODEL, "context": CONTEXT,
                "runtime": "llamacpp", "speculative": speculative, "port": 8899}
    if any(active.get(key) != value for key, value in expected.items()):
        raise RuntimeError("The active server does not match the parent-owned benchmark arm.")
    if active.get("memory_limit") not in ("48G", "49152M"):
        raise RuntimeError("The server must have the fixed 48 GiB memory cap.")
    if speculative == "mtp" and active.get("draft_tokens") != 2:
        raise RuntimeError("The MTP arm must use exactly two draft tokens.")


class Benchmark(fixture.Benchmark):
    """Reuse the fixture with native llama.cpp token counting and process counters."""

    def __init__(self, args: argparse.Namespace):
        self.output, self.targets, self.timeout = args.output.resolve(), [16384], args.timeout
        self.model = MODEL
        self.workspace = self.output / "workspace"
        self.session = self.output / "sessions/conversation.jsonl"
        self.cursor = self.part = self.counter = 0
        self.current_phase = "prepare"
        self.started = time.time()
        self.controller = args.controller_pid
        self.speculative = args.speculative
        self.server_output = args.server_output.resolve()
        self.report: Record = {"status": "preparing", "targets": self.targets, "thinking": "xhigh",
            "output_limit": 4096, "checks": [], "memory_floor_gib": 8, "server_context_capacity": CONTEXT,
            "model": MODEL, "speculative": self.speculative, "temperature": 1.0, "top_p": 0.95,
            "seed": None, "seed_note": "Unset; the runtime selects a seed per request.",
            "server_output": str(self.server_output),
            "history": "Existing constructed public-source OMP repair and strict resume fixture."}
        paths = sorted((ROOT / "state/sources").glob("prism-e3de715-*.cpp"))
        if not paths:
            raise RuntimeError("The existing benchmark public-source corpus is missing.")
        self.corpus = "\n".join(f"\nPublic source file: {path.name}\n" + path.read_text() for path in paths)
        self.report["sources"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
        self.report["fixture_sha256"] = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        self.facts = {"16384": {"release_key": "release-qwen27-20261111", "target_branch": "branch-qwen27-comparison",
                                 "deadline_tag": "deadline-2026-11-12"}}

    def check(self) -> Record:
        """Check parent ownership and the Linux memory floor between requests."""
        if time.time() - self.started > 2 * self.timeout + 120:
            raise TimeoutError("The complete workload exceeded its bounded run time.")
        if (self.server_output / "aborted.json").exists():
            raise RuntimeError("The parent memory guard stopped the server.")
        active: Record = json.loads((ROOT / "state/active.json").read_text())
        check_identity(active, self.controller, self.speculative)
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        if available < 8 * 2**30:
            raise RuntimeError("Available Linux memory is below 8 GiB.")
        return active

    def count(self, body: Record) -> int:
        """Count the actual request with the running server's chat template."""
        self.check()
        prompt = request("/apply-template", body)["prompt"]
        return len(request("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"])

    def sample(self, phase: str) -> None:
        """Record process memory and the actual cgroup limits."""
        active = self.check()
        process = Path("/proc") / str(active["pid"])
        group = next(line[3:] for line in (process / "cgroup").read_text().splitlines() if line.startswith("0::"))
        scope = Path("/sys/fs/cgroup") / group.lstrip("/")
        memory = {key: (scope / key).read_text().strip() for key in
                  ("memory.current", "memory.peak", "memory.max", "memory.swap.max", "memory.swap.current", "memory.events")}
        if memory["memory.max"] != str(48 * 2**30) or memory["memory.swap.max"] != "0" or memory["memory.swap.current"] != "0":
            raise RuntimeError("The live server lacks the fixed memory cap or zero-swap boundary.")
        row = {"at": time.time(), "phase": phase, "model_memory": memory,
               "process_status": (process / "status").read_text(),
               "process_io": (process / "io").read_text() if (process / "io").exists() else None}
        with (self.output / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def run_omp(self, prompt: str, phase: str, capture: bool = False) -> Record:
        """Retain native timings around the unchanged real OMP workflow."""
        self.check()
        before = {} if capture else metrics()
        delta: dict[str, float] = {}
        try:
            result: Record = fixture.Benchmark.run_omp(self, prompt, phase, capture)
        finally:
            if not capture:
                try:
                    after = metrics()
                    delta = {key: value - before.get(key, 0) for key, value in after.items() if "_total" in key}
                    evidence: Record = {"before": before, "after": after, "delta": delta}
                except (OSError, ValueError) as error:
                    evidence = {"before": before, "error": str(error)}
                fixture.save(self.output / f"{phase}-native.json", evidence)
        if capture:
            tools: list[Record] = result.get("tools", [])
            edits: list[Record] = [tool["function"]["parameters"] for tool in tools if tool.get("function", {}).get("name") == "edit"]
            if len(edits) != 1 or not {"path", "old_string", "new_string"}.issubset(edits[0].get("required", [])):
                raise RuntimeError("The captured edit tool is not the guarded replacement format.")
            if result.get("temperature") != 1.0 or result.get("top_p") != 0.95 or not 1 <= int(result.get("max_tokens", 0)) <= 4096:
                raise RuntimeError("The captured request changed sampling or exceeded the output bound.")
            if result.get("seed") is not None:
                raise RuntimeError("The captured request changed the unseeded sampling setting.")
        else:
            result["native_metrics_delta"] = delta
            result["actual_input_tokens_rendered"] = result.pop("actual_input_tokens_offline")
            fixture.save(Path(result["artifacts"]).with_suffix(".summary.json"), result)
        return result


def stop_owned_worker(output: Path) -> None:
    """Stop only an OMP worker whose environment names this private output."""
    worker = output / "worker.json"
    if not worker.exists():
        return
    pid = int(json.loads(worker.read_text())["pid"])
    process = Path("/proc") / str(pid)
    try:
        if process.stat().st_uid != os.getuid() or os.fsencode("OMP_BENCH_ROOT=" + str(output)) not in (process / "environ").read_bytes().split(b"\0"):
            return
        subprocess.run(["systemctl", "--user", "stop", f"omp-local-{pid}.scope"], timeout=20, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if process.exists():
            os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, FileNotFoundError):
        pass


def main() -> None:
    """Preview by default; attach only to an explicitly identified owned server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-output", type=Path, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--speculative", choices=("none", "mtp"), required=True)
    parser.add_argument("--timeout", type=int, default=1200, help="Maximum seconds per real OMP process.")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.controller_pid < 1 or not 60 <= args.timeout <= 1800:
        parser.error("Use a positive controller PID and a 60..1800 second phase limit.")
    if not args.run:
        print(json.dumps({"mode": "preview", "model": MODEL, "context": CONTEXT, "filled_input_target": 16384,
            "speculative": args.speculative, "phases": ["repair with recall", "strict resume"], "memory_cap_gib": 48,
            "linux_floor_gib": 8, "windows_memory_checks": False, "server_owned_by": "parent server.py"}, indent=2))
        return
    os.environ.update(PI_EDIT_VARIANT="replace", XDG_RUNTIME_DIR="/run/user/1000",
                      DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus", OMP_BENCH_MAX_REQUESTS="8")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "aborted.json").symlink_to(args.server_output.resolve() / "aborted.json")
    benchmark = Benchmark(args)
    try:
        benchmark.sample("admission")
        benchmark.run()
    finally:
        stop_owned_worker(args.output)


if __name__ == "__main__":
    main()
