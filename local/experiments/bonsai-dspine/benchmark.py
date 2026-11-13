"""Compare an isolated Bonsai server. Preview unless --run is set."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/bonsai-next"))
base = importlib.import_module("benchmark")
Probe: Any = importlib.import_module("native_probe").Probe

Record = dict[str, Any]
GUARD = ROOT / "experiments/deepseek-strata/network_guard.py"


def argument(command: list[str], *names: str) -> str:
    """Read one required option without accepting ambiguous duplicates."""
    positions = [index for index, value in enumerate(command) if value in names]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"Supply exactly one value for {names}.")
    return command[positions[0] + 1]


class Experiment(Probe):  # pyright: ignore[reportUntypedBaseClass]
    """Reuse OMP fixtures with a private, network-restricted native server."""

    def __init__(self, args: argparse.Namespace, launch: Record):
        launch = {"model": "bonsai2-27b", "memory_gib": 40, "windows_floor_gib": 4, **launch}
        Probe.__init__(self, args, launch)
        command = cast(list[str], launch["command"])
        self.capacity = int(argument(command, "--ctx-size", "-c"))
        if argument(command, "--port") != "8899" or argument(command, "--host") != "127.0.0.1":
            raise ValueError("Use the private loopback endpoint 127.0.0.1:8899.")
        if argument(command, "--alias", "-a") != self.model_name or self.model_name != "bonsai2-27b":
            raise ValueError("Use the bonsai2-27b alias for the existing OMP fixture.")
        if not Path(command[0]).is_absolute() or Path(command[0]).name != "llama-server":
            raise ValueError("Use an absolute llama-server path, without shell or cap wrappers.")
        if "--metrics" not in command:
            raise ValueError("Enable --metrics so the existing OMP counters are available.")
        if args.mode == "omp" and self.capacity != 262144:
            raise ValueError("The reused OMP launcher requires 262144 context capacity.")
        self.unit = f"bonsai-dspine-model-{os.getpid()}.scope"
        self.environment.update(CAPPED_UNIT=self.unit)
        self.lock: Any = None
        self.model: subprocess.Popen[Any] | None = None
        self.report.update(context_capacity=self.capacity, mode=args.mode,
                           network_policy="No outgoing connections; accepted loopback replies remain available")
        if args.mode == "screen":
            self.report.update(target_input_tokens=None, temperature=0, thinking="off",
                               history="Three short prompts; no filled conversation history.")

    def prepare(self) -> None:
        """Let the 16K fixture compact while retaining its recent tool results."""
        base.Benchmark.prepare(self)
        if self.args.history_tokens == 16384:
            path = self.output / "config/omp.json"
            settings = json.loads(path.read_text())
            settings["compaction"]["keepRecentTokens"] = 4096
            base.save(path, settings)

    def start_model(self) -> None:
        """Hold the model lock, start one capped process, and verify its ownership."""
        self.lock = (ROOT / "state/server.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError("Another managed model operation holds server.lock") from None
        if base.ACTIVE.exists():
            raise RuntimeError("Managed model state exists; this experiment does not adopt it")
        base.localmodel.check_free(8899)
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 8899))
        first = base.available()
        if first["linux_available_bytes"] < 8 * 2**30 or first["windows_available_bytes"] < self.windows_floor_gib * 2**30:
            raise RuntimeError("Available RAM is below the Linux 8 GiB or configured Windows floor")
        command = cast(list[str], self.launch["command"])
        self.report["runtime_binary_sha256"] = hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()
        wrapped = [str(base.CAPPED), "--mem", f"{self.memory_gib}G", "--time", f"{self.args.total_timeout}s",
                   str(base.PYTHON), str(GUARD), *command]
        self.report["executed_command"] = wrapped
        self.server_log = (self.output / "server.log").open("w")
        self.model = subprocess.Popen(wrapped, cwd=ROOT, env=self.environment,
            stdin=subprocess.DEVNULL, stdout=self.server_log, stderr=subprocess.STDOUT, start_new_session=True)
        threading.Thread(target=self.monitor, daemon=True).start()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            self.check()
            if self.model.poll() is not None:
                raise RuntimeError("Model startup failed; inspect server.log")
            try:
                if base.localmodel.request(8899, "/health").get("status") == "ok":
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError("The isolated server did not become ready in four minutes")
        group = subprocess.check_output(["systemctl", "--user", "show", self.unit,
            "--property=ControlGroup", "--value"], env=self.environment, text=True).strip()
        if not group:
            raise RuntimeError("The owned model scope has no cgroup")
        self.scope = Path("/sys/fs/cgroup") / group.lstrip("/")
        limits = {name: (self.scope / name).read_text().strip() for name in ("memory.max", "memory.swap.max")}
        if limits != {"memory.max": str(self.memory_gib * 2**30), "memory.swap.max": "0"}:
            raise RuntimeError(f"Incorrect actual model limits: {limits}")
        binary = Path(command[0]).resolve()
        pids: list[int] = []
        for value in (self.scope / "cgroup.procs").read_text().split():
            try:
                if (Path("/proc") / value / "exe").resolve() == binary:
                    pids.append(int(value))
            except OSError:
                pass
        if len(pids) != 1:
            raise RuntimeError(f"Expected exactly one owned llama-server in the scope; found {pids}")
        state = base.localmodel.endpoint(8899)
        if state["model"] != self.model_name or state["context"] != self.capacity:
            raise RuntimeError(f"The live endpoint does not match this case: {state}")
        status = (Path("/proc") / str(pids[0]) / "status").read_text()
        security = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                    for line in status.splitlines() if line.startswith(("Seccomp:", "Seccomp_filters:", "NoNewPrivs:"))}
        if security.get("Seccomp") != "2" or security.get("NoNewPrivs") != "1":
            raise RuntimeError(f"The owned process has no effective seccomp filter: {security}")
        base.save(self.output / "server-props.json", base.localmodel.request(8899, "/props"))
        self.report.update(live_settings=state, cgroup=str(self.scope), limits=limits,
                           server_pid=pids[0], security=security)
        base.save(self.output / "summary.json", self.report)

    def stop_model_locked(self) -> None:
        """Stop only this experiment's unique scope and child process group."""
        if self.model is not None:
            subprocess.run(["systemctl", "--user", "stop", self.unit], env=self.environment,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=False)
            try:
                self.model.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.model.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.model.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.model.pid, signal.SIGKILL)
                    self.model.wait(timeout=5)
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def screen(self) -> None:
        """Record short greedy code, math, and prose fixtures with correctness checks."""
        self.workspace.mkdir()
        code = self.chat("screen-code", {"max_tokens": 768, "messages": [{"role": "user", "content":
            "Return only Python source for these three functions, with no code fences or imports. "
            "clamp_index(index,length) clamps into [0,length-1], raising ValueError for length<=0. "
            "clamp_many(values,lower,upper) returns a new list clamped into [lower,upper], preserving the input, "
            "and raises ValueError for reversed bounds. clamp_span(start,end,length) independently clamps both "
            "endpoints into [0,length], and raises ValueError for negative length or start>end. "
            "Use only min, max and ValueError calls. No annotations, defaults, decorators, or helper functions."}]})
        math = self.chat("screen-math", {"max_tokens": 192, "messages": [{"role": "user", "content":
            "There are 37 crates with 24 items each. Ship 19 crates, then add 75 loose items. "
            "Return only JSON with remaining_items and shipped_items as integers."}]})
        prose = self.chat("screen-prose", {"max_tokens": 256, "messages": [{"role": "user", "content":
            "Mira owns the release. The deadline is Friday. Two checks remain: validate cancellation recovery "
            "and preserve the revision marker through compaction. Return only JSON with owner, deadline, "
            "and checks. Use a list containing exactly those two check descriptions."}]})
        source = code["message"]["content"] or ""
        safe = False
        try:
            tree = ast.parse(source)
            safe = len(tree.body) == 3 and all(isinstance(node, ast.FunctionDef) and not node.decorator_list
                and not node.args.defaults for node in tree.body)
            safe &= not any(isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute, ast.Global, ast.Nonlocal))
                or isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                or node.func.id not in {"min", "max", "ValueError"}) for node in ast.walk(tree))
        except SyntaxError:
            pass
        result = None
        if safe:
            (self.workspace / "clamp.py").write_text(source)
            result = subprocess.run([str(base.CAPPED), "--mem", "256M", "--time", "30s", str(base.PYTHON),
                str(GUARD), str(base.PYTHON), "-c", base.VALIDATOR, "2"], cwd=self.workspace,
                env=dict(self.environment, CAPPED_UNIT=f"bonsai-dspine-check-{os.getpid()}.scope"),
                capture_output=True, text=True, timeout=35)
        try:
            math_ok = json.loads(math["message"]["content"]) == {"remaining_items": 507, "shipped_items": 456}
        except (TypeError, ValueError):
            math_ok = False
        try:
            prose_ok = json.loads(prose["message"]["content"]) == {
                "owner": "Mira", "deadline": "Friday", "checks": [
                    "validate cancellation recovery", "preserve the revision marker through compaction"]}
        except (TypeError, ValueError):
            prose_ok = False
        self.report["fixture_checks"] = {"code_semantics": result is not None and result.returncode == 0,
                                         "math": math_ok, "prose_fields": prose_ok}
        self.report["code_validator"] = {"safe_ast": safe, "stdout": result.stdout if result else "",
                                         "stderr": result.stderr if result else ""}
        self.report["all_quality_checks_passed"] = all(self.report["fixture_checks"].values())

    def run(self) -> None:
        """Use the existing complete OMP workflow or the bounded native screen."""
        if self.args.mode == "omp":
            base.Benchmark.run(self)
        else:
            Probe.run(self)


def main() -> None:
    """Preview or run an explicit native-server command without changing defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("screen", "omp"), default="screen")
    parser.add_argument("--history-tokens", type=base.localmodel.tokens, choices=(16384, 65536, 196608), default=16384)
    parser.add_argument("--phase-timeout", type=int, default=900)
    parser.add_argument("--total-timeout", type=int, default=2400)
    parser.add_argument("--temperature", type=float, default=1.0, help="OMP temperature; the native screen is greedy.")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    launch = cast(Record, json.loads(args.launch_file.read_text()))
    command = launch.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in cast(list[Any], command)):
        parser.error("The launch file must contain a nonempty native command array")
    if args.output.exists() or not 30 <= args.phase_timeout <= 900 or not 60 <= args.total_timeout <= 7200:
        parser.error("Use a new output directory and bounded phase/total deadlines")
    experiment = Experiment(args, launch)
    if not args.run:
        print(json.dumps({"status": "preview-only", "launch": experiment.launch, "mode": args.mode,
            "output": str(args.output), "context_capacity": experiment.capacity,
            "history_tokens": args.history_tokens if args.mode == "omp" else None,
            "model_memory_gib": experiment.memory_gib, "omp_memory_gib": 4, "swap": 0,
            "linux_floor_gib": 8, "windows_floor_gib": experiment.windows_floor_gib,
            "network_guard": str(GUARD)}, indent=2))
        return
    os.umask(0o077)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda _number, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    experiment.run()
    if not experiment.report.get("all_quality_checks_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
