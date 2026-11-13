"""Manage the existing rental-GPU backend with a persistent localmodel-style CLI."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from types import FrameType
from typing import Any, Generator, TextIO, cast
import urllib.request

HERE = Path(__file__).resolve().parent
BACKEND = HERE / "start-remote-model.sh"
sys.path.insert(0, str(HERE))
MODELS = {"auto": "auto", "deepseek": "deepseek-auto", "mistral": "mistral-b300"}
HELP = """remotemodel COMMAND [MODEL] [options]

Commands:
  start [auto|deepseek|mistral]  Prepare missing files, then serve in the foreground.
  fetch [auto|deepseek|mistral]  Prepare weights and runtime without starting inference.
  list                         Show model families; no GPU or network is required.
  status                       Show the controller recorded for this root and port.
  stop                         Stop only that verified controller and its children.

Options:
  --root DIRECTORY             Persistent data/build root (default /workspace/remote-models).
  --port PORT                  Loopback API port (default 8897).
  --profile PROFILE            Use an existing hardware profile for this model family.
  --dry-run, --show-config     Preview the backend plan and stages without writes/downloads.
  --tui                        Show the shared Strata monitor; fetch also prepares its dependencies.
  --no-fetch                   Start only from prepared files, without preparation downloads/builds.
  --install-deps               Allow the backend to install system dependencies during preparation.
  --backend-help               Show every existing hardware, quantization, cache and build flag.

Existing backend flags pass through, including --context 1M, --quant NAME,
--spec dspark/none, --dspark N, --expert-ram-gb N, --gpu-count N and --build-jobs N.
Explicit --root overrides REMOTEMODEL_ROOT, then QWEN_ROOT. Build archives retain
the backend default /workspace/archive. Preparation reuses verified weights and builds.
Start does not set up SSH. DeepSeek/Strata and Mistral serving retain the backend's
outgoing-network block. There is no automatic fallback after startup fails.
With --tui: q closes the monitor; Ctrl-C stops this launcher's server.
Legacy invocations beginning with backend options retain the old script interface.
"""


def parser() -> argparse.ArgumentParser:
    """Parse wrapper controls while retaining backend options unchanged."""
    result = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    result.add_argument("--root", default=os.environ.get("REMOTEMODEL_ROOT", os.environ.get("QWEN_ROOT", "/workspace/remote-models")))
    result.add_argument("--port", type=int, default=int(os.environ.get("QWEN_PORT", "8897")))
    result.add_argument("--profile")
    result.add_argument("--dry-run", "--show-config", dest="dry_run", action="store_true")
    result.add_argument("--tui", action="store_true")
    result.add_argument("--no-fetch", action="store_true")
    result.add_argument("--install-deps", action="store_true")
    result.add_argument("--backend-help", action="store_true")
    result.add_argument("--help", "-h", action="store_true")
    return result


def read_plan(command: list[str]) -> dict[str, Any]:
    """Read complete JSON objects from the backend's read-only preview output."""
    result = subprocess.run([*command, "--show-config"], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr + result.stdout).strip())
    decoder = json.JSONDecoder()
    plans: list[dict[str, Any]] = []
    position = 0
    for line in result.stdout.splitlines(keepends=True):
        if line.startswith("{"):
            try:
                value, _ = decoder.raw_decode(result.stdout[position:])
            except ValueError:
                pass
            else:
                if isinstance(value, dict):
                    item = cast(dict[str, Any], value)
                    plan = item.get("plan", item)
                    if isinstance(plan, dict) and {"profile", "model_id", "engine", "port", "root"} <= plan.keys():
                        plans.append(item if "plan" in item else {"plan": item})
        position += len(line)
    if len(plans) != 1:
        raise RuntimeError("The backend did not return one complete model plan.")
    return plans[0]


def identity(pid: int) -> dict[str, Any]:
    """Identify one process across PID reuse and machine restarts."""
    process = Path("/proc") / str(pid)
    fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        raise ProcessLookupError(pid)
    return {"pid": pid, "start_ticks": int(fields[19]), "uid": process.stat().st_uid,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "command": [os.fsdecode(part) for part in (process / "cmdline").read_bytes().split(b"\0") if part]}


def owns(record: dict[str, Any]) -> bool:
    """Reject stale records and processes owned by another user."""
    saved = record.get("identity", {})
    try:
        pid = saved["pid"]
        return type(pid) is int and pid > 1 and saved["uid"] == os.getuid() and identity(pid) == saved
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return False


def read_record(path: Path) -> dict[str, Any] | None:
    """Read a regular controller record without following a link."""
    if path.is_symlink():
        raise RuntimeError(f"Controller record must not be a symbolic link: {path}")
    if not path.exists():
        return None
    value: Any = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid controller record: {path}")
    return cast(dict[str, Any], value)


def save_record(path: Path, record: dict[str, Any]) -> None:
    """Replace the private controller record atomically."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        json.dump(record, output, indent=2)
        output.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def controller(path: Path, plan: dict[str, Any], phase: str) -> Generator[dict[str, Any], None, None]:
    """Serialize management of one API port and retain process ownership."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another remotemodel operation owns this root and port.") from error
        previous = read_record(path)
        if previous and owns(previous):
            raise RuntimeError("A remotemodel controller is already running for this root and port.")
        record = {"identity": identity(os.getpid()), "phase": phase, "plan": plan}
        save_record(path, record)
        try:
            yield record
        finally:
            current = read_record(path)
            if current and current.get("identity") == record["identity"]:
                path.unlink()


def stop(path: Path) -> None:
    """Signal a verified PID handle; never reuse an unrelated process ID."""
    record = read_record(path)
    if not record or not owns(record):
        raise RuntimeError("No verified remotemodel controller is running for this root and port.")
    descriptor = os.pidfd_open(int(record["identity"]["pid"]))
    try:
        if not owns(record):
            raise RuntimeError("The recorded controller ended before it could be stopped.")
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    finally:
        os.close(descriptor)
    deadline = time.monotonic() + 60
    while owns(record) and time.monotonic() < deadline:
        time.sleep(0.2)
    if owns(record):
        raise RuntimeError("The controller is still stopping. Inspect its server log.")
    print("Stopped the remote model controller.")


def stop_child(child: subprocess.Popen[str]) -> None:
    """Stop only the backend process group created by this controller."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def run_child(command: list[str]) -> int:
    """Run preparation with signal-safe cleanup of its process group."""
    child = subprocess.Popen(command, text=True, start_new_session=True)
    try:
        return child.wait()
    finally:
        stop_child(child)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep readiness checks on their original loopback endpoint."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def healthy(port: int) -> bool:
    """Read only local readiness, without proxies or redirects."""
    request = urllib.request.Request(f"http://127.0.0.1:{port}/health")
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(request, timeout=2) as response:
            value: Any = json.loads(response.read(65537))
            return isinstance(value, dict) and cast(dict[str, Any], value).get("status") == "ok"
    except (OSError, ValueError):
        return False


def serve(command: list[str], path: Path, record: dict[str, Any], monitor_command: list[str] | None) -> int:
    """Keep the backend in the foreground and optionally attach telemetry."""
    output: TextIO | None = None
    if monitor_command is not None:
        output = path.with_suffix(".log").open("a", buffering=1)
        print(f"Starting the server; launch log: {output.name}", flush=True)
    try:
        child = subprocess.Popen(command, text=True, stdout=output,
                                 stderr=subprocess.STDOUT if output else None, start_new_session=True)
        try:
            record["phase"] = "serving"
            save_record(path, record)
            if monitor_command is not None:
                deadline = time.monotonic() + 3600
                while child.poll() is None and not healthy(int(record["plan"]["port"])):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Server readiness timed out. Inspect the launch log.")
                    time.sleep(0.5)
                if child.poll() is not None:
                    raise RuntimeError(f"Backend exited before readiness. Inspect {path.with_suffix('.log')}.")
                monitor_child = subprocess.Popen(monitor_command, text=True, start_new_session=True)
                try:
                    while monitor_child.poll() is None:
                        if child.poll() is not None:
                            raise RuntimeError(f"Backend exited while the monitor was open. Inspect {path.with_suffix('.log')}.")
                        time.sleep(0.2)
                    result = monitor_child.wait()
                finally:
                    stop_child(monitor_child)
                if result == 130:
                    return 130
                if result:
                    raise RuntimeError(f"The Strata monitor exited with status {result}.")
                print("Monitor closed. The server is still running; Ctrl-C stops it.", flush=True)
            return child.wait()
        finally:
            stop_child(child)
    finally:
        if output is not None:
            output.close()


def main(argv: list[str] | None = None) -> int:
    """Dispatch localmodel-style commands through the existing remote backend."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments in (["--help"], ["-h"], ["help"]):
        print(HELP)
        return 0
    if arguments == ["--backend-help"]:
        return subprocess.call(["bash", str(BACKEND), "--help"])
    if arguments[0].startswith("-"):
        os.execvp("bash", ["bash", str(BACKEND), *arguments])
    action = arguments.pop(0)
    if action not in ("start", "fetch", "list", "status", "stop"):
        raise RuntimeError(f"Unknown command: {action}. Run remotemodel --help.")
    model = arguments.pop(0) if action in ("start", "fetch") and arguments and not arguments[0].startswith("-") else "auto"
    if model not in MODELS:
        raise RuntimeError(f"Unknown model family: {model}. Run remotemodel list.")
    args, extra = parser().parse_known_args(arguments)
    if args.help:
        print(HELP)
        return 0
    if args.backend_help:
        return subprocess.call(["bash", str(BACKEND), "--help"])
    if not 1024 <= args.port <= 65535:
        raise RuntimeError("The API port must be between 1024 and 65535.")
    root = Path(str(args.root)).expanduser().absolute()
    path = root / "state" / f"remotemodel-{args.port}.json"
    if action in ("list", "status", "stop"):
        if extra or args.profile or args.tui or args.no_fetch or args.install_deps or args.dry_run:
            raise RuntimeError(f"{action} accepts only --root and --port.")
        if action == "list":
            print("auto       Detect DeepSeek/Strata or single-B300 Mistral.\ndeepseek   DeepSeek 1017 on Strata-supported GPUs.\nmistral    Mistral Large 4 on one B300.\nUse --profile for existing explicit hardware/legacy profiles; see --backend-help.")
        elif action == "stop":
            stop(path)
        else:
            record = read_record(path)
            print(json.dumps({"running": bool(record and owns(record)), "state": record}, indent=2))
        return 0
    blocked = {"--prepare-only", "--prepare-assets-only", "--smoke-test", "--smoke-full-context", "--smoke-long",
               "--archive-build-only", "--ssh-only", "--setup-ssh", "--ornith-prepare-only", "--deepseek-prepare-only",
               "--ornith-smoke-test", "--deepseek-smoke-test", "--ornith-dry-run", "--deepseek-dry-run"}
    if any(item.split("=", 1)[0] in blocked for item in extra):
        raise RuntimeError("Use start/fetch and --dry-run for lifecycle control; use the legacy script for archive-only, smoke or SSH actions.")
    if args.no_fetch and (action != "start" or args.install_deps):
        raise RuntimeError("--no-fetch requires start and cannot be combined with --install-deps.")
    if args.tui and action == "start" and not args.dry_run and not sys.stdin.isatty():
        raise RuntimeError("--tui requires an interactive terminal. Run without it or use SSH with a TTY.")
    profile = str(args.profile or MODELS[model])
    command = ["bash", str(BACKEND), "--root", str(root), "--port", str(args.port), "--profile", profile, "--no-ssh", *extra]
    preview = read_plan(command)
    plan = cast(dict[str, Any], preview["plan"])
    if model != "auto" and not str(plan["profile"]).startswith(model):
        raise RuntimeError(f"The selected profile does not serve {model}; no model was prepared or started.")
    if args.tui and plan["engine"] != "strata":
        raise RuntimeError("--tui currently requires a Strata profile. This hardware/profile uses another engine.")
    if args.no_fetch and plan["engine"] != "strata" and plan["profile"] != "mistral-b300":
        raise RuntimeError("--no-fetch requires Strata or Mistral; legacy profiles can build or download during serving.")
    preparation = [*command, "--prepare-only", *(["--install-deps"] if args.install_deps else [])]
    if args.dry_run:
        print(json.dumps({"backend": preview, "prepare": None if args.no_fetch else preparation,
                          "serve": command if action == "start" else None, "tui": bool(args.tui)}, indent=2))
        return 0
    if args.tui and not args.no_fetch:
        import monitor
        monitor.creation_python()
    with controller(path, plan, "preparing" if not args.no_fetch else "starting") as record:
        if action == "start":
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                    raise RuntimeError(f"Port {args.port} is already in use; no existing service was interrupted.")
        if not args.no_fetch:
            result = run_child(preparation)
            if result:
                return result
        monitor_command = None
        if args.tui:
            import monitor
            monitor_command = monitor.prepare(preview, install=not args.no_fetch)
        if action == "fetch":
            print("Preparation complete. No model server was started.")
            return 0
        return serve(command, path, record, monitor_command)


def interrupted(_signum: int, _frame: FrameType | None) -> None:
    """Unwind the controller so its owned child processes are stopped."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped this launcher's child processes.", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"remotemodel: {error}", file=sys.stderr)
        raise SystemExit(1) from error
