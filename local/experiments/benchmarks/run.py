"""Plan local model experiments offline. Only --run can start benchmark services."""
# pyright: strict

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from typing import Any, cast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / ".venv/bin/python"
MANAGER = ROOT / "bin/localmodel"
Record = dict[str, Any]


def read(path: Path) -> Record:
    """Read one local JSON object."""
    return cast(Record, json.loads(path.read_text()))


def commands(case: Record, directory: Path, hooks: list[str], timeout: int) -> Record:
    """Build commands without loading a runtime or inspecting an endpoint."""
    model = str(case["model"])
    launch = [str(MANAGER), "start", model, "--context", "256k", "--port", "8898", "--no-offload",
              "--cache-ram", "32768", "--kv-cache", str(case["kv"])]
    if model == "dory-gguf":
        launch += ["--rounds", str(case["rounds"]), "--batch-tokens", "2048", "--ubatch-tokens", "1024"]
    else:
        launch += ["--runtime", str(case["runtime"]), "--speculative", "mtp", "--draft-tokens", str(case["draft_tokens"]),
                   "--draft-kv-cache", str(case.get("draft_kv", "q4_0")), "--mtp-chain" if case["chain"] else "--no-mtp-chain",
                   "--reasoning-effort", str(case["effort"]), "--batch-tokens", "256", "--ubatch-tokens", "64"]
    launch += cast(list[str], case["flags"])
    common = [str(PYTHON), str(ROOT / "results/opencode-mtp-long-bench.py"), "--port", "8898", "--model", model,
              "--base-url", "http://127.0.0.1:8895/v1", "--source-tokens", "205000", "--timeout", str(timeout),
              "--phase-file", str(directory / "proxy/phase.json"), "--output", str(directory / "opencode.json")]
    if case.get("effort"):
        common += ["--reasoning-effort", str(case["effort"])]
    choices = {
        "opencode": common,
        "capacity": [str(PYTHON), str(ROOT / "results/check-bonsai.py"), "capacity", "--port", "8895", "--model", model,
                     "--context", "262144", "--prompt-tokens", "262079", "--output-tokens", "64", "--seed", "424242",
                     "--timeout", str(timeout), "--output", str(directory / "capacity.json")],
        "grammar107": [str(PYTHON), str(HERE / "replay-tools.py"), "--run", "--model", model,
                       "--output", str(directory / "grammar107.json"), "--timeout", str(timeout)],
    }
    if case.get("effort"):
        choices["grammar107"] += ["--reasoning-effort", str(case["effort"])]
    return {"launch": launch, "hooks": {name: choices[name] for name in hooks}}


def offline(matrix: Record, selected: list[Record], destination: Path, hooks: list[str], timeout: int) -> Record:
    """Validate local inputs only. Never call a model or a compiler."""
    paths = [Path(__file__), HERE / "matrix.json", ROOT / "localmodel.py", ROOT / "opencode_local.py",
             ROOT / "results/run-27b-mtp-chain.py", ROOT / "results/mtp-capture-proxy.py",
             ROOT / "results/opencode-mtp-long-bench.py", ROOT / "results/check-bonsai.py", HERE / "replay-tools.py",
             HERE / "tools-107.json", MANAGER, PYTHON]
    absent = [str(path) for path in paths if not path.exists()]
    if absent:
        raise ValueError("Missing harness files: " + ", ".join(absent))
    names = [case["name"] for case in matrix["cases"]]
    if len(names) != len(set(names)):
        raise ValueError("Matrix case names must be unique.")
    models = read(ROOT / "config/models.json")
    model_status: Record = {}
    runtime_status: Record = {}
    for case in selected:
        model = str(case["model"])
        item: Record = models.get(model, {})
        if case.get("draft_tokens"):
            item = item.get("mtp", item)
        weight = ROOT / "bonsai/models" / str(item.get("directory", "")) / str(item.get("filename", "missing"))
        model_status[model] = {"catalog_entry_present": bool(item), "weights_present": weight.is_file()}
        runtime = str(case.get("runtime", "dory-native"))
        if runtime == "stock":
            release = read(ROOT / "config/runtime.json")["release"]
            binary = ROOT / "bonsai/runtime" / release / ("llama-" + release) / "llama-server"
        else:
            binary = ROOT / "state/builds" / runtime / "bin/llama-server"
        runtime_status[runtime] = {"binary_present": binary.is_file(),
                                  "manifest_present": (ROOT / "experiments" / runtime / "manifest.json").is_file() if runtime != "stock" else None,
                                  "build_record_present": (binary.parent.parent / "localmodel-build.json").is_file() if runtime != "stock" else None}
    return {"mode": "offline plan; nothing started", "context": 262144,
            "workload": "staged-fixture-v2 with constructed public source history; actual request token counts are authoritative",
            "matrix_version": matrix["version"], "assets": model_status, "runtimes": runtime_status,
            "case_plans": [{"name": case["name"], "stage": case["stage"], "conditional": case.get("conditional"),
                            "commands": commands(case, destination / str(case["name"]), hooks, timeout)} for case in selected],
            "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.suffix in (".py", ".json")}}


def refuse_services() -> None:
    """Require an unused managed model state and unused private ports."""
    if (ROOT / "state/active.json").exists():
        raise RuntimeError("A managed model state exists. This runner never adopts or stops an existing service.")
    for port in (8898, 8895, 8896, 8899):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"Private benchmark port {port} is in use.") from error
    for path in Path("/proc").glob("[0-9]*/comm"):
        try:
            name = path.read_text().strip()
        except (OSError, PermissionError):
            continue
        if name == "ollama":
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open("http://127.0.0.1:11434/api/ps", timeout=3) as response:
                    idle = json.load(response).get("models") == []
            except Exception:
                idle = False
            if idle:
                continue
        if name in {"llama-server", "vllm", "ollama"}:
            raise RuntimeError("An existing inference service is running. Stop it explicitly before benchmarking.")


def windows_memory(directory: Path, environment: dict[str, str], helpers: Any) -> Record:
    """Read bounded Windows counters only after the model reports idle."""
    output = directory / "windows-memory.json"
    try:
        slots: list[Record] = helpers.get(8898, "/slots", timeout=3)
        if not slots or any(slot.get("is_processing") is not False for slot in slots):
            return {"status": "skipped_idle_unconfirmed"}
        diagnostic_env = dict(environment)
        diagnostic_env.update(CAPPED_UNIT=f"localmodel-windows-memory-{os.getpid()}-{time.monotonic_ns()}.scope",
                              HEAVY_PEAKS=str(directory / "windows-memory-peaks"))
        command = [str(ROOT.parent / "heavy-management/bin/capped"), "--mem", "1G", "--time", "60s",
                   str(PYTHON), str(ROOT / "results/sample-mtp-windows-gpu.py"), "--output", str(output)]
        with (directory / "windows-memory.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, env=diagnostic_env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=65, check=False)
        sample = read(output) if output.exists() else {}
        return {"status": sample.get("status", "error"), "summary": sample.get("summary", []),
                "error": sample.get("error"), "returncode": completed.returncode, "artifact": str(output)}
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}", "artifact": str(output)}


def run_case(case: Record, plan: Record, directory: Path, timeout: int, sample_windows: bool = False) -> Record:
    """Reuse the existing recorder, scope monitor and owned model cleanup."""
    spec = importlib.util.spec_from_file_location("sweep_helpers", ROOT / "results/run-27b-mtp-chain.py")
    assert spec is not None and spec.loader is not None
    helpers: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    refuse_services()
    directory.mkdir(parents=True, exist_ok=False)
    proxy_directory = directory / "proxy"
    proxy_directory.mkdir()
    row: Record = {"name": case["name"], "case": case, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                   "commands": plan, "resources": {}, "hooks": {}}
    model: subprocess.Popen[Any] | None = None
    proxy: subprocess.Popen[Any] | None = None
    worker: subprocess.Popen[Any] | None = None
    handles: list[Any] = []
    controller: int | None = None
    stopped = threading.Event()
    sampler: threading.Thread | None = None
    started = time.monotonic()
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("LLAMA_MTP_") or key in {"LAB_SPEC_DRAFT_N_MAX", "GGML_SYCL_KNOBS_FILE", "GGML_CUDA_FA_MMA_DECODE_MIN_KV"}:
            environment.pop(key)
    unit = f"localmodel-experiment-{case['name']}-{os.getpid()}.scope"
    environment.update(CAPPED_UNIT=unit, HEAVY_PEAKS=str(directory / "model-peaks"), LLAMA_MTP_DRAFT_TIMING="1",
                       LLAMA_ARG_LOG_VERBOSITY="4")
    row["diagnostic_environment"] = {key: environment[key] for key in ("LLAMA_MTP_DRAFT_TIMING", "LLAMA_ARG_LOG_VERBOSITY")}
    try:
        log = (directory / "server.log").open("w")
        handles.append(log)
        model = subprocess.Popen(plan["launch"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + 240
        while True:
            active = ROOT / "state/active.json"
            if active.exists():
                try:
                    state = read(active)
                except ValueError:
                    continue
                candidate = state.get("controller_pid")
                if not isinstance(candidate, int) or os.getpgid(candidate) != model.pid or state.get("port") != 8898:
                    raise RuntimeError("The active model is not owned by this trial.")
                controller = candidate
            try:
                if controller is not None and helpers.get(8898, "/health").get("status") == "ok":
                    break
            except OSError:
                pass
            if model.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Model startup failed or timed out. No build or download is attempted.")
            time.sleep(.5)
        row["live_settings"] = read(ROOT / "state/active.json")
        expected = {"model": case["model"], "context": 262144, "cache_ram_mib": 32768, "kv_cache": case["kv"]}
        if case.get("draft_tokens"):
            flags: list[str] = case["flags"]
            expected.update(runtime=case["runtime"], draft_tokens=case["draft_tokens"], mtp_chain=case["chain"],
                            reasoning_effort=case["effort"], prompt_lookup="--prompt-lookup" in flags,
                            draft_kv_cache=case.get("draft_kv", "q4_0"))
            for option, key in (("--draft-window", "draft_window"), ("--kv-vram-cells", "kv_vram_cells"),
                                ("--mma-decode-min-kv", "mma_decode_min_kv"), ("--lookup-tokens", "lookup_tokens")):
                expected[key] = int(flags[flags.index(option) + 1]) if option in flags else None
            expected["lookup_type"] = flags[flags.index("--lookup-type") + 1] if "--lookup-type" in flags else None
        else:
            expected["rounds"] = case["rounds"]
        if any(row["live_settings"].get(key) != value for key, value in expected.items()):
            raise RuntimeError("The actual model settings differ from the named case.")
        group_name = subprocess.check_output(["systemctl", "--user", "show", unit, "--property=ControlGroup", "--value"], text=True).strip()
        group = Path("/sys/fs/cgroup") / group_name.lstrip("/")
        row["cgroup"] = str(group)
        row["limits"] = {field: (group / field).read_text().strip() for field in ("memory.max", "memory.swap.max")}
        if row["limits"]["memory.swap.max"] != "0" or row["limits"]["memory.max"] == "max":
            raise RuntimeError("The inference scope must have a finite RAM cap and no swap.")
        sampler = threading.Thread(target=helpers.monitor, args=(group, directory, stopped, row["resources"]), daemon=True)
        sampler.start()
        log = (directory / "proxy.log").open("w")
        handles.append(log)
        proxy = subprocess.Popen([str(PYTHON), str(ROOT / "results/mtp-capture-proxy.py"), "--port", "8895", "--upstream-port", "8898",
                                  "--output-dir", str(proxy_directory)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 15
        while True:
            try:
                if helpers.get(8895, "/health").get("status") == "ok":
                    break
            except OSError:
                pass
            if proxy.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("The recording proxy did not initialize.")
            time.sleep(.2)
        for name, command in plan["hooks"].items():
            helpers.save(proxy_directory / "phase.json", {"name": name})
            helpers.save(directory / "trial.json", row)
            log = (directory / f"{name}.log").open("w")
            handles.append(log)
            worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = worker.wait(timeout=timeout * 6 + 300 if name == "opencode" else timeout + 120)
            except subprocess.TimeoutExpired:
                worker.terminate()
                code = worker.wait(timeout=30)
            row["hooks"][name] = {"returncode": code, "result": read(directory / f"{name}.json") if (directory / f"{name}.json").exists() else None}
            worker = None
            helpers.save(directory / "trial.json", row)
        row["status"] = "ok" if all(item["returncode"] == 0 for item in row["hooks"].values()) else "failed"
        if sample_windows:
            row["windows_memory"] = windows_memory(directory, environment, helpers)
            helpers.save(directory / "trial.json", row)
    except (Exception, KeyboardInterrupt) as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        for process in (worker, proxy):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        stopped.set()
        if sampler is not None:
            sampler.join(timeout=6)
        try:
            active = ROOT / "state/active.json"
            if controller is None and model is not None and active.exists():
                state = read(active)
                candidate = state.get("controller_pid")
                if isinstance(candidate, int) and os.getpgid(candidate) == model.pid and state.get("port") == 8898:
                    controller = candidate
            helpers.stop_owned(controller)
            if model is not None:
                if model.poll() is None and controller is None:
                    os.killpg(model.pid, signal.SIGTERM)
                model.wait(timeout=30)
        except Exception as error:
            row["cleanup_error"] = str(error)
        for handle in handles:
            handle.close()
        row["elapsed_seconds"] = time.monotonic() - started
        helpers.save(directory / "trial.json", row)
    return row


def interrupted(_number: int, _frame: Any) -> None:
    """Allow owned-process cleanup after SIGTERM."""
    raise KeyboardInterrupt


def main() -> int:
    """Print an offline plan unless --run is explicitly selected later."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute only after Matti explicitly authorizes benchmarks.")
    parser.add_argument("--json", action="store_true", help="Print only the offline JSON plan.")
    parser.add_argument("--cases", help="Comma-separated matrix names. Default: nonoptional cases that are not archived. Archived cases require explicit --cases or --stage.")
    parser.add_argument("--stage", help="Select one named stage instead of --cases.")
    parser.add_argument("--hooks", default="opencode,capacity", help="Comma-separated opencode,capacity,grammar107 hooks.")
    parser.add_argument("--output-directory", type=Path, default=HERE / "runs/pending")
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--windows-memory", action="store_true",
                        help="After all timed hooks, sample Windows VM GPU memory in a separate 1 GiB scope; failure is diagnostic only.")
    args = parser.parse_args()
    if args.cases and args.stage or args.timeout <= 0:
        parser.error("Select either --cases or --stage and a positive timeout.")
    matrix = read(HERE / "matrix.json")
    cases: list[Record] = matrix["cases"]
    wanted = str(args.cases).split(",") if args.cases else None
    selected = [case for case in cases if case["name"] in wanted] if wanted else [case for case in cases if case["stage"] == args.stage] if args.stage else [case for case in cases if case["stage"] != "optional" and not case.get("archived")]
    if not selected or wanted and set(wanted) != {case["name"] for case in selected}:
        parser.error("Select known matrix cases or stages.")
    hooks = str(args.hooks).split(",")
    if len(set(hooks)) != len(hooks) or any(name not in {"opencode", "capacity", "grammar107"} for name in hooks):
        parser.error("Use distinct known hook names.")
    destination: Path = args.output_directory.resolve()
    plan = offline(matrix, selected, destination, hooks, args.timeout)
    plan["windows_memory"] = args.windows_memory
    if not args.run:
        print(json.dumps(plan, indent=2))
        if not args.json:
            for case_plan in plan["case_plans"]:
                print(shlex.join(case_plan["commands"]["launch"]))
        return 0
    if destination.exists():
        parser.error("Use a new output directory. Existing run artifacts are preserved.")
    if any(case.get("conditional") for case in selected):
        parser.error("Conditional cases require a validated port and an explicit matrix review before execution.")
    if any(not value["weights_present"] for value in plan["assets"].values()):
        parser.error("Required weights are missing. Fetch them in a separate authorized step.")
    if any(not value["binary_present"] or value["manifest_present"] is False or value["build_record_present"] is False
           for value in plan["runtimes"].values()):
        parser.error("A runtime or native build record is missing. Build it in a separate authorized step.")
    refuse_services()
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    signal.signal(signal.SIGTERM, interrupted)
    destination.mkdir(parents=True)
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results: list[Record] = []
    for case, case_plan in zip(selected, plan["case_plans"], strict=True):
        row = run_case(case, case_plan["commands"], destination / str(case["name"]), args.timeout, args.windows_memory)
        results.append(row)
        (destination / "sweep.json").write_text(json.dumps(results, indent=2) + "\n")
        if row.get("cleanup_error") or row.get("error", "").startswith("KeyboardInterrupt"):
            break
    (destination / "sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(row["status"] == "ok" and not row.get("cleanup_error") for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
