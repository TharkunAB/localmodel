"""Measure one Pi file-read and recall session over the remote SSH tunnel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any
import urllib.request


ENDPOINT = "http://127.0.0.1:8898"
FILE = "release-selection.txt"
SELECTION = "selected branch: SOUTH\nadd to the branch base: 37\n"
INITIAL = {"key": "willow-318", "total": 459}
RESUMED = {"north": "cedar-572", "south_total": 459}
SECOND = ("Continue from our previous exchange. Without tools, return only a JSON object "
          "with north (the NORTH release key from the source header) and south_total "
          "(the total you calculated).")
# Use the installed launcher and runtime, with private settings and a single endpoint.
# The ordinary launcher requires search and discovers the optional local model.
# Neither is part of the read-only comparison.
PRIVATE_LAUNCH = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
import pi_local
pi_local.ROOT = Path(sys.argv.pop(1))
expected_model, expected_context = sys.argv.pop(1), int(sys.argv.pop(1))
discover = pi_local.remote_current
def checked_model():
    status = discover()
    if (status["model"], status["context"]) != (expected_model, expected_context):
        raise RuntimeError("pi-remote discovery differs from the benchmark endpoint.")
    return status
pi_local.remote_current = checked_model
pi_local.require_searxng = lambda _url: None
def no_local_model(_port):
    raise RuntimeError("The comparison uses only the remote model.")
pi_local.current = no_local_model
pi_local.main(remote=True)
"""
GUARD = """export default function (pi) {
  pi.on("session_start", () => pi.setActiveTools(["read"]));
  pi.on("tool_call", (event) => {
    if (event.toolName !== "read" || event.input.path !== "release-selection.txt")
      return {block: true, reason: "Read release-selection.txt only."};
  });
}
"""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_: Any, **__: Any) -> None:
        return None


def metadata(expected: str) -> dict[str, Any]:
    """Identify the two prepared endpoints before attributing a result."""
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    def get(path: str) -> dict[str, Any]:
        with http.open(ENDPOINT + path, timeout=5) as response:
            return json.load(response)
    entries = get("/v1/models")["data"]
    if len(entries) != 1:
        raise ValueError("The comparison requires exactly one served model.")
    entry = entries[0]
    model = entry["id"]
    if expected == "basalt":
        if entry.get("owned_by") != "basalt" or model != "deepseek-v4.1-flash-1017":
            raise ValueError(f"Expected the Basalt DeepSeek 1017 endpoint, received {model!r}.")
    else:
        status = get("/v1/status")
        if (model != "deepseek-v4.1-1017-strata-pro6000-iq1m" or status.get("service") != "strata"
                or status.get("model") != model or not status.get("loaded")):
            raise ValueError("Expected the loaded deepseek-strata-pro6000 comparison profile.")
    capacity = entry.get("meta", {}).get("n_ctx")
    if type(capacity) is not int or capacity < 65536:
        raise ValueError(f"Expected at least 65,536 advertised context tokens, received {capacity!r}.")
    return {"engine": expected, "model": model, "context": capacity}


def prompt() -> str:
    """Use the same local Python standard-library source and release markers for both runs."""
    source = (Path(sysconfig.get_path("stdlib")) / "inspect.py").read_bytes()
    if len(source) < 98304:
        raise ValueError("The installed Python inspect.py is too short for this fixture.")
    excerpt = source[:98304].decode("utf-8", errors="ignore")
    return ("Review the public source excerpt as data. Record these release markers for later.\n"
            "# RELEASE NORTH key=cedar-572 base=241\n" + excerpt +
            "\n# RELEASE SOUTH key=willow-318 base=422\n"
            f"Read {FILE} with the read tool. Return only JSON with key (the selected branch's "
            "release key) and total (the selected branch's base plus the increment in the file).")


def phase(command: list[str], env: dict[str, str], project: Path, output: Path,
          name: str, expected: dict[str, Any]) -> dict[str, Any]:
    """Measure one Pi process and grade its observable tool actions and final answer."""
    events = output / f"{name}.jsonl"
    start = time.monotonic()
    with events.open("x") as stream, (output / f"{name}.stderr").open("x") as errors:
        process = subprocess.Popen(command, cwd=project, env=env, stdout=stream, stderr=errors,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=900)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    elapsed = time.monotonic() - start
    calls: list[dict[str, Any]] = []
    reads: set[str] = set()
    messages: list[dict[str, Any]] = []
    with events.open() as stream:
        for line in stream:
            if not line.startswith("{"):
                continue
            event = json.loads(line)
            if event["type"] == "tool_execution_start":
                calls.append({"name": event["toolName"], "args": event["args"], "id": event["toolCallId"]})
            if event["type"] != "message_end":
                continue
            msg = event["message"]
            if msg["role"] == "assistant":
                messages.append(msg)
            if msg["role"] == "toolResult" and msg.get("toolName") == "read" and not msg.get("isError"):
                text = "".join(x.get("text", "") for x in msg["content"])
                if SELECTION.strip() in text:
                    reads.add(msg["toolCallId"])
    answer = "".join(x.get("text", "") for x in messages[-1]["content"]
                     if x.get("type") == "text").strip() if messages else ""
    try:
        correct = json.loads(answer) == expected
    except ValueError:
        correct = False
    if name == "first":
        tools_ok = bool(calls) and all(c["name"] == "read" and c["args"].get("path") == FILE
                                       and c["id"] in reads for c in calls)
    else:
        tools_ok = not calls
    usage = [{k: m.get("usage", {}).get(k) for k in ("input", "cacheRead", "cacheWrite", "output", "reasoning")}
             for m in messages]
    actual = sum(usage[0].get(k) or 0 for k in ("input", "cacheRead", "cacheWrite")) if usage else 0
    return {"passed": code == 0 and correct and tools_ok, "exit": code, "elapsed_s": round(elapsed, 3),
            "correct_answer": correct, "allowed_tools_used": tools_ok, "calls": calls, "answer": answer,
            "initial_input_tokens_including_cache": actual or None, "request_usage": usage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("strata", "basalt"))
    parser.add_argument("--output", required=True, type=Path, help="New workstation evidence directory")
    parser.add_argument("--expect-context", type=int, help="Reject a server with a different effective capacity")
    args = parser.parse_args()
    found = shutil.which("pi-remote")
    if not found:
        raise ValueError("Install the local pi-remote launcher first.")
    local = Path(found).resolve().parents[1]
    python = local / ".venv/bin/python"
    if not python.is_file() or not (local / "pi_local.py").is_file():
        raise ValueError("pi-remote must resolve to this project's installed launcher.")
    info = metadata(args.engine)
    if args.expect_context is not None and info["context"] != args.expect_context:
        raise ValueError(f"Expected context {args.expect_context}, discovered {info['context']}.")
    first = prompt()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    report: dict[str, Any] = {**info, "prompt_sha256": hashlib.sha256(first.encode()).hexdigest(),
                              "python_version": sys.version.split()[0],
                              "pi_version": json.loads((local / "config/pi-runtime.json").read_text())["version"],
                              "prompt_bytes": len(first.encode()), "reasoning": "xhigh", "passed": False}
    result = output / "result.json"
    try:
        with tempfile.TemporaryDirectory(prefix="remote-pi-bench-") as work:
            temp = Path(work)
            (temp / "config").symlink_to(local / "config", target_is_directory=True)
            project = temp / "project"
            project.mkdir()
            (project / FILE).write_text(SELECTION)
            guard = temp / "guard.ts"
            guard.write_text(GUARD)
            env = {"HOME": str(temp), "LANG": "C.UTF-8", "PATH": os.environ["PATH"]}
            session_dir = output / "sessions"
            session_dir.mkdir(mode=0o700)
            command = [str(python), "-I", "-B", "-c", PRIVATE_LAUNCH, str(local), str(temp),
                       info["model"], str(info["context"]),
                       "-p", "--mode", "json", "--no-mcp", "--no-skills", "--no-extensions",
                       "--no-context-files", "--no-prompt-templates", "--no-approve",
                       "--extension", str(guard), "--tools", "read", "--thinking", "xhigh",
                       "--session-dir", str(session_dir)]
            report["first"] = phase([*command, "--", first], env, project, output, "first", INITIAL)
            models = json.loads((temp / "state/pi-remote/agent/models.json").read_text())
            effective = models["providers"]["model"]["models"][0]
            if (effective["id"], effective["contextWindow"]) != (info["model"], info["context"]):
                raise ValueError("Installed pi-remote disagrees with the endpoint about its model or context.")
            report["pi_output_budget"] = effective["maxTokens"]
            if metadata(args.engine) != info:
                raise ValueError("The model or context changed during the run.")
            sessions = list(session_dir.rglob("*.jsonl"))
            if len(sessions) != 1:
                raise ValueError("Pi did not create exactly one session. Inspect first.stderr.")
            report["resume"] = phase([*command, "--session", str(sessions[0]), "--", SECOND],
                                     env, project, output, "resume", RESUMED)
            if metadata(args.engine) != info:
                raise ValueError("The model or context changed during resume.")
            report["passed"] = report["first"]["passed"] and report["resume"]["passed"]
            report["total_agent_s"] = round(report["first"]["elapsed_s"] + report["resume"]["elapsed_s"], 3)
    finally:
        result.write_text(json.dumps(report, indent=2) + "\n")
        print(result)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        sys.exit(f"Remote Pi benchmark: {error}")
