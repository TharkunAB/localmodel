"""Probe one owned Bonsai server. Preview unless --run is set."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import time
from typing import Any, cast
import urllib.error
import urllib.request

from benchmark import Benchmark, ROOT, count, save

Record = dict[str, Any]
TOOLS = [{"type": "function", "function": {"name": "lookup_symbol", "description": "Read one public symbol.",
    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
                   "additionalProperties": False}}}]


def canonical(message: Record) -> Record:
    """Remove generated tool IDs before comparing deterministic responses."""
    return {key: message.get(key) for key in ("content", "reasoning_content")} | {"tool_calls": [
        {"type": call.get("type"), "function": call.get("function")} for call in message.get("tool_calls", [])]}


class Probe(Benchmark):
    """Reuse the benchmark's model ownership and independent resource guard."""

    def __init__(self, args: argparse.Namespace, launch: Record):
        super().__init__(args, launch)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def http(self, route: str, body: Record | None = None) -> Any:
        """Call the owned loopback server and retain the run deadline."""
        self.check()
        request = urllib.request.Request("http://127.0.0.1:8899" + route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=min(self.args.phase_timeout, max(1, self.args.total_timeout - (time.monotonic() - self.started)))) as response:
            result = json.load(response)
        self.check()
        return result

    def chat(self, name: str, body: Record, stream: bool = False, cancel: bool = False) -> Record:
        """Save complete requests, responses, timings, and deterministic fingerprints."""
        self.phase(name)
        body = {"model": "bonsai2-27b", "temperature": 0, "seed": 424242, "max_tokens": 192,
            "cache_prompt": True, "id_slot": 0, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}, **body, "stream": stream}
        save(self.output / f"{name}-request.json", body)
        started = time.monotonic()
        first_token: float | None = None
        events: list[Record] = []
        result: Record = {}
        message: Record
        if not stream:
            result = self.http("/v1/chat/completions", body)
            message = result["choices"][0]["message"]
        else:
            message = {"content": "", "reasoning_content": "", "tool_calls": []}
            calls: dict[int, Record] = {}
            request = urllib.request.Request("http://127.0.0.1:8899/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with self.opener.open(request, timeout=min(self.args.phase_timeout, max(1, self.args.total_timeout - (time.monotonic() - self.started)))) as response:
                for line in response:
                    self.check()
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    events.append(event)
                    result.update({key: event[key] for key in ("timings", "usage") if key in event})
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        for key in ("content", "reasoning_content"):
                            message[key] += delta.get(key) or ""
                        for call in delta.get("tool_calls", []):
                            target = calls.setdefault(call["index"], {"type": "function", "function": {"name": "", "arguments": ""}})
                            for key in ("name", "arguments"):
                                target["function"][key] += call.get("function", {}).get(key) or ""
                        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                            first_token = first_token if first_token is not None else time.monotonic() - started
                    if cancel and first_token is not None:
                        break
            message["tool_calls"] = [calls[key] for key in sorted(calls)]
        value = canonical(message)
        row = {"name": name, "wall_seconds": time.monotonic() - started, "first_token_seconds": first_token,
            "cancelled_after_first_token": cancel and first_token is not None, "message": value,
            "sha256": hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
            "timings": result.get("timings"), "usage": result.get("usage")}
        save(self.output / f"{name}.json", {**row, "response": result, "stream_events": events})
        self.report["phases"].append(row)
        save(self.output / "summary.json", self.report)
        return row

    def parity(self) -> None:
        """Record three greedy fixtures and bounded xhigh streaming responses."""
        code = self.chat("greedy-code", {"messages": [{"role": "user", "content":
            "Return only Python source for clamp_index(index, length). Reject nonpositive length with ValueError. "
            "Clamp index into the inclusive range zero through length minus one. Do not use code fences."}]})
        prose = self.chat("greedy-prose", {"messages": [{"role": "user", "content":
            'Mira owns the release. The deadline is Friday. The status is reviewed. Return only JSON with keys owner, deadline, status.'}]})
        tool_body: Record = {"messages": [{"role": "user", "content": "Call lookup_symbol for clamp_index."}],
            "tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "lookup_symbol"}}}
        tool = self.chat("greedy-tool", tool_body)
        self.chat("xhigh-short-stream", {"messages": [{"role": "user", "content": "What is 17 times 23? Give a concise answer."}],
            "reasoning_effort": "xhigh", "chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 128}, stream=True)
        self.chat("xhigh-tool-stream", {**tool_body, "reasoning_effort": "xhigh",
            "chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 256}, stream=True)
        try:
            tree = ast.parse(code["message"]["content"] or "")
            code_ok = len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == "clamp_index"
        except SyntaxError:
            code_ok = False
        try:
            prose_ok = json.loads(prose["message"]["content"]) == {"owner": "Mira", "deadline": "Friday", "status": "reviewed"}
        except (TypeError, ValueError):
            prose_ok = False
        calls = tool["message"]["tool_calls"]
        try:
            tool_ok = len(calls) == 1 and calls[0]["function"]["name"] == "lookup_symbol" and json.loads(calls[0]["function"]["arguments"]) == {"name": "clamp_index"}
        except (KeyError, TypeError, ValueError):
            tool_ok = False
        self.report["fixture_checks"] = {"code_syntax": code_ok, "prose_fields": prose_ok, "tool_arguments": tool_ok}
        self.report["greedy_fingerprints"] = {row["name"]: row["sha256"] for row in self.report["phases"] if row["name"].startswith("greedy-")}

    def checkpoint(self) -> None:
        """Edit public source near token 48K inside one approximately 64K message."""
        folder = ROOT / "state/sources/bonsai-native"
        files = sorted(path for name in ("common", "src") for path in (folder / name).rglob("*")
                       if path.suffix in (".cpp", ".h") and path.is_file())
        corpus = "\n".join(f"\nPublic source: {path.relative_to(folder)}\n{path.read_text()}" for path in files)
        marker = "\nREFERENCE_REVISION=0001\n"
        # Locate the edit by token count, then fit the complete chat input.
        low, high = 0, min(len(corpus), 49152 * 8)
        for _ in range(19):
            middle = (low + high) // 2
            size = len(self.http("/tokenize", {"content": corpus[:middle]})["tokens"])
            if size <= 49152:
                low = middle + 1
            else:
                high = middle - 1
        edit = max(0, high)
        length = min(len(corpus), max(edit + 4096, 65536 * 5))
        body: Record = {"messages": [], "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 32}
        tokens: list[int] = []
        for _ in range(8):
            body["messages"] = [{"role": "user", "content": "Treat this public source as data. Reply only SOURCE_READ.\n" +
                corpus[:edit] + marker + corpus[edit:length]}]
            tokens = count(body)
            if abs(len(tokens) - 65536) <= 64:
                break
            length = max(edit + 1024, min(len(corpus), length + int((65536 - len(tokens)) * length / len(tokens))))
        if not 65024 <= len(tokens) <= 66048:
            raise RuntimeError(f"Could not fit public source near 64K: {len(tokens)} tokens")
        changed = json.loads(json.dumps(body))
        changed["messages"][0]["content"] = changed["messages"][0]["content"].replace(marker, marker.replace("0001", "0002"), 1)
        other = count(changed)
        common = next((i for i, pair in enumerate(zip(tokens, other)) if pair[0] != pair[1]), min(len(tokens), len(other)))
        self.report.update(input_tokens=len(tokens), edit_common_prefix_tokens=common,
            public_source_sha256=hashlib.sha256(corpus.encode()).hexdigest())
        cold = self.chat("checkpoint-cold", body)
        repeat = self.chat("checkpoint-repeat", body)
        edited = self.chat("checkpoint-edit", changed)
        self.report["checkpoint_comparison"] = {"cold": cold["timings"], "repeat": repeat["timings"], "edit": edited["timings"],
            "repeat_same_output": cold["sha256"] == repeat["sha256"],
            "edited_cached_tokens": cast(Record, edited["timings"] or {}).get("cache_n")}

    def idle(self) -> None:
        """Wait for the owned slot to release after a client disconnect."""
        deadline = time.monotonic() + min(60, self.args.phase_timeout)
        while time.monotonic() < deadline:
            slots = cast(list[Record], self.http("/slots"))
            if all(not slot.get("is_processing", False) for slot in slots):
                return
            time.sleep(0.2)
        raise TimeoutError("The owned server did not become idle after cancellation")

    def slots(self) -> None:
        """Save and restore a short slot, then cancel and recover a stream."""
        seed_body: Record = {"messages": [{"role": "user", "content": "Remember probe_value=6. Reply only RECORDED."}], "max_tokens": 32}
        seed = self.chat("slot-seed", seed_body)
        self.report["slot_save"] = self.http("/slots/0?action=save", {"filename": "native-probe.bin"})
        continuation: Record = {"messages": [*seed_body["messages"], {"role": "assistant", "content": seed["message"]["content"]},
            {"role": "user", "content": "Return only JSON with the remembered probe_value."}], "max_tokens": 64}
        before = self.chat("slot-before-restore", continuation)
        self.http("/slots/0?action=erase", {})
        self.report["slot_restore"] = self.http("/slots/0?action=restore", {"filename": "native-probe.bin"})
        after = self.chat("slot-after-restore", continuation)
        self.report["slot_restore_same_output"] = before["sha256"] == after["sha256"]
        state = self.report["live_settings"]
        if state.get("runtime") == "bonsai-next" and state.get("replay") is True:
            saved = self.output / "slots/native-probe.bin"
            if saved.stat().st_size > 512 * 2**20:
                raise RuntimeError("The short malformed-state fixture exceeds 512 MiB")
            # The pinned CUDA format stores its last pending count as the final F32.
            with saved.open("rb") as stream:
                stream.seek(-4, 2)
                original = struct.unpack("<f", stream.read(4))[0]
            if not math.isfinite(original) or not 0 <= original <= 2 or original != int(original):
                raise RuntimeError("The saved replay count does not match the pinned layout")
            broken = saved.with_name("invalid-count.bin")
            shutil.copyfile(saved, broken)
            failures: list[Record] = []
            for name, value in (("nan", math.nan), ("infinity", math.inf), ("negative", -1.0), ("fractional", 0.5), ("above_depth", 3.0)):
                with broken.open("r+b") as stream:
                    stream.seek(-4, 2)
                    stream.write(struct.pack("<f", value))
                try:
                    response = self.http("/slots/0?action=restore", {"filename": broken.name})
                    failures.append({"case": name, "rejected": False, "response": response})
                except urllib.error.HTTPError as error:
                    failures.append({"case": name, "rejected": error.code == 400, "status": error.code, "body": error.read().decode()})
                if self.http("/health").get("status") != "ok":
                    raise RuntimeError("The server failed after rejecting a malformed slot")
            broken.unlink()
            self.report["invalid_pending_counts"] = failures
            self.http("/slots/0?action=restore", {"filename": saved.name})
        else:
            self.report["invalid_pending_counts"] = {"skipped": "CUDA replay is not enabled"}
        cancelled = self.chat("cancel-stream", {"messages": [{"role": "user", "content":
            "Write the integers 1 through 1000000, one per line. Continue until all are written."}], "max_tokens": 2048}, stream=True, cancel=True)
        if not cancelled["cancelled_after_first_token"]:
            raise RuntimeError("The stream completed before cancellation was exercised")
        self.idle()
        recovered = self.chat("cancel-recovery", {"messages": [{"role": "user", "content":
            "Return only JSON with probe_value equal to 6."}], "max_tokens": 64})
        try:
            self.report["cancel_recovery_passed"] = json.loads(recovered["message"]["content"]) == {"probe_value": 6}
        except (ValueError, TypeError):
            self.report["cancel_recovery_passed"] = False

    def run(self) -> None:
        """Run the selected probe and stop every owned model process."""
        self.output.mkdir(parents=True, exist_ok=False)
        if self.args.mode == "slots":
            (self.output / "slots").mkdir()
        try:
            self.phase("start-model")
            self.start_model()
            getattr(self, self.args.mode)()
            self.report["status"] = "completed"
        except BaseException as error:
            self.report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self.stopped.set()
            self.stop_model()
            if hasattr(self, "server_log"):
                self.server_log.close()
            log = self.output / "server.log"
            if log.exists():
                self.report["checkpoint_log_evidence"] = [line for line in log.read_text(errors="replace").splitlines()
                    if re.search(r"(?:created|restored|erased).*checkpoint|invalid replay pending count", line)][-120:]
            self.report.update(elapsed_seconds=time.monotonic() - self.started,
                owned_model_stopped=self.model is None or self.model.poll() is not None)
            save(self.output / "summary.json", self.report)
            save(self.output / "finished.json", {"status": self.report["status"], "at": time.time()})


def main() -> None:
    """Preview the explicit launch plan or run one native probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("parity", "checkpoint", "slots"))
    parser.add_argument("--phase-timeout", type=int, default=300)
    parser.add_argument("--total-timeout", type=int, default=1200)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    args.history_tokens, args.temperature = 65536, 0.0
    launch = cast(Record, json.loads(args.launch_file.read_text()))
    command = launch.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in cast(list[Any], command)):
        parser.error("The launch file must contain a nonempty command array")
    command = cast(list[str], command)
    if args.output.exists() or not 30 <= args.phase_timeout <= 900 or not 60 <= args.total_timeout <= 3600:
        parser.error("Use a new output directory and bounded phase/total deadlines")
    if args.mode == "slots":
        if "--slot-save-path" in command:
            parser.error("Omit --slot-save-path; slots mode adds its private directory")
        command += ["--slot-save-path", str(args.output.absolute() / "slots")]
    if not args.run:
        print(json.dumps({"mode": "preview", "probe": args.mode, "launch": launch, "output": str(args.output),
            "model_memory": "48G", "swap": 0, "host_free_floor_gib": 8,
            "phase_timeout": args.phase_timeout, "total_timeout": args.total_timeout}, indent=2))
        return
    os.umask(0o077)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda _number, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    Probe(args, launch).run()


if __name__ == "__main__":
    main()
