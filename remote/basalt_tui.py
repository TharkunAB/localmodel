"""Read Basalt's local /metrics endpoint and display a single-session terminal view."""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import select
import signal
import sys
import termios
import time
from types import FrameType
from typing import Any


def metrics(port: int) -> dict[str, Any]:
    """Read one bounded response directly from the loopback listener."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", "/metrics", headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}")
        payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError("metrics response too large")
        value = json.loads(payload)
        if not isinstance(value, dict) or not all(isinstance(value.get(k), dict) for k in ("engine", "live", "hardware")):
            raise ValueError("unexpected metrics response")
        return value
    finally:
        connection.close()


def number(value: Any, scale: float = 1, digits: int = 0) -> str:
    """Format a telemetry number. Missing samples use a dash."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return "-"
    return f"{value / scale:,.{digits}f}"


def display(data: dict[str, Any]) -> list[str]:
    """Render the live values and the most recently completed request."""
    live, engine = data["live"], data["engine"]
    state = live.get("state")
    if state not in ("loading", "reading", "generating", "unloaded", "asleep", "idle"):
        state = "unknown"
    rows = [f"Basalt   {state}", "",
            f"Context limit     {number(engine.get('max_context'))} tokens",
            f"Current request   {number(live.get('prompt_tokens'))} input   {number(live.get('generated'))} output",
            f"Current rates     {number(live.get('prefill_tok_s_mean'), digits=1)} prefill   "
            f"{number(live.get('tok_s'), digits=1)} decode   tok/s"]
    # The single-card profile has no verified mapping for multiple NVML samples.
    gpus = data["hardware"].get("gpus")
    if isinstance(gpus, list) and len(gpus) == 1 and isinstance(gpus[0], dict):
        gpu = gpus[0]
        rows.append(f"GPU {number(gpu.get('index'))}             {number(gpu.get('util'))}% load   "
                    f"{number(gpu.get('mem_used'), 2**30, 1)} / {number(gpu.get('mem_total'), 2**30, 1)} GiB VRAM")
    else:
        rows.append("GPU readings      unavailable")
    rows.append("")
    recent = data.get("requests") or []
    if isinstance(recent, list) and recent and isinstance(recent[0], dict):
        last = recent[0]
        prompt, reused, ms = (last.get(key) for key in ("prompt_tokens", "reused", "prompt_ms"))
        prefill = None
        if type(prompt) is int and type(reused) is int and 0 <= reused < prompt and isinstance(ms, (int, float)) and ms > 0:
            prefill = 1000 * (prompt - reused) / ms
        rows.extend([f"Last request      {number(prompt)} input   {number(reused)} reused   {number(last.get('output_tokens'))} output",
                     f"Last rates        {number(prefill, digits=1)} uncached prefill   "
                     f"{number(last.get('decode_tok_s'), digits=1)} decode   tok/s",
                     f"Wall time         {number(last.get('duration_s'), digits=1)} s"])
    else:
        rows.append("No completed requests yet.")
    return rows


def monitor(port: int) -> int:
    """Update changed rows and restore the terminal on every normal or signal exit."""
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    keys = termios.tcgetattr(fd)
    keys[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
    keys[6][termios.VMIN], keys[6][termios.VTIME] = 1, 0
    previous: list[str] = []
    try:
        termios.tcsetattr(fd, termios.TCSANOW, keys)
        sys.stdout.write("\x1b[?1049h\x1b[2J\x1b[?25l")
        while True:
            try:
                rows = display(metrics(port))
            except (http.client.HTTPException, OSError, ValueError):
                rows = ["Basalt", "", "Waiting for local /metrics"]
            cols, lines = os.get_terminal_size(sys.stdout.fileno())
            rows = (rows + [""])[:max(0, lines - 1)] + ["q closes monitor   Ctrl-C stops server"]
            rows = [row[:max(0, cols - 1)] for row in rows]
            for i in range(max(len(previous), len(rows))):
                row = rows[i] if i < len(rows) else ""
                if i >= len(previous) or row != previous[i]:
                    sys.stdout.write(f"\x1b[{i+1};1H\x1b[K{row}")
            sys.stdout.flush()
            previous = rows
            until = time.monotonic() + 1
            while (remaining := until - time.monotonic()) > 0:
                if not select.select([fd], [], [], remaining)[0]:
                    break
                key = os.read(fd, 1)
                if key in (b"q", b"Q", b""):
                    return 0
                if key == b"\x03":
                    return 130
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, original)
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()


def interrupted(_signal: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Use a valid TCP port.")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("The monitor needs an interactive terminal.")
    signal.signal(signal.SIGTERM, interrupted)
    return monitor(args.port)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
