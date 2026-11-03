#!/usr/bin/env python3
"""Queue Strata release maintenance when the local machine has capacity."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, cast
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state/strata-watch.json"
MANIFEST = ROOT / "experiments/deepseek-strata/manifest.json"
API = "https://api.github.com/repos/Niko1221/Strata/releases/latest"


def version(tag: str) -> tuple[int, ...]:
    """Accept stable numeric release tags only."""
    if not re.fullmatch(r"v?\d+(?:\.\d+){1,3}", tag):
        raise ValueError(f"Unexpected stable release tag: {tag!r}")
    parts = tuple(int(part) for part in tag.removeprefix("v").split("."))
    return parts + (0,) * (4 - len(parts))


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a local file."""
    value: Any = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def busy_reason() -> str | None:
    """Defer work if local resource checks fail."""
    if shutil.disk_usage(ROOT).free < 10 * 2**30:
        return "less than 10 GiB free disk space"
    memory = Path("/proc/meminfo").read_text()
    available = re.search(r"^MemAvailable:\s+(\d+) kB$", memory, re.MULTILINE)
    if available is None or int(available[1]) * 1024 < 72 * 2**30:
        return "less than 72 GiB available Linux RAM"
    if os.getloadavg()[0] > min(8, (os.cpu_count() or 1) / 2):
        return "CPU load is too high for a benchmark"
    if (ROOT / "state/active.json").exists():
        return "localmodel has an active server"
    for query in ("--query-compute-apps=pid", "--query-gpu=memory.used,utilization.gpu"):
        result = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        if "compute-apps" in query:
            if result.stdout.strip():
                return "a GPU compute process is running"
        else:
            rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
            if not rows or any(int(row[0]) > 2048 or int(row[1]) > 5 for row in rows):
                return "the GPU is not idle"
    return None


def check(thread: str, codex: str, check_only: bool) -> None:
    """Check the release and queue at most one reminder per day."""
    request = Request(API, headers={"Accept": "application/vnd.github+json", "User-Agent": "localmodel-strata-watch"})
    with urlopen(request, timeout=30) as response:
        release: dict[str, Any] = json.loads(response.read(1024 * 1024))
    tag = str(release["tag_name"])
    latest = version(tag)
    if release.get("draft") or release.get("prerelease"):
        raise ValueError("The latest release is not stable")
    installed = str(read_json(MANIFEST)["source_tag"])
    state = read_json(STATE) if STATE.exists() else {}
    if state.get("completed_tag") == tag or (latest <= version(installed) and state.get("pending_tag") != tag):
        print(f"Current: installed {installed}; latest {tag}.")
        return
    if check_only:
        print(f"Maintenance needed: installed {installed}; latest {tag}. No work queued.")
        return
    if state.get("pending_tag") == tag and time.time() - float(state.get("queued_at", 0)) < 86400:
        print(f"Already queued: {tag}. Pending work is retried after 24 hours.")
        return
    reason = busy_reason()
    if reason:
        print(f"Deferred {tag}: {reason}. The next timer check will retry.")
        return
    message = (
        f"Scheduled Strata maintenance authorized by Matti: official stable release {tag} is available. "
        f"Work in {ROOT}. Recheck the official release and current installed pins. "
        "Read docs/strata-watch.md and follow its update and benchmark procedure. "
        "This notification is an instruction to do the update and measurements, not only report availability. "
        "Recheck resources before starting; do not interrupt any running model or other work. "
        f"When measurements and cleanup are complete, set completed_tag to {tag!r} in {STATE} "
        "and report results here. If work is deferred or incomplete, leave completed_tag unchanged."
    )
    daemon = json.loads(subprocess.run(
        [codex, "app-server", "daemon", "version"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout)
    if daemon.get("status") != "running":
        raise RuntimeError("Codex daemon is unavailable; leave the release pending")
    endpoint = "unix://" + str(daemon["socketPath"])
    subprocess.run(
        [codex, "queue", "--remote", endpoint, "--thread", thread, "--message", message],
        check=True, timeout=60,
    )
    state.update(pending_tag=tag, queued_at=time.time())
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE)
    print(f"Queued {tag} into thread {thread}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread", required=True, help="Existing Codex conversation UUID")
    parser.add_argument("--codex", default="codex", help="Path to the installed Codex CLI")
    parser.add_argument("--check-only", action="store_true", help="Check releases without queuing work")
    args = parser.parse_args()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check(args.thread, args.codex, args.check_only)


if __name__ == "__main__":
    main()
