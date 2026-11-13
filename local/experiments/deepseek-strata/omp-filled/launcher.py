"""Run the normal OMP launcher with private benchmark state."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

import omp_local

BENCHMARK = Path(os.environ["OMP_BENCH_ROOT"]).absolute()
HELPERS = Path(__file__).resolve().parent
original_runtime = omp_local.runtime
original_collaboration = omp_local.collaboration
original_current = omp_local.current
original_which = omp_local.shutil.which


def runtime() -> Path:
    """Use the reviewed runtime with its existing checksum checks."""
    omp_local.ROOT = REPOSITORY
    try:
        return original_runtime()
    finally:
        omp_local.ROOT = BENCHMARK


def collaboration() -> dict[str, Any]:
    """Keep the configured collaboration origin and certificate policy."""
    omp_local.ROOT = REPOSITORY
    try:
        return original_collaboration()
    finally:
        omp_local.ROOT = BENCHMARK


def current() -> dict[str, Any]:
    """Keep live model discovery and bound each benchmark response."""
    status = original_current()
    expected = os.environ.get("OMP_BENCH_MODEL", "deepseek-v4.1-1017-iq1-m")
    if status["model"] != expected:
        raise RuntimeError(f"Expected {expected}, found {status['model']}.")
    status["output"] = 4096
    return status


def which(command: str, mode: int = os.F_OK | os.X_OK, path: str | None = None) -> str | None:
    """Add the recorder to this benchmark process only."""
    if command == "bun":
        return str(HELPERS / "bun-observed")
    return original_which(command, mode=mode, path=path)


os.environ["PI_OPENAI_STREAM_FIRST_EVENT_TIMEOUT_MS"] = "43200000"
os.environ["PI_OPENAI_STREAM_IDLE_TIMEOUT_MS"] = "43200000"
agent = BENCHMARK / "state/omp-local/agent"
agent.mkdir(parents=True, exist_ok=True)
instructions = agent / "AGENTS.md"
if not instructions.exists():
    instructions.write_text("")
omp_local.ROOT = BENCHMARK
omp_local.runtime = runtime
omp_local.collaboration = collaboration
omp_local.current = current
omp_local.shutil.which = which
omp_local.main()
