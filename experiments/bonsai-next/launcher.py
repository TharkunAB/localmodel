"""Run the reviewed OMP launcher with private benchmark state."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import omp_local

BENCH = Path(os.environ["OMP_BENCH_ROOT"]).absolute()
saved_runtime = omp_local.runtime
saved_collaboration = omp_local.collaboration
saved_current = omp_local.current
saved_provider = omp_local.model_provider
saved_which = omp_local.shutil.which


def runtime() -> Path:
    """Retain the existing runtime checksum checks."""
    omp_local.ROOT = ROOT
    try:
        return saved_runtime()
    finally:
        omp_local.ROOT = BENCH


def collaboration() -> dict[str, Any]:
    """Retain the existing collaboration restrictions."""
    omp_local.ROOT = ROOT
    try:
        return saved_collaboration()
    finally:
        omp_local.ROOT = BENCH


def current() -> dict[str, Any]:
    """Discover the live model and bound each benchmark response."""
    status = saved_current()
    expected = os.environ.get("OMP_BENCH_MODEL", "bonsai2-27b")
    if status["model"] != expected or status["context"] != 262144:
        raise RuntimeError(f"This benchmark requires {expected} with 256K capacity.")
    status["output"] = 4096
    status["temperature"] = float(os.environ.get("OMP_BENCH_TEMPERATURE", "1"))
    return status


def provider(status: dict[str, Any]) -> dict[str, Any]:
    """Use the same sampling seed in each controlled case."""
    result = saved_provider(status)
    result["models"][0]["compat"]["extraBody"]["seed"] = 424242
    return result


def which(command: str, mode: int = os.F_OK | os.X_OK, path: str | None = None) -> str | None:
    """Record this benchmark's requests without changing the installed runtime."""
    if command == "bun":
        return str(HERE / "bun-observed")
    return saved_which(command, mode=mode, path=path)


agent = BENCH / "state/omp-local/agent"
agent.mkdir(parents=True, exist_ok=True)
(agent / "AGENTS.md").write_text("")
omp_local.ROOT = BENCH
omp_local.runtime = runtime
omp_local.collaboration = collaboration
omp_local.current = current
omp_local.model_provider = provider
omp_local.shutil.which = which
omp_local.main()
