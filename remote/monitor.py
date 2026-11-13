"""Prepare and run the shared Strata display on a rental GPU host."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, cast

REPOSITORY = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPOSITORY / "local/config/strata-monitor-requirements.txt"
sys.path.insert(0, str(REPOSITORY / "local"))
from strata_tui import create_app, run_app  # noqa: E402


def digest(path: Path) -> str:
    """Hash a regular source file without following symbolic links."""
    if not path.is_file() or path.resolve() != path.absolute():
        raise RuntimeError(f"The monitor source is missing or linked: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_source(source: Path, receipt_sha: str, revision: str) -> None:
    """Verify the pinned upstream receipt and every source file before import."""
    if not re.fullmatch(r"[a-f0-9]{64}", receipt_sha) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("The monitor requires pinned source and receipt hashes.")
    receipt = source / "source-files.json"
    if digest(receipt) != receipt_sha:
        raise RuntimeError("The Strata monitor source receipt differs from its pin.")
    commit = source / "SOURCE_COMMIT"
    digest(commit)
    if commit.read_text().strip() != revision:
        raise RuntimeError("The Strata monitor source revision differs from its pin.")
    files = cast(dict[str, str], json.loads(receipt.read_text()))
    for name, expected in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or digest(source / name) != expected:
            raise RuntimeError(f"The Strata monitor source file changed: {name}")
    extras = [path for path in source.rglob("*.py") if str(path.relative_to(source)) not in files]
    if extras:
        raise RuntimeError("The Strata monitor source contains unpinned Python files.")


def expected_packages() -> dict[str, str]:
    """Read the shared hash-locked runtime package versions."""
    return dict(line.split()[0].split("==", 1) for line in REQUIREMENTS.read_text().splitlines()
                if line and not line[0].isspace() and not line.startswith("#"))


def installed_packages(python: Path) -> dict[str, str]:
    """Read runtime versions and exclude the standard venv bootstrap tools."""
    code = ("import importlib.metadata as m,json,sys;"
            "assert sys.version_info >= (3,11), 'The monitor needs Python 3.11+';"
            "print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version "
            "for d in m.distributions() if d.metadata['Name'].lower() not in ('pip','setuptools')}))")
    result = subprocess.check_output([str(python), "-I", "-B", "-c", code], text=True, timeout=30)
    return cast(dict[str, str], json.loads(result))


def creation_python() -> str:
    """Find Python 3.11 or later for the shared monitor dependency lock."""
    candidates = [sys.executable, *(shutil.which(name) for name in ("python3.12", "python3.11", "python3"))]
    for python in dict.fromkeys(path for path in candidates if path):
        probe = subprocess.run([python, "-I", "-B", "-c",
                                "import sys;sys.exit(not ((3,11) <= sys.version_info[:2] < (4,0)))"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        if probe.returncode == 0:
            return python
    raise RuntimeError("--tui needs Python 3.11+ with venv support. Install it or omit --tui.")


def prepare(plan: dict[str, Any], install: bool = True) -> list[str]:
    """Verify or prepare the cached monitor without changing the server environment."""
    if plan["plan"].get("engine") != "strata":
        raise ValueError("--tui requires a Strata profile.")
    port = plan["plan"]["port"]
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("The monitor port must be between 1 and 65535.")
    source = Path(plan["server"]["cwd"]).absolute()
    pins = plan["runtime_pins"]
    receipt_sha, revision = str(pins["source_receipt_sha256"]), str(pins["source_revision"])
    verify_source(source, receipt_sha, revision)
    expected = expected_packages()
    venv = source.parent / ("monitor-" + expected["textual"])
    python = venv / "bin/python"
    if venv.is_symlink():
        raise RuntimeError("The monitor environment must be a regular directory.")
    lock = venv.with_name(venv.name + ".lock")
    flags = (os.O_CREAT | os.O_RDWR) if install else os.O_RDONLY
    try:
        descriptor = os.open(lock, flags | os.O_NOFOLLOW, 0o600)
    except FileNotFoundError:
        raise RuntimeError("The TUI is not prepared. Start once without --no-fetch.") from None
    with os.fdopen(descriptor, "r" if not install else "a") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if install else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is preparing the Strata monitor.") from None
        if not python.exists():
            if not install:
                raise RuntimeError("The TUI is not prepared. Start once without --no-fetch.")
            subprocess.run([creation_python(), "-I", "-B", "-m", "venv", str(venv)], check=True)
        actual = installed_packages(python)
        if actual != expected:
            if not install:
                raise RuntimeError("The TUI packages differ from their pins. Start once without --no-fetch.")
            if set(actual) - set(expected):
                raise RuntimeError(f"The isolated monitor has unexpected packages. Recreate {venv}.")
            subprocess.run([str(python), "-I", "-B", "-m", "pip", "--isolated", "--disable-pip-version-check",
                            "install", "--require-hashes", "--only-binary", ":all:", "--no-deps",
                            "-r", str(REQUIREMENTS)], check=True)
            if installed_packages(python) != expected:
                raise RuntimeError("The installed TUI packages differ from their pins.")
    return [str(python), "-I", "-B", str(Path(__file__).resolve()), "--source", str(source),
            "--receipt-sha", receipt_sha, "--revision", revision, "--port", str(port)]


def main() -> int:
    """Run the read-only display after checking its source and dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt-sha", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    source = args.source.absolute()
    verify_source(source, args.receipt_sha, args.revision)
    if installed_packages(Path(sys.executable)) != expected_packages():
        raise RuntimeError("The TUI packages differ from their pins. Start once without --no-fetch.")
    return run_app(create_app(source, args.port))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
