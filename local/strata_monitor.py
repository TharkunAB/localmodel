"""Run the pinned Strata telemetry monitor in a separate Python environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from project_paths import CAPPED, read_manifest  # noqa: E402
from strata_tui import create_app as shared_create_app, run_app  # noqa: E402

SOURCE = ROOT / "state/sources/strata-official-0.2.13"
VENV = ROOT / "state/runtimes/strata-monitor-8.2.8"
PYTHON = VENV / "bin/python"
REQUIREMENTS = ROOT / "config/strata-monitor-requirements.txt"


def command(port: int) -> list[str]:
    """Build the loopback monitor command without installing anything."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("The monitor port must be between 1 and 65535.")
    return [str(PYTHON), "-I", "-B", str(Path(__file__).resolve()), "--port", str(port)]


def ensure() -> None:
    """Prepare the monitor under a memory cap before the server starts."""
    env = os.environ.copy()
    env.update({"XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
                "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{os.getuid()}/bus",
                "HEAVY_PEAKS": str(ROOT / "state/strata-monitor-setup-peaks")})
    subprocess.run([str(CAPPED), "--mem", "2G", "--time", "5m", sys.executable,
                    "-I", "-B", str(Path(__file__).resolve()), "--prepare"], env=env, check=True)


def verify_source() -> None:
    """Use the existing receipt verifier before importing the monitor source."""
    info = read_manifest(ROOT / "experiments/deepseek-strata/manifest.json")
    if Path(info["source_directory"]) != SOURCE:
        raise RuntimeError("The monitor requires the reviewed Strata 0.2.13 source.")
    spec = importlib.util.spec_from_file_location(
        "stock_strata_runtime", ROOT / "experiments/deepseek-strata/runtime.py")
    assert spec and spec.loader
    stock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stock)
    stock.verify_source(SOURCE, info)


def prepare() -> None:
    """Install only the locked monitor packages into their own environment."""
    verify_source()
    if VENV.is_symlink():
        raise RuntimeError("The monitor environment must be a regular directory.")
    if not PYTHON.exists():
        subprocess.run(["uv", "venv", str(VENV), "--python", "python3.12"], check=True)
    subprocess.run(["uv", "pip", "sync", "--python", str(PYTHON), "--require-hashes",
                    "--only-binary", ":all:", str(REQUIREMENTS)], check=True)
    code = "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))"
    actual = json.loads(subprocess.check_output([str(PYTHON), "-I", "-B", "-c", code], text=True))
    expected = dict(line.split()[0].split("==", 1) for line in REQUIREMENTS.read_text().splitlines()
                    if line and not line[0].isspace() and not line.startswith("#"))
    if actual != expected:
        raise RuntimeError("The monitor Python packages differ from their pins.")


def create_app(port: int) -> Any:
    """Verify the local source before creating the shared telemetry display."""
    command(port)
    verify_source()
    return shared_create_app(SOURCE, port)


def main() -> int:
    """Prepare dependencies or run the telemetry display."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return 0
    return run_app(create_app(args.port))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
