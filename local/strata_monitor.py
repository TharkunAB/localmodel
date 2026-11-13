"""Run the pinned Strata telemetry monitor in a separate Python environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
from types import FrameType
from typing import Any, cast
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from project_paths import CAPPED, read_manifest  # noqa: E402

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


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep telemetry requests on the configured loopback endpoint."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


def create_app(port: int) -> Any:
    """Create the read-only upstream app with local transport and quit actions."""
    command(port)
    verify_source()
    sys.path.insert(0, str(SOURCE))
    client = importlib.import_module("strata_cli.client")
    monitor = importlib.import_module("strata_cli.monitor")
    binding = importlib.import_module("textual.binding")
    rich_console = importlib.import_module("rich.console")
    rich_text = importlib.import_module("rich.text")

    class LocalEndpoint(client.Endpoint):
        """Retain upstream normalization with proxy-free loopback requests."""

        def get(self, path: str) -> dict[str, Any]:
            if path not in ("/v1/telemetry", "/v1/status", "/metrics", "/health"):
                raise ValueError("The monitor can read telemetry endpoints only.")
            request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                             headers={"Accept": "application/json"})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    if "application/json" not in response.headers.get("Content-Type", ""):
                        raise client.ClientError("server did not return JSON telemetry")
                    body = response.read(client.MAX_BODY + 1)
                    if len(body) > client.MAX_BODY:
                        raise client.ClientError("telemetry response exceeds the 2 MiB limit")
            except urllib.error.HTTPError as error:
                raise client.ClientError(f"telemetry request failed: HTTP {error.code}",
                                         4 if error.code in (401, 403) else 3,
                                         http_status=error.code) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                raise client.ClientError(f"cannot reach Strata server: {type(error).__name__}") from error
            try:
                data = json.loads(body)
            except (ValueError, UnicodeDecodeError, RecursionError) as error:
                raise client.ClientError("server returned malformed JSON telemetry") from error
            if not isinstance(data, dict):
                raise client.ClientError("server returned non-object telemetry")
            return cast(dict[str, Any], data)

    class TelemetryMonitor(monitor.MonitorApp):
        """Detach on q and request server shutdown on Ctrl-C."""

        ENABLE_COMMAND_PALETTE = False
        BINDINGS = [
            binding.Binding("ctrl+c", "interrupt", "Stop server", priority=True),
            binding.Binding("b", "none", show=False),
            binding.Binding("7", "none", show=False),
        ]

        def action_quit(self) -> None:
            self.closing = True
            self.exit(0)

        def action_interrupt(self) -> None:
            self.closing = True
            self.exit(130)

        def action_browser(self) -> None:
            """Keep this monitor inside the terminal."""

        def _paint(self, size: Any = None) -> None:
            monitor.MonitorApp._paint(self, size)
            widget = self.query_one("#content")
            content = widget.content
            if isinstance(content, rich_console.Group):
                hint = content.renderables[-1]
                if isinstance(hint, rich_text.Text) and "   b open in browser" in hint.plain:
                    content.renderables[-1] = rich_text.Text(hint.plain.replace("   b open in browser", ""), style="dim")
                    widget.update(content)

        def _help_line(self) -> str:
            return str(monitor.MonitorApp._help_line(self)) + "  Ctrl-C stop server"

    app = TelemetryMonitor(LocalEndpoint(f"http://127.0.0.1:{port}"), None)
    app.quiet_note = "Ctrl-C stops Strata"
    return app


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
    app = create_app(args.port)
    def terminate(_signum: int, _frame: FrameType | None) -> None:
        app.closing = True
        app.exit(143)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        result = app.run()
        return int(result) if result is not None else int(app.return_code or 0)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
