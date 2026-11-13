"""Share the read-only Strata terminal display between local and remote launchers."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import signal
import sys
from types import FrameType
from typing import Any, cast
import urllib.error
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep telemetry requests on the configured loopback endpoint."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


def create_app(source: Path, port: int) -> Any:
    """Create the read-only upstream app with local transport and quit actions."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("The monitor port must be between 1 and 65535.")
    sys.path.insert(0, str(source))
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


def run_app(app: Any) -> int:
    """Run the display and restore its signal handler after exit."""
    def terminate(_signum: int, _frame: FrameType | None) -> None:
        app.closing = True
        app.exit(143)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        result = app.run()
        return int(result) if result is not None else int(app.return_code or 0)
    finally:
        signal.signal(signal.SIGTERM, previous)
