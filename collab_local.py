"""Control the private OMP collaboration service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
UNIT = "omp-collab.service"


def run(args: list[str]) -> None:
    """Run a service command with the current user bus."""
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={env['XDG_RUNTIME_DIR']}/bus")
    subprocess.run(args, env=env, check=True)


def install() -> None:
    """Install the repository unit without replacing another unit."""
    destination = Path.home() / ".config/systemd/user" / UNIT
    source = ROOT / "config" / UNIT
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"Another unit exists at {destination}")
    else:
        destination.symlink_to(source)
    run(["systemctl", "--user", "daemon-reload"])
    run(["systemctl", "--user", "enable", UNIT])


def main() -> None:
    """Run a service command without a model dependency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "start", "stop", "restart", "status", "logs", "info"])
    args = parser.parse_args()
    if args.command == "install":
        install()
    elif args.command in ("start", "stop", "restart", "status"):
        run(["systemctl", "--user", str(args.command), UNIT, "--no-pager"])
    elif args.command == "logs":
        run(["journalctl", "--user", "-u", UNIT, "-n", "50", "--no-pager"])
    else:
        config = json.loads((ROOT / "config/omp-collab.json").read_text())
        print(f"Browser: {config['webUrl']}\nRelay:   {config['relayUrl']}", flush=True)
        print(f"CA:      {ROOT / config['caFile']}", flush=True)
        run(["openssl", "x509", "-in", str(ROOT / config["caFile"]), "-noout", "-fingerprint", "-sha256"])


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"omp-collab: {error}", file=sys.stderr)
        raise SystemExit(1) from error
