#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)
root=$(cd -- "$here/../.." && pwd)
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
export HEAVY_PEAKS="$root/state/qwen-engram-benchmark-peaks"
unset CAPPED_UNIT
exec "$root/libexec/capped" --mem 4G \
    "$root/state/venvs/strata-official-0.2.8/bin/python" -B "$here/run.py" "$@"
