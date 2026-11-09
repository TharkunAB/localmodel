#!/bin/bash
set -euo pipefail
cd /home/tritlo/Code/local-models
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export HEAVY_PEAKS="$PWD/results/strata-candidates-20261109/harness/peaks"
exec /home/tritlo/Code/heavy-management/bin/capped --mem 2G --time 40m \
  "$PWD/state/venvs/strata-official-0.2.8/bin/python" \
  "$PWD/experiments/strata-candidates-20261109/benchmark/run.py" \
  --model deepseek-full --expected-source-tag v0.2.12 --expected-native-version 0.1.58 \
  --mem 64 --expert-ram-gb 48 --cpu-threads 12 --prefill-chunk 4096 --speculative none \
  "$@"
