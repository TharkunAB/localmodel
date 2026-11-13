#!/bin/bash
set -euo pipefail
cd /home/tritlo/Code/local-models
export STRATA_BENCH_RELEASE="${1:?Specify v0.2.12 or v0.2.13 before the case arguments}"
shift
case "$STRATA_BENCH_RELEASE" in
    v0.2.12) native_version=0.1.58 ;;
    v0.2.13) native_version=0.1.59 ;;
    *) echo 'Expected v0.2.12 or v0.2.13.' >&2; exit 2 ;;
esac
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export HEAVY_PEAKS="$PWD/results/strata-0213-20261109/$STRATA_BENCH_RELEASE/peaks"
exec /home/tritlo/Code/heavy-management/bin/capped --mem 2G --time 40m     "$PWD/state/venvs/strata-official-0.2.8/bin/python"     "$PWD/experiments/strata-0213-20261109/benchmark/run.py"     --model deepseek-full --expected-source-tag "$STRATA_BENCH_RELEASE"     --expected-native-version "$native_version" --mem 64 --expert-ram-gb 48     --cpu-threads 12 --prefill-chunk 4096 --speculative none "$@"
