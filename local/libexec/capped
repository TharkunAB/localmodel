#!/bin/bash
# Run a command under a memory cap (a systemd user scope with MemoryMax and
# no swap, so the kernel kills the scope, not the VM) and an optional time cap.
# When the command ends, the scope's peak memory is appended to
# ${HEAVY_PEAKS:-~/.local/state/heavy/peaks} (date, command, cap, peak in MiB, exit code),
# so that later caps can follow the measured peaks.
# Usage: capped [--mem SIZE] [--time DURATION] command...
# Exit 124 means the time cap, 137 the memory cap.
# When CAPPED_UNIT is set, the scope gets that name, so that heavy can stop it.
mem=16G
time=
while [ $# -gt 0 ]; do
    case "$1" in
        --mem) mem=$2; shift 2 ;;
        --time) time=$2; shift 2 ;;
        *) break ;;
    esac
done
label=$(basename "${1:-?}")
case $label in bash | sh) label="$label $(basename "${2:-?}")" ;; esac
if [ -n "$time" ]; then
    set -- timeout --kill-after=30 "$time" "$@"
fi
export CAPPED_LABEL="$label" CAPPED_MEM="$mem"
exec systemd-run --user --scope --quiet ${CAPPED_UNIT:+--unit="$CAPPED_UNIT"} -p MemoryMax="$mem" -p MemoryHigh="$mem" -p MemorySwapMax=0 \
    "$(dirname "$(readlink -f "$0")")/capped-record" "$@"
