# Official Strata 0.2.12 and 0.2.13 comparison

Use one controller and two evidence directories. Set `STRATA_BENCH_RELEASE` to
`v0.2.12` or `v0.2.13`. The controller and bootstrap retain this variable. The
production launcher removes inherited `STRATA_*` variables from the engine.

Run the old release before updating production runtime pins. Freeze the new
release after its signed artifacts and source pins are installed. Do not change
this harness or workload files between runs. Each release has separate runtime
pins. A version, file hash, or namespace mismatch stops the run. The benchmark
lock is shared across both evidence directories.

Use three separate cases for each release: 16,384, 32,768, and 65,536 input tokens.
The fixed context capacity is 196,608 tokens. Use full Unsloth IQ1_S weights,
six active experts, a 48 GiB expert cache, a 64 GiB cgroup cap, zero swap,
12 CPU threads, 4,096-token prefill chunks, a 700 MiB VRAM reserve, and no DSpark.
The sampling defaults are temperature 1, top-p 0.95, and seed 20261109. The short
warmup retains the earlier temperature-0 prompt. Every case starts from the same
frozen expert profile. This is a trained-routing replay.

The source harness is `experiments/strata-candidates-20261109/benchmark`.
The copied inputs are in `results/strata-0213-20261109/<release>`. They retain the
workload, source corpus, and routing profile from the morning comparison.
`source-ancestry.json` records the original harness hashes. The custom native
binary and community cache-routing controls are excluded from this harness.

The parent process owns SearXNG. Start it only if it is not already running.
Do not stop a service that was present before this work. Finish artifact downloads
and extraction before timed cases. Verify that no model or GPU workload is active.
Admission requires 72 GiB Linux available memory, GPU memory at most 2,048 MiB,
GPU utilization at most 5%, and one-minute CPU load at most 8. The memory guard
stops the owned worker and server below 8 GiB Linux available memory. It also
stops them on swap use or repeated memory-read failure. There are no Windows
memory checks. Keep the production network guard and `PI_EDIT_VARIANT=replace`.

From the repository root, freeze and preview the old release:

```sh
STRATA_BENCH_RELEASE=v0.2.12 state/venvs/strata-official-0.2.8/bin/python experiments/strata-0213-20261109/benchmark/freeze.py
experiments/strata-0213-20261109/benchmark/run-case.sh v0.2.12 --name official-16k --target 16384 --dry-run
```

Run the three cases serially after resource admission. Each command stops its
owned model before returning. Case directories must not already exist.

```sh
experiments/strata-0213-20261109/benchmark/run-case.sh v0.2.12 --name official-16k --target 16384
experiments/strata-0213-20261109/benchmark/run-case.sh v0.2.12 --name official-32k --target 32768
experiments/strata-0213-20261109/benchmark/run-case.sh v0.2.12 --name official-64k --target 65536
STRATA_BENCH_RELEASE=v0.2.12 state/venvs/strata-official-0.2.8/bin/python experiments/strata-0213-20261109/benchmark/cache_io.py
```

After the old cases finish and cleanup succeeds, install the new runtime pins.
Repeat these commands with `v0.2.13`. Use the same case names in its separate
evidence directory. Do not replace an existing `runtime-pins.json` or
`harness-pins.json`.

Check `case-summary.json` after every case. Require `server_stopped`,
`profile_restored`, `lane_profile_restored`, `state_config_restored`,
`active_state_restored`, and `normal_settings_unchanged`. Check worker cleanup,
port 8899, and `state/active.json` before the next case. Preserve failed cases.
The controller verifies the native executable hash, loaded libraries, seccomp,
launch arguments, API sampling, memory limits, and the captured edit schema.

`cache_io.py` records actual input, native decode and prefill, repair and resume
wall time, memory, GPU use, I/O, quality, and cleanup. Missing I/O is not zero.
Shared disk counters do not establish model cache misses. The existing semantic
repair check can accept JSON inside prose. The separate
`quality.repair_final_json` field requires the final visible reply to contain only
one JSON object with the four requested keys and values. It excludes reasoning
blocks and does not change the existing semantic score.
