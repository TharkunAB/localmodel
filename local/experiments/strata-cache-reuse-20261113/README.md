# Qwen cache-reuse backport

This candidate applies the three code and test changes from
[Strata commit 13eedf5524](https://github.com/Niko1221/Strata/commit/13eedf552417a6a791a7d13c6597e24300ee3a09)
to the pinned 0.2.13 source at `d2f096475817d4666c364c820b0ca830f751cfc5`.
`upstream.patch` preserves the upstream hunks. They apply with zero fuzz and
line offsets only. The generated `data/settings-register.json` update is
omitted. No other upstream commit is included.

The pinned source limits speculative cache commits at the remaining output
budget or the first EOS only on HIP. This patch enables the same limit for
CUDA at both speculative commit sites. It retains the merged-prompt exception.
`STRATA_SERVE_STOP_BOUND=0` disables the limit. The intended effect is to keep
the cached conversation consistent with the emitted response. Output identity
and continuation reuse on this GPU require a separate inference check.

The copied source, build directory, and runtime receipt are separate from the
signed release. The launcher uses the official executable by default. Select
this candidate with `--cache-reuse-fix`; omit it or use `--no-cache-reuse-fix`
to use the official executable again.

The signed Qwen 0.1.59 binary records source `f-0213-rc 712fec6b4297` in
`BUILD.json`. This candidate uses `d2f096475817d4666c364c820b0ca830f751cfc5`
plus the backport. A comparison against the signed binary therefore includes
source and toolchain differences. Use this same candidate binary with
`STRATA_SERVE_STOP_BOUND=0` and `STRATA_SERVE_STOP_BOUND=1` to isolate the
effect of the fix. Keep the signed binary as a separate reference.

## Build and checks

Run from the repository root. The first command prints the build plan. The
second prepares and compiles the candidate. It requires the installed official
0.2.13 source and the CUDA 13.0.88 toolchain from the earlier cache-route build.
It verifies their recorded file hashes before compilation. It uses the
receipt-pinned vendored ggml in 0.2.13, GCC 13, CMake, and Ninja.

```sh
uv run --no-project --python .venv/bin/python experiments/strata-cache-reuse-20261113/build.py
env XDG_RUNTIME_DIR=/run/user/1000 \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
  HEAVY_PEAKS="$PWD/state/strata-cache-reuse-build-peaks" \
  /home/tritlo/Code/heavy-management/bin/capped --mem 16G --time 90m \
  nice -n 19 uv run --no-project --python .venv/bin/python \
  experiments/strata-cache-reuse-20261113/build.py --run
```

The build requires a 16 GiB cgroup limit, zero swap, and nice 19. It uses two
compiler jobs and builds only the `strata` target for CUDA `89-real`. No model
loads during preparation or compilation. The upstream CPU test checks 2,736
accepted-window, output-budget, and EOS combinations. It also checks the four
environment-switch cases. Assertions remain active in the Release test.

`source-files.json` pins only the three changed files. All other source files
must match the installed official source receipt. `build-receipt.json` records
the binary, library hashes, build arguments, and dependency receipt. Its hash
is pinned in `runtime-pins.json`. This receipt describes the build only. Keep
later inference evidence separate. `qwen_cache_reuse.verify()` checks these
records and every prepared source file before the launcher starts the candidate.
It also rejects a changed official Qwen source revision or runtime path until
the backport is reviewed against that update.

The candidate requires its own `libstrata_kq.so` next to its executable. The
engine loads that exact path when it needs K-quant prompt kernels. The library
search path puts the candidate directory before the official Qwen CUDA 13
directory. The linker check covers both the engine and the separate kernel
library with this order. The official CUDA runtime and cuBLAS files remain
unchanged.

The CPU test and linker check establish bounded build evidence. They do not
establish GPU correctness, emitted-token identity, or faster continuation.
HIP, SYCL, Metal, and multi-GPU builds are outside this local backport check.

## Local validation, November 13

The guarded check used Unsloth IQ1_M, MTP4, 192K context capacity, a 64K GPU
KV window, 36 GiB of resident expert budget, and a 64 GiB process cap.
The first request contained 12,004 tokens. Sixteen requests used output limits
of 1 to 16 tokens. The later requests grew to 12,475 input tokens.
The candidate with the fix off, the candidate with it on, and the official
binary each received the same recorded request bodies.

All 16 outputs and output token-byte sequences matched across the three runs.
Each run emitted 29 tokens in total. These truncated responses were code
fences, so this sequence does not establish coding quality.
The fix changed accepted draft rows from 12 of 28 to zero of 28 in this
sequence. This confirms that it excludes rows beyond the output boundary.
It did not disable MTP: a separate normal-budget request accepted 24 of 40
drafts and returned the expected TypeScript sum function. The official binary
returned the same complete function.

All runs retained the long prefix. Continuations read 449 fresh tokens with
the fix off and 459 with it on. The official binary matched the off case.
This workload did not reproduce a full-prefix fallback or show a cache-reuse
gain. The official executable remains the default. Use the verified candidate
explicitly when needed:

```sh
localmodel start qwen3.8 --cache-reuse-fix --tui
```

A separate GHC build was active. The runs are correctness checks, not clean
speed measurements. Peak model cgroup memory was approximately 64 GiB.
Every run had zero swap and zero OOM events. Available Linux memory stayed
above 61 GiB. The native processes retained the outgoing-network filter.
All three controllers stopped and the previous server configuration was
restored. Evidence and both driver versions are under
`results/strata-cache-reuse-20261113/`.

The initial runner mistook the idle Ollama service for an active model.
The corrected check requires its `/api/ps` result to contain no loaded models.
The first inter-run port check also rejected a socket in TIME_WAIT. It now
checks for a listening server. These runner corrections did not change
the request bodies, candidate executable, or inference settings.

The live `--tui` check received telemetry from Qwen at the normal settings.
Pressing `q` retained the server and its metadata. The terminal modes and
alternate screen were restored. `localmodel stop` then stopped the controller
and native engine. The previous configuration was restored. Separate PTY and
fake-server checks covered Ctrl-C, external termination, startup failure,
monitor failure, and server exit. Strict Pyright passed for the changed modules.

The 2026-11-13 build passed all 171 compilation steps and peaked at 4,018 MiB.
The final incremental link pass peaked at 253 MiB. Both runs exited with code
zero. The effective memory limit was 16 GiB with zero swap. Readiness verified
all 6,112 prepared source files. Strict Pyright and both linker checks passed.
Local build logs are `results/strata-cache-reuse-build-20261113.log` and
`results/strata-cache-reuse-finalize-20261113.log`. Resource records are in
`state/strata-cache-reuse-build-peaks`.
