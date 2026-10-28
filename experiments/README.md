# Long-context candidates

The completed comparison covers nine 27B Bonsai configurations. All passed
the six OpenCode phases, filled capacity, and 107-schema replay. See the
[final measurements](../results/native-long-context-20261019/summary.md).
Matti canceled the 9B tests. Their existing raw data is retained, and no new
9B presets are added. The matrix has 14 definitions, of which nine are 27B cases.
The original
[13-case snapshot](../results/native-long-context-20261019/matrix-initial-13.json)
is preserved.

At 20:37 CEST, Matti stopped further Dory work. The
[Dory archive](../results/native-long-context-20261019/dory-current/summary.md)
retains the completed results. Dory4 passed the full OpenCode workflow and capacity check.
Its 107-schema hook was canceled. No Dory recovery runs or new presets follow.

The six-case continuation uses `experiments/benchmarks/runs/20261019-targets`.
It retains the archived Dory rounds 1, 2, and 4, followed by the three 27B cases:
active KV spill, stock medium reasoning, and MTP 2 with a 16K draft window and
MMA threshold 256. All three 27B continuation cases are complete.
The default is `27b-256k-native`: MTP 2, chaining on, `xhigh`, a 16K draft
window, and a 32 GiB prompt snapshot limit. Stock `27b-256k` remains available.
Plan commands remain offline by default.
Execution requires an explicit `--run` option.

## Native runtimes

- `bonsai-native` uses the current Prism revision with focused native patches.
  The patches add a bounded MTP window, wide-GQA MMA dispatch, a faster grammar
  copy, and explicit active KV spill to RAM. See [the source review](bonsai-native/source-audit.md).
- `dory-native` uses the pinned head of llama.cpp PR 30306. The model profile is
  `dory-gguf`. It uses NVIDIA's official Q5_K_M weights. See [the Dory notes](dory-native/README.md).

Source archives, revisions, file hashes, and build settings are in each
candidate's `manifest.json`. Source files are under `state/sources/`.
Build output goes under `state/builds/`. Neither runtime replaces the release
binary. The candidate manifest and binary must match the recorded build before
the manager starts a native candidate.

Both private CUDA 13 builds succeeded. The official Dory Q5_K_M file was fetched
and its pinned size and SHA256 were verified. Tool calls, prompt reuse, RAM
snapshot restore, cancellation, and generation after cancellation passed the
smoke checks. Explicit tiered KV placement passed on WSL at a 32K context with
1K and 8K VRAM heads. The full 256K spill case also passed the OpenCode workflow,
capacity check, and 107-schema replay. It moved 2.25 GiB of active KV to RAM.
The medium-effort case passed the same checks. The
[executed source snapshots](../results/native-long-context-20261019/executed-sources)
retain the exact benchmark source used by the runs.

## Print plans

Run these commands from `~/Code/local-models`:

```sh
uv run --no-project --python .venv/bin/python python experiments/build-native.py bonsai-native
uv run --no-project --python .venv/bin/python python experiments/benchmarks/run.py --cases 27b-native-kv-spill,27b-stock-medium,27b-native-window16k-mma

localmodel start 27b-256k-native --dry-run
localmodel start 27b-256k-native-mma --dry-run
localmodel start 27b-256k --prompt-lookup --lookup-type ngram-mod --lookup-tokens 32 --dry-run
localmodel start 27b-256k-native-spill --dry-run
localmodel start 27b-256k-medium --dry-run
```

The plan commands do not compile source, fetch weights, start servers, query
the GPU, or run inference. `localmodel start --dry-run` prints a server command.

## Build or repeat explicitly

The current builds and official Dory file are ready. This command describes
the explicit Bonsai build step for a repeat. Each build uses two compiler jobs, a memory
cap, and no swap through `heavy-management`. The build script refuses a known
running model server. It does not run inference after compilation.

```sh
uv run --no-project --python .venv/bin/python python experiments/build-native.py bonsai-native --run
```

The benchmark runner executes only with its explicit `--run` option.
See [the benchmark matrix](benchmarks/README.md) for cases and commands. Each case
starts a fresh server. The runner retains its own logs and request records.
It refuses an existing model server. It does not stop an unrelated service.

## Runtime controls

`--runtime bonsai-native` selects the patched Bonsai runtime. `--draft-window
16k` limits only the MTP draft context. The target retains its configured
context. Omit this flag to use the full draft context.

`--prompt-lookup --lookup-type ngram-mod --lookup-tokens 32` enables lookup with
its own draft limit. `--draft-tokens 2` still limits neural MTP drafting. This
lookup mode also works with the stock runtime.

`--kv-vram-cells N` keeps the first N active KV positions in VRAM. The native
runtime stages the remaining positions from RAM for GPU attention. This is
different from `--cache-ram 32768`, which retains prompt snapshots. The manager
adds 8 GiB to its process cap when active KV spill is selected. Both controls
can be used with GPU transformer layers and GPU attention computation.

`--mma-decode-min-kv 256` selects the native wide-GQA MMA threshold. Use zero
to disable that dispatch change. The grammar-copy change is built into the
native candidate.

The measured native choice is `27b-256k-native`: a 16K draft window with the
MMA override off. The optional `27b-256k-native-mma` adds threshold 256.
Both measured 64.5 tokens/s on the long workflow. The combined case reached
66.2 tokens/s at filled capacity, compared with 65.4 for the window-only case.
These single trials do not establish an additional MMA benefit at MTP 2.
The window-only run completed before the reported CPU load.

Use `27b-256k-native-spill` for measured GPU headroom. It traded long-history
throughput for a 2.25 GiB RAM tail. `27b-256k-medium` selects medium reasoning.
The default uses the native window configuration with xhigh. The OpenCode workflows began at
about 213K input tokens. Filled capacity used 262079 input tokens and 64 output
tokens. The lookup-friendly capacity result is separate from general workflow speed.

`dory-gguf --rounds 1`, `--rounds 2`, and `--rounds 4` select Dory loop counts.
Its defaults use a 256K window, Q8 last-loop KV, batch 2048, physical batch
1024, and a 32 GiB prompt snapshot limit. Use `--cache-ram 0` to reproduce the
reported no-snapshot-cache setting. Dory's last-loop predictions depend on
prefill chunk size. Context shifting is disabled because the PR does not
support it. OpenCode must compact before the context is full.

Run `opencode-local` after a model is ready. It discovers the model, context
capacity, and reasoning field from the running server. The Dory GGUF profile
uses the llama.cpp reasoning field. The existing vLLM Dory profile stays
separate.

## Review and verification limits

Source review covers the candidate changes and their local integration. It is
not a proof of native memory safety. SHA256 verifies artifact identity. It
does not prove that Q5 weights preserve every BF16 prediction. The
[Bonsai source review](bonsai-native/source-audit.md),
[Dory source review](dory-native/README.md#constraints-and-review),
[existing Bonsai tensor audit](../results/weights-audit.md), and
[Dory retained-vector audit](../results/native-long-context-20261019/dory-gguf-audit.md)
record the scope and limits. The Dory file has the official digest, but 158 of
217 retained BF16 vectors differ from the published BF16 checkpoint.
The audit does not establish export equivalence.

The final 27B report records all nine completed configurations and their checks.
It combines the baseline, candidate, and target continuation directories with
`--models bonsai2-27b`. Dory remains in its separate archive. See the
[saved-result command](benchmarks/README.md#read-saved-results).
The native full-draft and 16K-draft cases completed before the reported CPU load
began at about 18:45 CEST. The lookup case overlaps that interval. Later cases
also need the report's load qualification. The filtered queue snapshot found
no running jobs at 19:27 CEST, but this does not establish a quiet machine.
See [load events](benchmarks/load-events.json) and the
[filtered queue snapshot](../results/native-long-context-20261019/pueue-load-snapshot.json).
Published throughput remains separate from local measurements.
