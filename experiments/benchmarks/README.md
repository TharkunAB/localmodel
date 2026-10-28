# Native benchmark matrix

The completed comparison covers nine 27B Bonsai configurations. All passed
the six OpenCode phases, filled capacity, and 107-schema replay. See the
[final report](../../results/native-long-context-20261019/summary.md).
Matti canceled the 9B tests. Existing 9B raw data is retained, and no new 9B
presets are added. The matrix retains 14 definitions, including nine 27B cases.
The two 9B and three Dory definitions are archived and excluded by default.
Explicit case or stage selection can still read those definitions in an offline plan.
The original [13-case snapshot](../../results/native-long-context-20261019/matrix-initial-13.json)
is retained in the result directory.
Matti stopped further Dory work at 20:37 CEST. Its completed results remain in
the [Dory archive](../../results/native-long-context-20261019/dory-current/summary.md).
Dory4's schema hook was canceled. All three 27B continuation cases are complete.
Both private CUDA 13 builds succeeded. The official Dory Q5 file was fetched
and its pinned size and SHA256 were verified. The interactive default is now
`27b-256k-native`: MTP 2, chaining on, `xhigh`, a 16K draft window, and a 32 GiB
prompt snapshot limit. The stock benchmark control remains `27b-256k`.

`run.py` prints and validates an offline plan by default.
It reads local files only. It does not contact an endpoint, tokenize text,
start a service, compile a runtime, fetch weights, or inspect GPU counters.
Missing model assets appear in the plan. They do not trigger installation.

```sh
.venv/bin/python experiments/benchmarks/run.py --stage baseline
.venv/bin/python experiments/benchmarks/run.py --stage native --json
.venv/bin/python experiments/benchmarks/run.py --cases 27b-native-kv-spill,27b-stock-medium,27b-native-window16k-mma --json
```

`matrix.json` contains named stages rather than a full factorial sweep.

| Stage | Cases | Purpose |
|---|---|---|
| baseline | 27B stock, MTP 2, chaining on | Retain the measured production reference |
| native | 27B native, full draft attention and 16K draft attention | Isolate the draft window change |
| lookup | Native 16K, lookup 32 and MTP 2 | Keep lookup length separate from MTP depth |
| mma | Native MTP 1, added MMA dispatch off and threshold 256 | Compare the single-token decode route |
| combined | Native MTP 2, 16K draft window, MMA threshold 256 | Check whether the measured improvements combine |
| models | Archived Dory rounds 1, 2 and 4; canceled 9B definitions | Retain prior scope and raw results outside the current 27B report |
| optional | Native KV spill; stock medium reasoning | Separate optional hypotheses from defaults |

The full draft window omits `--draft-window`.
The 16K window passes `--draft-window 16384`.
The added MMA dispatch override is zero for the window and lookup comparisons.
Other MMA kernels remain available.
The optional KV case keeps 131072 KV cells in VRAM.
It passed the full OpenCode workflow, filled-context capacity, and 107-schema replay.
Tool calls, RAM snapshot restore, and
cancellation passed the native smoke checks. Explicit tiered KV placement
passed on WSL at a 32K context with 1K and 8K VRAM heads.
The full-context runtime placement check confirmed 2.25 GiB of active KV in RAM.
The separate 32 GiB limit retains prompt snapshots.
Dory uses Q8 KV and rounds. It does not receive a reasoning-effort override.

The six-case continuation is stored in `runs/20261019-targets`:

- Archived `dory-round1`, `dory-round2`, and `dory-round4`.
- Completed `27b-native-kv-spill` and `27b-stock-medium`.
- Completed `27b-native-window16k-mma`: MTP 2, a 16K draft window, and MMA threshold 256.

Print the three 27B continuation cases with:

```sh
.venv/bin/python experiments/benchmarks/run.py --cases 27b-native-kv-spill,27b-stock-medium,27b-native-window16k-mma
```

The completed window comparison uses MTP 2 with the MMA override disabled.
The separate MMA comparison uses MTP 1 with the full draft context.
The continuation checked the combined window and MMA configuration at MTP 2.
The window-only and combined cases both measured 64.5 tokens/s on long OpenCode
history. Their filled-capacity rates were 65.4 and 66.2 tokens/s.
These single trials do not establish an additional MMA speed benefit at MTP 2.
The measured native choice is the window-only `27b-256k-native` preset.
Its run also completed before the reported CPU-load interval.

The default hooks are `opencode,capacity`.
`opencode` reuses the staged fixture and native OpenCode history importer.
It reads public pinned source excerpts. It does not read private conversations.
It records three related code repairs, fact recall, compaction, and follow-ups.
It requests about 205000 source tokens. The actual request counts determine
the measured input length. A configured 256K window is not a filled input.
Each reply can use 16000 output tokens. External semantic tests check repairs.

`capacity` reuses the exact-token capacity harness through the recorder.
It sends 262079 input tokens and requests 64 output tokens with a one-token
context guard. It checks capacity and speed. It does not measure answer quality.
It runs after the OpenCode workflow, so the first OpenCode turn remains cold.

`grammar107` is an optional separate hook. `tools-107.json` contains 107
constructed schemas. The replay checks the returned tool name and JSON arguments.
It never executes a returned tool call. It is a parser and grammar test,
not an OpenCode coding task. It sizes about 60000 public reference tokens only
when `--run` is selected. The tool schemas add to this input. Actual request
usage determines the total. The selected call contains twelve structured
checks and release metadata. This requires a nontrivial JSON response.
This controlled fixture does not reproduce a published throughput result.
Use `replay-tools.py --source-tokens 0` to plan the separate short replay.
Plan the default replay with:

```sh
.venv/bin/python experiments/benchmarks/run.py --cases 27b-stock-mtp2 --hooks grammar107
.venv/bin/python experiments/benchmarks/replay-tools.py --model bonsai2-27b
```

The explicit `--run` option executes the selected plan.
Use a new output directory for every attempt. Start with the baseline stage.
Build and fetch prerequisites in separate steps before execution.
The runner refuses existing inference services and occupied private ports.
It starts a fresh server for each case and stops only the server that it owns.
It reuses the recording proxy, resource sampler, and owned cleanup helpers.
The inference scope must have a finite RAM cap and no swap.
OpenCode runs in its private 4 GiB scope. Existing user OpenCode and search
services stay outside this experiment.

Each case retains the launch command, live settings, scope limits, resource
samples, exact request and response artifacts, native timing counters, tool
and correctness results, and errors. Compare cold prefill, warm workflow costs,
actual input counts, and output lengths separately. A model must pass the
repair, tool, recall, and compaction checks before a speed result can select it.

## Host load qualification

[load-events.json](load-events.json) records reported CPU load from about
18:45 CEST on 2026-10-19. The end of that interval is not established.
The native full-draft and 16K-draft cases completed before that reported start.
The lookup case overlaps it. The report marks overlapping measured cases.
Do not describe their workflow wall times as quiet-machine results.
CPU and RAM load can affect prefill, tools, compaction, and total workflow time.
Retain native GPU decode measurements with the same qualification.

The filtered live queue snapshot found no running jobs at 19:27 CEST.
Seven CPU jobs ran from 19:03 to 19:19:42 CEST. This does not exclude other CPU
or Windows activity. The separate data-directory queue was stale.
Use the [filtered snapshot](../../results/native-long-context-20261019/pueue-load-snapshot.json),
not raw queue records. An interval with no reported overlap does not establish
an idle machine.

See the [Bonsai source review](../bonsai-native/source-audit.md),
[Dory source review](../dory-native/README.md#constraints-and-review),
[Bonsai tensor audit](../../results/weights-audit.md), and
[Dory retained-vector audit](../../results/native-long-context-20261019/dory-gguf-audit.md).
Artifact hashes establish identity. They do not prove numerical equivalence.

## Read saved results

`summarize.py` reads saved JSON and small text logs. It does not contact model
services or read weights. Its default input is `runs/20261019-quiet-baseline` under
this directory. The target-only report combines that baseline, the candidate
directory, and `runs/20261019-targets`. It filters the model profiles to
`bonsai2-27b`, so retained 9B and Dory records do not enter the comparison.
It writes `summary.json` and `summary.md` to
`results/native-long-context-20261019` under the repository root.

```sh
.venv/bin/python experiments/benchmarks/summarize.py \
  --root-directory experiments/benchmarks/runs/20261019-quiet-baseline \
  --root-directory experiments/benchmarks/runs/20261019-quiet-candidates \
  --root-directory experiments/benchmarks/runs/20261019-targets \
  --models bonsai2-27b
```

The report separates actual input counts, cold prefill, warm follow-ups,
native generation, compaction, capacity, and grammar replay.
It marks missing, unselected, running, and failed hooks separately.
The [executed source snapshots](../../results/native-long-context-20261019/executed-sources)
preserve the exact benchmark code. Dory results remain in the separate archive.
Published rates are external references. They are not local measurements.
