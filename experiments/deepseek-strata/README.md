# DeepSeek 1017 UD-IQ1_S on SSD

The active backend uses official [Strata](https://github.com/Niko1221/Strata).
The API source is pinned to the 0.2.7.1 tag, commit
`46c152a1310f77cd2f0ce2c7cd213962df14fe83` (Control 0.1.53).
The native engine is the signed Linux CUDA 13 package `strata-dsv41` 0.1.54.
The API and engine use upstream
code without local inference patches. The older server avoids the
[0.2.8 parser regression](../../docs/strata-0.2.8-review.md).
The weights are Unsloth Dynamic V3.1 UD-IQ1_S at revision
`8a6ec9cf118e785a362ea94ee123c61c70553be6`. All five files have pinned
SHA-256 values. The last shard contains the IQ4_NL Engram table.
The complete download is 232,979,608,420 bytes, about 217 GiB.

The default model profile has 256K context capacity, a hard 64 GiB RAM cap,
a 36 GiB expert RAM cache, 4096-token prefill chunks, four CPU expert threads,
and port 8899. Swap and routed-expert skipping are off. OMP and OpenCode
send explicit `xhigh` requests. DSpark is optional. The launcher
passes `--experts 6`. The five-expert Faster mode is off. These defaults
leave RAM for other programs and for buffers and charged file cache inside
the process cap.

The historical performance measurements used the previous ISTA GSQ-RCO
IQ1_M weights. They do not measure the new Unsloth weights.
The performance measurements under the historical heading came from Abyss.
Do not use them as stock Strata performance results. Filled 256K context
remains unverified.
The advertised 1M context has not been tested here.

## Start and tune

Run these commands from the existing installation:

```sh
localmodel fetch deepseek
localmodel start deepseek
```

Keep SearXNG running. Start `omp-local --mem 8` from the project where you
want to work. OMP discovers the live model and context. Its RAM cap is
separate from the model cap. `opencode-local` can use the same API.
Use `localmodel status --json` to inspect the current settings.
Use `localmodel stop` when you finish or before you switch models.

`localmodel start deepseek --dry-run` prints the command and configuration.
These controls apply to this backend:

| Flag | Default | Function |
| --- | --- | --- |
| `--context` | `256k` | Maximum context capacity |
| `--mem` | `64` | Hard process-tree RAM cap in GiB; swap disabled |
| `--expert-ram-gb` | `36` | Expert RAM cache in GiB, inside the hard cap |
| `--cpu-threads` | `4` | CPU expert compute threads |
| `--gpu-expert-fraction` | `1` | Fraction from 0 to 1 of the available GPU expert budget |
| `--vram-reserve-mib` | `700` | VRAM reserve outside model allocations |
| `--engram-cache-mib` | `256` | Hashed-embedding row-cache budget; zero disables |
| `--engram-threads` | `16` | Hashed-embedding SSD read threads |
| `--prefill-chunk` | `4096` | Positive prompt-processing chunk size in tokens |
| `--thinking-budget` | `32768` | Maximum reasoning tokens per request |
| `--speculative` | `none` | Use `dspark` to load and enable the optional drafter |
| `--draft-tokens` | `5` with DSpark | Maximum DSpark candidates to verify, from 1 to 64 |
| `--dspark-conf` | `0.65` | DSpark confidence threshold; requires `--speculative dspark` |
| `--reasoning-effort` | `xhigh` | Harness request effort; maps to DeepSeek maximum |
| `--port` | `8899` | Local API port |

The old `--gpu-expert-cache-gb` flag is rejected. The stock engine takes a
fraction rather than a GiB value. `--expert-ram-gb` still uses GiB. The
launcher converts this value to the decimal GB unit of stock `--ssd-cache`.
The 36 GiB default becomes `--ssd-cache 38.654705664`.

Use `--expert-ram-gb N` to change the expert cache within the 64 GiB process cap.
Do not set the expert cache equal to the process cap. The engine also needs
RAM for dense weights, buffers, and charged file cache. The hard cap includes
child processes. The normal server does not monitor free Windows RAM.
`--no-offload` cannot fit these weights on this GPU. Bonsai MTP, llama.cpp
KV cache, and Flint expert-cache controls do not apply to this runtime.

## Optional DSpark drafter

```sh
localmodel fetch deepseek --speculative dspark
localmodel start deepseek --speculative dspark --draft-tokens 5
# Start without draft allocations:
localmodel start deepseek --speculative none
# Remove only the draft weights and their pack:
localmodel remove deepseek --speculative dspark --dry-run
localmodel remove deepseek --speculative dspark
```

The Q8_0 drafter uses 7,976,914,240 bytes on disk. Its SHA-256 is pinned.
The draft pack has a separate receipt and does not copy the main weights.
The normal `localmodel remove deepseek` command removes both variants.

Strata allocates draft experts before main-model experts. Some draft experts
can remain in RAM. They remain subject to the 64 GiB process cap, but are
outside the `--expert-ram-gb` store. They can reduce the GPU cache available
to the main model. The depth setting limits verification candidates. The
drafter still computes its fixed five-position block.

This mode passes native `--dspark on`. The native `auto` mode normally
disables DSpark when expert weights greatly exceed GPU capacity. Native
`off` keeps the draft allocations; this launcher uses `none` to unload them.
See the [Strata guide](https://github.com/Niko1221/Strata/blob/v0.2.8/docs/DEEPSEEK41.md).

### Short IQ1_S checks, 2026-10-30

All three configurations loaded at 262144-token capacity under a 64 GiB
cap with swap disabled. The GPU was an RTX 4070 SUPER with 12 GiB VRAM.
The requests used 17, 29, and 308 prompt tokens. They generated 8, 42, and
41 tokens. Each configuration ran once. The run order was DSpark2, none,
then DSpark5. Temperature was zero and thinking was disabled.

| Configuration | Code decode tok/s | Tool-call decode tok/s | Draft acceptance, all requests | Peak cgroup RAM | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| No draft | 4.8 | 5.2 | — | 35.64 GiB | 10268 MiB |
| DSpark2 | 4.0 | 4.4 | 59 / 63 | 42.87 GiB | 10348 MiB |
| DSpark5 | 4.6 | 5.5 | 73 / 85 | 35.24 GiB | 10348 MiB |

All runs answered the known-answer question correctly and produced the
requested structured tool call. They returned identical code. That code
only removes adjacent duplicates, so this is not evidence of coding quality.
No run swapped or recorded an OOM event. Linux available memory stayed
above 64.65 GiB. Peak RAM includes charged file cache and depends on prior
cache state. Do not use these peaks as a stable memory requirement.

DSpark5 had mixed timing results. DSpark2 was slower in both longer replies.
Keep DSpark optional until a representative long-conversation test shows a
benefit. Five draft tokens is the default when DSpark is selected.
These checks do not fill the context or run OMP. All test servers were stopped.
Raw responses and samples are in the local, untracked directory
`results/deepseek-iq1s-dspark-20261030/`.

### 16K OMP comparison, 2026-10-30

Both runs used the real OMP launcher with `xhigh`, 256K capacity, a 64 GiB
process cap, a 36 GiB expert cache, and six active experts. The conversation
contained constructed tool history from pinned public source. The task,
three recall facts, and validator were the same in both runs. The earlier
attempt was canceled at the user's request and is excluded.

| Metric | No draft | DSpark5 |
| --- | ---: | ---: |
| Initial actual input tokens | 15963 | 15961 |
| Initial prefill tok/s | 275.41 | 276.10 |
| Weighted decode tok/s | 4.98 | 4.60 |
| Generated tokens | 562 | 1347 |
| Repair plus resumed follow-up | 211.19 s | 422.34 s |
| Model requests | 5 | 7 |
| Tool errors | 0 | 1 |
| Draft acceptance | — | 881 / 1114 (79.08%) |
| Peak model cgroup RAM | 38.28 GiB | 39.50 GiB |
| Peak total GPU memory | 10268 MiB | 10348 MiB |
| Repair, 100 assertions, three-fact recall, strict JSON resume | Pass | Pass |

Both runs reused earlier context. The final requests contained 16750 and
18977 tokens respectively. DSpark recovered from one command rejected by
the benchmark's approval rules. The test fixture remained unchanged.
Neither run swapped or recorded an OOM event. Linux available memory stayed
above 60.80 GiB. All benchmark servers and workers stopped. The existing
SearXNG service was left running.

DSpark decoded about 7.7% slower in this comparison. Its longer elapsed time
also reflects more generated text and tool steps. Do not attribute the full
time difference to drafting. These are single passes with background CPU
activity and sampling enabled. They do not establish a general slowdown or
measure a filled 256K context. Automatic compaction was disabled.
Keep local DeepSeek drafting optional on this evidence.

Both local and remote native engine pins matched the current Strata 0.2.8
release during the requested recheck. The 0.2.7.1 frontend remains pinned
because the [parser regression](../../docs/strata-0.2.8-review.md) still
reproduces in the current release. No dependency changed for the repeat.

The local, untracked [report](../../results/deepseek-iq1s-dspark-omp-20261030/summary.md)
contains per-request timings, context reuse, memory events, and cleanup
evidence. Its directory also contains the driver and comparison script.

## Historical stock startup checks with GSQ IQ1_M

At 256K capacity, the stock automatic 8192-token prefill chunk left 0.58 GB
of GPU expert weights, or 79 experts. A 512-token chunk left 3.24 GB, or
396 experts. Both starts retained the same context capacity. Prompt buffers
are allocated before expert placement. The smaller chunk leaves more VRAM
for experts. These allocation checks do not establish prompt or decode speed.

The first guarded API runs used 36 GiB and 20 GiB expert caches. They reached
37,223 MiB and 20,918 MiB of charged process-scope RAM. Windows free RAM
crossed the 8 GiB floor, so the guard stopped both runs. Neither recorded
an OOM event. A concurrent `bench-huge-list` process used about 37 GiB of
RAM during these checks. It was left unchanged. That configuration used
the smaller 8 GiB cache. Its short OMP check passed.

## Official runtime and API

[manifest.json](manifest.json) pins the source archive, signed engine feed,
engine archive, binary, Python dependencies, model files, and SHA-256 values.
The installer verifies the feed signature and the package file manifest.
It uses the bundled CUDA libraries from the official prebuilt engine.
It does not use the old Abyss or Project Maya checkouts, local patches,
GGML build, compiler tools, or the Dory private CUDA toolkit.

The isolated Python environment has four pinned packages: Jinja2,
MarkupSafe, regex, and psutil. The selected serving and tokenizer paths do not need
the full Strata desktop dependency set. The packer uses the Python standard
library. The installer uses `uv` to create this environment.

Inspect the runtime plan, check the installed files, or install missing
pinned files without loading the model:

```sh
python3 experiments/deepseek-strata/runtime.py
python3 experiments/deepseek-strata/runtime.py --check
python3 experiments/deepseek-strata/runtime.py --run
```

The launcher runs stock `serve/server.py` and `strata-dsv41` through an
outgoing-connection guard. It does not change inference or API code.
The model configuration sets `arch` to `deepseek41`. This selects the
DeepSeek chat template, DSML tool-call parser, stop token, and reasoning
mapping. Request-level `xhigh` maps to DeepSeek `max` (100). The stock
server rejects `xhigh` and `max` in its configuration-default field, so that
field is omitted. Bare API requests use the model's `high` default (75).
OMP and OpenCode send their explicit `xhigh` setting. `--thinking-budget`
maps to the stock `reasoning_budget_tokens` setting. The API supports streaming,
`reasoning_content`, forced tool choice, `/v1/models`, `/props`, and `/slots`.
OMP and OpenCode use the same model ID and live context discovery.

The API binds only to `127.0.0.1` and starts with `--offline`. It has no MCP
servers, remote image sources, or before-load commands. The stock Host and
Origin checks remain active. No extra CORS origins are configured. The API
request monitor is off. Hardware metrics remain local to the server.
Offline mode is an application policy. The launcher also applies
operating-system restrictions. Runtime startup does not download model
files or engine updates.

The launcher installs a libseccomp filter before it starts Strata. The filter
blocks outgoing TCP and Unix-socket connections, UDP, addressed sends,
descriptor transfers, and io_uring. Native children retain the filter.
The stock API continues to accept local requests and send replies.
The guard closes inherited descriptors other than standard input and output.
A policy failure prevents startup. There is no unrestricted fallback.
The host already has libseccomp. No new Python package is required.
File access and listening sockets are not restricted by this guard.

`localmodel fetch deepseek` is a separate network operation. It downloads
the pinned weights and runtime. These serving restrictions do not apply to
OMP or OpenCode tools. Those tools still need your project files and SearXNG.
The review covers the selected source paths and package checks. It does not
prove that the signed native binary matches a reproducible source build.
See the [audit and model-free checks](../../results/strata-official/20261025-security/audit.md).

The latest 16K OMP runs did not retain exact RAM expert-cache hit or miss
counts. The native one-shot command reports those counters, but its server
protocol does not export them. The stock Python API also discards the
coarser native `STAT` lines. Older `glm stat` entries came from Abyss.
Do not use those entries as current official Strata miss rates.

Disk-backed conversation caching is off (`STRATA_DS4_SLOT_KEEP=0`).
Matching prefixes can be reused in RAM. Automatic shared-system-prefix
checkpoints remain on. This does not guarantee reuse after a harness rewrites
old messages. Stock prompt-skip mode 1 remains enabled. It omits work that
cannot affect the requested final rows, subject to floating-point summation
order. Routed-expert skipping is off. Approximate prompt-skip mode 2 is not
enabled. No speculative drafter is loaded.
The native `expert-profile.bin` file records expert-routing counts.
It does not contain conversation text.

## Storage and lifecycle

The five files occupy 232,979,608,420 bytes, or about 217 GiB.
The first four shards occupy 122.38 GB. The IQ4_NL Engram shard
occupies 110.60 GB. The directory is
`/home/tritlo/Code/local-models/models/deepseek41-1017-unsloth-iq1-s`.
This uses WSL's ext4 filesystem in the existing C: virtual disk.

The stock packer creates `ds4.txt`, `tensors.txt`, `engram.bin`, and five
files in `tokenizer/`. A separate receipt records their checksums.
`tensors.txt` points to the existing GGUF files and tensor offsets.
The packer reads those files without changing or requantizing their weights.
It reads the Engram metadata directly. It does not use the former
Abyss BF16 conversion pack or `--dense-override` and `--skip-tensors`.

`localmodel fetch deepseek` installs the pinned runtime, resumes missing
weight downloads, and verifies each completed file's size and SHA-256 value.
A file-identity receipt avoids repeated full reads while verified files stay
unchanged. The download does not execute code from Hugging Face.
Pack creation runs under a separate no-swap RAM cap.

Preview model deletion before you run it:

```sh
localmodel stop
localmodel remove deepseek --dry-run
localmodel remove deepseek
```

Removal deletes the five model files and generated pack. It keeps the
runtime, configuration, and benchmark evidence. It rejects active use,
unknown files, and symbolic links in the managed directory.
Run `localmodel fetch deepseek` to restore the files.

## Strata 0.2.6: 48 GiB and 64 GiB benchmarks

Both completed cases used six routed experts, xhigh requests, a 4096-token
prompt chunk, and 262144-token capacity. The constructed OMP history filled
about 16K tokens. Tool exchanges then grew the conversation. Both cases
passed the code repair, 100 assertions, three recalled facts, and resume.
The non-target functions and test file stayed unchanged.

The 48 GiB cap with a 36 GiB expert cache gave 257.7 prompt tokens/s and
5.41 decode tokens/s. Peak model-scope RAM was 40.28 GiB. Windows kept at
least 14.88 GiB free. The repair and resumed request took 221.6 seconds.

The 64 GiB cap with a 48 GiB expert cache gave 262.0 prompt tokens/s and
5.58 decode tokens/s. Peak model-scope RAM was 59.05 GiB. Windows kept at
least 10.11 GiB free. The repair and resumed request took 593.4 seconds.
The model generated 2639 tokens, compared with 654 in the 48 GiB case,
and recovered from two failed tools. Both settings passed all quality checks.
These are single passes with different responses. The rates do not show
a clear decode gain from the larger cache.

The first 64 GiB attempt crossed the 8 GiB Windows floor and was stopped.
It did not reach the model's memory cap or an OOM. Before the retry, another
job that used about 12 GiB finished. A cache check found 8.50 GiB of model
files in Linux file cache. The retry released clean cached pages for those
five GGUF files. It did not change weights or drop other files' cache.
Its routing profile was warmer. Do not treat this as an isolated RAM A/B test.

Use `--mem 64 --expert-ram-gb 48 --prefill-chunk 4096` for the larger profile.
The default at the time used 48 GiB, a 36 GiB expert cache, and 4096-token chunks.
The current cap is 64 GiB after the WSL memory increase.
The normal server does not include the benchmark's Windows headroom guard.
The model and guard were stopped after validation. Filled 256K remains untested.
See the [complete report](../../results/strata-official/20261025-update-bench/summary.md)
and [raw summary](../../results/strata-official/20261025-update-bench/summary.json).

## Earlier stock Strata 0.2.4.2 validation

The real OMP launcher passed a file-read tool roundtrip with its normal
core prompt and explicit `xhigh` requests. The initial input was 2801 tokens.
Prefill measured 55.2 tokens/s. Decode measured 4.0 tokens/s for the tool call
and 3.6 tokens/s for the final answer. The follow-up reused 2862 tokens and
processed 23 new tokens. The complete roundtrip took 73.8 seconds.

Peak model scope RAM was 16.79 GiB against the 48 GiB cap. Swap use was zero.
The scope recorded no memory-cap hits or OOM events. Windows free RAM stayed
above 16.68 GiB; Linux available RAM stayed above 38.18 GiB. Another job used
about 37 GiB of RAM. These are functional smoke-check figures on a busy
machine. They do not establish performance at a filled 16K or 256K context.

The stock source, signed engine, and three Python dependencies passed
integrity checks. Twenty-one local CLI and read-only lifecycle/tokenizer
checks passed. Offline token counts matched both real API inputs exactly.
A temporary dummy-data removal fixture also passed before cleanup. No real
weights were removed. All five original checksum identities still match.
The obsolete Abyss/Maya/GGML source trees, old build, build environment,
conversion pack, custom API wrapper, build scripts, and patches were removed.
Historical measurements remain available. The server and test guard exited.

See the [report](../../results/strata-official/20261024-migration/summary.md)
and adjacent raw records. The cleanup list is
`state/strata-official-cleanup.json` in the project directory.

## Historical Abyss measurements

These results used the former patched Abyss runtime on 2026-10-24.
They do not measure official Strata. All rows used four CPU expert threads,
exact routing, and reasoning off.
Decode rates exclude loading and prompt processing. Peak RAM includes the
scope's charged file cache. VRAM includes the Windows desktop.

| Check | Context | RAM cap / expert cache | Decode tok/s | Prompt tok/s | Peak RAM GiB | Peak VRAM MiB | First token incl. load, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Code generation | 4K | 32 / 20 GiB | 8.06 | 4.67 | 22.5 | 11664 | 44.69 |
| Code generation | 4K | 48 / 36 GiB | 9.59 | 5.38 | 38.6 | 11664 | 74.18 |
| Short prompt | 64K | 48 / 36 GiB | 7.03 | 4.69 | 38.5 | 11847 | 53.73 |
| Short prompt | 128K | 48 / 36 GiB | 7.05 | 4.92 | 38.8 | 11611 | 51.01 |
| Short prompt | 256K | 48 / 36 GiB | 6.79 | 4.60 | 38.5 | 11779 | 45.39 |
| Short prompt | 256K | 64 / 48 GiB | 8.17 | 5.70 | 50.5 | 11688 | 68.36 |

The code prompt has 40 tokens. Its answer has 205 generated tokens and stops
at EOS. The generated merge function passed its three assertions and 15,876
additional input pairs. The capacity prompt has 17 tokens and generates
59 tokens. Compare rows within each prompt group.

Peak whole-card VRAM was 11,611 to 11,847 MiB. At 256K, the native attention
KV allocation was 0.79 GiB. The automatically sized GPU expert cache was
3.89 GiB in the 48 GiB test.

The final 64 GiB run sampled Windows free memory at a minimum of 6.4 GiB.
Linux available memory stayed above 19.2 GiB. The runner stopped earlier
attempts under the previous 8 GiB Windows floor. These stops were headroom
checks, not out-of-memory failures. The final run used Matti's 4 GiB Windows
floor and an 8 GiB Linux floor. It checked both every ten seconds.
The normal server enforces its process cap. It does not run this host monitor.

IQ1_M expert tensors have no batched MMQ prompt path in this build.
Prompt processing uses one token at a time. A cold 10K-token prompt can take
about half an hour at the measured rate. Cached requests reuse matching
prefixes. This makes cold harness startup and compaction important limits.
The model's advertised 1M context has not been tested here.

Raw measurements are in `results/deepseek-strata/`. The generated code check
is in `state/deepseek-generated-code-check.json`. The headroom record is
`state/deepseek-64g-headroom.json`.

### Historical OMP integration check

The real OMP launcher passed a reply, a file-read tool roundtrip, and a resumed
conversation. The test used private state, one read tool, reasoning off, and
a short system prompt. The model used its 32 GiB cap and 256K context. OMP
had a separate 2 GiB cap. Production OMP settings and sessions stayed unchanged.

The cold tool request read 1,056 tokens in 153.18 seconds and generated
55 tokens at 7.6 tokens/s. OMP then executed the read tool. The result request
reused 1,055 tokens and processed 76 new tokens. Its answer took 9.55 seconds.
The complete tool roundtrip took 171.35 seconds.

The resumed turn recalled the file contents without another read. It reused
1,130 tokens, processed 32 new tokens, and completed in 7.78 seconds.
The live model scope peaked at 22.74 GiB with no swap or OOM events.
This is an API, tool, resume, and prefix-cache smoke check. It does not
validate long coding tasks, compaction, full tool sets, skills, or reasoning.
Evidence is in `state/deepseek-omp-smoke/summary.json`.

### Historical filled 16K OMP coding check

The separate filled-input check passed on 2026-10-24. It used a 48 GiB model
cap, a 36 GiB expert RAM cache, four CPU expert threads, and 256K configured
capacity. OMP had a separate 4 GiB cap. The test used the normal OMP core
prompt, read/edit/bash tools, `xhigh`, and a 4096-token output limit.
Constructed history came from pinned public source files. Private benchmark
state disabled skills, extensions, goal continuation, and automatic compaction.
Production settings and sessions stayed unchanged.

The first API request contained 15,969 input tokens. The final task request
contained 16,968. The model repaired the requested `clamp()` boundary bug,
passed 100 semantic assertions, and recalled all three held-out release facts.
The first test command used an unavailable `python` executable. The model
recovered with `python3`. Future benchmark prompts use `python3`.

| Measurement | Result |
| --- | ---: |
| Cold prefill | 2230.1 seconds; 7.2 tokens/s |
| Complete coding and recall task | 2495.7 seconds; 41.6 minutes |
| Decode, weighted across requests | 6.06 tokens/s; 701 generated tokens |
| Peak model scope RAM | 43.54 GiB |
| Peak whole-card VRAM | 11488 MiB |
| Minimum sampled Windows free RAM | 11.35 GiB |
| Minimum sampled Linux available RAM | 31.47 GiB |

The guard checked both hosts every five seconds. Neither host crossed the
8 GiB floor. Swap use was zero. The model scope recorded no memory-limit or
out-of-memory events. The primary task passed. This one case does not
establish general coding accuracy or published model-card scores.

All five follow-up requests within the same OMP process reused the prompt
prefix. A fresh OMP resume replaced an earlier file-read result with
`[Superseded by a newer read of this file]`. The system prompt and tool
schemas stayed equal. The native engine started a cold 17,002-token refill.
The probe was cancelled after 133 seconds, 752 processed tokens, and no output.
Resume is not verified by this longer check. This history pruning can make
long resumed sessions expensive even when automatic compaction is disabled.

Matti limited the run to 16K. The 32K, 64K, and 128K filled-input cases were
cancelled before generation. The driver now defaults to the 16K target.
The guard and benchmark workers exited. The model was stopped after the run.
Normal serving enforces its cap but does not retain this benchmark guard.
The [report](../../results/deepseek-filled-omp/20261024-0352/summary.md) and
adjacent raw records contain the measurements and request evidence.

## Historical runtime details

The measurements above used Abyss revision
`1f866d4480085008e7360db4204948b5b87c66ac`, derived from Strata through
Project Maya. That setup used local layout and API patches, a private CUDA
toolkit, and an extra GGML source tree. Its packer converted 88 small
projections to BF16 or F32. Original GGUF files retained their published
checksums, but those conversions could change arithmetic.

The stock migration removes those requirements. The historical reports and
raw request, memory, and timing records remain as evidence. Their runtime
paths and build commands describe the former setup. They are not current
installation instructions. The old `build.py` and Abyss benchmark commands
are not the stock Strata installation path.

## Update on 2026-10-27

Source 0.2.7 and engine 0.1.52 passed file and signature verification.
A filled 16K OMP run passed repair, 100 assertions, recall, and resume.
It measured 284 prompt tokens/s and 5.31 decode tokens/s with a 48 GiB cap,
36 GiB expert cache, and 4096-token chunks. Peak charged RAM was 41.09 GiB.
The server stopped afterward. See results/strata-027-20261027/comparison.json.
Vision is explicitly off. Six experts, the RAM cap, and the outgoing-connection
guard remain unchanged. The existing pack and its 0.2.6 generator remain pinned
separately from the serving source. The earlier benchmark section describes 0.2.6 only.
