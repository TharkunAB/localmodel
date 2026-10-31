# Strata 0.2.9 review

Reviewed on 2026-10-31. This review covers the changed Python serving code,
the DeepSeek packer, and the two new Linux CUDA Qwen GDN defaults. It is not a
complete security audit. The parser checks did not execute tools, load weights, use
a GPU, or connect to a remote deployment. One API fixture used a temporary
loopback server with scripted token input.

## Update decision

Use the full official 0.2.9 release: source, Python API, and packer at commit
`c62d5e3a15b40ea51c96aae9e8d36928965f53ab`, with native engines 0.1.55.
The earlier API hold is removed. This selects the stock server defaults and
the release's other parser fixes.

The quoted `</think>` counterexample from the
[0.2.8 review](strata-0.2.8-review.md#parser-regression) remains reproducible.
It is a constructed output case. No matching failure has been observed in a
real OMP workload, and these checks do not estimate its frequency. The finding
is retained below as a known limitation; it does not hold this release.

## Parser results

The original reproduction has 24 cases per source version. It declares a real
`bash` schema with a string `command` argument. It covers Qwen and DSML syntax,
recovery on and off, and chunk sizes of one character, seven characters, and
the whole input. The test inspects events only.

- 0.2.7.1 keeps all 24 quoted examples as text.
- 0.2.8 and 0.2.9 emit tool calls in the 12 cases with a quoted stray
  `</think>`. Their other 12 fenced examples remain text.
- Disabling recovery does not prevent this conversion.

The cause is unchanged. After reasoning ends, `_stray_think_at()` accepts a
line-start `</think>` inside a code fence. The content parser then clears its
Markdown state. A quoted call that follows becomes a tool-call event.
See the pinned
[detection code](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/serve/frontend.py#L1338)
and [state reset](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/serve/frontend.py#L1617).
DSML translates into this parser. Its implementation did not change from
0.2.8 to 0.2.9.

A second matrix has 448 cases per version. It adds thinking on and off,
two-character chunks, four- through six-backtick fences, tilde fences,
indented blocks, undeclared names, and genuine calls after closed code blocks.
All 128 genuine-call controls pass on each version. The failures among the
320 quoted or undeclared cases are:

| Source | Cases that emitted a quoted call |
| --- | ---: |
| 0.2.7.1 | 192 |
| 0.2.8 | 208 |
| 0.2.9 | 84 |

These selected edge cases are not a security score. The new version fixes the
four- and five-backtick nesting cases and ordinary indented Qwen examples.
Its 84 failures comprise 16 fenced stray-think cases, 32 six-backtick cases,
8 space-indented DSML cases, 8 tab-indented DSML cases, and 20 indented
stray-think cases. Ordinary indented DSML fails with recovery enabled.
The six-backtick and indentation defects also occur in the old source.
All failures produce final calls; none is only a streamed fragment.

All 16 methods in the upstream Markdown, quoted-value, tool-corpus, and
history-tool test modules pass. Those tests do not cover the counterexample
above.

A separate fixture replays the example through stock 0.2.9
`/v1/chat/completions`. It uses the upstream mock engine, tokenizer class with
a small vocabulary, DeepSeek template, and default thinking and recovery.
The closing-think special token passes through normal token decoding.
Streaming and nonstreaming responses both keep the ordinary fenced example as
text. Adding the quoted `</think>` produces a declared `get_weather` call and
`finish_reason=tool_calls`. The tool is not executed. This confirms that the
case does not require our older config. It does not show that a real model
will generate that output during ordinary use.

Audit scripts and machine-readable results are retained locally under
`state/strata-update-20261031/parser/`. `original-*.jsonl` records the original
reproduction. `extended-*.json` records all matrix cases. `stock-api-repro.py`
and `stock-api-repro.json` record the HTTP fixture. These files are ignored
runtime artifacts, not files in the published package.

## Serving code and outgoing connections

No new outgoing request or command execution was found in the reviewed parser
changes. The requirements file, telemetry module, MCP client, image helpers,
and offline helper are unchanged from 0.2.7.1. New server hardware checks call
`nvidia-smi` with fixed arguments. New replica storage uses local shared memory.

The new `with_history_tools()` accepts a tool name from an earlier assistant
tool call when the current request declares at least one tool. It can therefore
accept a name that is absent from the current tool list. An empty current tool
list remains empty. This is an intentional compatibility change included in
the selected API. Clients must still enforce their current allowed tools.

The existing seccomp wrapper remains the outgoing-connection boundary. It is
inherited by the engine and its child processes. The offline option alone is
not that boundary. Optional image and MCP code still exists in upstream source;
our serving config and network guard restrict its use. This review does not
claim that source inspection proves the behavior of every native binary.

## Existing Unsloth weights and packer

The new packer handles an ISTA-DAS set whose metadata declares six shards while
the caller supplies five model shards and a separate Engram file. The change
counts that explicit Engram file as the missing sixth shard. See
[engram_is_last_shard](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/tools/dsv41/pack_gguf.py#L366).

Our existing Unsloth five-shard set does not need this format change. The local
upgrade rebuilt the small normal and DSpark indexes with the 0.2.9 packer.
All eight checked pack assets are byte-identical to their previous versions.
The old packs and receipts remain in `state/strata-update-20261031/`.
The weights retain their existing checksum evidence and file identities.

Qwen's four preparation scripts, expert profile, and draft vocabulary are
byte-identical between the two source releases. Its prepared files are reused.
Their checked file identities and hashes did not change. The old artifact
receipt is retained, and its manifest binding was updated after validation.
The Python dependency versions also did not change. The existing environment
directory still has `0.2.8` in its name; it contains no installed Strata API.
Both launchers execute the verified 0.2.9 source directory.

## Qwen Linux CUDA GDN defaults

The stock 0.2.9 API supplies `STRATA_GDN_TC=1` and `STRATA_GDN_CONVL2=2` for
our Linux CUDA Qwen profile. The launcher does not duplicate those settings.
It retains `STRATA_ROUTE_PRIOR=0`. These reviewed values do not require a new
dependency, a new model pack, or a network permission. They apply to prefill.
They do not promise a decode speed increase.

The upstream server enables them only on Linux for `qwen4exp` with CUDA. It
excludes DeepSeek, Qwen3.6, Metal, SYCL, Vulkan, and Windows. It preserves an
explicit value. All seven upstream GDN-default test methods pass in CPU-only
checks.

The native TC dispatch requires at least 128 tokens in a prefill chunk,
compute capability 8 or later, and at least 48 streaming multiprocessors.
It checks shared memory before launch. The full path needs 124,944 bytes per
block. The split path for SM86, SM89, and SM120 needs 97,808 bytes, or 98,832
bytes with fused output normalization. Thus the full-path requirement does
not exclude Ada. Unsupported devices and failed scratch allocations use the
existing kernels.

The TC path allocates 60,305,408 bytes, about 57.51 MiB, of scratch space per
GPU on first use and retains it. This is not proportional to the configured
context limit. The initial native-only trial used a 700 MiB Qwen reserve. It
passed, but the native log reported only 468 MiB free. The launcher now
reserves 768 MiB. The extra 68 MiB covers the new scratch allocation. The
native-only repeat had no low-VRAM warning. It does not establish headroom
under other desktop loads or replace the full-release workload check.

`CONVL2=2` fuses convolution, SiLU, and Q/K normalization for chunks longer than
64 tokens. It uses registers and does not add a separate workspace. It changes
the FP32 reduction order. TC also changes rounding: it uses split FP16 products
with FP32 accumulation. Neither setting promises bitwise-identical output.
Do not enable the separate experimental `STRATA_GDN_TC_NP` settings.

These conditions come from the pinned
[native dispatch](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/prefill/kernels.cu#L1306),
[shared memory and scratch definitions](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/prefill/gdn_tc.cuh#L44),
and [fused convolution](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/prefill/kernels.cu#L401).

The [release measurements](https://github.com/Niko1221/Strata/releases/tag/v0.2.9)
report Qwen IQ3_S prefill gains of 5.5% at 16K on RTX PRO 5000 and 2.6% at 16K
on RTX A4000. The source records mean KL divergence 0.0019, maximum 0.0125,
and 55 of 61 suite tasks against 54 of 61. Upstream also reports a 30-minute
soak. These checks support a local trial. They do not establish the speed or
quality of our IQ2_XS profile on the RTX 4070 SUPER.

## Native DeepSeek behavior

Three native defaults also change on Linux CUDA:

- Post-prompt expert-cache settling drops from 6,000 to 800 ms
  (`DS4_SETTLE_MS`). See the
  [settling budget](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_model.cu#L7153).
- Idle settling is now enabled. It can move experts for up to 6,000 ms between
  requests (`DS4_SETTLE_IDLE_MS`). The server cancels it and joins the worker
  when the next request arrives. See the
  [idle loop](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_main.cpp#L716)
  and [cache work](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_model.cu#L7204).
- The CPU lane spins for 250 ms after its last answer before sleeping, up from
  2 ms (`DS4_LANE_SPIN_MS`). This can increase CPU use. See the
  [worker loop](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_model.cu#L7461).

These changes can affect time to the first token, cache state between tool
calls, and CPU use without a large change in full-prompt throughput. The new
automatic fast-attention and mapped-file changes that support the large
Strix Halo gains apply to Windows HIP. They do not enable those paths in our
Linux CUDA config. See the
[attention defaults](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_kernels.cu#L2080)
and [mapped-file default](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_model.cu#L3183).

## Local measurements

Both final runs used the full 0.2.9 API and packer with native 0.1.55 on the
RTX 4070 SUPER. Both configured 262144 context tokens. They used the existing
OMP repair, 100-assertion, three-fact recall, and resumed-follow-up workload.
The 8 GiB Linux available-memory floor and zero-swap process caps remained active.

| Final configuration | Initial input | Prefill tok/s | Decode tok/s | Generated tokens | Peak cgroup RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek IQ1_S, no draft, 48 GiB expert cache, 70 GiB cap | 65307 | 419.71 | 5.21 | 527 | 59.16 GiB |
| Qwen IQ2_XS, MTP3, 32 GiB expert cache, 64 GiB cap, 768 MiB VRAM reserve | 16035 | 431.28 | 47.95 | 856 | 64.00 GiB |

Both final runs passed code, all 100 assertions, recall, and strict JSON resume.
DeepSeek made five model requests with no tool errors. Qwen made seven requests
and recovered from one invalid edit request. Its MTP acceptance was 72.08%.
Neither run recorded an OOM or used swap. DeepSeek recorded no memory-limit
events. Qwen reclaimed file cache at its cap; `memory.events max` reached 105535.
Linux available RAM stayed above 51.56 GiB and 61.82 GiB respectively.
Neither final native log reported low VRAM. All benchmark servers and OMP
workers stopped. The existing SearXNG service stayed running.

The earlier DeepSeek native 0.1.54 run used the same cache and cap. It reached
5.22 decode and 423.04 prefill tokens/s. The new result is effectively unchanged.
Repair plus resume took 302.29 s instead of 392.64 s, but the new run generated
527 tokens instead of 956. Do not attribute the elapsed-time difference to the
release alone.

The earlier Qwen native 0.1.52 run reached 35.49 decode and 520.10 prefill
tokens/s. The final configuration has about 35% higher decode throughput and
17% lower prefill throughput in these runs. Its reserve also increased from
700 to 768 MiB. This comparison does not isolate the engine, API, GDN kernels,
or reserve change. Repair plus resume took 80.35 s, compared with 90.39 s before.
The output lengths differ: 856 versus 1300 tokens.

### Earlier native-only cases

These runs used native 0.1.55 with the older 0.2.7.1 API. They are retained
as evidence, but they do not describe the selected full 0.2.9 configuration.

| Configuration | Prefill tok/s | Decode tok/s | Quality result |
| --- | ---: | ---: | --- |
| DeepSeek 64K, 48 GiB cache, 70 GiB cap | 419.94 | 5.28 | All checks pass; two extra Python checks required approval |
| Qwen 16K, 700 MiB reserve | 549.00 | 47.19 | All checks pass; native log reported 468 MiB free VRAM |
| Qwen 16K, 768 MiB reserve | 443.07 | 49.08 | Code and recall pass; correct resumed JSON was fenced |

The two earlier Qwen runs each had two shell-approval rejections. The reserve
increase removed the low-VRAM warning. The strict-format failure is retained;
a later passing run does not erase it or prove which setting caused it.
Qwen also warns that its 32 GiB expert budget does not cover the prompt cache's
lent slots. These runs do not establish performance on 50K-plus Qwen prompts.

These are single stochastic passes. The reference runs came from earlier dates.
The driver released clean cached pages for each selected model's raw files;
it did not flush the global OS page cache. Automatic compaction was disabled.
This is not filled-256K, compaction, remote-GPU, or general quality evidence.
Raw requests, timings, drivers, memory records, and every attempted case remain
in the local directory `results/strata-029-20261031/`. `comparison.json` includes
the reference records and the earlier native-only cases.

## Deployment checks

Both local runtime installers verified the pinned source and signed native
packages. Qwen's live native environment confirmed that the stock API supplied
the GDN values. The actual API and native processes retained seccomp. Separate
smoke checks rejected IPv4 and IPv6 connects, UDP sockets, and connects after
child execution without contacting an external host.

The remote script passed 18 offline hardware/profile checks and four pack
migration checks. Its embedded verifier passed with the new source under the
network restriction. Modified feed data and a wrong binary hash were rejected.
No RunPod was accessed or restarted. These checks do not measure remote speed.
