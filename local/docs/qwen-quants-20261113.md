# Qwen Flash-Next quant measurements — 2026-11-13–14

**Keep IQ1_M as the default.** Its weighted repair/resume decode was
**50.86 tok/s**, compared with **39.32 for Q2_K_XL** and **37.11 for IQ3_XXS**.
All three completed four code/recall checkpoints through approximately 245K
input tokens and retained the four requested facts after OMP compaction.
IQ3's measured decode rate was 27.0% lower than IQ1_M's. Its combined
repair/resume and compaction time was 14.4% longer. IQ1_M also pins 14.43 GiB
less expert memory. IQ3 remains an optional choice with higher published
coding scores.

These are single-run workflow observations. Generated output, tool use and
later conversation histories differ. The repairs are simple and do not
establish general coding quality or a local quality ranking between quants.

The default expert budget is **48 GiB for all three quants**. The default
quant remains IQ1_M. The `lowram` profile selects zero resident expert
budget, and the model cap remains 64 GiB. All three retain Q8_0 Engram,
256K capacity and four MTP draft tokens. The completed comparison below uses
the 48 GiB budget and the `long` case. A separate Q2_K_XL run used 36 GiB;
the interrupted IQ3 run at that budget is excluded.

## Completed 48 GiB comparison

| Quant | Generated tokens / native decode seconds | Weighted decode tok/s | First prefill tok/s | Repair/resume seconds | Compaction seconds | Combined seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ1_M | 4,362 / 85.7656 | 50.8596 | 1,326.69 | 331.43 | 160.73 | 492.17 |
| Q2_K_XL | 2,670 / 67.9038 | 39.3203 | 1,203.24 | 311.36 | 221.17 | 532.53 |
| IQ3_XXS | 3,280 / 88.3784 | 37.1131 | 1,360.95 | 341.69 | 221.22 | 562.90 |

Weighted decode covers repair/resume generation only. It excludes prefill,
tool execution, startup and compaction. Combined time sums the measured
repair/resume and compaction phases; it excludes setup, prompt sizing and
startup. First prefill covers 15,997 new tokens with no reused prefix.

| Quant | Native pinned experts GiB | Peak sampled GPU MiB | Minimum Linux available GiB | MTP accepted / offered, repair/resume |
| --- | ---: | ---: | ---: | ---: |
| IQ1_M | 32.01 | 11,718 | 53.95 | 2,972 / 4,071 (73.00%) |
| Q2_K_XL | 42.44 | 11,657 | 49.01 | 1,909 / 2,466 (77.41%) |
| IQ3_XXS | 46.44 | 11,624 | 44.82 | 2,253 / 3,201 (70.38%) |

All three reached approximately 64 GiB in the model cgroup, including file
cache, without an OOM event or swap use. Each passed 4/4 semantic code checks,
4/4 recall checks and 4/4 strict resume checks. Each passed 0/4 strict repair
JSON checks because its final replies included prose and fenced JSON.
All three observed compaction and passed the four-fact retention check.

## Published quality and compatibility

[Unsloth's November 4 Dynamic v3.1 charts](https://unsloth.ai/docs/models/qwen3.8-next)
report these percentages from repeated stochastic benchmarks. Their error bars
overlap; these are not measurements of our Q8 Engram configuration. Q2_K_XL is
absent from these coding panels. IQ2_XS is a different quant.

| Benchmark | IQ1_S | IQ1_M | IQ3_XXS |
| --- | ---: | ---: | ---: |
| DeepSWE v1.1 | 54.0 | 49.6 | 55.1 |
| Terminal Bench 2.1 | 68.3 | 69.5 | 70.8 |
| LiveCodeBench v6 | 88.9 | 89.5 | 90.5 |
| Aider Polyglot pass@2 | 82.2 | 83.1 | 87.1 |

IQ1_S is incompatible with pinned Strata: its layer 18 has
[different gate/up types](https://github.com/Niko1221/Strata/blob/d2f096475817d4666c364c820b0ca830f751cfc5/tools/test_setup_unsloth_v31.py#L520),
IQ2_S and IQ2_XXS. The
[packer rejects this](https://github.com/Niko1221/Strata/blob/d2f096475817d4666c364c820b0ca830f751cfc5/tools/iq_pack.py#L452);
the [native format shares `gu_type`](https://github.com/Niko1221/Strata/blob/d2f096475817d4666c364c820b0ca830f751cfc5/src/core/expert_source.cpp#L5674)
between them. These published scores remain a reason to try
IQ3_XXS on harder coding tasks; the local fixture does not verify that advantage.

## Common settings and provenance

All completed cases used official Strata API **v0.2.13**, source
`d2f096475817d4666c364c820b0ca830f751cfc5`, and native **0.1.59**. The CUDA 13
[engine archive](https://github.com/Niko1221/Strata/releases/download/v0.2.13/strata-engine-0.1.59-linux-x86_64-cuda-cuda13.tar.gz)
has SHA256 `42ab04fa3201e41bef5fac28c05f5c35c909419ec937d879b505bf1516742610`.
The executable has SHA256
`0ba914b28df1c8e09c0b5209c2ca6040c75fb650a7408ff3d9ef3ad159a13eb9`.

The four original Q2_K_XL shards are pinned to Unsloth revision
`f5d175de2cf7f720d45d6064019a81fcf056e986`. Their sizes and SHA256 hashes are in
[q2_k_xl.json](../experiments/qwen-strata/q2_k_xl.json), whose recorded SHA256
is `99326fa66b6735324b0e88a1a214ba9814bbb24ad92e49d092286f6299c04f03`.
Each case retains `provenance.json`, frozen code hashes and artifact identities.
The four IQ3_XXS shards use the same Unsloth revision. Their overlay is
[iq3_xxs.json](../experiments/qwen-strata/iq3_xxs.json), with recorded SHA256
`19776a5b08367d05757c834b868acdc03406fd3bc2c45a1e241e3d243f4262f5`.
IQ1_M retains its three original shards at revision
`d96201194f103c906994f283f504209dc8d2a10c`. Its base
[manifest.json](../experiments/qwen-strata/manifest.json) has recorded SHA256
`b02e477f9c6c9733b46ac1d2cc9ad0cd87d383e4d7788c12b38cb9ffd6ef17c1`.
The [shared Engram report](qwen-engram-20261113.md) records the Q8_0 table pin
and the earlier IQ1_M measurements. The comparison above uses the fresh
48 GiB run from November 14, not those earlier measurements.

All completed cases used a 64 GiB model cap, zero swap, 262,144-token capacity, INT8 KV,
65,536 GPU KV cells, direct Engram reads and a 1,024 MiB VRAM reserve.
Four MTP drafts map to native `--spec 5`; startup logs confirm the loaded draft
layer. Prompt lookup, adaptive fetch, route priors and expert tail skipping
were off. Sampling used temperature 1.0, top-p 0.95 and seed 20261113.
The Linux available-memory stop floor was 8 GiB.

## Q2_K_XL results

The 36 GiB case used `matched64`: one OMP conversation with repair targets at
16K, 32K, 48K and 64K. The 48 GiB case used `long`: targets at 16K, 64K, 128K
and 240K, followed by compaction. Automatic compaction was disabled during the
repair checkpoints and enabled explicitly for the compaction phase.

| Expert budget and case | Generated tokens / native decode seconds | Weighted decode | MTP accepted / offered | Repair/resume wall time |
| --- | ---: | ---: | ---: | ---: |
| 36 GiB, `matched64` | 2,608 / 64.4095 | 40.4909 tok/s | 1,920 / 2,304 (83.33%) | 200.97 s |
| 48 GiB, `long` | 2,670 / 67.9038 | 39.3203 tok/s | 1,909 / 2,466 (77.41%) | 311.36 s |

Weighted decode divides all generated repair/resume tokens by the sum of native
decode time. It excludes prefill, tool execution, startup and compaction.
The wall times include repair and resume work. There were 20 model requests
and 12 tool calls with no tool errors in the 36 GiB case. The 48 GiB case used
21 requests and 13 tool calls, with one recovered tool error.

The next table records the first request at each repair checkpoint. Prefill
rates cover only its new tokens. Only the first checkpoint had no reused prefix.
Checkpoint decode includes that repair's follow-up requests and its resume.

| Case | Repair input | Reused / new tokens | Prefill tok/s | Checkpoint decode tok/s | Repair/resume wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 36 GiB | 15,997 | 0 / 15,997 | 722.54 | 39.48 | 60.30 |
| 36 GiB | 32,316 | 17,093 / 15,223 | 1,307.60 | 39.42 | 50.24 |
| 36 GiB | 48,624 | 33,242 / 15,382 | 1,387.40 | 43.29 | 40.16 |
| 36 GiB | 65,073 | 49,479 / 15,594 | 1,371.20 | 40.65 | 50.28 |
| 48 GiB | 15,997 | 0 / 15,997 | 1,203.24 | 40.93 | 50.24 |
| 48 GiB | 65,232 | 17,231 / 48,001 | 1,527.66 | 39.57 | 60.24 |
| 48 GiB | 130,608 | 66,057 / 64,551 | 1,449.84 | 37.66 | 80.37 |
| 48 GiB | 245,386 | 131,541 / 113,845 | 1,396.21 | 38.62 | 120.51 |

The first prefill took 22.1398 seconds at 36 GiB and 13.2949 seconds at 48 GiB.
This is **not an isolated RAM A/B test**. The two case kinds encode different
future checkpoint keys and padding in their first messages. Their captured
request hashes differ despite having the same first repair task and token count.
Later requests also retain different generated output. Cache warmth, placement
and workload length differ. The speed changes are observations; these runs
do not prove a causal prefill gain or decode regression from the RAM budget.

Both runs passed **4/4 semantic code checks, 4/4 recall checks and 4/4 strict
resume checks**. The independent validators ran 100, 248, 251 and 2,519 assertions
at the successive repair stages, with tests unchanged. Both runs scored **0/4
on the separate strict repair-final-JSON check** because replies included prose
and fenced JSON. Semantic success does not satisfy that formatting requirement.

## Q2_K_XL expert residency and memory

At the 36 GiB budget, native logs reported 36.00 GiB of pinned experts and
roughly 6.9 GiB assigned to the file tier during initial sizing. None of the
1,033 GPU cache slots borrowed for prefill had a duplicate expert in RAM.
The engine warned that these experts would need file reads during long prompts.
The final counters recorded 54,613 logical expert blob reads from files.

At the 48 GiB budget, the engine pinned **42.44 GiB**, an increase of **6.44 GiB**.
All 971 borrowed prefill slots had their experts in RAM. The prefill planner
reported a zero-size expert file tier, and request counters recorded **zero
logical expert blob reads from files**. This does not mean zero disk I/O:
startup, the direct-read Engram table and other data still use files. The byte
counter also retained 2,285.2 MB while the expert blob count was zero. These
counters are not a measurement of physical SSD traffic.

Both model cgroups reached approximately **64 GiB** and recorded nonzero
`memory.events.max` counts, but no OOM events or swap use. File-cache charges
contribute to those peaks. At the end, ordinary file cache (`file - shmem`) was
18.44 GiB in the 36 GiB case and 8.54 GiB in the 48 GiB case. Equal cgroup peaks
do not imply equal pinned memory or prove that the larger budget is free.
Shared cache accounting and reclaim can differ across runs.

Peak sampled GPU memory was 11,484 MiB and 11,657 MiB, respectively. The lowest
sampled Linux available memory was 54.00 GiB and 49.01 GiB. Startup took 82.07 s
and 94.07 s. These are single-run resource observations, not guaranteed limits.

## Q2_K_XL compaction at 48 GiB

OMP persisted a `compaction` entry after **246,349 conversation tokens**.
The phase took **221.169 seconds**. The final response retained the requested
release key, target branch, deadline tag and probe value as an unfenced JSON
object. Both `observed` and `retention_passed` are true. This checks those four
facts; it is not a full audit of the generated summary.

The phase made three summary requests and one final reply request. Submitted
input sizes were 61,530, 825, 8,416 and 24,052 tokens; the native engine processed
the 825-token request first. Keep these requests separate from the repair/resume
rates above. Including compaction, the run recorded 5,071 accepted drafts from
7,290 offered. The 36 GiB `matched64` case did not run a compaction phase.

## IQ3_XXS at 48 GiB

The completed `long` case generated **3,280 repair/resume tokens in 88.3784
native decode seconds**, or **37.1131 tok/s**. Repair/resume wall time was
341.69 seconds. It used 21 model requests and 13 tool calls, with one rejected
extra validation command. The permitted validator had already passed at that
checkpoint. MTP accepted **2,253 of 3,201 drafts (70.38%)**. These counts exclude
compaction.

| Repair input | Reused / new tokens | Prefill tok/s | Checkpoint decode tok/s | Repair/resume wall seconds |
| ---: | ---: | ---: | ---: | ---: |
| 15,997 | 0 / 15,997 | 1,360.95 | 37.52 | 40.19 |
| 65,181 | 16,814 / 48,367 | 1,372.21 | 38.36 | 70.34 |
| 130,693 | 66,171 / 64,522 | 1,352.52 | 34.92 | 100.60 |
| 245,437 | 132,093 / 113,344 | 1,317.76 | 38.50 | 130.55 |

The first prefill took 11.7543 seconds. Q2_K_XL and IQ3_XXS used the same first
prompt in their 48 GiB `long` cases. Both first-prompt identity files record
SHA256 `91bf02e4912cc51b02d31e37dc7c93ffd895c478046f49a85f7a762ddc81bf42`
and 15,997 tokens. Later generated histories differ. IQ3 generated 3,280 tokens
across repair/resume work, compared with Q2's 2,670. The full-run wall times
therefore compare different output workloads. Each configuration ran once;
the measurements do not establish a fixed-work speed ratio.

IQ3 passed **4/4 semantic code checks, 4/4 recall checks and 4/4 strict resume
checks**. The validators ran 100, 248, 251 and 2,519 assertions, with tests
unchanged. As with Q2, it passed **0/4 strict repair-final-JSON checks** because
its replies included prose and fenced JSON. These results do not show a local
quality advantage over Q2.

The engine pinned **46.44 GiB** of experts. All **1,074** borrowed prefill cache
slots retained a duplicate expert in RAM. The prefill planner reported a
zero-size expert file tier. Final counters recorded **zero logical expert blob
reads from files**, while the separate byte counter retained **2,649.8 MB**.
These counters do not measure physical SSD traffic or prove zero disk I/O.
The direct-read Engram table and startup still use files.

The model cgroup reached approximately **64 GiB** and recorded 229,836
`memory.events.max` events. It recorded no OOM events or swap use. Final ordinary
file cache (`file - shmem`) was 6.23 GiB. Peak sampled GPU memory was
**11,624 MiB**, and minimum sampled Linux available memory was **44.82 GiB**.
Startup took **90.07 seconds**. These measurements include file-cache accounting;
they do not imply that 46.44 GiB is the process's total RAM use.

OMP persisted a compaction after **246,797 conversation tokens**. The phase
took **221.219 seconds**, with `observed` and `retention_passed` both true.
The final response was an unfenced JSON object with the correct release key,
target branch, deadline tag and probe value. This checks four facts, not every
statement in the summary. Submitted compaction inputs were 61,535, 881, 8,733
and 23,882 tokens. Including compaction, MTP accepted 5,763 of 8,114 drafts.

### Interrupted 36 GiB case

The user stopped `iq3_xxs-ram36g-q8256-matched64-r1` after its first two
checkpoints to change the budget. This partial run is excluded from completed
comparisons. Its original case summary retains an interruption and cleanup
error: the worker exited while cleanup inspected its `/proc` environment.
The separate `cleanup-recheck.json` confirms that the worker and its process
group were gone, port 8899 was closed, the active record was absent, and the
original state/profile restoration checks passed. The original record also
confirms that the model server stopped. The recheck supplements the failure
record; it does not replace it or turn the partial run into a completed result.

## Fresh IQ1_M baseline at 48 GiB

The completed `long` case generated **4,362 repair/resume tokens in 85.7656
native decode seconds**, or **50.8596 tok/s**. Repair/resume wall time was
331.43 seconds. It used 29 model requests and 19 tool calls, with three tool
errors. MTP accepted **2,972 of 4,071 drafts (73.00%)** during these phases.
These saved measurements exclude compaction. An earlier progress total of
4,872 tokens included the first 510-token compaction summary and is superseded.

| Repair input | Reused / new tokens | Prefill tok/s | Checkpoint decode tok/s | Repair/resume wall seconds |
| ---: | ---: | ---: | ---: | ---: |
| 15,997 | 0 / 15,997 | 1,326.69 | 53.54 | 70.35 |
| 65,353 | 17,906 / 47,447 | 1,608.97 | 49.73 | 70.29 |
| 130,623 | 66,676 / 63,947 | 1,511.40 | 49.58 | 80.33 |
| 245,502 | 131,975 / 113,527 | 1,457.03 | 49.05 | 110.47 |

The first prefill took 12.0578 seconds. IQ1_M's captured first request differs
from Q2 and IQ3 only in the dynamic date: November 14 instead of November 13.
The raw first-prompt SHA256 is
`ebb4baed456a21ca523d19b503ea8c1daa8c2622f60fd18fc9e863ab50f27f44`.
All other request fields and content match, including the 15,997-token size.
The original requests and their distinct hashes remain unchanged. The shared
targets, seed and settings support a workflow comparison; later generated
histories, output lengths and tool use prevent a fixed-work speed ratio.

IQ1_M passed **4/4 semantic code checks, 4/4 recall checks and 4/4 strict resume
checks**, with 100, 248, 251 and 2,519 validator assertions and unchanged tests.
Its separate strict repair-final-JSON result was **0/4**, because replies
included prose and fenced JSON. Semantic success does not remove that failure.

The engine pinned **32.01 GiB** of experts at the 48 GiB budget. All **1,641**
borrowed prefill slots retained their experts in RAM. The prefill planner
reported a zero-size expert file tier. Final counters recorded **zero logical
expert blob reads from files** and a separate **2,784.4 MB** byte count. As with
the other runs, these are not physical SSD measurements.

The model cgroup reached **64 GiB**, including file cache. It recorded 185,837
`memory.events.max` events, no OOM events and zero swap use. Final ordinary file
cache (`file - shmem`) was 16.10 GiB. Peak sampled GPU memory was **11,718 MiB**;
minimum sampled Linux available memory was **53.95 GiB**. Startup took
**68.05 seconds**. The 48 GiB expert budget is a ceiling, not a forced allocation.

OMP persisted a compaction after **246,683 conversation tokens**. It took
**160.733 seconds**, with `observed` and `retention_passed` both true. The final
unfenced JSON contained the correct release key, target branch, deadline tag
and probe value. Submitted compaction inputs were 63,043, 792, 9,033 and 24,290
tokens. Including compaction, MTP accepted 5,998 of 8,207 drafts.

## Evidence and cleanup

The completed directories under `local/results/qwen-quants-20261113/` are
`q2_k_xl-ram36g-q8256-matched64-r1`, `q2_k_xl-ram48g-q8256-long-r1` and
`iq3_xxs-ram48g-q8256-long-r1`, plus `iq1_m-ram48g-q8256-long-r1`.
They retain measurements, native logs, request identities, provenance,
memory samples and cleanup records. All four recorded stopped owned servers,
workers and guards, closed port 8899, restored state and expert profiles,
unchanged normal OMP settings, and preservation of the preexisting search service.

After measurement, Q2_K_XL's remaining core and prepared files were removed,
freeing 55.19 GiB. Its 26.82 GiB bundled IQ4_NL table had first been renamed into
the IQ3_XXS directory. Both manifests pin that file to the same SHA256; no bytes
were changed. The separate Q8_0 table remains available. The archived receipts
and transfer record are in `removed-artifact-records/q2_k_xl/` beneath the results
directory. This cleanup preserves the settings, pins and measurement evidence.

The IQ3_XXS core and prepared files were then removed, freeing 85.73 GiB for
the fresh IQ1_M fetch. Its receipts and manifests remain in
`removed-artifact-records/iq3_xxs/`. The shared Q8_0 table, quant profile and
measurement results remain available.

IQ1_M's core weights and prepared files are restored. Q2_K_XL and IQ3_XXS core
weights remain removed; their profiles can fetch them again. The final IQ1_M
server and benchmark processes are stopped.
