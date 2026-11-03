# DeepSeek Coder comparison on Strata 0.2.9

For the current runtime, see the [Strata 0.2.10 remeasurement](strata-0.2.10-coder.md).

2026-11-02. The selected `deepseek` default is ISTA GSQ-RCO Coder IQ2_XXS
with resident expert copies and a 64 GiB process cap. This retains more RAM
for other programs; the paired 70 GiB run improved decode speed by only 2%.
Some experts remain file-backed at 64 GiB. Use `--mem 70G` for the measured
full-residency configuration. The larger profile
remains available as `deepseek-full`. Coder passed the OMP workload at 16K,
32K, and 64K input. At the matched 64 GiB demand-cache settings, it decoded
37.7% faster than the full Unsloth UD-IQ1_S control.

The 70 GiB resident run held all 7,680 routed experts across RAM and VRAM.
It decoded 13.32 tokens/s with a routing profile collected from this workload.
The default does not install that benchmark profile. A normal cold-start
check at 70 GiB also achieved full residency and passed an actual `omp-local` read
and strict JSON response. The Engram table remains on SSD. These results select
a practical default; they do not establish an optimum for every workload.

## Method

The machine used an RTX 4070 SUPER with 12 GiB VRAM and an AMD Ryzen 9 7950X.
WSL exposed 12 physical cores and 24 logical CPUs. Both profiles used the
same pinned Strata API and packer 0.2.9, native engine 0.1.55, 12 CPU workers,
4,096-token prefill chunks, 256 MiB Engram cache, and 16 Engram readers.
Both used all six selected experts. DSpark was disabled.

Process caps were 64 GiB, except for the explicitly labelled 70 GiB resident
case. Every model process tree had zero swap allowance. A guard stopped the
run if Linux available RAM fell below 8 GiB. Windows RAM was not monitored.
The outgoing-connection guard and offline API stayed enabled.

Context capacity was 262,144 tokens. The same constructed OMP conversation
grew through approximately 16K, 32K, and 64K input. It used public source
text, code repair, three-fact recall, required read/edit/bash calls, independent
code assertions, and strict JSON resume checks. Earlier assertions remain
in later checkpoints. Automatic compaction was disabled. These are not
filled-256K or compaction measurements.

Reasoning was xhigh, temperature 1.0, and top-p 0.95. Each request allowed
4,096 output tokens. No sampling seed was fixed. The server reserved part
of that output budget for the final answer. Output trajectories can differ.

The cold cases started without a routing profile. The trained-profile cases
started with the same frozen routing counts from the first Coder demand-cache
run. All cases started with empty saved conversation slots. The driver
requested eviction of clean weight pages with `POSIX_FADV_DONTNEED`. It did
not clear the global OS page cache. Thus, cold means no saved routing counts
or conversation slots; it does not mean a globally cold SSD cache. Each
completed server stopped afterward. The original routing profile, including
an originally absent profile, was restored.

## Matched demand-cache results

Both cases used a 48 GiB expert demand cache. Prefill rates below measure
the first repair request at each checkpoint and count only uncached tokens.
Decode rates divide all repair/resume output tokens by native decode time.

| Profile | Input | Prefill tokens/s | Decode tokens/s | Checks |
| --- | ---: | ---: | ---: | --- |
| Full Unsloth UD-IQ1_S | 16K | 259.94 | 8.04 | Pass |
| Coder IQ2_XXS | 16K | 377.77 | 10.90 | Pass |
| Full Unsloth UD-IQ1_S | 32K | 356.02 | 8.25 | Pass |
| Coder IQ2_XXS | 32K | 512.96 | 11.55 | Pass |
| Full Unsloth UD-IQ1_S | 64K | 427.49 | 8.23 | Pass |
| Coder IQ2_XXS | 64K | 589.92 | 11.41 | Pass |

The full control generated 1,942 tokens at 8.1604 tokens/s. Coder generated
1,959 tokens at 11.2366 tokens/s. Repair plus resume took 503.51 seconds
for the full profile and 372.40 seconds for Coder, a 26.0% reduction.
These times include OMP and tool overhead. They exclude startup, shutdown,
and the short warm-up request.

Both cases passed code, recall, unchanged-test, required-tool, and strict
resume checks at every checkpoint. Neither had a tool error. Each reached
the 64 GiB process cap. The full control recorded 8,765 `memory.max` events;
Coder recorded 6,608. These indicate reclaim pressure. Neither case recorded
an OOM or model swap use. Whole-run peaks are not per-checkpoint peaks.

These are single sampled conversations with similar output lengths. They
support a local speed improvement. They do not establish equal general
coding ability or a global optimum.

## Expert placement and resident copies

The full profile placed 50 of 15,360 routed experts in VRAM, about 0.67 GB.
Coder placed 240 of 7,680 in VRAM, about 2.38 GB. Its CPU side held 7,440
experts with 64.2 GB of payload. The 48 GiB demand cache had 5,017 slots.
Each slot used the largest expert size, about 10.3 MB. This padding prevents
the cache from holding all CPU experts.

The resident-copy comparison uses native `--mmap-experts off --ssd-cache 0`.
It copies variable-size experts into RAM within the cgroup-aware allowance.
It retains an 8 GiB margin and leaves remaining experts backed by files.
It does not force `DS4_COPY_GB`, which would override that allowance.

The cold resident run initially copied 6,964 of 7,438 CPU-side experts into
RAM, about 59.4 GB. Another 242 experts were in VRAM. The remaining 474
experts stayed file-backed. Startup took 46 seconds, versus 11 seconds for
the demand-cache run.

| Cold resident mode | Prefill tokens/s | Decode tokens/s | Checks |
| --- | ---: | ---: | --- |
| 16K | 496.55 | 13.07 | Pass |
| 32K | 646.75 | 13.12 | Pass |
| 64K | 756.17 | 12.91 | Pass |

The resident run generated 2,417 tokens at 13.0289 tokens/s. Repair plus
resume took 362.04 seconds. It generated 23.4% more output than the Coder
demand-cache run, so workflow time improved less than decode throughput.
All checks passed with no tool errors. The peak was approximately 64 GiB,
with 148,342 `memory.max` events and three `memory.high` events. There was
no OOM or model swap use. The larger fixed allocation caused more reclaim.

Resident selection happens at startup. Without a prior routing profile,
all counts are zero and selection follows layer/index order. A saved profile
can change that selection. Comparisons using learned counts must use the
same frozen profile for both modes and identify that condition separately.

## Comparison with the same saved routing profile

The following runs used the same 30,720-byte routing profile. It came from
`coder-64g-48cache-growing/profile-after.bin`, with SHA-256
`317875a8a047719357eb8bc05670e0a7948a0d1626df3834cb14f7474c9d9b03`.
These are replays with routing counts from this workload. They are separate
from the cold comparisons. The profile was identical at each launch;
the engine could update its working counts during each run.

Prefill columns are the first repair request at the named input size.
Decode is total generated tokens divided by total native decode seconds,
not an arithmetic average of request rates. Generated tokens include
reasoning and final output from repair and resume calls.

| Configuration | Prefill 16K / 32K / 64K tokens/s | Decode tokens/s | Generated tokens | Workflow seconds | Recovered tool errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 GiB cap, 48 GiB demand cache | 406.46 / 511.40 / 607.19 | 11.51 | 2,735 | 462.46 | 2 |
| 64 GiB cap, resident copies | 502.17 / 659.97 / 771.87 | 13.06 | 2,718 | 372.00 | 0 |
| 70 GiB cap, resident copies | 497.11 / 669.56 / 767.02 | 13.32 | 2,285 | 342.11 | 1 |

All three trained-profile cases passed every code, recall, unchanged-test,
required read/edit/bash, and strict JSON resume check at all three checkpoints.
The demand-cache case recovered from one bash error and one read error at
16K. The 70 GiB resident case recovered from one bash error at 16K. The
64 GiB resident case had no tool errors. All attempts remain in the output
and timing totals. The checkpoint assertion counts were 100, 248, and 251; later counts include earlier assertions.

At 70 GiB, Strata reported 242 experts in VRAM and all 7,438 remaining experts
copied into RAM. All 7,680 routed experts were resident. The CPU payload was
64.0 GB in the native log; this is decimal GB, not the process cap. The
103.00 GiB Engram table still used SSD reads and the 256 MiB row cache.
Full expert residency therefore does not imply zero disk reads. The trained
64 GiB resident case copied 6,902 CPU experts, about 59.3 GB. With its 242
VRAM experts, 536 experts remained file-backed. The trained and cold 64 GiB
cases copied different expert counts because the saved profile selected
a different mix of expert sizes.

The trained demand-cache case reached its 64 GiB cap and recorded 1,904
`memory.max` events. The 64 GiB resident case reached approximately 64 GiB
and recorded 122,022 `memory.max` events and seven `memory.high` events.
The 70 GiB resident case reached its cap and recorded 114,802 `memory.max`
events. No case recorded an OOM or model swap use. These whole-run peaks
include file cache and other process memory. The events show reclaim pressure
even when all expert copies are resident.

During the 70 GiB run, Linux available RAM stayed above 37.14 GiB.
The whole-card sampled VRAM peak was 11.27 GiB. Mean GPU utilization was
72.69% across the run and 94.0% during samples labelled as generation.
Mean power was 93.70 W; sampled peak power was 206.96 W. These measurements
include the whole GPU and are not isolated per-process hardware counters.

The 70 GiB resident run decoded 15.7% faster than the trained demand-cache
run. Workflow time fell 26.0%, but it also generated 16.5% fewer tokens.
Startup took 44.14 seconds versus 22.03 seconds.

Against the trained 64 GiB resident case, the 70 GiB case decoded only 2.0%
faster. Prefill rates were similar. Workflow time fell 8.0%, while output
length fell 15.9%. Startup was 44.14 versus 31.05 seconds. The extra 6 GiB
therefore establishes full expert residency, not a large measured speed gain.
The 64 GiB resident setting remains useful when RAM matters more. The speed,
output length, startup, and RAM cost must be considered separately. One sampled
run per setting does not establish a universal ranking.

## Storage and verification

The user authorized removal of the four Unsloth transformer shards after
the fresh control completed. They totalled 122,384,317,828 bytes. The original
receipts, packs, routing profiles, and benchmark evidence were retained.
The optional 7.43 GiB DSpark file was retained.

The five Coder files total 72,123,567,456 bytes, or 67.17 GiB. Each download
matched its pinned SHA-256. The shared Engram file adds 103.00 GiB.
Only one physical copy is stored. Both profile directories have a hard link
to that copy. Removing one profile preserves the other profile's link.

The Engram payload was verified against both releases. The original
Unsloth file has a 192-byte header. The Coder release has a 416-byte header.
Hashing the published Coder header followed by the entire local payload
produced the published Coder file hash. The original full-file hash also
matched its pin. This proves equality of the 110,595,290,400-byte payload.
It does not prove the other pruned tensors against an unpruned IQ2_XXS set,
which was not downloaded.

Tokenizer vocabulary, merges, token types, and tokenizer JSON were identical.
The embedded chat-template changes were reviewed. They adjust reasoning-effort
validation and preserve tool content blocks. The runtime and network guard
were not changed.

## Quality boundary

The [publisher's model card](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF)
reports removal of half the routed experts. It reports SWE-bench Verified
77.00 versus 81.20 for the base model, and LiveCodeBench v6 88.57 versus
89.71. These are publisher measurements at reasoning effort 50. They are
not measurements of this machine or of our Unsloth UD-IQ1_S control.
The publisher accepts capability loss outside its selected domains.

Passing this OMP workload is a compatibility and correctness check. It does
not reproduce either published benchmark. The local recommendation applies
to coding workflows; it is not a claim that pruning preserves every capability.

## Evidence and limits

Local raw evidence is under `results/deepseek-coder-20261102/`:

- `full-64g-growing/`: fresh full control before weight replacement.
- `coder-64g-48cache-growing/`: matched Coder demand-cache run.
- `coder-64g-resident-cold/`: cold resident-copy comparison.
- `coder-64g-cache-profile/`: demand cache with the frozen routing profile.
- `coder-64g-resident-profile/`: 64 GiB resident copies with the same profile.
- `coder-70g-resident-profile/`: 70 GiB resident copies with the same profile.
- `default-smoke/`: installed defaults, cold residency, and OMP discovery check.
- `compatibility/`: bounded GGUF headers, tensor sizes, and Engram hash proof.
- `previous-unsloth/`: original receipts, pack assets, and routing profiles.
- `prepare-state.json`: exact replacement and verified download record.
- Each case's `case-summary.json`, network records, `engine.log`, and memory
  samples: throughput, quality, placement, and resource evidence.
- `cache_io.py`: read-only collector for the case artifacts.
  `cache-io-comparison.json` contains all six completed cases.

The original 70 GiB installed-default smoke used the commands in `default-smoke/command.json`
and `default-smoke/omp-command.json`. Startup took 44.65 seconds without an
existing routing profile. It confirmed the 70 GiB cap, zero swap allowance,
256K context, resident mode, retained slots, and outgoing-connection guard.
The installed `omp-local` command read a file and returned the expected strict
JSON. All 7,438 CPU-side experts were copied into RAM. The server and guard
stopped after the check. This short probe is not a throughput measurement.

This WSL kernel exposes neither cgroup `io.stat` nor `/proc/PID/io`.
Exact per-process SSD bytes and expert-cache hit/miss rates are unavailable.
The observer records shared block-device counters. Those include other WSL
activity and cannot be attributed exclusively to the model. Native CPU-lane
percentages include prompt and decode work; they are not decode cache-hit rates.
