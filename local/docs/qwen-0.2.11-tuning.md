# Qwen3.8 Flash-Next: Strata 0.2.11 measurements

Measured on 2026-11-05 with the existing ISTA GSQ-RCO IQ2_XS weights and MTP
artifacts. Strata 0.2.11 improves prefill substantially on this machine.
The existing configuration remains the default. It passed every check in both
the release comparison and the longer control.

An optional 65,536-cell GPU KV window increased weighted decode from 53.36 to
70.05 tokens/s through approximately 128K input. All code and recall checks
passed. However, all four resume answers used Markdown fences around the correct
JSON value. They failed the explicit bare-JSON requirement. Keep this limitation
with the speed result. Removing suffix lookup did not remove it.

## Use the measured options

The default remains 256K capacity, a 64 GiB cap, a 32 GiB resident expert budget,
INT8 KV entirely in VRAM, MTP depth 3, automatic CPU workers, and no suffix lookup.
To try the faster configuration:

```sh
localmodel start qwen3.8 --kv-vram-cells 65536
```

Stop the existing server first. This flag retains the full 256K capacity.
It keeps a 64K KV window in VRAM and streams the remaining positions from RAM.
The native engine keeps a full host KV copy. Its log reports about 3.09 GiB of
pinned main KV at this capacity. The host copy is inside the existing memory cap.
No native patch or new weights are required.

The launcher also exposes these controls:

```sh
# MTP depth and CPU workers. Zero workers selects the native automatic count.
localmodel start qwen3.8 --draft-tokens 3 --cpu-threads 0

# Optional suffix lookup alongside MTP. This did not show a clear gain here.
localmodel start qwen3.8 --prompt-lookup --lookup-tokens 12
```

`--draft-tokens 3` permits up to three neural drafts. The native `--spec 4` includes
the accepted token. With lookup enabled, `--lookup-tokens 12` sets the verification
window. It does not request eleven neural drafts. The native engine reports
`spec=12`, `mtp_max=4`, and `lookup=3` for that combination. The last value is
the minimum suffix match length.

Public launcher dry runs reproduce the measured KV-only and KV-plus-lookup
configurations exactly. The new controls do not change the default configuration.
No remote configuration changed.

## Matched release comparison

The old control used API/source 0.2.9 and native 0.1.55. The new control used
API/source 0.2.11 and native 0.1.57. Both used the unchanged default settings.
Each conversation grew through 16K, 32K, and 64K targets.

| Measurement | 0.2.9 | 0.2.11 |
|---|---:|---:|
| Weighted decode, tokens/s | 51.65 | 51.38 |
| First-request prefill at 16K, tokens/s | 575.61 | 1294.07 |
| First-request prefill at 32K, tokens/s | 577.37 | 1349.21 |
| First-request prefill at 64K, tokens/s | 571.48 | 1608.99 |
| Native prefill tokens / seconds | 63961 / 119.6114 | 64596 / 54.0835 |
| Generated tokens / decode seconds | 2367 / 45.8315 | 1785 / 34.7442 |
| Sum of API phase spans, seconds | 166.331 | 89.737 |
| Code, recall, and strict resume checks | All pass | All pass |

Prefill improved by 2.25 to 2.82 times at these checkpoints. Decode stayed
approximately unchanged. The new native log selects the small-expert-cache
unfused prefill path. Its automatic prefill cap changes from 768 to 1024.
Its reported PCIe work fraction changes from 0.55 to 0.25. This comparison
measures the release as a whole. It does not isolate one internal change.

The new run also generated fewer tokens. Of the reduction in native time,
65.53 seconds came from prefill and 11.09 seconds came from decode. Do not
attribute the whole task-time reduction to faster kernels.

## Short tuning runs

Each row below used a fresh 16K task on 0.2.11. All used MTP depth 3 and automatic
workers unless stated otherwise. All code and recall checks passed.

| Configuration | Decode, tokens/s | First prefill, tokens/s | Strict resume |
|---|---:|---:|---|
| Full GPU KV, suffix lookup, window 12 | 52.89 | 1250.63 | Pass |
| 64K GPU KV, suffix lookup, window 12 | 70.86 | 1558.51 | Pass |
| Same, MTP depth 2 | 63.14 | 1560.88 | Pass |
| Same, MTP depth 5 | 65.67 | 1561.34 | Pass |
| Same, MTP depth 3, eight CPU workers | 71.77 | 1559.56 | Fenced JSON |

MTP depth 3 gave the strongest decode result among the tested depths with the
automatic worker count. Eight workers improved that short decode measurement
by only 1.3 percent. Keep automatic workers. The VM exposes 12 physical cores;
the engine selected eleven workers plus the host thread.

Suffix lookup alone did not clearly improve throughput. The wider verification
window is available for other workloads. It remains disabled by default.

## Long-context comparison

These three fresh conversations grew through 16K, 32K, 64K, and 128K targets.
Each used MTP depth 3 and automatic workers. The final first requests contained
130,917, 130,876, and 130,658 input tokens respectively.

| Measurement | Default | 64K GPU KV only | 64K GPU KV + lookup |
|---|---:|---:|---:|
| Weighted decode, tokens/s | 53.36 | 70.05 | 69.84 |
| Decode at the 128K checkpoint, tokens/s | 53.10 | 69.63 | 73.62 |
| New-suffix prefill at 128K, tokens/s | 1548.20 | 1930.00 | 1925.74 |
| Generated tokens / decode seconds | 2121 / 39.7460 | 2991 / 42.6958 | 3937 / 56.3756 |
| Native prefill tokens / seconds | 129637 / 96.5005 | 129155 / 80.0440 | 128044 / 81.3139 |
| Sum of API phase spans, seconds | 137.653 | 124.141 | 139.899 |
| Code and recall checkpoints passed | 4/4 | 4/4 | 4/4 |
| Strict resume checks passed | 4/4 | 0/4 | 0/4 |
| Recoverable tool errors | 0 | 0 | 1 |

KV-only decode at 16K/32K/64K/128K was 69.58/70.95/70.02/69.63 tokens/s.
Its first-request prefill was 1561.17/1761.80/1908.20/1930.00 tokens/s.
There was no decode collapse after the conversation exceeded the GPU KV window.
The final coding checkpoint passed 2,519 assertions in every long run.

KV-only increased weighted decode by 31.28 percent. Its API span fell by
9.82 percent despite generating 41 percent more tokens. The lookup combination
generated 86 percent more tokens and did not improve the total API span.
These are sampled trajectories, not fixed-output latency tests.

Both long streaming runs returned the correct resume value inside Markdown
fences. Resume itself worked. The exact output-format check failed. The default
passed all four checks. This is a reason to retain the default, but these single
runs do not establish that streaming causes a general quality regression.

## Memory and expert traffic

The default long run had about 846 MiB of GPU expert cache. KV-only had 3824 MiB
and 2773 expert slots. Main GPU KV fell from 3168 MiB to 792 MiB. The native
allocator also changed its prefill workspace and cache allocation.

The reported GPU expert hit fraction rose from about 35.3 to 67.3 percent.
These are means of rounded per-request counters. They are not physical SSD
miss rates. Default decode reported 1333 file expert blobs; KV-only reported
zero. KV-only prefill still reported 1,308 file-blob reads. The file-backed PLE
table also remains. Linux page cache can serve file-backed reads. Zero decode
expert blobs does not mean zero disk reads for the whole engine. Per-cgroup
physical I/O counters were unavailable.

Every completed case reached the 64 GiB hard cap. Linux reclaimed charged file
cache. No case used model swap or had an OOM. In the long KV-only run, minimum
available Linux RAM was 58.32 GiB and sampled VRAM peaked at 11190 MiB. Sampled
GPU load during generation averaged 92.3 percent. These GPU samples cover the
whole device and are not continuous process-only measurements.

## Procedure and limits

The host was WSL2 with an RTX 4070 SUPER 12 GiB and 24 visible logical CPUs.
Each case ran serially with a 64 GiB model cap, zero model swap, and an 8 GiB
available-Linux-memory stop floor. The observer and coding process had separate
small caps. No Windows memory monitor was used. Inference network restrictions
remained active. No new weights, dependencies, or community patches were installed.

All cases used the same pinned public corpus, repair fixture, recall facts,
workspace guard, expert profile, sampling settings, and isolated OMP setup.
The tests used xhigh reasoning, temperature 1.0, top-p 0.95, and a 4096-token
response limit. Cached graph warmup is excluded from the reported OMP rates.
Native decode is total generated tokens divided by total native decode seconds.
Later first-request prefill rates measure the new suffix after prefix reuse.

OMP polls process completion every ten seconds. Its recorded workflow time
therefore includes polling delay plus startup and presentation. API phase spans
measure from the first request to the last timing event in each repair/resume
phase. They include tool gaps. Output token counts differ between cases.

The fixture requires replacement edits. It explicitly sets `PI_EDIT_VARIANT=replace`
for its OMP processes and checks the advertised schema before generation.
Normal Qwen OMP settings still use their existing edit mode. The first attempt
used hashline edits against a replacement-only guard. That attempt is retained
and excluded. It is not evidence about Qwen quality or throughput.

There were ten valid cases and 23 repair/recall checkpoints. All passed code
and recall; 14 of 23 strict resume checks passed. The nine failures were fenced
JSON with the correct value. The checks are repeated small coding tasks, not
a broad capability score. These runs test approximately 128K actual input within
256K capacity. They do not test filled 256K input or automatic compaction.

All benchmark servers, observers, guards, and scopes stopped. Port 8899 closed.
Normal settings and managed state were restored. The expert profile hashes
remained unchanged. The selected official runtimes and prepared artifacts verify.

## Evidence

Raw cases and excluded attempts are under `results/qwen-0211-20261105/`.
`comparison.json`, `final-summary.json`, and `final-summary.txt` collect the results.
`workload-pins.json` and `harness-pins.json` record the inputs and driver sources.
`cli-validation/` contains flag, lifecycle, type-check, and configuration checks.
Large raw results remain local and are ignored by Git.

The installed release is [Strata v0.2.11](https://github.com/Niko1221/Strata/releases/tag/v0.2.11).
Its source revision is `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`.
The weights revision is `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
The measurements here use that installed configuration. They do not reproduce
the different hardware and quantization in [community issue 5166](https://github.com/Niko1221/Strata/issues/5166).
