# Qwen Unsloth: 192K defaults

Measured on 2026-11-06 and 2026-11-07 with Unsloth Dynamic v3.1 UD-IQ1_M,
official Strata 0.2.11, and native engine 0.1.57. The selected configuration
reached 185,531 input tokens and averaged **53.20 decode tokens/s**.
The earlier 32 GiB / 768 MiB configuration and its tuning results follow below.

## Selected configuration

```sh
localmodel start qwen3.8
```

The defaults are 196,608 context tokens, 65,536 GPU KV cells, INT8 KV,
four MTP drafts, a 36 GiB expert RAM budget, a 64 GiB process-group cap,
and a 1024 MiB VRAM reserve. PLE reads use `direct`. CPU workers and prefill
chunks use the native automatic settings. Suffix lookup, routing priors,
and expert tail skipping are off. Model swap is disabled.

The native verification window includes the accepted token. Four drafts
therefore map to `--spec 5`. The measured run loaded MTP and produced
nonzero accepted-draft counters. The public launch configuration uses the
same settings as the selected run.

`omp-local` discovers the 192K capacity. Its normal compaction threshold
is 157,286 tokens, with a 16,000-token output allowance. The benchmark
disabled automatic compaction to test larger inputs directly.

## Headroom validation on 2026-11-07

The selected run generated 4,782 tokens in 89.8841 native decode seconds.
All four code, recall, and strict resume checkpoints passed. The smallest
reported free VRAM was 498 MiB, compared with 55 MiB in the earlier run.
There was one warning below 512 MiB. The reserve improves the measured
margin; it does not guarantee that other GPU activity cannot exhaust it.
The 36 GiB budget held 31.90 GiB of actual resident experts.

The earlier run averaged 55.25 tokens/s. The observed difference is 3.7%.
The reserve, RAM budget, and initial page-cache advice changed together.
Outputs also differ. This comparison does not isolate the reserve's cost.
See the [complete comparison](local-192k-comparison.md) for the selected
run's prefill, memory, correctness, and cleanup records.

## Earlier growing conversation: 32 GiB / 768 MiB

Each checkpoint used real OMP read, edit, and validation tools. It repaired
a small function and recalled facts from the conversation. A resumed OMP
process then answered a strict JSON probe. The context contained pinned
public source files, without repeated filler.

| Target | First request input | New-suffix prefill, tokens/s | Decode, tokens/s |
|---|---:|---:|---:|
| 16K | 16,013 | 1548 | 54.65 |
| 64K | 65,210 | 1920 | 57.62 |
| 128K | 130,637 | 1883 | 52.66 |
| 180K | 183,952 | 1783 | 56.27 |

The final resume request reached **185,151 input tokens**. Later prefill
rates measure new suffixes after prefix reuse, not cold prompts of those
total lengths. Aggregate decode was 3,290 tokens / 59.5461 seconds.
MTP accepted 2,270 of 2,992 proposed tokens, or 75.87 percent.

All four code, recall, and strict resume checkpoints passed. The validators
completed 3,118 assertions. There were no tool errors in this long run.
Across all 17 checkpoints in the study, code and recall passed 17/17.
Strict resume passed 12/17. Five earlier short answers returned the correct
value inside Markdown fences. Eight earlier tool errors were recovered
benchmark-guard refusals. They were not malformed protocol calls or crashes.

## Tuning results

All short cases below used 192K capacity and a 64K GPU KV window.
Expert tail skipping was disabled.

| Change | Expert budget, GiB | Decode, tokens/s |
|---|---:|---:|
| MTP2, direct PLE | 36 | 53.50 |
| MTP2, mmap PLE | 36 | 52.60 |
| MTP3, direct PLE | 32 | 53.91 |
| MTP4, direct PLE | 32 | 55.63 |
| MTP4, direct PLE, suffix lookup window 12 | 32 | 54.43 |

The 36 GiB budget did not add residency. Both budgets covered the expert
complement and all prompt-lend slots. Actual allocation was about 31.9 GiB.
The mmap and lookup alternatives did not improve these single sampled runs.
MTP4 had a modest lead and then passed the long validation. These runs do
not establish a global optimum or isolate small differences from sampling.

An earlier 256K control kept all KV in VRAM. The engine reported insufficient
VRAM for MTP and continued without it. That run averaged 35.54 tokens/s
through 128K input: 4,339 tokens / 122.096 decode seconds. A 192K control
with full GPU KV loaded MTP3 and reached 55.30 tokens/s in the short test.

Initial 192K/64K screens used the native tail-skip default. MTP2/3/4 reached
63.56/57.60/65.74 tokens/s. Those results include an approximation. Strata
can omit low-ranked experts absent from its GPU cache. CPU RAM residency
does not exempt them. The selected configuration explicitly sets
`STRATA_ROUTE_TAIL_SKIP=0`. See the pinned upstream
[skip logic](https://github.com/Niko1221/Strata/blob/c3b3f5035506f3e966f1b4b7b188aeaabae92f1d/src/core/expert_source.cpp#L4941-L4964)
and [GPU residency map](https://github.com/Niko1221/Strata/blob/c3b3f5035506f3e966f1b4b7b188aeaabae92f1d/src/program/generate.cpp#L9455-L9463).

## Memory and timing

The long run allocated 31.89 GiB expert RAM and 2,202 GPU expert slots,
about 2.93 GiB. The process-group peak was 64 GiB. File-cache reclaim
occurred at the cap. OOM events, OOM kills, and model swap were zero.
Available Linux memory stayed above 61.43 GiB. Whole-device VRAM peaked
at 11.72 GiB. Sampled decode GPU utilization averaged 95.6 percent;
prefill averaged 83.1 percent. These include desktop activity.

All 20 OMP requests reported zero logical expert file-blob reads. PLE
remained file-backed. The shared Linux root disk recorded 3.22 GiB of
reads during 341.54 seconds after readiness. This is guest block I/O;
WSL host caching is not measured. Other processes contribute to the
counter. Process-specific physical I/O was unavailable. Zero
expert file blobs therefore does not mean zero physical disk traffic.

Startup took 66.04 seconds. Warmup took 1.24 seconds. The API request
intervals totalled 175.92 seconds; native prefill and decode totalled
170.90 seconds. Polled OMP phases totalled 230.83 seconds. The full case
took 408.58 seconds. It also included 77 capture-only sizing requests,
token counting, setup, and cleanup. Do not use total case time as native
inference latency.

## Weights and limits

The old ISTA Qwen files were removed before the replacement download.
The three Unsloth shards total 68.01 GB. Their SHA256 values match the
pinned official Strata download table. The original GGUF bytes remained
unchanged after preparation. The existing official MTP files were reused.

The upstream packer uses `--compat-bf16`. The engine reads the original
Q8 hyper-connection projections with `STRATA_HC_Q8=1`, so it bypasses their
BF16 preparation copies. The remaining nonexact small-tensor conversion
is `blk.1.ple_conv1d.weight`, BF16 to F16, with maximum absolute error
2.9802322387695312e-08. This audit checks file identity and preparation;
it does not prove equivalence to the original unquantized checkpoint.

The historical ISTA result near 70 tokens/s used different VM storage,
less desktop VRAM, and expert skipping. It is not a matched quantization
comparison. The interrupted pre-reboot control completed no checkpoint
and is excluded. These are small repair and recall tasks, not a general
coding-capability evaluation. Filled 256K input and automatic compaction
were not tested.

The Qwen test stopped its server, worker, observer, and memory guard.
It verified port closure and restored saved profiles and normal settings.
Serving retained the outgoing-connection guard and loopback binding.

Local raw evidence is under `results/qwen-unsloth-20261106/`.
`final-results.json` includes all case denominators and cleanup records.
`preparation-audit.json` records weight checks. `production-parity.json`
and `production-preview-checks.json` record launcher validation.
