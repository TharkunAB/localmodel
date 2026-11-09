# Strata candidate measurements, 9 November 2026

The four requested candidates are prefill borrowing, cache-biased expert
routing, the DSML alias-string fix, and Linux arena admission. Production
defaults, official runtime files, and model weights remain unchanged.

All six OMP cases, three distribution runs, and the CPU checks are complete.
Keep borrowing enabled. Cache routing is a promising experimental option:
it improves measured throughput but changes expert selection and outputs.
Keep the official engine as the default until broader quality validation and
upstream review. The parser fix is useful; the arena change does not apply
to the current profiles. No candidate is installed in production.

## Configuration

The machine has an RTX 4070 SUPER with 12 GB VRAM, driver 610.88, and a Ryzen
9 7950X. WSL exposes 24 CPU threads and about 110 GiB RAM. The engine uses
12 CPU workers. The target is the installed full Unsloth DeepSeek IQ1_S.

All OMP cases use Strata API 0.2.12 and the native 0.1.58 source base. They
keep all six routed experts, 48 GiB expert-cache capacity, a 64 GiB hard cap,
zero swap, 4096-token prefill chunks, and 196608-token context capacity.
DSpark stays disabled. The API and native engine retain the outbound-network
guard. A separate guard stops the run below 8 GiB available Linux RAM.

OMP has a separate 4 GiB cap. The driver has a 2 GiB cap. Each case uses the
same initial expert profile and restores the production profile afterward.
Temperature is 1, top-p is 0.95, and seed is 20261109. The warmup uses
temperature 0. A fixed seed does not guarantee identical output: tool results,
case paths, and routing decisions can change the subsequent input.

The workload contains a public conversation history, one Python repair
fixture with 100 assertions, fact recall, and a resumed conversation. This is
a bounded task check, not a broad coding benchmark. The initial 16K target
produces about 15961 to 15963 native input tokens. Capacity and filled input
are recorded separately.

## OMP at 16K input

| Engine and setting | Initial prefill, tokens/s | Decode, tokens/s | Generated tokens | Repair and resume, seconds | Peak scope, GiB |
|---|---:|---:|---:|---:|---:|
| Official, current borrowing | 272.7 | 9.21 | 881 | 191.2 | 60.37 |
| Official, `DS4_BORROW=0` | 276.0 | 8.31 | 578 | 161.1 | 57.25 |
| Custom, routing off | 273.6 | 9.14 | 694 | 171.2 | 59.90 |
| Custom, routing `0.2,0.3` | 333.6 | 12.63 | 1007 | 150.8 | 55.91 |

Decode is generated tokens divided by native decode time across repair and
resume requests. Initial prefill uses the first request's new input and
native prompt time. Startup and warmup are excluded. These are single-case
measurements, not averages across repeated trials.

All four cases pass repair, recall, and strict resume checks. The official
control, borrowing-off case, and routing-on case each have one rejected
directory-listing command. The benchmark guard permits only its verification
command. Each rejected call has the exact declared `bash` name and valid
string arguments. These are policy refusals, not DSML parsing failures. Each
case recovers. The custom routing-off control has no tool errors.

All four cases report zero memory-limit and OOM events and zero swap. Their
cleanup checks pass, including stopped workers, stopped server, closed model
port, restored profiles and server configuration, and unchanged normal OMP
settings.

### Prefill borrowing

At the fixed 4096-token chunk size, disabling borrowing reduces the GPU expert
pool from 304 to 69 experts. Decode falls by 9.8%; initial prefill changes by
only 1.2%. The lower wall time follows a smaller generated response and does
not establish a throughput improvement. Keep borrowing enabled here.

The published report also concerns automatic prompt-chunk growth. Our explicit
chunk setting bypasses that growth. Its speed claim therefore does not
transfer directly to this configuration. Source details and the exact tested
flags are in the [borrowing notes](../experiments/strata-candidates-20261109/borrowing.md).

### Cache-biased routing

The two controls have nearly identical prefill and decode rates. Enabling
`STRATA_DS4_CACHE_ROUTE=0.2,0.3` in the same custom binary improves decode by
38.2% and initial prefill by 21.9% in this screen. It also changes expert
selection. It is not a lossless cache optimization.

The sampled root-device read counter falls from 271.9 GiB in the custom control
to 146.7 GiB with routing enabled, despite more generated tokens. This is
consistent with fewer uncached expert reads. These shared counters include
other processes and the WSL storage layer. They are not exact cache misses or
physical Windows SSD traffic. Per-model cgroup I/O counters are unavailable.

## OMP at 64K input

The larger comparison uses the same custom binary with routing off and on.
It fills about 65306 to 65308 native input tokens in the 192K-capacity server.

| Setting | Initial prefill, tokens/s | Decode, tokens/s | Generated tokens | Repair and resume, seconds | Peak scope, GiB |
|---|---:|---:|---:|---:|---:|
| Routing off | 430.3 | 9.51 | 1526 | 342.7 | 64.00 |
| Routing `0.2,0.3` | 513.1 | 12.15 | 805 | 221.7 | 61.57 |

Routing improves native decode by 27.8% and initial prefill by 19.2% here.
The 35.3% reduction in task wall time also includes a shorter response; it
must not be attributed entirely to throughput. Both cases pass the recorded
repair, recall, and resume checks. Neither has a tool-error event.

Both repair replies include additional prose despite a JSON-only instruction.
The existing repair checker searches accumulated text for JSON with the three
recalled fields. It does not require JSON-only output, exactly four keys, or
the requested `fixed_function` value. Both cases fail the requested reply
format. Both separate resume replies are exactly `{"probe_value": 6}` and
use no tools, as their stricter checker requires. Do not interpret the
recorded pass flags as perfect instruction following.

The control records 7871 hard-cap events and no OOM event. Routing records
zero hard-cap events. Neither uses swap. Available Linux RAM stays above
54.3 GiB. Sampled peak VRAM is 11.42 GiB for the control and 11.35 GiB for
routing, including other device users. The sampled shared root-device read
counter changes from 399.3 to 157.6 GiB; output lengths differ, and the same
I/O attribution limits as the 16K comparison apply.

Both larger cases stop their workers and server, close the model port, and
restore the checked production state. These are one case per setting at
each input size. They do not establish performance or quality at 192K filled
input, across arbitrary coding tasks, or under concurrent requests.

## Distribution check

The official engine and the custom routing-off engine produce byte-identical
scores at all 4995 teacher-forced positions in five public text segments.
This establishes a useful build control on this corpus.

Routing changes the top predicted token at 8.71% of those positions. Aggregate
perplexity changes from 5.783 to 5.735. Results differ by segment: code,
English, and other-language perplexity worsen slightly; documentation and
the agent transcript improve. This does not establish general quality parity
or improvement. The reference itself is IQ1_S, not a higher-fidelity target.

The scorer uses 8K capacity and 16-token windows. It does not establish
long-context quality. Exact results, limits of the approximate top-64 KL
metric, resource observations, and provenance are in the
[quality report](../experiments/strata-candidates-20261109/quality/README.md).

## DSML alias-string fix

PR 5598 passes all 156 targeted fixtures; the baseline passes 96. The fix
resolves an accepted tool alias before selecting its parameter schema. It
corrects the 60 alias cases that lose a true string marker. Upstream parser
and API suites pass in both versions, with one skipped test each. The API
tests use a scripted engine.

A separate pre-existing limitation remains: a true string marker can mutate
the parser's stored schema, affecting a later false marker for that parameter
in the same parser instance. This occurs with exact names in both versions.
The patch extends the same behavior to aliases. These tests do not establish
cross-request leakage. The patch is useful but is not a complete correction
for all type handling. It is not installed in production.

## Linux arena admission

The baseline passes 906 host checks; the patch passes 918. Hard-limit refusal
and file-source fallback checks pass. No hugepage pool or host setting changes.

The current Qwen profile passes `--resident-budget-gib 36`, which selects
mapped experts and bypasses `ArenaExpertSource::open`. The low-RAM profile
also uses the mapped source. The new admission credit additionally requires
an unlimited cgroup; our 64 GiB cap excludes it. DeepSeek uses another expert
store. A GPU timing test of these unchanged production paths would not test
the patch. There is no performance claim for this candidate.

## Source review and evidence

The three patches were pinned and reviewed before execution. The added code
introduces no networking, subprocess execution, or tool execution. The arena
change reads local `/proc/meminfo`. The cache-routing source has about 60 KiB
of bias buffers outside the usual per-device free list; process exit releases
them. Its lock-free cache hints follow existing code, but this review does
not prove concurrency safety. The arena helper has incomplete overflow
validation for fabricated memory-counter text; its live input is kernel-owned.

- [Experiment and reproduction](../experiments/strata-candidates-20261109/README.md)
- [16K data](../experiments/strata-candidates-20261109/benchmark/results-16k.json)
- [64K data](../experiments/strata-candidates-20261109/benchmark/results-64k.json)
- [Cache-routing source audit](../experiments/strata-candidates-20261109/cache-route/source-audit.md)
- [Parser and arena checks](../experiments/strata-candidates-20261109/parser-arena/README.md)
- Raw local evidence: `results/strata-candidates-20261109/`.

Primary sources:

- [Prefill borrowing report, issue 5570](https://github.com/Niko1221/Strata/issues/5570)
- [Cache-biased routing, draft PR 5596](https://github.com/Niko1221/Strata/pull/5596)
- [DSML alias strings, PR 5598](https://github.com/Niko1221/Strata/pull/5598)
- [Linux arena admission, PR 5531](https://github.com/Niko1221/Strata/pull/5531)
