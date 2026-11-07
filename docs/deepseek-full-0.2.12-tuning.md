# Full DeepSeek tuning on Strata 0.2.12

The 2026-11-07 sweep keeps the existing defaults. Nine OMP screens did not
establish a better configuration for long coding conversations. The two control
runs measured 9.19 and 9.16 decode tokens/s.

The target is full Unsloth UD-IQ1_S on Strata 0.2.12, native engine 0.1.58.
The control uses 192K context capacity, a 64 GiB server cap, a 48 GiB expert
cache, twelve CPU threads, 4096-token prefill chunks, a 700 MiB GPU reserve,
16 Engram readers, and no DSpark. All six selected experts run.
The expert cache is part of the server cap. OMP has a separate 4 GiB cap.
Model swap is disabled. The guard stops below 8 GiB available Linux RAM.
Outgoing networking stays blocked.

Each candidate changes one part of the control configuration. First requests
contain 15,961–15,963 tokens. Completed conversations reach 16,571–18,287
input tokens. These screens do not fill the configured 196,608-token capacity.

| Changed setting | Decode t/s | First prefill t/s | Output tokens | OMP seconds | Peak server GiB | Checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Control | 9.19 | 267.44 | 1029 | 211.48 | 60.75 | pass |
| CPU threads: 8 | 8.15 | 272.89 | 543 | 150.92 | 59.43 | pass |
| Prefill chunk: 2048 | 9.61 | 189.91 | 1408 | 272.09 | 62.73 | pass |
| GPU reserve: 1536 MiB | 8.48 | 275.33 | 1430 | 261.62 | 59.69 | pass |
| Expert cache: 40 GiB | 8.49 | 268.60 | 342 | 130.91 | 52.66 | pass |
| Expert cache: 52 GiB | 9.37 | 274.83 | 618 | 151.01 | 62.65 | pass |
| DSpark: depth 1 | 8.50 | 278.72 | 860 | 191.28 | 64.00 | pass |
| DSpark: depth 3 | 8.74 | 278.40 | 506 | 141.18 | 61.41 | pass |
| Control repeat | 9.16 | 270.11 | 948 | 201.27 | 60.83 | pass |

Decode divides generated tokens by native decode seconds. First prefill counts
new tokens in the initial request. OMP time includes repair and a separate
resume. Warmup and capture-only probes are excluded. These are separate sampled
trajectories. Shorter elapsed time alone does not establish a speedup.

The 2048-token chunk improves measured decode slightly but reduces first
prefill by 29%. It retains the same 304 GPU experts. Its decode increase also
appears in early requests, so output length alone does not explain it.
The prefill cost makes it a poor default for long conversations.

The 52 GiB cache gains about 2% decode against both controls. One pass does not
establish that small gain. It uses more RAM and was not tested at long context.
The 40 GiB cache saves about 8 GiB at peak but reduces measured decode.
Eight CPU threads also reduce decode. CPU settings 0 and 16 were skipped:
on this host, both use the same physical-core pool as the twelve-thread control.

The larger GPU reserve removes all routed GPU experts. Both DSpark depths also
leave zero routed GPU experts. Dense weights remain in VRAM. Draft acceptance
does not offset the drafter cost in these runs. Both depths decode more slowly
than the controls.

All nine screens pass repair, 100 assertions, hidden-marker recall, and strict
resume. Validators remain unchanged. Recorded tool errors are benchmark-guard
refusals, not parser or patch-validation failures. These include a known
workspace-root read restriction. No run records OOM or model swap. DSpark depth
1 reaches the cap and records 754 reclaim events; the other screens record none.
Available Linux RAM stays above 56.5 GiB.

The [earlier growing run](deepseek-full-0.2.12.md) remains the long-context
evidence: 184,758 actual input tokens, 8.99 decode tokens/s, and 373.09 new-prefill
tokens/s. Later requests reuse prefixes. No candidate justified another long run
or a default change.

Sampling uses temperature 1 and top_p 0.95 without a fixed seed. Paths and
sizing vary slightly between requests. GPU counters include desktop allocations.
CPU-lane percentages are not cache-miss rates. A brief analysis process started
in the 23:42 CET minute; its exact stop time is unavailable. Cache52's measured
work ended at 23:41:52.836. DSpark's measured work began at 23:43:07.614.
Startup overlap is possible; exact non-overlap is not proven.

The [saved JSON](../results/deepseek-full-tuning-analysis-20261107/summary.json)
contains exact timing denominators, arguments, resource counters, and provenance.
The [quality audit](../results/deepseek-full-tuning-analysis-20261107/quality-notes.md)
records input differences and guard refusals. Raw runs remain under
`results/deepseek-full-tuning-0212-20261107/`.
All nine cleanup checks pass. Normal settings and routing data are restored.
All owned model, OMP, guard, observer, and search processes are stopped.
Ports 8888 and 8899 are closed. No CUDA compute process remains.
