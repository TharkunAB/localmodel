# Local 192K comparison: Qwen, Coder and full DeepSeek

2026-11-07. These measurements used Strata 0.2.11. The selected Qwen configuration reaches 185,531 actual input tokens at 53.20 native decode tok/s with more free VRAM than the earlier baseline. Full Unsloth's lower-effort JSON Patch repair passes all code checks, but the public-check workflow is incomplete and a broader quality advantage is unproven. After this comparison, Matti selected full Unsloth as the `deepseek` default and removed the local Coder weights. Coder remains available through the explicit `deepseek-coder` alias.

Links into `results/` and the installed OMP source refer to local evidence files. They are available on this machine and are not published with the repository documentation.

All three weight sets pass the growing long-context test and the primary dependency-planner diagnostic. Full Unsloth averages 9.55 native decode tok/s, about 20% below Coder, while using a 48 GiB dedicated expert-cache budget. These are measurements from one local study, not a general model ranking.

## Configurations and method

The throughput and primary coding runs use the RTX 4070 SUPER with 12 GiB VRAM, official Strata 0.2.11/native 0.1.57, OMP 18.8.6, a **64 GiB model cap**, zero model swap and xhigh reasoning. The memory guard stops below 8 GiB available Linux RAM. Each candidate has **196,608 tokens of capacity**. Capacity is the maximum permitted context, not the amount actually processed. The separate secondary diagnostic changes only client reasoning effort and is identified below.

| Candidate | Expert RAM budget | Decoder | Other settings |
| --- | ---: | --- | --- |
| **Qwen3.8-Flash-Next Unsloth UD-IQ1_M v3.1: selected default** | **36 GiB** | MTP4 | 65,536 GPU KV cells, INT8 KV, direct PLE reads, **1,024 MiB VRAM reserve**; expert skipping and suffix lookup off |
| Qwen: earlier throughput and primary coding baseline | 32 GiB | MTP4 | Same settings except **768 MiB VRAM reserve** |
| DeepSeek GSQ-RCO Coder IQ2_XXS | 0 GiB private cache | None | 12 CPU workers, 4,096-token prefill, file-backed experts |
| Full DeepSeek Unsloth UD-IQ1_S v3.1 | 48 GiB | None | 12 CPU workers, 4,096-token prefill; long validation passed |

Zero private cache still lets Linux cache Coder's expert pages and reclaim them for compilation or other programs. Full DeepSeek uses a 48 GiB dedicated RAM-store budget, with 39.0 decimal GB of expert payload loaded at startup. Qwen holds about **31.9 GiB of resident expert payload**. Dedicated allocations are less reclaimable than clean file pages, even though every model has the same 64 GiB cap. The budget and payload sizes differ. Neither is a measured anonymous-memory peak. Keep that difference in mind when compiling alongside inference.

Qwen's native CPU and prefill settings remain automatic. Both DeepSeek defaults now use **192K capacity and a 64 GiB cap**: Coder has cache0; full Unsloth has cache48. Both use 4K prefill and no DSpark. Production reasoning remains **xhigh**. The primary Qwen coding diagnostics below used the earlier 32 GiB/768 MiB baseline; they were not repeated with the selected 36 GiB/1,024 MiB allocation.

The growing conversations use real OMP read, edit, validation and resumed-response flows. They contain public source text and test small repairs and recalled facts. Automatic compaction is disabled. These runs therefore do not test normal compaction behavior. Qwen's ordinary OMP threshold is about 157K input.

## Completed long-context results

| Run | Maximum actual input | Native generated tokens / decode seconds | Weighted decode | OMP repair + resume wall time | RAM / whole-GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Qwen selected, 36 GiB/1,024 MiB**, 16K/64K/128K/180K targets | **185,531** | **4,782 / 89.8841** | **53.20 tok/s** | **281.13 s** | **64.00 / 11.38 GiB** |
| Qwen baseline, 32 GiB/768 MiB, same targets | 185,151 | 3,290 / 59.5461 | **55.25 tok/s** | 230.83 s | 64.00 / 11.72 GiB |
| Coder, 16K/64K/128K/180K targets | 184,811 | 3,034 / 254.6245 | **11.92 tok/s** | 744.46 s | approximately 64.00 / 11.84 GiB |
| Full Unsloth, 16K/64K/128K/180K targets | 184,898 | 3,887 / 406.9776 | **9.55 tok/s** | 966.70 s | 64.00 / 11.61 GiB |

The Qwen baseline's first-checkpoint prefill is 1,548 tok/s; its final-checkpoint new-suffix prefill is 1,783 tok/s. Coder reaches 412 and 483 tok/s. Later requests reuse large prefixes; these are not cold 180K-prefill rates. Both runs pass all four code, recall and strict-resume checkpoints, including 3,118 assertions each. The Qwen baseline has no tool errors; Coder recovers from three early errors. Neither records OOM or swap. [Qwen baseline](qwen-unsloth-192k.md), [Coder evidence](../results/deepseek-192k-20261106/analysis-review/README.md).

The **selected Qwen headroom run** also passes all four checkpoints and 3,118 assertions. Its checkpoint decode rates are **54.84, 55.05, 52.30 and 50.99 tok/s**. Initial/new-suffix prefill is **1,354, 1,677, 1,619 and 1,517 tok/s**. Across 26 OMP requests, 181,280 new prefill tokens take 131.4725 native seconds; MTP accepts **3,257/4,449 drafts, or 73.21%**. One chained validator command is refused at 128K and the task recovers. It is not a read-parser failure.

Native free-VRAM reports improve from **361 MiB at startup and a 55 MiB minimum** to **714 MiB and 498 MiB**. Low-free-memory warnings fall from three to one. The 36 GiB expert budget loads **31.90 GiB of pinned expert payload**, close to the baseline's 31.89 GiB. The budget is a ceiling, not four additional GiB necessarily allocated. The new run reaches the 64 GiB cgroup cap, with 86,894 max events, no high/OOM/kill events and zero model swap. Minimum Linux available RAM is 64.61 GiB. All cleanup and restoration checks pass.

This trades a **3.71% lower sampled decode rate** for more measured VRAM headroom. The RAM ceiling, VRAM reserve and cache preparation change together: the new run advises both raw and prepared weight files, while the old run advises raw shards only. Outputs and request counts also differ. The comparison does not isolate a reserve-only performance cost. The selected reserve does not guarantee 512 MiB free under another desktop load. Startup takes 97.07 seconds and the full case 485.62 seconds, including preparation, sizing and cleanup. [Headroom analysis](../results/qwen-headroom-20261107/analysis/README.md), [exact denominators](../results/qwen-headroom-20261107/analysis/comparison.json).

Full Unsloth also passes all four checkpoints and 3,118 assertions. It recovers from three early tool errors. Its sampled decode is **19.8% below Coder's 11.9156 tok/s**; selected Qwen is about **5.6 times** as fast at native decode. These are aggregate observations, not causal whole-task speedups. Generated output, recovered errors and routing calibrations differ.

Of the six recovered Coder/full tool refusals, **one Coder `read('.')` call was a valid root listing rejected by the older guard**. Coder's other two calls requested internal history resources. Full Unsloth requested two internal history resources and `bash ls`. Those five actions were outside the stated benchmark policy. None was a code-correctness failure, and none was a raw/range-selector refusal. All occurred at the first 16K checkpoint. [Recorded attribution](../results/memory-io-20261107/tool-refusals.json).

| Full Unsloth checkpoint | First actual input | New tokens / native prefill seconds | Prefill tok/s | Repair + resume wall seconds |
| --- | ---: | ---: | ---: | ---: |
| 16K | 15,970 | 15,970 / 57.9123 | 275.76 | 302.28 |
| 64K | 65,158 | 45,822 / 105.9902 | 432.32 | 201.29 |
| 128K | 130,799 | 64,744 / 140.4881 | 460.85 | 241.50 |
| 180K | 183,909 | 52,211 / 116.8747 | 446.73 | 221.63 |

The final checkpoint reuses 131,698 input tokens. The final resumed request reaches **184,898 actual input tokens**. Across 23 OMP requests, native prefill processes 181,569 new tokens in **511.0047 seconds**; native prefill plus decode totals 917.9823 seconds. Summed HTTP response intervals are 925.18 seconds, OMP phases take 966.70 seconds, and the whole case takes 1,132.39 seconds. The latter includes 21.03 seconds of startup, a 7.02-second warmup, sizing and cleanup.

The model reaches its 64 GiB cap. Its cgroup records **2 high events and 65,535 max events**, which show memory pressure at the limits; they do not measure how many bytes were reclaimed. OOM events, OOM kills and model swap remain zero. Minimum available Linux RAM is **50.45 GiB**. Whole-device VRAM peaks at **11.61 GiB**. Server, worker, guard and observer cleanup completes, port 8899 closes, and profiles, lane state, generated configuration, active state and normal settings are restored. [Completed full run](../results/deepseek-unsloth-20261107/full-cache48-192k-long/case-summary.json), [OMP checks](../results/deepseek-unsloth-20261107/full-cache48-192k-long/summary.json).

The separate Coder 256K-versus-192K capacity pair uses only 16K/32K/64K input targets. Weighted decode is **12.51 versus 12.67 tok/s**, a 1.3% difference from one sample each. OMP wall time is 412.52 versus 381.96 seconds, but outputs contain 2,358 versus 2,157 generated tokens and follow different tool paths. This is not a fixed-work speedup. The longer run establishes usable context near 192K; the capacity reduction alone does not establish a repeatable throughput gain. [Paired review](../results/deepseek-192k-20261106/analysis-review/paired-report.md).

Weighted native decode divides total generated tokens by total native decode time. OMP wall time also includes prefill, tools and wrapper overhead. Total case time additionally includes startup, sizing requests and cleanup. Output lengths differ, and the older harness polls at 10-second intervals. Do not treat these measurements as interchangeable or rank capability from tokens/s.

## Memory and I/O limits

The three original long runs and the resident screen reached approximately 64 GiB of charged model memory, with zero swap and OOM events. The recordings lack `memory.stat` and process `smaps`, so anonymous and file-backed peaks cannot be separated afterward. Minimum host-wide Linux available RAM was 61.43 GiB for the Qwen baseline, 95.93 GiB for Coder, 50.45 GiB for full Unsloth and 45.43 GiB for resident mode. Available RAM includes reclaimable file cache. It is not remaining space inside the model's cap.

The kernel's `file` category includes tmpfs and shared memory. `shmem` identifies its swap-backed subset; it does not mean that swap was used. Subtracting `shmem` from `file` separates the ordinary filesystem category. It does not guarantee that every such page is immediately reclaimable. [Kernel cgroup v2 definitions](https://docs.kernel.org/admin-guide/cgroup-v2.html). A separate [snapshot appendix](../results/memory-io-20261107/README.md#appendix-snapshots-during-short-coding-tasks) records all three candidates during short tasks. Those snapshots occur at different points and do not replace the missing long-run peak breakdowns.

A later full-Unsloth snapshot during its planner task records **48.70 GiB anonymous memory and 2.30 GiB shared RAM**. This confirms that the earlier short-task snapshot did not capture its later allocation. It remains one observation, not a peak or a matched comparison with another candidate. [Saved later snapshot](../results/coding-quality-20261107/runs/full-cache48-quality-v4/memory-breakdown-snapshot-late.json).

The **Qwen baseline's** 20 OMP requests record **812,561/1,481,397 GPU expert-cache hits, or 54.85%**. The other 668,836 lookups are GPU-cache misses, not SSD misses. A separate 469,323 routed experts use the PCIe/other-GPU path. These requests report zero expert file blobs and 922,144 decode RAM blobs. The final KV report records **99.10% VRAM hits** over 27,605,688 block reads and 1,003.7 MiB read from host RAM. The selected Qwen run also reports zero logical expert file blobs. These counters do not include all other I/O or prove spare GPU capacity.

DeepSeek's saved native logs provide **no actual RAM-store hit/miss or logical expert-file-read counters**. Coder maps approximately 63.0 decimal GB of CPU-lane experts; full Unsloth maps approximately 113.3 GB. Their CPU-lane percentages describe execution routing, not cache misses. Resident mode's larger copied payload leaves less cap space for file-backed working data. Its extra cap-pressure events support concern about that trade-off, but do not quantify reclaimed bytes or stalled time.

During each OMP request span, adjacent samples bound shared guest-device reads at **3.215–3.216 GiB for Qwen**, **121.982–122.257 GiB for Coder**, **1,071.599–1,074.481 GiB for full Unsloth** and **244.970–246.585 GiB for resident mode**. These are shared Linux block counters. They combine expert, Engram/PLE and other activity, and Windows caching can satisfy reads without physical SSD traffic. They are neither model-specific I/O nor SSD miss rates. Different request spans and generated outputs prevent a matched per-task or per-token I/O speedup claim. [Bounded analysis and evidence](../results/memory-io-20261107/README.md).

## Full Unsloth short screens

Each screen starts around 15,960 actual input tokens under 192K capacity. All use the same full-model weight pins, frozen routing counts and 64 GiB cap. The public warmup is excluded from decode rates.

| Expert cache / decoder / prefill | Native tokens / decode seconds | Decode tok/s | First prefill tok/s | OMP wall seconds | RAM peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 GiB / none / 4,096 | 968 / 130.9512 | 7.39 | 266 | 231.21 | 64.00 |
| **48 GiB / none / 4,096** | **550 / 59.4319** | **9.25** | **279** | **141.04** | **59.42** |
| 48 GiB / DSpark3 / 4,096 | 517 / 57.2871 | 9.02 | 285 | 151.13 | 61.59 |
| 48 GiB / DSpark5 / 4,096 | 3,344 / 368.5297 | 9.07 | 283 | 483.29 | 63.98 |
| 48 GiB / none / 8,192 | 610 / 74.3380 | 8.21 | 321 | 160.95 | 56.83 |

All screens eventually pass repair, recall and strict resume. DSpark5 needs extra turns, including recovery from one code assertion failure. Its long response explains much of its wall time; this single trajectory does not prove a capability loss. The cache48/no-draft short screen reaches at most 16,737 actual input tokens. Its separate long validation is reported above. [Completed screens and denominators](../results/deepseek-unsloth-20261107/analysis-review/README.md).

**Use 4K prefill and no DSpark for the full-model default.** Moving to 8K improves first prefill by 15.3%, but reduces weighted decode by 11.3%. Native startup also moves target experts out of VRAM: 304 with 4K prefill, zero with 8K. DSpark accepts 91.4% of drafts at depth 3 and 75.0% at depth 5, yet does not improve decode throughput. Its allocation leaves only two GPU experts across target and draft layers. These allocation changes plausibly explain the result; the screens do not isolate causality.

The dashboard's DSpark-off flag and negative expert-slot count are metadata bugs. Native layer allocation and actual draft counters prove that DSpark ran. Do not use those dashboard fields to infer decoder state or cache misses. [Metadata review](../results/deepseek-unsloth-20261107/analysis-review/metadata-issues.md).

A separate **resident-copy screen did not improve the selected full-model configuration**. At the same 192K capacity and 64 GiB cap, Strata copied **7,931/15,055 CPU-lane experts, or 59.4 decimal GB**, into RAM. The remaining experts stayed file-backed. Native decode was **934 tokens / 122.6252 seconds = 7.62 tok/s**, below the 48 GiB demand-cache sample's 9.25 tok/s. First prefill was **301.7 tok/s**. Repair and resume took **211.32 seconds**, with at most **17,203 actual input tokens**. All checks and cleanup passed. Memory reached the cap, with 60 high events and 913,860 max events; OOM and swap remained zero. This is one sampled trajectory. It supports retaining demand caching rather than making resident copy the default. [Resident allocation and cleanup](../results/deepseek-resident-20261107/full-resident-16k/case-summary.json), [repair](../results/deepseek-resident-20261107/full-resident-16k/009-16384-repair.summary.json), [resume](../results/deepseek-resident-20261107/full-resident-16k/010-16384-resume.summary.json).

## Completed primary coding diagnostics

| Candidate | JSON Patch repair | Dependency planner |
| --- | --- | --- |
| Qwen Unsloth UD-IQ1_M | **127/133; incomplete repair** | **1,546/1,546; pass** |
| Coder IQ2_XXS | **Incomplete: 900-second deadline** | **1,546/1,546; pass** |
| Full Unsloth UD-IQ1_S | **Incomplete: 900-second deadline** | **1,546/1,546; pass** |

These two fresh repository tasks use identical files and requirements across models. The primary protocol is **v4-read-compatible**: each task permits **24 requests, 4,096 output tokens per reply and 900 seconds**, with a 4 GiB tool cap. Independent tests check the edited code. Code correctness is separate from final prose and tool execution. Timeouts, truncation and infrastructure failures are inconclusive. No outcomes can be inferred from fixture self-tests. Two diagnostics will still not establish a general coding ranking. [Suite](../results/coding-quality-20261107/README.md).

The tool policy is more restrictive than everyday OMP use. It permits declared workspace reads, edits to two modules, and only the exact `python3 check.py` shell command. It also applies the documented Python source restrictions. Custom test commands, network access and subagents are unavailable. A refusal of an otherwise ordinary development command can therefore reflect the diagnostic policy. The shared wall-time limit also favors faster configurations: a slower decoder can complete fewer reasoning tokens and tool turns before the same deadline. The request and output limits are ceilings, not equal amounts of completed work. This tests bounded progress on this hardware and does not isolate model capability.

Qwen completes JSON Patch in **20 requests and 310.17 seconds**, and the planner in **13 requests and 132.07 seconds**. Both public checks and required tool workflows pass. Neither task has a truncated reply or a time/request-budget interruption. JSON Patch's failed independent checks therefore represent an incomplete repair in this sample, rather than an infrastructure cutoff. [JSON Patch result](../results/coding-quality-20261107/runs/qwen-quality-v4/json_patch/summary.json), [planner result](../results/coding-quality-20261107/runs/qwen-quality-v4/planner/summary.json).

The six JSON Patch failures are upstream cases 37–42. They test one underlying defect: array `replace` calls insertion without removing or overwriting the old element. Replacing `/0` in `["foo"]` with `"bar"` returns `["bar", "foo"]` instead of `["bar"]`. The same defect affects numeric, boolean, null and nested-array replacement values. The written contract requires RFC 6902 replacement. An independent replay in separate offline, read-only views reproduces all six candidate failures and all six reference passes. The generated code was not changed, and no feedback was supplied to another run. These are six checks of one defect, not six independent capability failures. [Contract](../results/coding-quality-20261107/fixtures/json_patch/README.md), [candidate](../results/coding-quality-20261107/runs/qwen-quality-v4/json_patch/workspace/patch.py), [validator result](../results/coding-quality-20261107/runs/qwen-quality-v4/json_patch/validation.stdout.json).

Coder's first JSON Patch attempt stops as **`inconclusive_timeout` after 900.55 seconds and seven requests**. It completes no edits; both source files still match the initial fixtures. The resulting **49/133 checks are the unchanged baseline**, not the score of a submitted repair. No completed reply hits the output-token limit. Valid root, raw and range reads work. The first `python3 check.py` call produces the expected baseline assertion failure. The next `python3 smoke.py` call is rejected because it bypasses the required public-check wrapper. The task scope is cleaned before the planner starts. This is an incomplete attempt under the time and tool constraints, not a completed low-quality solution. [Timed-out task](../results/coding-quality-20261107/runs/coder-quality-v4/json_patch/summary.json).

Coder's planner completes in **nine requests and 332.19 seconds**, with **1,546/1,546 independent checks**, a passing public check and the required tool workflow. [Planner result](../results/coding-quality-20261107/runs/coder-quality-v4/planner/summary.json), [lifecycle cleanup](../results/coding-quality-20261107/runs/coder-quality-v4/lifecycle-summary.json).

Full Unsloth's JSON Patch attempt stops after **900.52 seconds and five requests**, with no completed edits or tool-policy errors. Its raw status is **`inconclusive_runtime`**, with error **`Deadline exceeded`** and final stop reason **`aborted`**: OMP's own deadline fires before the outer timeout handler. The cause is a deadline, and the raw label is preserved. Its **49/133 checks are the unchanged baseline**, not a submitted repair's correctness score. [Raw task result](../results/coding-quality-20261107/runs/full-cache48-quality-v4/json_patch/summary.json).

Full Unsloth's planner completes in **eight requests and 604.40 seconds**, passing **1,546/1,546 independent checks**, the public check and required tool workflow. All three candidates therefore complete the planner under this protocol. Qwen completes more JSON Patch work within the time limit, but neither DeepSeek timeout establishes the correctness of a completed solution. [Planner result](../results/coding-quality-20261107/runs/full-cache48-quality-v4/planner/summary.json), [lifecycle cleanup](../results/coding-quality-20261107/runs/full-cache48-quality-v4/lifecycle-summary.json).

The successful planner runs separate native generation from prefill and whole-task time:

| Candidate | Generated tokens / native decode seconds | Weighted decode | Native prefill seconds | Task wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 5,940 / 109.7696 | 54.113 tok/s | 9.8681 | 132.0692 |
| Coder | 4,186 / 282.2980 | 14.828 tok/s | 34.1997 | 332.1855 |
| Full Unsloth | 5,527 / 534.9240 | 10.332 tok/s | 57.7198 | 604.4042 |

These denominators include the completed planner requests only. Different output lengths and tool paths prevent treating the wall-time ratios as a controlled decoding speedup. All three edited packages pass the same planner checks.

Two earlier Qwen runs remain **excluded pilots** because their guard rejected valid raw reads and root listings. The corrected guard uses the installed OMP parser and passes focused and real offline reader checks. Fixtures and task semantics are unchanged.

The 12-request pilot reaches its JSON Patch request limit at 126.07 seconds, leaving 55/133 checks; its public failure occurred before edits. Its planner passes 1,546/1,546. The old-read 24-request pilot passes both code suites, but JSON Patch retains `inconclusive_truncated` after recovering from an earlier clipped reply. These samples illustrate run variability and do not replace the primary results. Original protocols and results remain in [the 12-request archive](../results/coding-quality-20261107/protocol12-snapshot/snapshot.json), [the 24-request archive](../results/coding-quality-20261107/protocol24-old-read-snapshot/snapshot.json) and their linked pilot directories.

## Completed secondary effort-50 diagnostic

| Candidate | JSON Patch at low effort |
| --- | --- |
| Coder IQ2_XXS, cache0 | **Incomplete: 900-second deadline** |
| Full Unsloth UD-IQ1_S, cache48 | **133/133 code checks; workflow incomplete** |

This separate diagnostic runs JSON Patch once per DeepSeek variant with OMP `--thinking low`. Captured real OMP requests contain `reasoning_effort: low`; both installed templates render effort **50 on their 1–100 scale**. This is not a 50-token budget. The captures run offline and forward no model requests. The fixtures, validation, 24-request limit, 4,096 output tokens per request, 900-second deadline, 192K capacity, 64 GiB model cap, cache settings and cleanup policy remain the same. Production defaults stay xhigh. Keep these results separate from the primary outcomes; they do not replace a timeout or establish a published-benchmark score. [Secondary protocol and independent review](../results/coding-effort-20261107/README.md), [request mapping proof](../results/coding-effort-20261107/capture-check.json).

Coder's low-effort attempt stops as **`inconclusive_timeout` after 900.57 seconds and six requests**. Actual request settings verify low effort and the 4,096-token limit on every recorded request. It completes no edits, and **49/133 is again the unchanged initial baseline**. No completed reply has a length stop or a runtime error. This is another incomplete attempt. It establishes neither an improvement from low effort nor the quality of a completed repair. [Coder low-effort result](../results/coding-effort-20261107/runs/coder-low/json_patch/summary.json), [cleanup](../results/coding-effort-20261107/runs/coder-low/lifecycle-summary.json).

Full Unsloth completes **five edits in nine requests and 894.56 reported seconds**. Its edited code passes **133/133 independent checks**, and actual requests verify low effort with the 4,096-token limit. It does not run the required public check. Consequently, `code_passed` is true, while `public_check_passed`, `workflow_complete` and `task_passed` are false. The raw process exits with status 0 and the runner records `completed`; the final answer is empty and every assistant stop reason is `toolUse`. These fields must not be treated as successful workflow completion. [Raw result](../results/coding-effort-20261107/runs/full-cache48-low/json_patch/summary.json), [independent checks](../results/coding-effort-20261107/runs/full-cache48-low/json_patch/validation.stdout.json).

The early inline Python probe is refused by the exact `python3 check.py` policy. The final `read('patch.py')` and `read('pointer.py')` calls are valid, but OMP skips them with synthetic `assistant_stop_aborted`, `executed: false` results and **`Deadline exceeded`**. The provider finishes with `tool_calls`. OMP's deadline-before-dispatch branch explains this exit; it is not a read-parser failure. The frozen runner misses the synthetic tool cancellation because it checks assistant errors and process exit. Its raw label is retained. OMP's wall-clock deadline and the wrapper's monotonic duration also differ. [Saved tool events](../results/coding-effort-20261107/runs/full-cache48-low/json_patch/stdout.jsonl), [installed OMP branch](/home/tritlo/.bun/install/global/node_modules/@oh-my-pi/pi-agent-core/src/agent-loop.ts:1590).

All full-low scopes, server and state cleanup checks pass. This trial supports **correct code under the secondary low-effort setting**, with an unfinished public-check workflow. It does not replace the primary timeout, prove a repeatable benefit from low effort, or establish an overall capability ranking. [Lifecycle cleanup](../results/coding-effort-20261107/runs/full-cache48-low/lifecycle-summary.json).

## Published capability evidence

These are publisher claims, not measurements from this machine. Benchmark names, quants and harnesses must match before scores can be compared.

**Weight provenance:** the checked full-Unsloth HEAD, `2688c0c3` from November 2, has the same IQ1_S and DSpark file hashes and sizes as our installed `8a6ec9cf` pin from October 30. The newer revision updates documentation; it does not provide a changed DeepSeek quant for this comparison. The fetch restored missing transformer files and reused the verified shared Engram and draft artifacts. All five full-model shards passed local hash verification. The publisher revision comparison and the local verification are separate evidence. [Revision and hash comparison](../results/deepseek-unsloth-20261107/RESEARCH.md), [completed fetch verification](../results/deepseek-unsloth-20261107/fetch-full.log).

| Published variant | Published results | Attribution limit |
| --- | --- | --- |
| Full DeepSeek Unsloth UD-IQ1_S v3.1 | DeepSWE v1.1 **65.8%**; Terminal-Bench 2.1 **69.5%**; AutomationBench v1.0.6 **51.5%** | Explicit IQ1_S attribution in [Unsloth's documentation](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) |
| Coder IQ2_XXS | SWE-bench Verified **77.00%**; LiveCodeBench v6 **88.57%** | Publisher's base scores are 81.20% and 89.71%; these benchmarks differ from the row above. [Coder card](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) |
| Qwen Flash-Next Unsloth **IQ1_S** v3.1 | DeepSWE v1.1 **54.0%**; Terminal-Bench 2.1 **68.3%** | This is **not our IQ1_M quant**. Do not assign these scores to the local configuration. [Publisher documentation](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) |

Unsloth's newer DeepSeek evaluations use 512K capacity, reasoning effort 50 and up to 65,536 output tokens per reply. DeepSWE uses 113 tasks with at least three passes. Terminal-Bench uses 82 of 89 tasks after seven reference failures; Aider uses pass@2. These settings differ from our OMP tests. The Coder card specifies effort 50, but the inspected evidence does not establish a matching harness and trial protocol. [Published methods](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/dynamic-3.1-ggufs), [source review](../results/deepseek-unsloth-20261107/RESEARCH.md).

The published Qwen protocol uses 256K context without YaRN, preserved thinking, and xhigh reasoning. It does not use the DeepSeek effort-50 setup. [Qwen methods](https://unsloth.ai/docs/models/qwen3.8-next).

**Aider correction:** Unsloth's November 2 prose reports Coder at 47.6% against a 78.2% comparator. It does not identify that comparator's quant. The chart could not be verified, so **78.2% must not be attributed to UD-IQ1_S**. The older IQ1_S table reports 36.52% Aider pass@1 with a 32K setup. Do not combine that table with the newer pass@2 result. [Dated guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash).

A bounded official-source follow-up did not resolve the Aider chart label or verify a current IQ1_M agent score. The Qwen guide's older IQ1_M table contains token-agreement and KLD metrics, not coding-task scores. It describes a 74.5 GB artifact, while the current IQ1_M repository shows approximately 68 GB, and it does not pin that table to v3.1. Do not transfer those fidelity metrics to our installed files. The current verified 54.0%/68.3% agent scores remain explicitly IQ1_S. [Follow-up sources and retrieval limits](../results/deepseek-unsloth-20261107/public-score-followup.md).

Coder removes half the routed experts; full Unsloth retains the expert pool at lower mixed precision. Coder's advertised 1.05-bit figure counts removed parameters as zero and excludes Engram. Its retained parameters average about 2.06 bits. These are different compression trade-offs, not a ranking implied by the quant names. [Coder method](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF).

## Recommendation and launch commands

Use **Unsloth Qwen at 192K with MTP4, a 36 GiB expert budget and 1,024 MiB reserve** for interactive OMP work. The `deepseek` alias now selects **full Unsloth**. It is about 20% slower than Coder in this 0.2.11 long sample, uses a 48 GiB dedicated cache budget and has no demonstrated broad quality advantage here. This selection keeps the full expert pool and uses Unsloth target weights. The deadline-incomplete workflow and small diagnostic scope still limit the quality comparison.

```bash
localmodel start qwen3.8        # Selected interactive configuration
localmodel start deepseek       # Full Unsloth, cache48
localmodel start deepseek-full  # Same full Unsloth configuration
```

Run one server at a time. All three now default to 192K capacity and a 64 GiB cap; production reasoning remains xhigh. The secondary low-effort trials do not change that default.

Completed lifecycle runs stopped their servers and helpers and restored normal settings. GPU peaks include desktop allocations. Shared WSL disk counters do not prove model-only physical SSD traffic or cache miss rates. These are single sampled trajectories with no fixed sampling seed. Keep older results on the previous VM storage separate from these comparisons.
