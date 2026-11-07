# Strata 0.2.12 measurements

Measured on 2026-11-07. Measurements and cleanup are complete. The normal Qwen default remains MTP4. Explicit DFlash2 uses confidence 0.8. The long DFlash2 run decoded at 58.41 tok/s against MTP4's 52.17, but its different tool history required extra prefill and increased total model HTTP time. These runs do not establish a matched whole-workflow speedup.

After these measurements, Matti selected file-backed experts and adaptive fetch 1 as the local Qwen memory defaults. This trades speed for reclaimable RAM. The mapped MTP4, fetch 1 row below describes that configuration. The long MTP4 and DFlash2 comparisons used resident experts. Use `--expert-ram-gb 36` to restore that memory layout; adaptive fetch then defaults to off.

The raw evidence links below point to local, ignored files under `results/strata-0212-20261107`. Those files are not included in the published repository. Official upstream claims have separate public links. No target GGUF weights changed. Owned model and OMP scopes stopped, port 8899 closed, state was restored, and the temporary search service stopped.

## Settings and method

Hardware: RTX 4070 SUPER, 12 GB, under WSL. Each model has a 64 GiB cgroup limit and zero swap. Runs stop below 8 GiB Linux available RAM. Windows memory is not monitored. The model API and its native children retain the outgoing-connection restriction. OMP workers and their tools have a separate 4 GiB limit.

Old controls use Strata 0.2.11 and native 0.1.57. New screens use Strata 0.2.12 and native 0.1.58. Target weights and routing profiles are unchanged between each model's controls. The workload, public source corpus, code fixture, tool guard and validator are frozen. OMP uses xhigh, a 4,096-token response limit, temperature 1 and top-p 0.95. No sampling seed is fixed. These are single trajectories, not repeated estimates of a population mean.

- Qwen target: Unsloth Qwen3.8-Flash-Next UD-IQ1_M Dynamic v3.1, revision `d96201194f103c906994f283f504209dc8d2a10c`. The manual screens use 196,608-token capacity, int8 KV, a 65,536-token GPU KV window, 1,024 MiB VRAM reserve, direct PLE reads, and no suffix drafting. Routing prior and tail skipping are off; HC_Q8 is on.
- Resident Qwen MTP: 36 GiB expert RAM budget and four drafts, native `--spec 5`. The tuned DFlash2 screens retain that memory layout, use five drafts, native `--spec 6`, and no MTP fallback. The first uses confidence 0.35; the comparison changes it to 0.8. Native logs confirm the tuned selector and no loaded MTP layer.
- Adaptive-fetch screens: the same manual Qwen MTP4 settings, except expert RAM budget zero selects `--mmap-experts`. Compare modes 0, 1 and 2 within this layout. Resident mode disables adaptive fetch.
- DeepSeek Coder: ISTA GSQ-RCO Coder IQ2_XXS with the retained shared Engram shard. Settings are 196,608-token capacity, SSD cache zero, 12 CPU threads, 4,096-token prefill chunks, six experts, 700 MiB VRAM reserve, vision off and no drafter. Engram uses 256 MB cache and 16 threads. The isolated slot cache has one slot and a 2 GiB bound.

Rates below are sums of native tokens divided by sums of native time. They include completed OMP repair/resume requests only. They exclude the warmup and captured sizing requests. Prefill counts newly processed tokens, not reused context. The report matches each request to a native metric row; it does not average request rates. Strata 0.2.12 changes API timing accounting, so dashboard values alone are not a valid runtime comparison.

## Completed short screens

| Arm | Native decode, tok/s | New prefill, tok/s | Reported speculation acceptance | Native output tokens | Largest actual input | Matched requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen MTP4, 0.2.11 | 55.27 | 1,298.88 | 75.56% | 2,329 | 66,047 | 17/17 |
| Qwen MTP4, 0.2.12 | 53.63 | 1,319.75 | 71.89% | 3,917 | 66,421 | 17/17 |
| Qwen tuned DFlash2, confidence 0.35 | 54.24 | 1,454.71 | 62.46% | 2,871 | 66,031 | 16/16 |
| Qwen tuned DFlash2, confidence 0.8 | 58.57 | 1,452.02 | 81.11% | 2,672 | 66,144 | 16/16 |
| Qwen mapped MTP4, fetch 0 | 44.26 | 667.09 | 74.75% | 2,701 | 66,241 | 16/16 |
| Qwen mapped MTP4, fetch 1 | 44.46 | 690.38 | 74.47% | 2,934 | 66,257 | 15/15 |
| Qwen mapped MTP4, fetch 2 | 42.68 | 593.18 | 70.97% | 2,785 | 66,141 | 16/16 |
| Qwen factory Auto, 128K capacity | 61.63 | 1,617.99 | 87.23% | 3,732 | 66,155 | 17/17 |
| Qwen factory Auto, 192K capacity | 56.82 | 1,421.03 | 85.16% | 3,319 | 66,434 | 15/15 |
| Coder, 0.2.11 | 12.55 | 356.54 | No draft | 2,592 | 66,141 | 18/18 |
| Coder, 0.2.12 successful repetition | 11.74 | 412.22 | No draft | 2,681 | 66,002 | 16/16 |

The original new-Coder attempt timed out and is excluded from this completed-run table. Its successful same-budget repetition is included. Before the timeout, the first attempt's four completed requests measured 12.00 decode and 318.05 prefill tok/s. Its fifth request was cancelled and has no complete API metric match. Those partial values belong to the timed-out attempt and are not a completed-workload comparison.

Qwen MTP changed from 55.27 to 53.63 decode tok/s while generating 68% more output. This pair does not establish a causal runtime regression. Tuned DFlash2 at confidence 0.35 measured about 1.1% higher decode than new MTP, with different answers and output counts. Its whole case took 363.1 seconds versus 414.1 seconds for MTP; startup alone took 84.3 versus 122.9 seconds. Do not assign the whole-case difference to draft throughput.

The confidence-0.8 comparison retains the tuned pack and manual memory layout. It measured about 8.0% higher decode than confidence 0.35, while aggregate new-prefill remained close: 1,452.02 versus 1,454.71 tok/s. All code, recall, tool and strict-format checks passed. It generated 2,672 tokens versus 2,871, so this remains one sampled comparison. Its whole case took 339.0 seconds, including 78.7 seconds of startup. The completed matched long results are separate below.

Fetch 1 measured about 0.5% higher decode than fetch 0. Fetch 2 was slower in this screen. All three mapped configurations were slower than resident MTP in both decode and new-token prefill. The mapped configuration starts faster and uses less charged RAM, but that is a separate trade-off.

Factory Auto uses ordinary DFlash2 and different suffix lookup behavior. Its accepted/offered counts are 2,301/2,638 at 128K and 2,117/2,486 at 192K. These are aggregate speculation counters, not neural-drafter accuracy. The 128K case took 334.1 seconds, including 45.5 seconds of startup; 192K took 306.4 seconds, including 39.7 seconds of startup. Different output counts and startup times prevent these wall times from isolating inference speed.

Sources: [runtime controls](../results/strata-0212-20261107/bench-analysis/runtime-controls.json), [draft and memory screens](../results/strata-0212-20261107/bench-analysis/draft-and-mmap-screens.json), [adaptive modes](../results/strata-0212-20261107/bench-analysis/adaptive-012.json), [factory screens](../results/strata-0212-20261107/bench-analysis/stock-screens.json), [draft confidence](../results/strata-0212-20261107/bench-analysis/draft-confidence.json), and [Coder attempts](../results/strata-0212-20261107/bench-analysis/coder-final.json).

## Filled context and quality boundaries

192K is the manual configuration's capacity; factory Auto has completed screens at 128K and 192K capacity. Neither capacity is filled in these completed screens. Manual Qwen request inputs span 16,004–19,018 tokens at the 16K checkpoint, 32,377–33,543 at 32K, and 65,076–66,421 at 64K. Within those ranges, the confidence-0.8 screen spans 16,004–17,928, 32,391–33,454 and 65,151–66,144. Factory 128K spans 16,009–19,250, 32,555–33,540 and 65,121–66,155; factory 192K spans 16,009–18,117, 32,396–33,504 and 65,107–66,434. Old Coder spans 15,972–18,220, 32,433–33,262, and 65,229–66,141 respectively. Its successful new-release repetition spans 15,972–18,520, 32,491–33,251 and 65,228–66,002.

Later checkpoints reuse earlier context. For example, new resident MTP processes 16,005 new tokens for the first checkpoint, then 13,540 and 31,551 at the initial 32K and 64K requests. Their prefill rates are 1,500.5, 1,715.1 and 1,843.4 tok/s. These are not cold full-32K/full-64K measurements. Fetch 0 processes comparable suffixes at 1,029.4 and 907.2 tok/s, so its prefill loss is not restricted to the first cold request.

All completed Qwen screens and both completed Coder runs pass the code, recall and required-tool checks at all three checkpoints. Old Qwen and tuned DFlash2 at confidence 0.35 return the correct resume JSON values inside Markdown fences at all three checkpoints. The strict JSON check rejects those fences. Their raw lifecycle status remains `inconclusive_workload`; the report does not turn these into full passes. New MTP, tuned DFlash2 at confidence 0.8, all three adaptive-fetch screens, and both factory screens pass strict resume formatting too.

The first new-Coder attempt repairs the first code defect and passes its public 100-assertion check. It then spends the remaining phase budget on its final response and reaches the 300-second deadline. Its refused `ls -la` and `history://16384` calls violate the guard policy; a valid workspace read succeeds. Review found no launch-argument, routing, template or tool-parser discrepancy that explains the timeout. The cancelled final request generates 1,745 tokens over about 140 native decode seconds. The same-budget repetition passed all three checkpoints and stopped cleanly. It measured 11.74 decode and 412.22 new-prefill tok/s, against old Coder's 12.55 and 356.54. Output counts were 2,681 versus 2,592. Whole-case times were 595.93 versus 603.76 seconds. Keep the original timeout alongside this successful repetition: two new-release attempts produced one timeout and one complete run. This is not evidence of a general runtime speedup.

These small deterministic code checks do not measure general coding capability or establish exact semantic equivalence between drafters. The generated paths and output lengths differ. See [Coder timeout review](../results/strata-0212-20261107/bench-analysis/coder-timeout-review.md).

## Matched long-context runs

Both 192K-capacity configurations passed code, recall, required-tool and strict-format checks at all four checkpoints. MTP generated 4,448 tokens across 21 matched requests; tuned DFlash2 at confidence 0.8 generated 4,744 across 25. All forwarded requests matched native metric rows.

| Target checkpoint | Largest actual input, MTP / DFlash2 | Decode tok/s, MTP / DFlash2 | First new-prefill tokens, MTP / DFlash2 | First new-prefill tok/s, MTP / DFlash2 |
| --- | ---: | ---: | ---: | ---: |
| 16K | 18,715 / 18,013 | 53.38 / 58.92 | 16,009 / 16,010 | 1,329.9 / 1,594.9 |
| 64K | 66,272 / 67,175 | 52.28 / 59.00 | 46,347 / 47,276 | 1,865.1 / 2,070.3 |
| 128K | 131,958 / 131,920 | 52.00 / 57.74 | 64,576 / 63,354 | 1,867.5 / 1,977.3 |
| 180K | 185,243 / 185,415 | 49.95 / 57.66 | 52,046 / 52,109 | 1,730.3 / 1,888.3 |

Later prefill entries are new suffixes after reuse, not cold processing of the whole input. The largest inputs are 185,243 and 185,415 tokens within the 196,608-token capacity. Aggregate native decode was 52.17 tok/s for MTP and 58.41 for DFlash2. Aggregate new-prefill rates were 1,545.45 and 1,701.80 tok/s. Reported speculation acceptance was 71.59% and 81.01%.

| Timing category | MTP4 | Tuned DFlash2, confidence 0.8 |
| --- | ---: | ---: |
| Native decode | 85.267 s | 81.225 s |
| Native prefill | 117.225 s | 145.330 s |
| Sum of model HTTP response durations | 203.40 s | 227.75 s |
| OMP phase wall time | 271.19 s | 291.22 s |
| Startup | 73.34 s | 73.72 s |
| Whole benchmark case | 609.42 s | 627.97 s |

OMP phase wall includes tools, client work and polling. The whole case also includes startup, local prompt sizing and client setup. It is not an ordinary user's pure inference time. Both runs peaked near 64 GiB. MTP and DFlash2 recorded 155,582 and 171,116 cgroup max events respectively, with no high/OOM events or swap. Both passed cleanup. Source: [long comparison](../results/strata-0212-20261107/bench-analysis/long-comparison.json).

### Why DFlash2 needed more prefill

DFlash2 processed 247,322 new prompt tokens; MTP processed 181,165. Most of the difference came from the 64K resume. OMP replaced one earlier 568-character `clamp.py` read result with `[Superseded by a newer read of this file]`. The changed tool trajectory had read the file again. This stale-read maintenance is separate from automatic conversation compaction.

The rendered prompts still shared 65,782 prefix tokens, but Strata reused a compatible recurrent-state checkpoint at only 3,589 tokens. It therefore reread **62,193 unchanged tokens** before the actual edit. This resume read 63,586 tokens in total, against 34 for the corresponding MTP resume. Its HTTP duration was 32.08 seconds versus 1.35 seconds. That 30.73-second difference exceeds the 24.35-second difference between the complete runs' summed HTTP times.

The native log reported zero parked-conversation evictions. Strata keeps a limited set of recurrent-state checkpoints inside a parked conversation. The exact earlier checkpoint replacement sequence is not logged, so it cannot be reconstructed. Zero parked-conversation evictions does not imply that every earlier recurrent checkpoint remained available. Similar OMP replacements at the later DFlash2 resumes retained much longer prefixes and did not cause the same reread.

The API/native timing records agree. The evidence shows a real history-edit and prefix-reuse cost; it does not identify a DFlash-specific cache defect or an accounting error. Faster measured decode did not give a matched whole-workflow win here. Changing stale-read pruning or checkpoint retention needs a separate controlled test. Neither was changed. The [local prefill diagnosis](../results/strata-0212-20261107/bench-analysis/prefill-diagnosis.md) records the exact requests, tokenizer reproduction and source paths.

## Memory, I/O and adaptive counters

| Arm | Model cgroup peak, GiB | Whole-GPU peak, MiB | Linux available minimum, GiB | Observed guest root-device reads, GiB |
| --- | ---: | ---: | ---: | ---: |
| Old Qwen MTP4 | 64.00 | 11577 | 61.20 | 1.39 |
| New Qwen MTP4 | 59.32 | 11212 | 65.73 | 1.68 |
| Tuned DFlash2, confidence 0.35 | 64.00 | 11233 | 66.40 | 1.46 |
| Tuned DFlash2, confidence 0.8 | 64.00 | 11344 | 63.32 | 1.36 |
| Mapped fetch 0 | 44.83 | 11539 | 94.77 | 30.84 |
| Mapped fetch 1 | 44.83 | 11539 | 95.05 | 30.86 |
| Mapped fetch 2 | 48.81 | 11540 | 90.67 | 30.91 |
| Factory Auto, 128K | 61.07 | 11852 | 67.22 | 1.74 |
| Factory Auto, 192K | 62.89 | 11846 | 68.29 | 1.73 |
| Old Coder | 64.00 | 11821 | 96.56 | 77.88 |
| New Coder successful repetition | 64.00 | 11815 | 96.61 | 71.22 |

These cgroup totals include charged file-backed pages. The regular sampler did not retain `memory.stat`, so the anonymous/file-cache split at these peaks cannot be recovered. A zero private expert copy still uses RAM through mapped pages and the OS page cache. Some runs reached the cgroup cap and recorded reclaim/limit events. Confidence 0.8 recorded 110,954 max events but no high/OOM events or swap; factory 192K recorded no high/max/OOM events or swap. No completed or cancelled arm above recorded a model OOM or swap use. Whole-device VRAM measurements include the desktop and are rounded to MiB. The long MTP and DFlash2 runs reached 11,540 and 11,339 MiB respectively.

A separate [single live snapshot](../results/strata-0212-20261107/extra-memory-snapshots.jsonl), taken at 13:21:57 CET while `new-qwen-dflash2-conf08-long` was running, records anonymous memory 5.890 GiB, file 57.699 GiB, shared memory 32.844 GiB, and kernel memory 0.303 GiB. Its saved stage label is `running-65536`; that label is not an exact input-token measurement. The file category already includes shared memory. Subtracting shared memory leaves about 24.855 GiB in the ordinary file category, but that does not prove all of it is immediately reclaimable. Do not add shared memory to file a second time. This is not a peak breakdown or a measurement of another case. The completed long-run results above come from the saved lifecycle and request records, not this snapshot.

Native logs show limited GPU headroom in both 192K candidates. The completed confidence-0.8 screen's lowest start-of-request warning reports 320 MiB free, below the native 512 MiB warning threshold. Factory 192K logs zero free MiB for a prefill super-chunk and disables that layer-major prefill path. These native observations are distinct from the whole-device VRAM peak table and do not establish spare capacity for larger workloads.

Guest disk deltas cover the observed post-ready interval, including warmup and sizing. They include other processes and remain WSL guest counters. They are not model-only physical SSD traffic. File-scoped cache advice is advisory; Windows host cache state is uncontrolled. Startup timings therefore need the same caveat.

Adaptive modes 1 and 2 have positive native activation markers. The serving interface does not expose a separate adaptive admission/hit counter. Matched GPU-cache hit fractions are 683,691/1,614,720 = 42.34% for mode 0, 747,633/1,755,360 = 42.59% for mode 1, and 761,750/1,721,760 = 44.24% for mode 2. These are GPU lookup counts, not SSD miss counts. In resident mode, additional PCIe routes bypass this lookup denominator and must be reported separately.

Mapped modes account for about 1.06, 1.14 and 1.10 TB of repeated logical expert payload. They do not read that many physical SSD bytes. Their zero reported RAM/file blob counters omit mapped access paths. Do not interpret those zeros as zero file access or zero misses. The [analysis method](../results/strata-0212-20261107/bench-analysis/README.md) records the counter boundaries and denominators.

Every completed or cancelled arm described above stopped its owned server, emptied its scope, closed port 8899 and restored state. The temporary search service was also stopped after the study.

## Factory comparison

Factory setup uses the canonical IQ1_M model name. Its Auto policy selects the **ordinary** DFlash2 pack, native window 6 and confidence 0.8. It does not select the newly tuned `strata1` pack for IQ1_M. The ordinary source, intermediate pack and smaller selector pack were prepared with official code and matched every pinned size and SHA256. Target GGUFs were unchanged.

Factory settings use the full expert arena, native 700 MiB reserve and stock suffix behavior. WSL setup disables KV streaming and automatic PLE RAM loading. Its routing profile hash matches the manual arm; routing prior remains off, tail skipping remains zero, and HC_Q8 remains on. This is a whole-configuration comparison, not a single cache or drafter change.

For this text-only workload, setup received its official `--vision no` option. It generated identical configurations from the observed host RAM and from a 64 GiB RAM observation. The benchmark retained loopback binding, `--offline`, the outgoing-connection restriction, and the 64 GiB cap with zero swap.

The first stock-128K start reached readiness but stopped before the workload because the benchmark parser did not recognize the valid `expert arena:` log. A later read-only review found two guard imports that still used the original directory depth. The v2 harness adds the exact full-arena success marker and corrects those imports from four parent levels to five. It preserves the workload and genuine factory numerical settings. The failed preflight is retained separately and supplies no throughput result.

The first factory startup reported only 63 MiB CUDA free and recommended a 1,149 MiB reserve instead of the stock 700 MiB. Both completed v2 factory screens retained genuine settings and passed all checks, with no cgroup high/max/OOM events or swap. Their model peaks were 61.07 GiB at 128K capacity and 62.89 GiB at 192K. The later 192K prefill headroom warning remains a material limit. Factory settings have not been tested with filled input near 180K here. The explicit tuned-confidence-0.8 configuration and matched MTP4 control completed that longer test. The Coder repetition also completed. The normal MTP default remains; the optional tuned DFlash2 setting uses confidence 0.8.

## Published RTX 3060 claims

The [official 0.2.12 release](https://github.com/Niko1221/Strata/releases/tag/v0.2.12) reports RTX 3060 12 GB results on **Unsloth UD-IQ4_XS-v31**, a different target quant. Prefill changes from 87 to 264 tok/s at 2K and from 361 to 981 at 32K. Reported decode changes are code +37%, agent +34%, and JSON extraction -18.9%; no absolute decode rates are given for this row.

The publisher used 2–3 rounds, fresh engines, alternating order, greedy 300-token answers, and each version's own defaults. The final 0.1.58 packages were not timed again. The headline gain includes a file-tier policy fix and changed drafter selection. Configured context capacity and complete RAM/cap settings for that row are not published. It is neither a threefold decode claim nor a prediction for our IQ1_M workload. Exact source links and known gaps are in [the public-source review](../results/strata-0212-20261107/stock-defaults/research/README.md).
