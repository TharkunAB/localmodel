# Full Unsloth DeepSeek on Strata 0.2.12

2026-11-07. `deepseek`, `deepseek-full`, `deepseek-iq1-s`, and `strata` now
select full Unsloth Dynamic v3.1 UD-IQ1_S. The Coder weights were removed
from this machine. The optional `deepseek-coder` fetch definition remains.

The 0.2.12 repeat passed all four OMP checkpoints through 184,758 actual input
tokens. Decode averaged 8.99 tokens/s, 5.8% below the matched 0.2.11 run.
New-token prefill averaged 373.09 tokens/s, 5.0% above the previous run.
This single comparison does not show a decode improvement from the update.

## Configuration and method

- GPU: RTX 4070 SUPER, 12 GiB VRAM. Host: Ubuntu on WSL2.
- Weights: Unsloth DeepSeek-V4.1-Flash-1017 UD-IQ1_S, revision
  `8a6ec9cf118e785a362ea94ee123c61c70553be6`.
- Context capacity: 196,608 tokens. Targets: 16K, 64K, 128K and 180K.
- Model cap: 64 GiB, with zero model swap. Expert cache: 48 GiB inside that cap.
- Twelve CPU expert workers, 4,096-token prefill chunks and all six selected experts.
- DSpark off. Reasoning xhigh, temperature 1.0, top-p 0.95. No fixed sampling seed.
- Same frozen OMP workload, corpus, chat template and trained routing baseline.
- Runtime: source/API 0.2.12 and native engine 0.1.58. The baseline used 0.2.11/0.1.57.
- Offline serving and the outgoing-connection guard remained enabled.
- Linux available-memory stop threshold: 8 GiB. Windows memory was not monitored.

Both runs start with empty conversation slots. Later checkpoints reuse earlier
prefixes. Automatic compaction is disabled to test the larger input directly.
New-prefill rates count newly processed tokens, not the whole cached context.
The experiment does not measure normal compaction behavior.

## Results

| Metric | 0.2.11 / native 0.1.57 | 0.2.12 / native 0.1.58 |
| --- | ---: | ---: |
| Generated tokens | 3,887 | 2,626 |
| Native decode seconds | 406.978 | 291.986 |
| Decode tokens/s | 9.551 | 8.994 |
| Newly processed prompt tokens | 181,569 | 182,176 |
| Native prefill seconds | 511.005 | 488.294 |
| New-prefill tokens/s | 355.318 | 373.086 |
| Maximum actual input | 184,898 | 184,758 |
| Model HTTP seconds | 925.176 | 787.761 |
| OMP repair and resume seconds | 966.696 | 825.694 |
| Whole-case seconds | 1,132.386 | 1,016.620 |
| Startup seconds | 21.033 | 21.030 |
| Peak model memory, GiB | 64.000 | 64.000 |
| Peak whole-device VRAM, GiB | 11.609 | 11.624 |
| Minimum available Linux RAM, GiB | 50.455 | 47.269 |
| Model swap / OOM events | 0 / 0 | 0 / 0 |

The exact new decode denominator is 2,626 / 291.9864 seconds.
The exact new-prefill denominator is 182,176 / 488.2944 seconds.
HTTP durations use the recorder's monotonic `elapsed_ms` fields.
Whole-case time also includes startup, sizing and cleanup.

| Target input | Old decode tokens/s | New decode tokens/s | Old first-request prefill tokens/s | New first-request prefill tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 16K | 9.81 | 9.09 | 275.76 | 272.08 |
| 64K | 9.42 | 8.77 | 432.32 | 466.56 |
| 128K | 9.26 | 8.97 | 460.85 | 465.28 |
| 180K | 9.25 | 9.08 | 446.73 | 442.64 |

All four repair, recall, required-tool and strict resume checks passed.
The new run passed 3,118 code assertions. It generated 32.4% fewer tokens
and followed a different tool path. Its shorter wall time does not establish
a fixed-work speedup. The single sampled pair does not isolate run variance.

The one recovered refusal in the new run was `read(".")`. The preserved
benchmark guard rejected a valid workspace-root listing. The baseline had
three calls outside that guard's allowlist. These refusals are separate from
code correctness and native tool-parser behavior.

## Resources and limits

Whole-device GPU utilization averaged 65.6% over recorded samples, 95.6%
during generation and 71.6% during prompt processing. Peak utilization was
100%. Mean sampled power was 87.9 W, with a 183.4 W peak. These figures
include the desktop and are sample means, not process-only measurements.

The largest current-memory sample reached 64.000 GiB. It recorded 48.618 GiB
anonymous memory and 14.987 GiB file memory. The file count includes
6.873 GiB shared memory, leaving 8.114 GiB in the other file category.
Do not add shared memory to the file total. Kernel memory was about 0.393 GiB.
These counters are sampled separately and can differ slightly from the cap.
The older run did not record `memory.stat`, so its breakdown is unavailable.

The new run recorded 85,995 memory-cap events and no OOM or swap. Cap events
show reclamation pressure; they do not count reclaimed bytes. Shared WSL disk
counters include other processes and do not establish model-only SSD misses.
Windows host caching was not controlled.

## Disk cleanup and verification

Removed `models/deepseek41-1017-gsq-rco-coder-iq2-xxs`, including its generated
pack. This freed 67.18 GiB of unshared data. The 103 GiB Engram file remains
in the full-model directory with the same inode. Its hard-link count changed
from two to one. The removal command refreshed the verified full-model receipt.
Full weight identities, the prepared pack and the signed runtime passed checks.
Benchmark results, runtimes, Qwen weights and the optional DSpark files remain.

All model, worker, guard and observer cleanup checks passed. Routing state,
lane calibration, generated configuration and normal OMP settings were restored.
The owned search service was stopped. Ports 8899 and 8888 closed. Whole-device
VRAM returned to approximately 887 MiB of desktop use.

The local evidence is in
[the comparison](../results/deepseek-full-0212-20261107/analysis/comparison.json),
[the run](../results/deepseek-full-0212-20261107/full-cache48-192k-long/case-summary.json),
[the preparation](../results/deepseek-full-0212-20261107/preparation.json), and
[the deletion receipt](../results/deepseek-full-0212-20261107/cleanup/after.json).
These ignored evidence files are not included in the Git repository.
