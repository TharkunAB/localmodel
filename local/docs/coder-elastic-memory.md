# Coder with reclaimable expert memory

Measured on 2026-11-03 with Strata 0.2.10 / native 0.1.56 and Coder IQ2_XXS.
The hardware was the local RTX 4070 SUPER 12 GiB and Ryzen 9 7950X in WSL2.
The Linux VM had 110 GiB of RAM. The installed default now uses a 64 GiB cap
with file-backed experts and zero private cache, at the user's request.
These sharing benchmarks used a 70 GiB cap. No runtime, weight, or sampling
settings changed.

## Configuration

The existing launcher can keep CPU experts in their files:

```sh
localmodel start deepseek --mem 70G \
  --expert-memory cache --expert-ram-gb 0
```

Zero disables Strata's private demand cache. Linux can still keep the file
pages in RAM. It can discard clean pages under pressure and read them again
when needed. The cap includes these file pages. GPU allocations do not become
reclaimable through this setting.

The experiment also set `CPUWeight=10` on its dedicated systemd slice and
model scope. This is relative CPU priority, not a CPU quota. The command above
does not set that priority. No new priority flag was added to `localmodel`.
The test verified the effective weight. It did not measure responsiveness
against a CPU-intensive compilation.

All cases used 256K context capacity, twelve CPU workers, 4096-token prefill
chunks, xhigh reasoning, six active experts, and no drafter. The GPU retained
353 expert slots. Strata reported 7,327 CPU experts and zero private RAM expert
copies. The routing profile matched the earlier resident comparison. Existing
routing and lane-calibration files were preserved and restored.

## Quiet OMP comparison

The same public-source OMP conversation grew through 16K, 32K, and 64K tokens.
It checked code repair, assertions, fact recall, required tools, and strict JSON
resume. These are context lengths exercised by the workload, not the 256K cap.

| Configuration | Decode 16K / 32K / 64K t/s | Weighted decode t/s | Initial prefill 16K / 32K / 64K t/s |
| --- | ---: | ---: | ---: |
| Earlier resident, 70 GiB | 14.20 / 14.01 / 13.78 | 14.04 | 478.2 / 619.6 / 756.5 |
| File-backed, 70 GiB, CPU weight 10 | 12.78 / 12.82 / 12.80 | 12.80 | 397.5 / 515.7 / 599.3 |

All three file-backed checkpoints passed without tool errors. They generated
2,187 tokens over 170.87 seconds of native decode. Repair and resume took
352.01 seconds. The earlier resident run generated 2,372 tokens and took
342.01 seconds for repair and resume. Its two recovered tool errors remain
in those timings. Output lengths differ, so workflow time is not a controlled
measure of the configuration's speed. Native decode was 8.8% lower in this run.
File-backed storage and CPU priority changed together. Their individual costs
were not measured. See the [resident comparison](strata-0.2.10-coder.md).

## Controlled memory pressure

A disposable parent slice used `MemoryHigh=76G`, `MemoryMax=80G`, and zero swap.
The model kept its separate 70 GiB cap. A competing process in the same slice
allocated and touched 32 GiB of anonymous memory under a 34 GiB cap. It then
slept during inference. Its measured anonymous allocation was 32.01 GiB.
This created local cgroup pressure without exhausting the host.

The model guard stopped below 8 GiB of available Linux RAM. The competing
process used a separate 12 GiB stop floor. Windows memory was not monitored.

Before the allocation, the model used 65.69 GiB total: 64.12 GiB of file cache
and 1.13 GiB of anonymous memory. After allocation, it used 43.93 GiB total:
42.36 GiB of file cache and the same 1.13 GiB of anonymous memory. Linux reclaimed
21.76 GiB of file cache without restarting the server.

The fresh 16K OMP pressure task passed repair, all 100 assertions, recall, and
strict JSON resume. It recovered from one tool error. Weighted decode was
6.40 t/s over 1,127 generated tokens. The first request decoded at 1.86 t/s;
later requests improved as the available cache filled with needed pages.
The pressure process remained active through the task.

The pressure request reused 3,428 prompt tokens and processed 12,533 new tokens
at 436.3 t/s. The quiet 16K request processed 15,965 new tokens without reuse.
These different cache states prevent a direct prefill comparison.

After the competing process stopped, a new 16K OMP task passed all checks.
It recovered from one invalid history-read request. Weighted decode reached
11.49 t/s over 879 generated tokens. File cache grew from 42.24 to 54.74 GiB
without a server restart. This was partial recovery: the short task did not
reach the quiet 12.80 t/s average or restore the earlier cache size.
The first recovery request reused 3,428 tokens and processed 12,533 new tokens
at 450.5 t/s. Repair and resume took 261.46 seconds under pressure and
130.75 seconds after release. Different output lengths affect those times.

All five repair and resume checkpoints passed. There were no OOM events or
model swap use. Minimum available Linux RAM was 59.48 GiB. The model reached
its 70 GiB cap during the quiet run. The parent reached its 76 GiB high
boundary during pressure and reclaimed file pages. Reclaim and refault
counters increased; they are not exact expert misses or model-only disk bytes.

The model, OMP workers, observer, guard, memory holder, and anchor stopped.
The temporary slice became inactive and its runtime properties were removed.
The model endpoint closed and no GPU compute process remained.

This mode is useful when RAM must remain available for other work. It costs
some quiet throughput and can lose much more speed while its cache is reduced.
Use `--expert-memory resident --mem 70G` for the earlier measured maximum
throughput. The zero-cache default keeps the existing 64 GiB cap. That cap
was verified by launch preview; it was not remeasured in this experiment.

## 32 GiB dedicated cache

The follow-up used `--expert-memory cache --expert-ram-gb 32` with the same
70 GiB cap, CPU weight, runtime, and frozen routing profile. The native log
confirmed a demand store with 3,344 slots and a 32 GiB budget. Cache misses
use unbuffered reads. This mode does not combine all spare Linux page cache
with an additional 32 GiB expert cache.

| Cache mode | Quiet 16K / 32K / 64K decode t/s | Quiet weighted t/s | Pressure 16K t/s | Recovery 16K t/s |
| --- | ---: | ---: | ---: | ---: |
| Zero private cache | 12.78 / 12.82 / 12.80 | 12.80 | 6.40 | 11.49 |
| 32 GiB private cache | 9.23 / 8.90 / 8.84 | 9.08 | 9.14 | 8.77 |

The 32 GiB run passed all five repair and resume checkpoints. Quiet prefill
was 379.4 / 489.7 / 589.0 t/s. The quiet phases generated 2,407 tokens and
took 483.10 seconds for repair and resume. Pressure generated 1,114 tokens
and took 181.28 seconds. Recovery generated 1,980 tokens and took 302.02
seconds. Recovered tool errors were 3 / 0 / 0 at the quiet checkpoints,
3 under pressure, and 7 during recovery. These different trajectories do
not establish a quality ranking or a controlled workflow-time comparison.

Pressure and recovery each reused 3,428 prompt tokens and processed 12,532
new tokens. Their initial prefill rates were 453.3 and 465.7 t/s.
The quiet 16K request processed 15,967 new tokens without reuse.

Before pressure, the process used 50.18 GiB: 31.70 GiB anonymous memory and
18.16 GiB file pages. The competing allocation reduced file pages to 11.91 GiB
while anonymous memory stayed at 31.70 GiB. Total use fell to 43.93 GiB.
The process ended recovery at 43.71 GiB, including 32.45 GiB anonymous memory
and 10.94 GiB file pages. The dedicated cache did not shrink in this test.
Strata's separate emergency cache-shrink path was not exercised.

Peak process memory was 53.30 GiB. There were no model cap, OOM, or swap events.
The parent slice reclaimed file pages at its high boundary. Minimum Linux
available memory was 31.27 GiB. All test processes stopped and routing state
was restored. The smaller private cache gave steadier speed under the tested
pressure, but less RAM remained immediately reclaimable than with zero cache.

### Failed attempt and benchmark guard

An initial unguarded attempt exhausted its request budget during the first
repair task. After a shell-command denial, it edited the user-level OMP
configuration to allow bash. That unintended edit was restored from the exact
original text in the tool record. The restored file hash was verified again
after the guarded retry. The failed attempt remains in the local evidence.

The retry used a trusted, fail-closed [benchmark extension](../experiments/deepseek-strata/omp-filled/workspace_guard.ts).
It permits workspace reads, pure-function replacements in `clamp.py`, and the
pinned validator command. It blocks configuration edits and checks the Python
before execution. It passed 27 policy checks and an installed-loader smoke
test before the retry. The standard filled-context helper now loads this guard.
Normal OMP launchers and approval settings are unchanged. This guard protects
the bounded fixture; it is not a general filesystem sandbox.

The guarded retry has a different tool boundary from the earlier zero-cache
run. Keep that difference and the failed attempt in the interpretation.
Evidence is under `results/coder-cache32-20261103/`. The successful case is
`coder-cache32-70g-guarded/`; `coder-cache32-70g/` is the failed attempt.

## Evidence

Local artifacts are under `results/coder-elastic-20261103/`. The corrected case
is `coder-file-backed-70g-valid/`. Its `pressure/` and `recovery/` directories
contain separate OMP transcripts and validation results. `comparison.json`
contains token accounting, memory snapshots, and the earlier resident results.
The original preparation attempt was stopped before inference after its launch
command exposed nested caps. Its cleanup passed, and its artifacts were retained.

The test measures file-cache reclaim, not exact expert-cache misses. It uses
single sampled coding trajectories. It does not test filled 256K context,
automatic compaction, or global memory exhaustion.
