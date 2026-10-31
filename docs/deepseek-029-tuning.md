# DeepSeek tuning on Strata 0.2.9

2026-10-31. The selected default uses 12 CPU workers and retained conversation
checkpoints. A growing 16K, 32K, and 64K OMP repeat passed every check. One earlier
checkpoint failed its required tool-execution check; it remains in this report.
These measurements do not establish a global optimum.

## Conditions and method

The runs used DeepSeek-V4.1-Flash-1017 Unsloth UD-IQ1_S, official Strata API and
packer 0.2.9, and CUDA engine 0.1.55. Hardware was one 12 GiB RTX 4070 SUPER and
an AMD Ryzen 9 7950X. WSL exposed 24 logical CPUs. Each run configured 262,144
context capacity, a 48 GiB expert RAM cache, a 70 GiB cgroup cap, and zero swap.
The memory guard stops a run if available Linux RAM falls below 8 GiB. It did not
monitor Windows memory. The outgoing-connection guard remained enabled.

Each case ran the same constructed OMP conversation, code repair, three-fact
recall, and resume workload. Initial prompts contained 15,963 or 15,964 tokens.
The run used xhigh reasoning, temperature 1.0, and top-p 0.95. It did not fix a
sampling seed. Automatic compaction was disabled. These are approximately 16K
input tests, not filled 256K tests.

Before each run, the driver installed the same 61,440-byte target routing
profile. Its SHA-256 was
`61c25cae339ef229a594d8e8135fabed474d13bd4a41f5b2a1d0c0950aa56c9c`.
The driver requested eviction of clean weight-file pages with
`POSIX_FADV_DONTNEED`. It did not flush the global OS page cache. It restored
the original profile and stopped each server afterward. No case skipped routed
experts.

Except where shown, settings were a 4,096-token prefill chunk, 256 MiB Engram
cache, 16 Engram readers, and no DSpark. DSpark cases used confidence 0.65.
Conversation slots were off in this screen.

## Completed screen

Decode throughput is total generated tokens divided by total native decode
time. Prefill throughput is the first repair request's uncached input divided
by its native prompt time. Workflow time covers repair plus resume, including
OMP and tool overhead. These columns exclude the short warm-up, server startup,
and shutdown. Resource peaks below cover the whole run.

| Case | CPU requested / effective | Change | Prefill t/s | Decode t/s | Generated tokens | Tool errors | Workflow seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline16 | 4 / 4 | Baseline | 234.9 | 5.20 | 918 | 1 | 292.0 |
| dspark2 | 4 / 4 | DSpark, draft limit 2 | 266.8 | 4.96 | 975 | 1 | 312.2 |
| dspark5 | 4 / 4 | DSpark, draft limit 5 | 263.8 | 4.98 | 1,172 | 1 | 373.0 |
| cpu8 | 8 / 8 | More CPU workers | 268.0 | 7.68 | 595 | 1 | 171.0 |
| cpu12 | 12 / 12 | More CPU workers | 264.2 | 8.80 | 4,058 | 4 | 635.3 |
| cpu16 | 16 / 12 | Same effective worker count | 270.5 | 8.51 | 1,924 | 4 | 342.2 |
| chunk2048 | 12 / 12 | 2,048-token prefill chunk | 182.7 | 8.39 | 460 | 0 | 171.4 |
| cpu12-dspark2 | 12 / 12 | DSpark, draft limit 2 | 267.3 | 8.38 | 918 | 1 | 221.5 |
| engram1024 | 12 / 12 | 1,024 MiB Engram cache | 270.1 | 8.18 | 890 | 0 | 221.7 |

The native engine reported 12 CPU workers when the launcher requested 16. Its
pool uses the physical cores visible through the WSL topology. The `cpu16`
result is another 12-worker observation. It is not evidence for 16 workers.

All nine cases passed the final checks. They retained 62 repair/resume model
request attempts, 11,910 generated tokens, 57 tool calls, and 13 tool errors.
Each case passed the code checks, 100 assertions, fact recall, unchanged-test
check, required read/edit/bash use, and strict resume check. Tool errors required
recovery; final success does not mean every tool call succeeded.

The output length varied from 460 to 4,058 tokens. The 12-worker case also
reprocessed a long prefix on resume. Its higher decode rate did not make its
whole workflow faster. These single sampled trajectories cannot isolate changes
in model quality or predict task completion time from decode speed alone.

The 12-worker observations reached 8.51–8.80 decode t/s versus 5.20 with four
workers. DSpark acceptance was 86.7% at limit 2 and 81.5% at limit 5 with four
workers. It was 84.8% at limit 2 with 12 workers. These limits use the same
drafter. These acceptance rates did not
produce a measured decode gain. The smaller prefill chunk reduced initial
prefill throughput. The larger Engram cache also showed no measured decode gain.

Whole-run cgroup peaks ranged from 55.75 to 62.17 GiB. These counts include
charged file cache. Sampled whole-card GPU memory ranged up to 10,416 MiB.
Available Linux RAM stayed above 47.40 GiB. No run recorded an OOM, a memory-cap
event, a guard abort, or a cleanup failure.

DSpark depths 1 and 3, a 1,024-token chunk, and 32 Engram readers were not run.
They remain unmeasured. The adaptive screen tested other settings after the
first DSpark results.

## Resume prefix loss and the isolated probe

The `cpu12` resume used 21,234 input tokens. It reused only 3,428 and processed
17,806 again. Native prompt processing took 52.95 seconds. Nine output tokens
took another 1.224 seconds. The OMP phase took 60.75 seconds in total.

The recorded system message and tools were identical. OMP replaced the earlier
file read at message index 34 with `[Superseded by a newer read of this file]`.
The last repair and resume prompts shared their first 16,214 tokens. Native state can
restore only a retained checkpoint at or before that point. With conversation
slots off, it retained the shared prefix at 3,428 but not the longer conversation
anchor. The pinned source defines
[four checkpoint roles](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_model.cu#L1104)
and [retains the conversation anchor only when slots are enabled](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_main.cpp#L927).

The isolated probe replayed the first repair, last repair, and resume bodies
with 12 workers and 4,096-token chunks. Each request generated one token. This
preserved the recorded input prefixes without substituting newly generated
history. The probe did not execute tools. Each case used an empty slot directory
and the same saved routing profile. Disk warmth was not normalized.

| Slot retention | First prompt seconds | Last repair reused / read | Resume reused / read | Resume prompt seconds |
| --- | ---: | --- | --- | ---: |
| Off | 55.02 | 15,963 / 4,749 | 3,428 / 17,806 | 47.15 |
| On, one RAM slot | 55.12 | 15,963 / 4,749 | 15,962 / 5,272 | 26.19 |

Retention reduced this resume's prompt time by 44.5%. It reused 12,534 additional
tokens. The first uncached prompt took essentially the same time. This supports
a checkpoint-retention benefit for this specific history rewrite. It does not
establish the benefit across all OMP conversations. The one-token probe's decode
rates are not steady-state decode measurements.

Both probes stopped their server and restored the routing profile. Neither
recorded a memory-cap event. The enabled case left about 63 MiB of slot data plus
metadata in its isolated output directory. `SLOT_CAP=1` counts conversations in
RAM. The probe's separate 0.25 GiB disk budget is soft. The native engine keeps
the newest file even when that file exceeds the budget.

## Growing conversation at 16K, 32K, and 64K

The control used four workers and disabled conversation slots. The candidate
and its repeat used twelve workers and retained slots. Each chain used one OMP
conversation with three successive repairs. It preserved earlier fixes and used
distinct deterministic recall facts at each checkpoint. The checks executed
100, 248, and 251 assertions at the successive checkpoints. These counts include
repeated checks of earlier fixes. Automatic compaction remained disabled.

New and reused tokens below describe the first repair request at each checkpoint.
Prefill rates count only new tokens. Workflow time includes repair and resume.

| Run | Target | New / reused input | Prefill t/s | Decode t/s | Generated | Workflow s | Result / tool errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Control | 16K | 15,962 / 0 | 268.8 | 5.21 | 620 | 221.1 | Pass / 0 |
| Control | 32K | 15,468 / 16,857 | 330.2 | 5.27 | 437 | 171.1 | Pass / 0 |
| Control | 64K | 32,236 / 33,040 | 397.3 | 5.24 | 454 | 201.1 | Pass / 0 |
| Candidate | 16K | 15,961 / 0 | 271.8 | 8.14 | 1,309 | 282.1 | Fail / 2 |
| Candidate | 32K | 14,281 / 18,148 | 316.4 | 8.19 | 636 | 151.1 | Pass / 0 |
| Candidate | 64K | 31,872 / 33,343 | 399.6 | 8.13 | 690 | 191.4 | Pass / 0 |
| Repeat | 16K | 15,962 / 0 | 271.5 | 8.07 | 541 | 151.1 | Pass / 0 |
| Repeat | 32K | 15,550 / 16,778 | 357.5 | 8.28 | 409 | 121.0 | Pass / 0 |
| Repeat | 64K | 32,263 / 33,015 | 416.2 | 7.82 | 441 | 161.5 | Pass / 0 |

The first candidate failed the required bash-execution check at 16K. OMP
rejected `ls -la; echo ---; python3 --version` and a `cd ... && python3 verify.py ...`
command. The benchmark allows only the literal verification commands. The
response then reported that it could not execute the tests. Independent code
checks, all 100 assertions, recall, and strict resume passed. The 32K and 64K
checkpoints passed every check. The repeat used the same settings and approval
policy, with an empty cache. All three checkpoints passed with no tool errors.
The first failure remains in the results.

| Whole chain | Native decode t/s | Generated tokens | Repair + resume s | Total run s | Peak cgroup GiB | Minimum available Linux GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 5.24 | 1,511 | 593.3 | 671.9 | 61.48 | 48.33 |
| Candidate | 8.15 | 2,635 | 624.6 | 700.0 | 63.86 | 47.04 |
| Repeat | 8.05 | 1,391 | 433.6 | 520.0 | 63.97 | 48.18 |

The confirmed candidate produced about 8 tokens/s across the growing workload,
compared with 5.24 for the control. Its repeat completed the repair/resume work
in 433.6 seconds versus 593.3 seconds, but it also generated 8% fewer tokens. The
first candidate generated 74% more tokens and took 624.6 seconds. These sampled
paths show why a decode gain does not guarantee the same gain in task time.
No chain recorded an OOM, swap use, or a memory-cap event. Resource peaks cover
every checkpoint in the chain. These runs filled about 64K input. They did not
fill 256K or test automatic compaction.

## Restart and launcher checks

The restart check ran the installed `localmodel start deepseek` command with
no tuning flags. It loaded a saved 66,165-token conversation from disk and read
one new prompt token. Native prompt processing took 0.5403 seconds. The strict
JSON answer was `{"probe_value":6}`. The full check took 24.5 seconds, including
about 21 seconds of model startup. This was a correctness and cache-reuse check,
not a steady-state decode benchmark.

Strict Pyright checks passed for both changed launcher modules. The installed
launcher passed 21 focused checks for defaults, overrides, backend rejection,
cache identity, dry-run behavior, and symlink rejection. The real launch changed
existing 0755 cache directories to 0700. The RAM cap, zero-swap setting, outgoing
connection guard, and offline mode remained enabled.

Every test server stopped. Both original routing profiles match their saved
bytes. Benchmark-created cache directories were moved into the result folders.
The normal cache namespace is empty for the next launch. No weights were removed.

## Selected defaults

`localmodel start deepseek` now uses 12 CPU workers and retained conversation
checkpoints. It keeps 256K capacity, the 48 GiB expert cache, 70 GiB process cap,
4096-token prefill chunks, 256 MiB Engram cache, and all six routed experts.
DSpark remains optional because the measured draft settings did not improve
throughput. A smaller chunk reduced prefill speed. A larger Engram cache did
not show a gain. These are the best settings supported by this experiment.
This was an adaptive search, not an exhaustive search.

Use `--cpu-threads 8` to leave more CPU capacity for other work. Use
`--no-strata-slots` to disable retained checkpoints. Slots use one saved
conversation in RAM and a 2 GiB disk budget per weights/runtime/context
configuration. The newest slot can exceed that disk budget. Saved state includes
prompt tokens. The launcher separates incompatible configurations with pinned
hashes and uses private directories under `state/deepseek-slots/`.

## Evidence

Raw results remain in the local, untracked `results/deepseek-029-tuning-20261031/` directory.

- [Completed-case collection](../results/deepseek-029-tuning-20261031/comparison.json) and each case's `case-summary.json`,
  `summary.json`, request logs, and `engine.log`.
- [Prefix-off result](../results/deepseek-029-tuning-20261031/prefix-off/probe-summary.json).
- [Prefix-on result](../results/deepseek-029-tuning-20261031/prefix-on/probe-summary.json).
- [Probe procedure and slot identity limits](../results/deepseek-029-tuning-20261031/prefix-cache-probe.md).
- [Installed-default restart check](../results/deepseek-029-tuning-20261031/restart-defaults/restart-summary.json).
- [Launcher checks](../results/deepseek-029-tuning-20261031/launcher-checks.json).
- Original pruning inputs: `cpu12/009-16384-repair.network.jsonl` and
  `cpu12/010-16384-resume.network.jsonl`.
