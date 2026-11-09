# DSML parser and Linux arena checks

The checks use isolated copies of Strata v0.2.12 at
`146df3c468739147f520ac3816de806f71232571`. The installed source and runtime
remain unchanged. Both upstream pull requests were open and unmerged when
checked on 2026-11-09. Their exact heads and patch hashes are in `pr-5598.json`
and `pr-5531.json`.

## DSML argument types: PR 5598

The patch resolves an accepted tool alias before it reads the parameter schema.
This preserves a `string="true"` marker when the generated name differs from
the declared name. No new tool aliases are added.

The fixed-fixture probe has 156 cases. It varies six name forms, four values,
both string markers, and three stream chunk sizes. It also checks ambiguous
aliases, unknown names, an unrelated namespace, and declared-name priority.
The baseline passes 96 cases and fails 60. The candidate passes all 156.
Each failure that the patch fixes is an alias with a true string marker.

The upstream parser and API suites pass with both versions. The baseline runs
130 tests and skips one. The candidate runs 131 tests and skips one. Neither
suite has a failure. The API tests use a scripted engine; they do not load a
model. The candidate includes the upstream regression test.

One separate diagnostic remains unresolved. In a single parser instance,
`string="true"` updates the stored schema. A later call to the same tool and
parameter with `string="false"` can therefore remain a string. The exact-name
case has this behavior in both versions. The patch gives aliases the same
behavior. This is limited to the parser instance; the probe does not establish
leakage between requests. The patch is not a complete correction for all DSML
type handling.

The runtime change adds one alias lookup. Review found no added file access,
networking, subprocess execution, or tool execution. It uses the existing alias
map and preserves declared-tool checks. This bounded review does not prove
the complete parser safe against every malformed output.

## Linux arena admission: PR 5531

The baseline header-only host test passes 906 checks. The candidate passes
918, including the new free-page, clean-file-cache, and reserved-hugepage
predicates. Existing hard-limit refusal and file-source fallback checks pass
in both builds. Tests use numerical inputs; no host hugepage pool changes.

The changed branch does not run in the current Qwen profile. The launcher
passes `--resident-budget-gib 36`. Its parser sets `o.mmap_experts=true`, so
the source selection bypasses `ArenaExpertSource::open`. The `lowram` profile
also selects the mapped source. Separately, the new credit requires
`hm.cgroup_limit == ~uint64_t{0}`. The finite 64 GiB production cap excludes
that branch. DeepSeek uses a different expert store.

No GPU timing comparison was run. The selected production path cannot use
the changed branch, so this experiment makes no speed claim. A full-arena
profile without a cgroup limit would be a different configuration. It was
not tested.

The patch adds a local `/proc/meminfo` read and diagnostic output. It does not
add network access or subprocess execution. The early hard-limit check stays
in place. The new helper has incomplete overflow validation for fabricated
memory-counter text; the live input is the kernel's `/proc/meminfo`, not model
output or a request parameter. No patch was installed in production.

## Resource use and reproduction

All checks ran in one `heavy-management/bin/capped` scope. Effective
`memory.max` and `memory.high` were 4 GiB. Swap was disabled. The recorded
peak was 87 MiB, rounded down by the existing recorder. There were no memory
limit or OOM events. The scope stopped. No GPU workload or model server ran.

Raw evidence is under `results/strata-candidates-20261109/parser-arena/`.
`run-cpu.sh` records the exact test commands. `parser_probe.py` records the
additional fixed fixtures. The source copies are under
`state/sources/strata-parser-arena-20261109/{baseline,candidate}`.
`summary.json` records results, hashes, and resource limits. These local
`results/` and `state/` paths are not part of the published repository.

## Quality-scorer lifecycle smoke

A separate CPU-only smoke checks the parent experiment's `quality/score.py`.
It substitutes a small executable that forks one sleeping child. It mocks
the weight metadata and GPU admission query. It retains the real network
guard, heavy-management launcher, and 64 GiB model scope with zero swap.
The outer test driver has a 4 GiB cap.

The initial attempt exposed missing optional `io.stat` telemetry. Both
process trees stopped, but the requested low-memory and signal checks had
not run. That failed evidence remains in `score-guard-smoke.json`.

After the scorer correction, three checks pass. An injected memory reading
below 8 GiB stops both child processes. SIGTERM to the controller does the
same. A simulated timeout from `systemctl stop` exercises fallback cleanup:
both processes stop, and the summary retains `cleanup_failed` and the
injected error. Every scope is empty before the independent test cleanup.
Every run verifies seccomp, no-new-privileges, the memory cap, and zero swap.
The smoke does not create real memory pressure or run GPU code.

The tested scorer SHA256 is
`9b8f071cc7b0c6d4e09799cef54f10abe48eaf94c59f185c92d694a9f2d65332`.
The completed evidence is `score-guard-smoke-r2.json` in the raw result
directory. Later edits to the scorer need their own relevant checks.

Primary sources:

- <https://github.com/Niko1221/Strata/pull/5598>
- <https://github.com/Niko1221/Strata/pull/5531>
