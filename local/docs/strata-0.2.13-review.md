# Strata 0.2.13 review and measurements

Date: 2026-11-09. Accepted with the existing tuning defaults. The required
compatibility, repair, recall, resume, and resource checks passed. The supplemental
JSON-only formatting limit is recorded below. The previous runtime remains
available for rollback.

## Release and verification

The [official stable release](https://github.com/Niko1221/Strata/releases/tag/v0.2.13)
updates the source API and packer to 0.2.13. Control and both CUDA engines are
0.1.59. The source revision is `d2f096475817d4666c364c820b0ca830f751cfc5`.
The previous source revision is `146df3c468739147f520ac3816de806f71232571`
from 0.2.12, with native engines 0.1.58.

The existing trusted verifier accepted signed feed 123 and both native packages.
The release checksums, package signatures, manifests, packaged files, and binary
hashes agree. The source archive has a pinned hash and a receipt for each file.
The annotated Git tag is unsigned. The native packages have signed build IDs
that start with `712fec6b4297`. This differs from the later release source tag.
These checks do not establish a reproducible build or a complete binary audit.

The four pinned Python dependencies are unchanged. The existing isolated
environment remains in use. The nine packer, reader, MTP, TLS, and vocabulary
inputs checked during preparation are byte-identical to 0.2.12.
No weights or prepared indexes need conversion. Receipt updates retain the old
source revision, verification basis, and migration record. Large weight files
use their pinned hashes and unchanged recorded file identities. This update
does not repeat a full scan of all weight bytes.

The optional Qwen DFlash2 URL moves to the new release. Its bytes, size, and hash
are unchanged. The selected draft settings remain unchanged.

## Relevant source changes

- The native Qwen engine now sets the Linux CUDA GDN and parallel defaults that
  the API previously set. Explicit overrides still take precedence.
- Automatic Q8 hyper-connection reads move into the native engine. The local
  Qwen launcher already enables this option explicitly.
- The Linux arena fallback excludes finite cgroup limits. It does not justify
  removing the memory cap or increasing the expert-cache budget.
- Template detection adds JSON tool calls. The existing Qwen template still
  uses XML calls. DeepSeek still uses its DSML parser. Seven upstream parser
  selection checks passed with mock objects.
- The tokenizer loader rejects unknown pre-tokenizers. The installed tokenizers
  use the supported `qwen35` and `deepseek3` values.

The control layer adds signed model catalog and shape-engine downloads to setup
and update commands. The current feed contains no entries for either feature.
Serving uses fixed local metadata and an explicit executable. The reviewed
serving, telemetry, and parser changes add no outbound requests.
The existing offline policy and inherited seccomp filter remain in place.

## Matched DeepSeek procedure

The comparison uses full Unsloth UD-IQ1_S, all six active experts, and the same
weights. It does not use the pruned Coder variant. Each release gets separate
16K, 32K, and 64K filled-input cases. Context capacity remains 196,608 tokens.
The actual initial input count comes from native API timing records.

Settings are a 48 GiB expert cache, 64 GiB cgroup limit, zero swap, 12 CPU threads,
4,096-token prefill chunks, 700 MiB VRAM reserve, and 16 Engram readers.
DSpark is disabled. Borrowing uses its normal default. Experimental cache-aware
routing is disabled. Sampling uses temperature 1, top-p 0.95, and seed 20261109.
Each case starts from the same frozen expert-routing profile. This is a replay
of an established routing profile, not a cold routing measurement.

The OMP fixture uses pinned public source history, a code repair, 100 assertions,
recall checks, and a resume probe. It uses replacement edits in an isolated
workspace. It is a bounded coding workflow, not a general coding-quality score.
The workspace guard can refuse generated shell commands. Record these refusals
separately from failed assertions and API or tool-parser failures.
Final repair JSON formatting is checked separately from the existing semantic
checks. Extra prose can pass the semantic check and fail the formatting check.

Decode throughput divides generated tokens by total native decode time across
repair and resume requests. Initial prefill excludes cached follow-up turns.
Workflow time includes both phases and excludes startup and the short warmup.
Output lengths can differ, so task time and generated tokens must be read with
throughput. One pass per case cannot establish a small statistical improvement.
GPU memory includes the desktop. Shared root disk reads include other processes
and Engram reads. They are not expert-cache miss counts. Missing counters are
not treated as zero.

The jobs run serially through `heavy-management`. The Linux memory floor is
8 GiB. The guard stops only owned workers and servers. It also stops on swap use
or repeated memory-read failure. CPU-only checks verified these stop paths before
inference. Workload and runtime hashes are frozen separately for each release.

## Measurements

Each cell lists 0.2.12 first, then 0.2.13. Both runs have the same initial native
input count. Throughput is in tokens per second. Time is in seconds.

| Target | Actual initial input | Initial prefill | Decode | Generated tokens | OMP workflow time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16K | 15,961 | 261.11 → 256.33 | 8.97 → 9.17 | 679 → 1,382 | 171.61 → 251.61 |
| 32K | 32,331 | 347.54 → 346.42 | 8.72 → 9.12 | 618 → 1,710 | 201.36 → 342.52 |
| 64K | 65,306 | 417.30 → 417.96 | 8.56 → 8.81 | 611 → 513 | 261.88 → 241.51 |

The observed decode changes are +2.3%, +4.6%, and +3.0%. Initial prefill changes
are -1.8%, -0.3%, and +0.2%. These single passes do not establish a general speed
gain. The 16K and 32K tasks take longer because they generate much more output.
The 64K task generates less output and finishes sooner. Keep the existing tuning
defaults. This release does not establish a faster complete coding workflow.

| Target | Peak cgroup RAM, GiB | Observed shared-root reads, GiB | Guard refusals |
| --- | ---: | ---: | ---: |
| 16K | 60.09 → 61.54 | 287.98 → 369.71 | 1 → 1 |
| 32K | 60.83 → 64.00 | 275.04 → 525.47 | 1 → 3 |
| 64K | 63.42 → 63.43 | 273.28 → 256.60 | 0 → 1 |

Whole-device peak VRAM stays between 11.776 and 11.791 GiB. Linux available RAM
stays above 54.1 GiB. All runs have zero swap and zero OOM events. The new 32K
case reaches the 64 GiB cgroup limit and records 3,693 `memory.events.max` events.
These are memory-limit pressure events, not OOM kills. Other cases record zero.
Cgroup RAM includes charged file-cache pages; it is not just process RSS.

Model-cgroup disk I/O and exact expert-cache hit/miss counters are unavailable.
The shared-root totals cover the observer interval, including warmup. They can
include other processes. Different generated lengths and tool paths also change
the work done. Do not interpret these totals as a fixed-work I/O regression or
as expert-cache misses.

All six runs pass code repair, the same 100 assertions, recall, and the strict
resume probe. Validators remain unchanged. Every recorded tool error is a
workspace-guard refusal. None is an assertion failure or a tool-parser error.

Supplemental JSON-only repair compliance passes in all three old cases and two
of three new cases. The new 16K response contains the correct JSON in a Markdown
fence, followed by prose. Its raw SSE content matches OMP's displayed content.
It finishes normally, with no leaked reasoning or DSML delimiters. The DSML
parser and chat template are byte-identical between releases. This supplemental
check does not change the pre-existing semantic or resume acceptance checks.
The formatting failure remains part of the result.

The initial requests match after normalization of the workspace path. Their
path tokens differ between release directories. At temperature 1, a fixed seed
does not make these byte-different requests produce identical generations.
Later tool choices and histories also differ. This comparison cannot attribute
the formatting difference to the engine, or establish broader coding quality.

The new 32K resume changes one earlier file-read message to
`[Superseded by a newer read of this file]`. Strata reuses 32,330 tokens and
processes 2,625 new tokens. The old resume reuses 33,188 and processes 29 new
tokens. Within-repair prefix reuse remains normal. The additional resume work
comes from OMP history replacement, not an observed cache-lookup failure.

All benchmark servers, workers, observers, guards, and owned SearXNG processes
stop after their cases. Routing profiles, lane profiles, runtime state, and
normal OMP settings are restored. The frozen matrix code remains unchanged.

## Qwen compatibility check

The first smoke attempt started the correct default but used an incorrect
assertion. The public launcher reports four draft tokens. The native engine
includes the accepted token in its five-token window. The attempt stopped before
generation and cleaned up completely. Its evidence is retained. The disposable
assertion was corrected without changing any production setting.

The corrected smoke passes with the normal public `localmodel start qwen3.8`
command. It verifies 192K capacity, a 36 GiB resident expert budget, a 64 GiB cap,
four MTP draft tokens, 64K GPU KV cells, int8 KV, and the 1,024 MiB reserve.
Native startup confirms that MTP loaded. The executable hash matches the signed
0.1.59 package and the native process retains seccomp mode 2.

A plain response and one forced `record_number(7)` tool call pass. No tool is
executed by this probe. OMP discovery reports 196,608 context tokens, 16,000 output
tokens, and a 157,286-token compaction threshold. This is a short compatibility
check, not a Qwen throughput or filled-context benchmark.

Startup takes 92.15 seconds and the whole check takes 103.18 seconds. Peak cgroup
memory reaches 64 GiB and records 68,983 memory-limit pressure events. There is
no swap or OOM event. The owned controller and scope stop. Port 8899 closes and
runtime state and normal settings are restored.

## Remote launcher

The prepared launcher changes four pin and version locations. Hardware
selection, weights, DSpark choices, limits, dependencies, and the B300 path remain
unchanged. Nine CPU-only hardware previews preserve their plans.
Cached verification accepts the new signed artifacts. Missing cache, modified
feed, incorrect archive pin, and incorrect manifest pin fail without downloads.

Socket probes confirm that IPv4, IPv6, UDP, and inherited child-process outbound
connections fail. Accepted inbound API connections can still send replies.
No remote deployment or remote GPU benchmark is part of this update.

## Evidence

The frozen driver is in `experiments/strata-0213-20261109/benchmark/`.
Local raw records are in `results/strata-0213-20261109/`.
The `preparation/` subdirectory contains release metadata, signature checks,
source diffs, dependency evidence, receipt backups, and remote validation.
Each release has its own runtime pins, workload pins, case records, and resource
samples. The old runtime remains available for rollback.
