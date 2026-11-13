# Strata 0.2.11 Coder comparison

The 2026-11-05 matched comparison found no material throughput change.
Weighted decode was **12.499 tokens/s on 0.2.10** and **12.460 on 0.2.11**
(-0.32%). Keep 0.2.11 for its serving fixes. Keep the current Coder settings.
See the [runtime review](strata-0.2.11-review.md) and
[official release](https://github.com/Niko1221/Strata/releases/tag/v0.2.11).

## Setup

Both runs used an RTX 4070 SUPER with 12 GiB VRAM and a Ryzen 9 7950X under WSL2.
The model was ISTA GSQ-RCO Coder IQ2_XXS at revision
`54229d38d588a66d1c6cff08af7b281b9326c656`. Weights did not change.
The old runtime was API 0.2.10/native 0.1.56. The new runtime was
API 0.2.11/native 0.1.57.

Settings were 256K context capacity, a 64 GiB model cgroup cap, zero swap,
zero private expert cache, 12 CPU workers, 4096-token prefill chunks, xhigh
reasoning, temperature 1.0, top-p 0.95, and no drafter. Linux cached file-backed
expert pages. The separate driver and OMP scopes had 2 GiB and 4 GiB caps.
The memory guard stopped below 8 GiB available Linux RAM. It did not monitor Windows.

Each run used the same guarded OMP fixture, public source corpus, sampling,
and frozen routing profile. It grew one conversation through about 16K, 32K,
and 64K input tokens. Each checkpoint required source repair, validator execution,
fact recall, and strict JSON after resume. The 17 workload hashes matched.
Each run used fresh slots and file-specific cache-release advice. This advice
does not reset the host's entire disk cache. No competing memory load was added.

## Measurements

| Approximate input | Decode 0.2.10 | Decode 0.2.11 | Prefill 0.2.10 | Prefill 0.2.11 |
| --- | ---: | ---: | ---: | ---: |
| 16K | 13.10 | 12.78 | 409.9 | 413.2 |
| 32K | 12.60 | 12.63 | 470.3 | 482.6 |
| 64K | 11.31 | 11.84 | 602.5 | 605.8 |

Rates are tokens/s. Decode includes repair and resume requests. Prefill is the
first request at each checkpoint. Later checkpoints reuse the prefix, so their
prefill rate measures the new suffix. The new run processed 15,965 new tokens
at 16K; 14,558 new plus 17,843 reused at 32K; and 31,919 new plus 33,264 reused
at 64K. This is not a filled-256K or automatic-compaction test.

Weighted decode divides generated tokens by total native decode time:
2,239 / 179.135 seconds before, and 2,244 / 180.103 seconds after.
The 25-token startup probe is excluded. Combined repair and resume time was
372.03 versus 371.97 seconds. Startup took 45.26 versus 27.05 seconds.
Total case time was 476.42 versus 461.08 seconds. One run per release does not
establish a startup improvement. Output lengths and tool paths differ.

Both runs passed every repair, recall, validator, and strict resume check.
The old run recovered from three guarded read refusals. The new run recovered
from one guarded read refusal and one guarded shell refusal. All occurred at 16K.
The validator and normal OMP settings stayed unchanged.

Both model scopes reached 64.00 GiB, with no OOM, OOM kill, or model swap.
Memory-limit events were 62,425 versus 61,011; these indicate reclaim at the cap.
Minimum available Linux RAM was 91.90 versus 92.11 GiB.
Peak device VRAM was 11,731 MiB in both runs. During generation, sampled GPU
utilization averaged 91.84% versus 97.24%, and power averaged 86.31 versus 85.54 W.
Higher sampled utilization did not produce higher aggregate decode throughput.

Placement stayed at 353 GPU experts and 7,327 file-backed CPU experts.
Shared root-disk reads were 69.10 versus 56.28 GiB during 426.6 and 430.4 seconds
of observation. These intervals do not cover each entire case. Reads include
Engram and any other disk users. They are not model-only SSD reads or complete
startup totals. This kernel did not expose cgroup `io.stat`. API cache-hit and
RAM/file-blob counters were null. No cache-miss percentage can be established.

## Verification and cleanup

Official source, signed native packages, and pinned dependencies verified.
Coder's eight regenerated pack assets were byte-identical. The previous receipt
was retained. Qwen's unchanged prepared files passed producer, dependency,
identity, and hash checks before their runtime receipt was migrated.
Qwen retains MTP3. Its 0.2.11 inference performance was not measured.
The remote launcher pins were updated and checked; no remote server was deployed.

Every live API and engine thread had `NoNewPrivs=1` and seccomp enabled.
The guard rejected IPv4/IPv6 connects and UDP creation, and survived exec into
a child. Serving replies worked through the benchmark. The outgoing block remains.

Focused upstream parser and stream checks passed 33 methods. A final simulated
60-request handoff test exceeded the shared three-minute limit. It remains
inconclusive; its successors alone require about 101 seconds of intentional delay.
No worker remained. The existing quoted-tool counterexample still reproduced
in 12 of 24 constructed cases. No tools were executed by that reproduction.
The serving fixes do not remove the need for client tool permissions.

The initial control driver lacked Jinja2 and failed before the OMP workload.
The corrected driver used the pinned serving environment for both measured runs.
The failure is retained outside the selected comparison. A Qwen migration helper
also stopped before mutation on an invalid dependency key. Its corrected checks
and applied migration are recorded separately.

All benchmark servers, OMP workers, observers, and guards stopped. Routing,
lane profiles, server configuration, and normal settings were restored.
Port 8899 closed, the active record disappeared, and no GPU compute job remained.
The previous runtime remains available for rollback.

Raw evidence is in `results/strata-0211-20261105/`: `final-summary.json`,
`comparison.json`, `cache-io-comparison.json`, both selected case directories,
the failed initial control, migration proofs, and `preparation-review/`.
These local artifacts are ignored by Git. `localmodel --stats` contains the
measurement summary. No inference measurements were added to `--help`.
