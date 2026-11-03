# Strata 0.2.10: local Coder update

Updated on 2026-11-03. Local DeepSeek now uses the official 0.2.10 API and
packer at commit `ffa80d73133ea5c548aefe57889a16e8085a938f`, Control 0.1.56,
and the signed Linux CUDA 13 native engine 0.1.56. The previous release was
0.2.9 at `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`, native 0.1.55.
Archive, feed, signature, manifest, and binary hashes are pinned in the
[runtime manifest](../experiments/deepseek-strata/manifest.json).

This comparison measures the local Coder IQ2_XXS profile. Qwen remains on
0.2.9 / 0.1.55. The remote script is unchanged. Earlier measurements retain
their original versions and settings.

## Compatibility and review

All five installed Coder transformer hashes match the release's official
Coder pins. No weights changed. The retained Engram still uses the previously
verified alternative header with the same tensor payload. The new packer
produced eight byte-identical pack, index, and tokenizer assets.

Resident expert copies still use the available-memory budget with an 8 GiB
reserve. The server now keeps up to four restore points by default. These
can reduce repeated prefill after a conversation changes. They also retain
more checkpoint state. Single-GPU prompt processing can borrow VRAM expert
storage for buffers and restore the experts afterward.

The new PCIe decode streaming path remains **off by default** in Strata.
We do not set `DS4_STREAM=1`. Upstream still requires identity, quality, and
performance checks for that path. Stock resident serving therefore does not
need a new `lane-times.txt` exception in the pack validator. The explicit
twelve-worker setting also keeps automatic worker tuning off.

The offline, signing, image, MCP, and DSML helpers are unchanged. No new
outgoing operation was found in the reviewed serving and native changes.
The existing seccomp guard remains active for the server and its children.
The live 0.2.10 API and native engine both reported `Seccomp: 2` and
`NoNewPrivs: 1`.
This bounded source review is not a complete native-binary audit.

Twenty-five upstream parser, restore-point, and context-limit tests passed
with fake engines. The existing 448-case parser matrix matches 0.2.9: all
128 genuine-call controls pass; 84 of 320 constructed quoted or undeclared
cases still emit calls. The quoted `</think>` limitation remains. These
cases do not estimate its frequency in real work. Clients must enforce
their allowed tools. See the [previous parser review](strata-0.2.9-review.md#parser-results).

## Comparison method

The control and update run separately on the same hardware. A second update
run tests a 70 GiB cap after the matched 64 GiB comparison. All runs use:

- A 64 GiB process cap, except for the labelled 70 GiB follow-up, zero swap,
  and automatic resident expert copies.
  The demand cache is zero. No forced copy budget is set. Some experts can
  remain file-backed when they do not fit. Engram remains on SSD.
- 262144-token capacity, twelve CPU workers, 4096-token prefill chunks,
  `xhigh` reasoning, six active experts, and no drafter.
- The same growing OMP workload at approximately 16K, 32K, and 64K tokens.
  It checks code repair, assertions, retained facts, and strict JSON resume.
- The same frozen routing profile, SHA-256
  `317875a8a047719357eb8bc05670e0a7948a0d1626df3834cb14f7474c9d9b03`.
  Each run starts with empty conversation slots. This is a comparison with
  a trained routing profile, not a cold-routing test.

The driver restores the original routing profile and any lane calibration
after the server stops. It records actual prompt tokens, reused tokens,
generated tokens, memory peaks, expert residency, and tool-check outcomes.
Weighted decode speed is total generated tokens divided by total native
decode time. Different output lengths can change total workflow time.
The 256K setting is capacity; these runs do not fill it.

## Results

| Strata | Cap GiB | Prefill 16K / 32K / 64K t/s | Weighted decode t/s | Generated tokens | Repair + resume seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.2.9 | 64 | 498.5 / 639.6 / 768.3 | 13.25 | 2,080 | 331.85 |
| 0.2.10 | 64 | 482.7 / 618.3 / 756.5 | 13.80 | 2,902 | 391.89 |
| 0.2.10 | 70 | 478.2 / 619.6 / 756.5 | 14.04 | 2,372 | 342.01 |

The new 64 GiB run decoded 4.2% faster than the old 64 GiB control.
Its initial prefill rates were 3.2%, 3.3%, and 1.5% lower at the three checkpoints.
The 70 GiB follow-up decoded 1.7% faster than the new 64 GiB run. Its prefill
rates were effectively unchanged. Prefill counts only uncached tokens in the
first repair request at each checkpoint. The later requests reuse earlier context.

All nine checkpoint repair, recall, unchanged-test, required-tool, and strict
JSON resume checks passed. The 64 GiB runs each recovered from one bash approval
denial. The 70 GiB run recovered from a bash denial and an initially failed
clamp assertion, then corrected the code. These attempts remain in the timings.
The checkpoints checked 100, 248, and 251 assertions; later counts include earlier ones.

Output lengths varied substantially. The new 64 GiB run generated 39.5% more
tokens than the control, so its total workflow took longer despite faster decode.
The shorter 70 GiB workflow also generated fewer tokens. These single sampled
trajectories do not establish faster completed coding tasks or a general quality
ranking. Workflow times exclude startup, shutdown, and the short warmup request.

| Strata / cap | GPU experts | RAM copies / CPU experts | File-backed experts | Startup seconds | Minimum Linux available GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.2.9 / 64 GiB | 242 | 6,902 / 7,438 | 536 | 32.04 | 40.88 |
| 0.2.10 / 64 GiB | 353 | 6,900 / 7,327 | 427 | 33.05 | 42.48 |
| 0.2.10 / 70 GiB | 353 | 7,327 / 7,327 | 0 | 41.94 | 39.36 |

The new release placed more experts in VRAM under the same launcher settings.
At 64 GiB, 427 experts remained file-backed, approximately 3.7 GB of payload
(about 3.5 GiB). A 68 GiB cap is an estimate for a possible intermediate setting;
it was not tested. At 70 GiB, all 7,327 CPU experts and 353 GPU experts were
resident. The separate 103 GiB Engram table remains SSD-backed.

All runs reached their process caps and recorded file-cache reclaim pressure.
None recorded an OOM or model swap use. Sampled GPU memory peaked near 11.25 GiB
in all three runs. Mean whole-card GPU utilization was 71% to 73%, or 93% to 95%
during generation samples. Mean power was 93 W to 95 W, with sampled peaks near
210 W. These are whole-device samples, not process-only hardware counters.

Exact expert-cache misses and per-process disk bytes are unavailable on this
WSL kernel. Shared block-device reads include Engram and other processes.
Full expert residency does not imply zero SSD traffic.

The later [memory-sharing comparison](coder-elastic-memory.md) changed the
default to a 64 GiB cap with zero private expert cache. Use
`localmodel start deepseek --expert-memory resident --mem 70G` for the
full-residency option measured here.
All benchmark servers, workers, guards, and observers stopped after the runs.


## Evidence and rollback

Local evidence is under `results/strata-0210-coder-20261103/`:

- `coder-029-control/`, `coder-0210-update/`, and `coder-0210-70g/` contain
  the run artifacts.
- `review/` contains the source review and parser checks.
- `promotion.json` records the identical pack assets and protected files.
- `upgrade-backup/` retains the previous DeepSeek and Coder manifests,
  Coder pack, full-model pack, and DSpark pack, including their receipts.

The old source and native runtime remain in
`state/sources/strata-official-0.2.9/` and
`state/runtimes/strata-official-0.1.55/`. A rollback must first stop the
server, then restore the saved manifests and matching packs together.
Do not mix receipts from different packer versions. Run the runtime and
pack verification before restarting. The evidence directories are local
artifacts and are not included in the published repository.

Upstream: [release notes](https://github.com/Niko1221/Strata/releases/tag/v0.2.10),
[native source](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/src/dsv41/ds4_model.cu),
[Coder pins](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/data/dsv41/coder-set.json).
