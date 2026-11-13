# DSpine with Bonsai 2 27B

Keep the trained MTP2 preset as the default. DSpine works with Bonsai and is
faster on the short code fixture. The long OMP measurements do not establish
a useful overall improvement. All seven completed OMP cases passed the
workflow checks. No production preset changed.

## Configuration

The machine has an RTX 4070 SUPER with 12 GiB VRAM. The target is the same
Bonsai 2 27B PTQ1 file with the ProCreations MTP head in every case. DFlash2
uses the existing Bonsai-trained Q4_K_M drafter. DSpine uses AJ's Q4_K_M
conversion of Tencent's dense Qwen3.8-27B drafter.

The experimental runtime adds five C++ source changes to the reviewed
`bonsai-next` tree. It restores selected output rows from Bonsai's Hadamard
basis and preserves DSpine's causal attention metadata. It retains the
existing greedy draft selector. The optional newer probabilistic selector
is not included. Both controls and candidates use the same existing GGML
CUDA kernels and CUDA 13 libraries.

OMP 18.8.6 runs with `xhigh` reasoning, temperature 1, and a fixed seed.
Each case has 256K context capacity, Q4 target KV, recurrent replay, a
256-token batch, a 128-token microbatch, and 8192-token checkpoints.
The server cap is 40 GiB. OMP has a separate 4 GiB cap. Swap is disabled.
The prompt snapshot cache may use 32 GiB.

MTP retains its normal full-GPU KV allocation. DFlash2 and DSpine keep
128K KV positions in VRAM and allocate the remaining capacity in RAM.
The target transformer layers stay on the GPU. The largest measured API
input was 68,436 tokens. These runs do not exercise attention over the RAM
KV tail or establish performance at filled 192K or 256K context.

## OMP results

The workflow repairs three functions through `read`, `edit`, and `bash`.
It also checks an edited old message, resume, cancellation recovery,
compaction, retained facts, and repairs after compaction. History contains
public source code. It does not contain private conversations.

Rates below are total measured tokens divided by total native inference
seconds. Prefill counts only newly evaluated tokens. Elapsed time includes
server startup, fixture preparation, and the workflow. Each row is one run.

| Initial input | Draft setting | Prefill t/s | Decode t/s | Generated tokens | Elapsed s |
| --- | --- | ---: | ---: | ---: | ---: |
| 16K | MTP2 control | 1,146.9 | 94.04 | 4,818 | 115.20 |
| 16K | DFlash2, 7 drafts | 1,120.6 | 86.48 | 7,207 | 139.56 |
| 16K | DSpine, 15 drafts, Q4 draft KV | 1,051.6 | 85.33 | 4,669 | 125.73 |
| 16K | DSpine, 7 drafts, Q4 draft KV | 1,106.3 | 81.05 | 12,086 | 207.56 |
| 16K | DSpine, 15 drafts, Q8 draft KV | 1,033.8 | 83.59 | 4,890 | 116.04 |
| 64K | MTP2 control | 969.6 | 89.35 | 7,104 | 205.41 |
| 64K | DSpine, 15 drafts, Q4 draft KV | 927.1 | 91.08 | 5,036 | 191.79 |

At 64K, DSpine's decode rate is about 2% higher and its prefill rate is
about 4% lower. It generates 29% fewer tokens. Its shorter elapsed time
therefore does not establish a speedup for a fixed amount of work.
Reasoning length and phase mix also differ in the 16K cases.

At 64K, sampled total GPU use peaks at 11.26 GiB for MTP2 and 10.28 GiB
for DSpine. These values include desktop GPU allocations. Peak server
cgroup memory is 6.13 GiB and 9.57 GiB respectively. DSpine's lower VRAM
use reflects its different KV placement. It is not a direct memory saving
from the drafting method. Linux available RAM stays above 94 GiB and
Windows free RAM above 66 GiB across the completed OMP cases. No OOM,
memory-limit, or swap event occurred.

## Short screen

The native screen uses 32K capacity, short prompts, temperature zero, and
thinking disabled. The code prompt has 127 input tokens and produces the
same 127-token answer in all six completed controls. The code validator
passes in every case.

| Setting | Code decode t/s |
| --- | ---: |
| No speculation | 59.8 |
| MTP2 | 117.1 |
| DFlash2, 7 drafts | 140.5 |
| DSpine, 7 drafts | 150.8 |
| DSpine, 15 drafts | 184.0 |

DSpine reduces the complete code-request time from 1.365 seconds with
MTP2 to 1.156 seconds, about 15%. The larger decode-rate increase does not
apply to the full request because prompt processing also takes time.
The second MTP control uses the ported runtime and gives 116.4 tokens/s.

The prose JSON check passes in every case. The arithmetic check fails with
and without speculation: the correct remaining count is 507, while the
runs return either 639 or 663. Code and prose are byte-identical across
the six controls. Arithmetic is not. These screens do not prove general
output equivalence or model accuracy.

## Verification and limits

Static review found no added networking, telemetry, file I/O, or subprocess
execution in the inference port. All inference runs use the existing
outgoing-network seccomp guard. The runtime still verifies draft tokens
against target logits. The short DSpine code run accepts 113 of 195 draft
proposals over 13 verification steps; it does not bypass verification.

The GGUF matches its published size and SHA256. Fourteen unquantized bridge
tensors match exact ranges from the pinned official Tencent checkpoint.
The bias requires exact BF16-to-F32 expansion. The quantized codebook and
other quantized tensors were not independently reproduced. This is a
bounded source and provenance check, not a full security proof.

Strict Pyright 1.1.414 passes for the experiment runner, provenance helper,
and build helper. Both source trees match their recorded hashes. The
original source and binary remain unchanged.

The first 16K OMP attempt could not compact because the reused fixture kept
20K recent tokens. The corrected private fixture keeps 4K for every 16K
comparison. The 64K fixture keeps 20K. One initial ported-control startup
failed because the experiment's port check did not allow TCP TIME_WAIT;
the corrected check retains the occupied-port guard. Neither failed attempt
is included in the tables. Raw evidence remains available.

All owned model, OMP, and search processes have stopped. Ports 8888 and
8899 are closed. No default, scheduled task, or remote deployment changed.

## Artifacts

- [Experiment setup and commands](../experiments/bonsai-dspine/README.md)
- [Source audit](../experiments/bonsai-dspine/source-audit.md)
- [Measurements and denominators](../experiments/bonsai-dspine/measurements.json)
- [Partial tensor provenance](../experiments/bonsai-dspine/bridge-provenance.json)
- Raw local evidence: `results/bonsai-dspine-20261108/`
