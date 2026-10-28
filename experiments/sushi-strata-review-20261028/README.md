# Sushi techniques on Strata and CUDA

Research date: 2026-10-28. No new inference measurements were taken.

Sushi's STQ decoding can be implemented on CUDA. Its published pack format
does not load in the installed Strata engines. A complete port requires
format ingestion, CPU and CUDA kernels, variable-rate expert storage, and
validation. The public encoder does not reproduce the tuned STQ packs.
The [quantization review](quantization.md) records those boundaries.

The first useful work is smaller: test memory allocation and DeepSeek
prefill chunk sizes, then measure prefix reuse across tool turns. Strata
already implements many of the techniques used by Sushi. The
[runtime review](runtime-techniques.md) identifies the existing mechanisms,
the missing parts, and the checks required before changing defaults.

## Current configuration

The host is Ubuntu on WSL2 with an RTX 4070 SUPER, 12282 MiB VRAM, and about
110 GiB of Linux RAM. No managed model was running during inspection.
The model endpoints on ports 8897, 8898, and 8899 did not respond.

| Command | Installed weights | Context capacity | Memory and decoding |
| --- | --- | --- | --- |
| `localmodel start` | Bonsai 2 27B PTQ1_0 with ProCreations MTP | 256K | Existing optimized preset; MTP2, 16K draft window, Q4 KV, recurrent replay |
| `localmodel start deepseek` | ISTA DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ1_M | 256K | 64 GiB cap, 36 GiB expert cache, six experts, DSpark off, 4096-token chunks |
| `localmodel start qwen3.8` | ISTA Qwen3.8-Flash-Next GSQ-RCO IQ2_XS | 256K | 64 GiB cap, 32 GiB resident budget, MTP3, int8 KV, automatic prefill |

The two Strata profiles now use source tag 0.2.7.1 and Control 0.1.53.
Their CUDA binaries remain 0.1.52. This is the intended upstream update for
Linux x86. The [update report](../../results/strata-official/20261028-update/summary.md)
records the signatures, source identity, cap checks, and test results.
Qwen's capacity was subsequently raised to 256K. It passed the 16K benchmark
at 520.1 prompt and 35.49 decode tokens/s after Windows memory monitoring was
removed. See the [Qwen repeat](../../results/strata-0271-linux-guard-20261028/summary.md).
DeepSeek passed its 64 GiB repeat
at 264.7 prompt and 5.15 decode tokens/s. See the
[repeat report](../../results/strata-0271-20261028/summary.md).

## Evidence about quality and speed

Our [2026-10-27 OMP runs](../../results/strata-027-20261027/summary.md) filled
about 16K tokens. Qwen measured 1395.8 prompt tokens/s and 61.68 decode
tokens/s with a 48 GiB cap. Code, tests, and recall passed; a resumed raw-JSON
response incorrectly used Markdown fences. DeepSeek measured 284.0 prompt
tokens/s and 5.31 decode tokens/s with a 48 GiB cap and passed its checks.
These are single sampled workflows. They do not measure the new cap,
filled 128K/256K contexts, or general capability.

Sushi's M5 Max numbers use different hardware, memory placement, prompts,
reasoning settings, and speculative decoding. Code and prose rates differ
substantially. Its successive restored prefill windows also differ from a
complete cold prompt. The runtime review links the exact benchmark protocol.
There is no matched published comparison of Sushi STQ against our two quants.

The bpw labels use different denominators. ISTA reports the transformer
backbone average. Sushi's headline describes routed experts and retains
other components at different precision. Our DeepSeek already has a 1.75 bpw
backbone. Its total download is 231.32 GB, against Sushi's 332.1 GB.
Our Qwen shards total 68.03 GB before separate MTP preparation. Sushi Qwen
uses about 76 GiB including its lookup table and MTP. Both Sushi packs are
larger than our installed base files. Compare full components before
attributing a quality difference to the quantizer.

## Recommended order

1. **Measure the 64 GiB process caps.** Keep the current expert budgets for
   the first comparison. Qwen previously reached its 48 GiB cap and reclaimed
   file cache. Then test a larger resident budget separately. For DeepSeek,
   the prior 36/48 GiB cache comparison showed no clear decode gain, so a
   larger cache is a hypothesis, not an established improvement.
2. **Compare DeepSeek 4096 and 8192 prefill chunks.** Larger chunks can reduce
   SSD expert passes. They also consume GPU memory. Measure the subsequent
   decode rate and expert misses as well as prompt speed.
3. **Evaluate a higher-quality Qwen GGUF before a full STQ port.** ISTA's
   IQ3_XXS is 75.8 GB versus 68.0 GB for IQ2_XS. Its published LiveCodeBench
   v6 score is 86.29 versus 83.43, with BF16 at 87.43. Those are publisher
   results, not our measurements. It needs a separately pinned profile and
   a matched OMP comparison. This uses an existing format and is a smaller
   integration than STQ. [ISTA model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF#results)
4. **Measure prefix reuse, then add a reasoning-boundary checkpoint if needed.**
   Sushi preserves the state after `</think>` so tool-call rerendering can
   reuse the reasoning prefix. This can matter more to task latency than
   increasing decode speed. Test cached state against cold execution before
   enabling it in a tool workflow.
5. **Prototype one STQ bank if quality per byte remains the goal.** Match the
   scalar decoder and packed fixtures first. Then benchmark CUDA rows 1, 2,
   4, and 8 with both resident and streamed weights. A full pack importer
   follows only if this proves useful. Do not requantize an already
   quantized pack and call that a faithful STQ comparison.

The optional Qwen Q2_0 pack is another speed candidate. The publisher reports
faster llama.cpp operation, but its coding score is lower and the timing does
not establish a gain on our Strata path. DFlash2 is a lower priority: the
installed Strata path is CLI-only and its 12 GB measurements mostly favor MTP.
DeepSeek DSpark can reduce latency only if accepted draft tokens pay for the
drafter's memory and extra expert reads.

Keep full routing and the existing isolation controls for these comparisons.
Expert dropping, residency-biased routing, approximate attention, and lossy
draft acceptance change the experiment. Report such changes separately.

## Scope completed

The source/control update and process caps are implemented and verified.
The research documents describe possible follow-up work. No Sushi weights
were downloaded. No STQ decoder, new quant profile, or inference optimization
was implemented. No model server was started or stopped.
