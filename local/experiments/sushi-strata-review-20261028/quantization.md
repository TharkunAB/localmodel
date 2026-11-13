# Sushi STQ on CUDA: quantization and importer review

Date: 2026-10-28. This is a source review. No STQ port, model conversion, or
inference benchmark was run for this review.

Sushi STQ can be implemented on CUDA. Its decoding operations do not require
Apple hardware. The public source is sufficient to decode the published
packs. Full pack support requires changes to the loader, GPU kernels, CPU
kernels, and expert storage. A change to a model filename is not sufficient.
No measured result establishes a quality or speed advantage over the
installed ISTA quants on this host.

## Evidence and scope

- Sushi source: commit
  `956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c`.
  The review includes this source revision, which contains changes after
  release 1.4.0. Source findings do not establish that every change is in the
  published 1.4.0 binary.
- Strata source: installed 0.2.7, commit
  `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`, in
  `state/sources/strata-official-0.2.7`.
- Host: WSL2, RTX 4070 SUPER with 12 GiB VRAM, AMD Ryzen 9 7950X,
  and approximately 110 GiB Linux memory. These values were read on the host.
- Local model sizes come from the pinned
  [Qwen manifest](../qwen-strata/manifest.json) and
  [DeepSeek manifest](../deepseek-strata/manifest.json).
- Sushi source files were read through GitHub. No local Sushi clone was
  created. The proposed port below has not been implemented.

## Compare complete data, not the headline bit rate

GB means 10^9 bytes. GiB means 2^30 bytes. The table describes stored model
data. It does not describe peak RAM or VRAM during inference.

| Model data | Headline rate and denominator | Non-lookup data, GB / GiB | Lookup data, GB / GiB | Total, GB / GiB |
| --- | --- | ---: | ---: | ---: |
| Installed Qwen GSQ-RCO IQ2_XS | 2.50 average bits over transformer weights | 39.226 / 36.532 | 28.800 / 26.822 | 68.026 / 63.354 |
| Sushi Qwen STQ | 2.75 average bits over routed experts; dense path uses affine 8-bit | about 49.5 / 46.1 | about 32.0 / 29.8 | about 81.5 / 75.9 |
| Installed DeepSeek GSQ-RCO IQ1_M | 1.75 average bits over transformer backbone weights | 120.721 / 112.430 | 110.595 / 103.000 | 231.316 / 215.430 |
| Sushi DeepSeek STQ | 1.75 average bits over routed experts; other components use separate formats | about 129 / 120 | about 203 / 189 | 332.1 / about 309.3 |

The local Qwen row includes its two original GGUF shards. It excludes the
separately prepared MTP and pack data. The local DeepSeek row includes four
backbone shards and the Engram file. It excludes a DSpark drafter, which the
current profile does not load. The Sushi packs include their draft and vision
components. Sushi component sizes are rounded publisher values, so their
sum need not equal the reported total exactly.

Sources: [ISTA Qwen model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
[Sushi Qwen model card](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw),
[ISTA DeepSeek model card](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF),
and [Sushi usage and disk sizes](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/docs/usage.md).

Sushi Qwen has more stored transformer data than our Qwen quant. A successful
port would not itself reduce our memory requirement. Our present expert
budget would also cover a smaller fraction of the larger pack.

Sushi DeepSeek keeps the original FP8 Engram tables. Our Engram file uses
IQ4_NL. The table-precision difference accounts for most of the download
increase. Any full-model quality difference can include this change, the
trunk precision, expert allocation, and inference numerics. It cannot be
attributed to STQ alone.

## Public decoding contract

The STQ codebooks use integer arithmetic and FP16 operations. The bitstream
has 16 by 16 weight tiles. Each tile contains `n` 16-bit words. Its rate is
`K = n / 16` bits per weight. The codeword for weight `t` ends at
`floor((t + 1) * n / 16)`. The 16-bit window wraps within the tile.

SPEC1 masks the codeword to the configured window. It multiplies the result
by a per-rate constant with 32-bit wraparound. A second constant masks the
product. The two resulting FP16 values are added with FP16 rounding. The
rounded sum is multiplied by 4096.

PAIR1 uses the odd weight's codeword for both weights of a pair. One masked
product produces the first weight. A rotated and separately masked product
produces the second. Each result uses the same rounded-half sum and scale.
PAIR1 requires the full 16-bit window.

The packed tile order follows the EXL3 tensor-core permutation. Input and
output scale vectors, `suh` and `svh`, surround H128 Hadamard transforms.
Their FP16 rounding is part of the reference behavior.

These operations can be expressed in CUDA. This is a conclusion from the
published algorithm. It is not a CUDA performance measurement.

Sources: [scalar decoder](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/src/exl3/expert_exl3.zig#L273),
[rate and bit-window arithmetic](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/src/exl3/expert_exl3.zig#L175),
[named codebook tables](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/src/exl3/codebook_tables.zig#L98),
and [pack contract](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/docs/pack-format.md#stq).

## Existing Strata support and missing parts

Strata source already has EXL3 CUDA kernels with H128 transforms. On SM80
and newer, the kernels decode weights into `mma.m16n8k16` operands. The
4070 SUPER is SM89. The source also has CPU expert kernels and a path that
shares one expert decode across speculative verification rows.

The source provides a useful starting point. It does not accept Sushi STQ
as written.

| Contract | Existing Strata EXL3 path | Required for Sushi STQ |
| --- | --- | --- |
| Rate | Integer `K` from 1 to 8 | Fractional `n/16`, including 1.625 |
| Codebook | 3inst, MCG, MUL1 | SPEC1 and PAIR1 with per-rate constants |
| Window | Fixed 16-bit unpacking | Explicit pack window, including 15 |
| Expert layout | Shared projection layout and stride | Per-expert rate groups and offsets |
| CPU kernels | MUL1-specialized expert path | SPEC1 and PAIR1 with the same decoded weights as CUDA |
| Pack metadata | Existing Strata metadata | Table identity, digest, window, group geometry, and source tensor mapping |

Sources: [Strata EXL3 types](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/include/strata/dsv41/exl3.hpp#L32),
[CUDA support predicate](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/src/dsv41/vendor/exl3.cu#L1351),
and [CPU decoder](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/src/dsv41/vendor/exl3_cpu.cpp).

Gyro support is not a compatible STQ decoder. Its pair-trellis format uses
128-weight blocks, 15-bit state masking, and a lookup table. Sushi uses a
different tile, bit window, and computed codebook. A conversion must preserve
decoded weights or introduce a new quantization error. A type-name change
cannot do that. See [Gyro state arithmetic](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/src/kernels/cpu/tq_states.hpp#L63)
and [Strata Gyro documentation](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/docs/GYRO.md).

## Rounding and scale handling

SPEC1 and PAIR1 intermediate halves can be subnormal. The half sum must round
before the 4096 scale is applied. A CUDA port must check this behavior across
all codewords. It must also check any compiler optimization that changes the
arithmetic order.

Sushi moves the 4096 scale after an FP32 GEMV sum on selected paths. It does
not move the scale across an FP16 store, because the unscaled value can
underflow. Its matrix-unit operands retain the scale. Do not copy the scalar
scale-folding change into an MMA path without the corresponding numerical
check. See [Sushi numerical rules](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/docs/engine-exl3-experts.md#L66).

## Requirements for complete pack import

An importer can change tensor names and physical byte order without changing
the represented weights. It must retain the original bitstreams, scales,
group assignments, codebook table, and window. It must refuse unknown
digests, unsupported rates, invalid shapes, and incomplete groups.

| Component | Qwen Sushi STQ | DeepSeek Sushi STQ |
| --- | --- | --- |
| Routed experts | SPEC1, mixed 2 to 4-bit banks, 512 experts per layer | PAIR1 at `n24`, `n26`, `n32`, and `n40`, with 384 experts per layer |
| Group mapping | Map each router index to its group and local index | Same; some groups have different down and gate/up rates |
| Dense path | MLX affine 8-bit tensors with their actual group sizes | SPEC1 trunk at `n64`; grouped projections must retain their group axes |
| Output head | Affine tensor | SPEC1 at `n96` |
| Embedding | Affine tensor and exact row-gather path | Affine 4-bit tensor and exact row-gather path |
| Sparse lookup table | Separate group-32 affine 4-bit n-gram file | Original FP8 Engram data and its scale layout |
| Draft model | Included MTP layer, with the same STQ reader | Three DSpark stages; PAIR1 `n32` experts and SPEC1 trunk |
| Remaining source tensors | Preserve dtype, orientation, names, and semantic role | Some off-grid linears stay FP8, including indexer `weights_proj` |
| Vision | Preserve the vision components for multimodal support | Preserve the vision components for multimodal support |
| Runtime metadata | Tokenizer, template, context scaling, and architecture fields | Tokenizer, template, architecture fields, and stage metadata |

DeepSeek's routed matrices use 5120 to 2304 and 2304 to 5120 geometry.
Its layer 23 includes a group with gate/up at `n32` and down at `n26`.
The importer must not assume one rate for the whole layer. The Qwen expert
geometry is 2560 to 640 and 640 to 2560. Both dimensions meet the H128 grid.
Sources: [group contract](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/docs/pack-format.md#rate-groups-per-expert-rates-uneven-expert-counts)
and [DeepSeek tensor map](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/docs/engine-exl3-experts.md#L381).

A prototype may deliberately omit vision or drafting. Such a prototype does
not yet implement the complete published pack. Its measured memory and
throughput must name those omissions.

## Public encoder boundary

The public conversion CLI accepts Qwen fixed-rate EXL3 MCG or MUL1. It exposes
calibration, Hessian collection, H128 rotation, LDLQ feedback, Viterbi search,
and scale search. It does not expose the STQ SPEC1 or PAIR1 conversion paths.
The documentation also states that the served packs' tuned allocation is
absent from the public encoder. See [encoder options and allocation](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/docs/encoder.md#what-is-written).

Thus, published-pack inference and reproduction of the published quant are
separate tasks. The public decoder is enough for the former. A new STQ
quantizer also needs an allocation method and its own quality evidence.

Do not use the existing ISTA GGUF as the source for a purported reproduction
of Sushi quality. That would quantize an already quantized checkpoint.
Similarly, decoding Sushi to FP16 and requantizing it into a supported GGUF
would add another lossy step. Use the original checkpoint for new
quantization research. Use lossless repacking to test the published pack.

## Quality evidence and limits

The Sushi Qwen card reports eight-category KLD of 0.0295. Its comparison
includes other Sushi and MLX packs. It does not include our ISTA IQ2_XS.
The fixture uses 16 prompts of 512 tokens, with BF16 KV and FP32 GDN state.
The card explicitly separates this fixture from its other KLD reading.
See the [Sushi Qwen quality table](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw#quality).

Strata reports DeepSeek IQ1_M KLD of 0.217 on 4,995 positions. That is a
different fixture from Sushi's results. Numerical proximity between those
KLD values is not evidence of equal quality. See [Strata DeepSeek evaluation](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/docs/DEEPSEEK41.md#the-official-2-bit-versions-the-ones-for-a-normal-pc).

The M5 speed measurements combine the model pack, memory system, kernels,
and speculative policy. They do not estimate 4070 SUPER performance. A
discrete GPU must also move cache misses over PCIe or compute them on the
CPU. A decoder microbenchmark must therefore include streamed cases.

## Proposed bounded prototype

This sequence is a proposal. None of these steps was run in this review.

1. Implement a separate CPU reference decoder. Check all 65,536 codewords
   for each required table row. Compare the packed PAIR1 fixtures with their
   committed decoded values. Include fractional rates, window masking,
   wraparound, permutation, subnormal sums, and invalid metadata.
2. Add one CUDA tile decoder and matvec path. Use a small real Qwen bank and
   a small real DeepSeek bank. Compare decoded FP16 values exactly. Compare
   products with FP64 accumulation of those same decoded values.
3. Measure rows 1, 2, 4, and 8. Run resident and streamed cases. Record
   decode time, product time, transfer bytes, and peak memory separately.
4. Add per-group storage and lossless pack import only if the first three
   steps pass. Extend CPU and GPU paths together. A GPU-only fast path must
   not silently send cache misses through an incompatible CPU decoder.
5. Check full-model logits and the actual OMP repair, recall, tool, and
   compaction workload. Match context, reasoning settings, cache budget,
   drafting policy, and vision state against ISTA. Keep isolated failures.

The committed [PAIR1 fixtures](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/src/exl3/fixtures/exl3_pair1_tiles.safetensors)
permit the first stage without a full model download. The source tests
also provide [geometry and decoded-product checks](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/docs/engine-exl3-experts.md#L423).

The work has five independent review boundaries: the format and reference,
the kernels, variable-rate storage, complete pack import, and end-to-end
validation. This scope is larger than a launcher update.

## Recommendation

Use the current native formats while measuring memory and cache changes on
the enlarged WSL allocation. If STQ remains a priority, implement the small
decoder prototype before a complete port or large download. The source makes
that prototype technically credible. Current evidence does not establish
that the complete Sushi packs improve our 12 GiB CUDA setup.
