# DSpine port review

Date: 2026-11-08. Scope: the isolated `bonsai-dspine` experiment.
The normal runtime and model defaults are unchanged.

## Source and scope

The parent source is the prepared `bonsai-next` runtime at PrismML commit
`e3de715222b2667c54a267441159a5729f11155e`, with its existing reviewed patch.
The new inference code comes from
[Ajay9o9/llama.cpp commit d493f841](https://github.com/Ajay9o9/llama.cpp/commit/d493f841b20241606c47a4e0537d3c786b5cf50b).
The downloaded upstream patch has SHA-256
`cec08cb1a5646197cd8c84820aa8a241663abb86b492974fef6d98cf3d05a5f2`.

`runtime.patch` applies to the prepared parent source, not to the unmodified
PrismML commit. It changes five files. The source hash list contains all 3,589
tracked parent files after the port. The manifest records the final patch hash.

The port includes tensor names, tensor loading, the code bridge, and the
candidate-score table. It also reads the causal-attention metadata in the
speculative runner. The new upstream base already reads this metadata. Our
older base unconditionally disabled causal attention. DSpine requires it.

Conversion and quantization tools are excluded. This experiment uses a pinned
preconverted GGUF. No GGML kernel or network component changes.

## Ternary output rows

The upstream candidate-score table reads selected target output rows directly.
That is incorrect for the stored Hadamard-rotated Bonsai output weights.

For stored row `z`, the normal forward projection computes `z · H(D x)`.
Here `H` is the normalized symmetric Hadamard matrix and `D` is the sign matrix.
The selected row must therefore become `D(H z)` before its dot product with
the original hidden state `x`.

The port uses the existing `build_embd_rows` path for a tied embedding table.
For an output-only transform, it uses the existing
`llama_mul_mat_hadamard` primitive, then the existing signs. It does not apply
the transform twice when the table already has an inverse-map entry.
A permuted output head causes an explicit error. The selected-row scores
also receive the output scale and the configured logit transforms.

A dependency-free FP64 arithmetic check compared both expressions at width
1,024, with 16 rows and 17 inputs. All 272 comparisons passed. The maximum
absolute error was `2.1316282072803006e-14`. This verifies the matrix order.
It does not establish CUDA graph parity or inference correctness.

## Selector compatibility

The existing runner detects the selector through `selector_top_k > 0`.
It reads `h_nextn` in rows of hidden width. Each row starts with candidate IDs,
then contains predecessor-major scores. Its greedy walk starts at row 1 and
selects the next predecessor by the largest score. These interfaces match the
upstream DSpine port. The first row remains the accepted anchor.

The port retains the existing greedy draft selection. It does not add the
new upstream probabilistic draft sampler. Target sampling remains separate.

## Security and failure handling

Static review found no added network requests, telemetry, file reads or writes,
subprocess execution, shell execution, or credential access. The additional
string handling reads one fixed boolean metadata value into an existing
bounded buffer.

The port rejects an incomplete codebook/gate pair, nonpositive dimensions,
an oversized selector table, a reduced draft vocabulary, mixed draft
architectures, and non-causal DSpine metadata. Existing tensor-loading checks
validate the expected weight shapes. Draft blocks outside the trained size do
not receive bridge writes.

This review does not prove memory safety for arbitrary hostile GGUF files.
Use the pinned files and the existing restricted benchmark process. The source
hash list and weight receipts provide provenance, not a general safety proof.

## Validation boundary

The original Bonsai source tree was copied. It was not modified.
The isolated source passes `git diff --check`. All new graph primitives are
present in the parent GGML API. Compilation and runtime checks belong to the
benchmark results and must pass before performance claims are made.

The drafter was trained for dense Qwen3.8-27B. Bonsai retains that architecture
but changes its weights and hidden-state distribution. Correct tensor shapes
and a successful launch do not establish useful acceptance or speed.
