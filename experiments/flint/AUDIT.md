# Flint runtime and model review

Date: 2026-10-22. Scope: the Flint support change, the official chat template,
and public artifact metadata. No GPU benchmark was run for this review.

## Installed verification

The setup downloaded the pinned GGUF and official runtime. Both full files
passed their size and SHA-256 checks. The runtime reports build 11803 and
commit `a37cc4610`. It reuses the existing private CUDA 12 libraries.

The actual GGUF architecture fields match the official configuration.
All 144 added special tokens match the official tokenizer configuration.
The embedded chat template matches the official template byte for byte.
The server's `/props` response removes its final newline.
The GGUF contains 779 tensors and a 150144-token vocabulary.
These checks do not compare quantized tensor values with the BF16 weights.

The server loaded a 262144-token window on the RTX 4070 SUPER with Q8 KV
and CPU MoE cutoff 33. A synthetic OpenAI API tool-call and result round trip
passed. Its reply included separate reasoning content. OpenCode and OMP
provider checks report the same context and a 32768-token output budget.
The test server stopped after these checks. No existing service was stopped.
This check did not fill the context window or benchmark long conversations.
Evidence is in `results/flint-setup-20261022/`.

## Pinned sources

- Runtime: official `llama.cpp` release `b11803`. Its source commit is
  [`a37cc46104eb50900d5ab4a9ba02a0abdac039db`](https://github.com/ggml-org/llama.cpp/commit/a37cc46104eb50900d5ab4a9ba02a0abdac039db).
- Flint support: [PR #30928](https://github.com/ggml-org/llama.cpp/pull/30928),
  merged on 2026-10-21. Reviewed PR head:
  `1906989dec34d5a60e7ca77d1843bca0c66c8fd8`. PR base:
  `0a1ef309cdbb9f2024f58a38fb4c42352c5db569`.
- GGUF: [`unsloth/flint-GGUF`](https://huggingface.co/unsloth/flint-GGUF/tree/20468edfc38ab43d6914cc88b08cbecb05d0fc62),
  revision `20468edfc38ab43d6914cc88b08cbecb05d0fc62`,
  file `flint-UD-Q3_K_XL.gguf`. Published size: 18,494,977,472 bytes.
  Published LFS SHA-256:
  `91b8d85ab6ff7fb095d9a972f9afa6f3fa2b9acb642867a079f3e013ea3c350c`.
- Official model: [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint/tree/2a247be0c0aec895ccb0ac7ca92229ec6df37288),
  revision `2a247be0c0aec895ccb0ac7ca92229ec6df37288`.
- Official [`chat_template.jinja`](https://huggingface.co/RecursiveSI/flint/blob/2a247be0c0aec895ccb0ac7ca92229ec6df37288/chat_template.jinja):
  2,328 bytes. SHA-256:
  `7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466`.

The GitHub API reports 18 changed files, 319 added lines, and 7 removed lines.
The added and removed lines in the merge match the final PR change. Some
surrounding source changed before the merge. Raw review files are in the ignored
`state/flint-audit/` directory.

## Runtime findings

No code in the Flint change adds network connections, uploads, telemetry,
subprocesses, shell execution, file access, server routes, or tool execution.
The new Hugging Face URL is input to the existing tokenizer-update maintenance
script. Inference does not call this URL.

The new code builds the model graph and loads its tensor metadata. It adds
sliding and full attention, attention sinks, per-head normalization, routed and
shared experts, the tokenizer, and a RoPE cache-shift guard. The shared activation
scaling helper changes its threshold from 32,768 to 1,024. This also affects
Mistral4 and Mellum. The PR reports a 10% to 30% speed cost from automatic scaling.
This report does not measure that cost on this machine.

No Jinja or tool parser code changes in this PR. The pinned runtime already
supports the template's `\x00` string escape. Its lexer checks for two hex digits
and rejects incomplete or invalid escapes. The existing automatic tool parser
supports tagged argument values and `<think>` reasoning. The installed API
tool round trip passed as recorded above.

Changed files:

```text
conversion/__init__.py
conversion/base.py
conversion/flint.py
convert_hf_to_gguf_update.py
gguf-py/gguf/constants.py
gguf-py/gguf/tensor_mapping.py
src/llama-arch.cpp
src/llama-arch.h
src/llama-graph.cpp
src/llama-graph.h
src/llama-kv-cache.cpp
src/llama-model.cpp
src/llama-vocab.cpp
src/llama-vocab.h
src/models/flint.cpp
src/models/models.h
tests/fusion/MTL.csv
tests/test-llama-archs.cpp
```

## Template and model findings

The official template only formats messages, tools, and reasoning. It contains
no network destination or host execution call. It removes NUL bytes and escapes
role markers in user and tool text. It keeps assistant reasoning in
`reasoning_content`. It does not use `reasoning_effort` or `enable_thinking`.
An `xhigh` request therefore does not select a trained reasoning level through
this template.

Tool calls use `<tool_call>`, `<arg_key>`, and `<arg_value>` markers. String
arguments are emitted without XML escaping. Content can contain these markers.
This is a parsing and prompt-injection boundary, not evidence of hidden host
execution. The harness must continue to control tool permissions. This source
review found no code that appears to exfiltrate data. It is not a proof that the
whole runtime or arbitrary model output is safe.

The official configuration supports 262,144 tokens, including output. It has
49 layers: 2 dense layers and 47 MoE layers. It uses 35 sliding-window layers
with a 513-token window and 14 full-attention layers. The model has 32 query
heads, 4 KV heads, and a head dimension of 128. The release adds no recurrent
loops or MTP prediction head.

The model card recommends temperature 1.0, top-p 0.95, and top-k 20. It recommends
at least 32,768 output tokens for hard problems and asks clients to preserve
returned reasoning in conversation history.

The [posted recipe](https://x.com/DogukanUrker/status/2113060329038970892)
uses a preview checkpoint. The requested file is from the final `flint-GGUF`
repository. A matching recipe does not establish matching speed or quality for
this file. No local tokens-per-second result is established by this audit.

The published GGUF hash identifies the download bytes. It does not prove that
quantization preserved every source weight. Quantization changes tensor values.
This review did not compare the full BF16 checkpoint with the GGUF. A GGUF runtime
does not require executing the model repository's Python serving kit.
