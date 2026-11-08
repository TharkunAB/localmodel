# Bonsai DSpine experiment

This experiment tests Tencent's Qwen3.8-27B DSpine drafter with the existing
Bonsai 2 27B target. It does not change production presets or binaries.

The source starts from the reviewed `bonsai-next` tree. `runtime.patch` adds
DSpine inference to that tree. Apply it after the existing Bonsai patches.
The source hashes in `source-files.sha256` describe the complete result.
The original patch and commit are recorded in `manifest.json`.

Two adaptations are required. The selected output rows must use Bonsai's
inverse Hadamard transform. The draft context must also honor the causal
attention metadata. The older runtime forced non-causal draft attention.
The port retains the existing greedy draft selector. It does not include
the newer optional probabilistic selector.

See [source-audit.md](source-audit.md) for the code review and its limits.
The experiment blocks outgoing connections with the existing seccomp guard.
It permits replies on accepted loopback API connections.

## Weights and build

The Q4_K_M drafter is 1,257,892,896 bytes. Its pinned revision is
`62e92414e114702e792ea7a4c37eba57b25bac8b` in
[`aj9o9/Qwen3.8-27B-DSpine-b16-GGUF`](https://huggingface.co/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF).
The file SHA256 is
`0c6c984ac18611288d7b8f497bc347c4921e8231b1b6abf28650481eaf35ff55`.

`verify-draft.py` compares 14 bridge tensors with the official Tencent
checkpoint at revision `ea909c0f5fab8865f9e14600334f55f7ce47308f`.
The BF16 projection tensors match byte for byte. The bias matches after
exact BF16-to-F32 expansion. This check does not establish equivalence for
the quantized codebook or other quantized tensors.

The experimental binary is `state/builds/bonsai-dspine/bin/llama-server`.
The build uses the unchanged GGML kernels from `bonsai-next`, installed into
`state/deps/bonsai-dspine-ggml`. The old CUDA toolkit directory is absent.
The experiment therefore uses the installed CUDA 13 libraries from
`state/runtimes/strata-qwen-0.1.58/engine`. This does not use the Strata engine.
Both the control and candidate use the same CUDA libraries.
The build has two compiler jobs, a 16 GiB memory cap, and no swap.

Exact commands and library hashes are in
`results/bonsai-dspine-20261108/build-receipt.json`.

To rebuild the prepared and verified source, use the existing capped builder:

```bash
.venv/bin/python experiments/build-native.py bonsai-dspine --run
```

This command requires the recorded GGML prefix and CUDA libraries. It checks
all source hashes before compilation. It does not fetch weights or prepare
the source. To prepare a fresh source tree, first prepare `bonsai-next`, copy
that tree to `state/sources/bonsai-dspine`, and apply `runtime.patch` there.

## Run a case

Run from the repository root. The launch files contain the exact commands
for this machine. Preview without `--run`:

```bash
.venv/bin/python experiments/bonsai-dspine/benchmark.py \
  --launch-file experiments/bonsai-dspine/launch/dspine15-32k.json \
  --output results/dspine-screen-new --mode screen --run
```

For the OMP workflow, start `searxng-local` in a separate terminal. Then run:

```bash
.venv/bin/python experiments/bonsai-dspine/benchmark.py \
  --launch-file experiments/bonsai-dspine/launch/dspine15-256k.json \
  --output results/dspine-omp-new --mode omp --history-tokens 64k --run
```

The runner rejects an existing managed model or occupied model port. It uses
a 40 GiB server cap, a separate 4 GiB OMP cap, and no swap. It stops below
8 GiB available Linux RAM or 4 GiB available Windows RAM. It stops its own
server on completion or failure. Stop the search service that you started
when the workflow ends.

The native screen uses three short prompts, temperature zero, and thinking
disabled. It is not a filled-context test. The OMP workflow uses temperature
one, `xhigh` reasoning, actual filled history, tool repairs, a changed old
message, resume, cancellation, and compaction. All cases use the same target
weights and seed. The 16K fixture keeps 4K recent tokens for compaction.
The 64K fixture keeps 20K. These settings affect private benchmark state only.

The 256K DFlash and DSpine launch files keep 128K KV positions in VRAM and
the remaining capacity in RAM. The MTP control retains its normal full-GPU
KV allocation. All target layers remain on the GPU. A filled 64K test does
not measure attention over the RAM tail or a filled 256K conversation.

## Results

[The completed report](../../docs/bonsai-dspine-20261108.md) keeps MTP2 as the
default. DSpine reaches 184 tokens/s on the short code fixture, but the seven
completed OMP cases do not establish a useful overall improvement.
All seven pass the workflow checks. The short arithmetic fixture fails with
and without speculation.

Raw requests, replies, counters, resource samples, and cleanup evidence are
under `results/bonsai-dspine-20261108/`. Compare generated token counts,
prefill work, completed task checks, and elapsed time together. The tracked
`measurements.json` retains the counters and summary hashes.
