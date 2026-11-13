# Prefill borrowing at a fixed chunk size

The comparison uses the official v0.2.12 source at
`146df3c468739147f520ac3816de806f71232571` and native 0.1.58.
Only the candidate environment differs: `DS4_BORROW=0` versus unset.

The source calls `chunk_default` only when `o.prefill_chunk < 0`
(`src/dsv41/ds4_model.cu:2795`). Both cases explicitly pass
`--prefill-chunk 4096`. They therefore bypass automatic chunk growth.
The automatic-growth aspect of issue 5570 is not active in this comparison.

Borrowing lets the prompt path use the memory occupied by tail expert slots.
`borrow_begin` temporarily removes those experts from VRAM. `borrow_end`
loads them again after prefill. Disabling borrowing allocates the prompt
buffers separately and can leave fewer decode expert slots.

The measured control has 304 VRAM experts and records successful returns
after prefill. The borrowing-off case has 69 VRAM experts and no borrowing
allocation or return markers. Both use the same 4K chunks and 192K capacity.
The live native environment verifies `DS4_BORROW=0` in the candidate.

At about 16K actual input, initial prefill changes from 272.7 to 276.0 tokens/s.
Weighted decode falls from 9.21 to 8.31 tokens/s. Both pass the task checks.
The candidate produces 578 tokens versus 881 in the control. Its shorter
wall time is therefore not evidence of improved throughput.

Keep borrowing enabled for this configuration. This experiment does not
evaluate automatic chunks, other GPU sizes, or multi-GPU allocation.

Evidence is in `benchmark/results-16k.json` and the per-case raw records:
`results/strata-candidates-20261109/harness/official-{control,borrow0}-16k/`.
Each directory contains `variant-final-markers.json` and `engine.log`.

Primary sources:

- [Issue 5570](https://github.com/Niko1221/Strata/issues/5570)
- [Pinned native source](https://github.com/Niko1221/Strata/blob/146df3c468739147f520ac3816de806f71232571/src/dsv41/ds4_model.cu)
