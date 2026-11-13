# Bonsai next native port review

Source: `/home/tritlo/Code/local-models/state/sources/bonsai-next`.
Base commit: `e3de715222b2667c54a267441159a5729f11155e` with the existing native patch already applied.
Scope: patches 0046, 0054, and 0070 only, with the adaptations below.

## Applied patches

- `0046-server-checkpoint-every-nt-N-also-checkpoints-inside.patch` SHA256 `f102ba37c6336c93046f307fb272dc05c66451c9721237a915c43d3bb8f312ce`.
- `0054-cuda-chunked-gated-delta-net-MMA-prefill-port-of-ggml.patch` SHA256 `98d13c1cf02f8ead3372d54ac5280e3401bcc790e48068ca47285cf06d931a1a`.
- `0070-recurrent-replay-rollback-LLAMA_RS_REPLAY-1-one-recu.patch` SHA256 `f9a11243cec81a0d8f50f0ccbff789d42904a70f3680e7368d77c17e6e416a49`.

0046 and 0054 applied without rejected hunks. 0070 required changes in recurrent memory and Qwen35 state selection. All other 0070 files applied without rejected hunks.

## Adaptations

- Preserve the existing ARC-LAB SYCL replay mode, fields, graph branches, API, and state layout. CUDA replay is mutually exclusive with SYCL replay.
- Rename the added CUDA operation constructor to `ggml_gated_delta_net_replay_raw`. The source already contains an incompatible SYCL operation constructor with the original name and the same C signature. A clean patch application alone does not detect that collision. The existing SYCL constructor remains unchanged.
- Use state layout tag 3 for CUDA raw pending inputs. Keep tag 2 for existing SYCL update records. Each reader rejects the other replay layout. Plain layout 0 remains readable. Loading plain state in CUDA replay clears pending inputs.
- Preserve existing SYCL arguments in `state_write_data`. Add an optional CUDA rejected-token mask vector. Keep the existing S-row range parameter for both replay modes.
- Allocate one S plane for CUDA replay, two S planes for SYCL replay, and all snapshot planes otherwise. Keep all convolution-state snapshot planes. Allocate separate pending-input tensors only in CUDA replay.
- Require the CUDA backend registry, F32 recurrent state, Qwen35 architecture, raw F32 gate tensors, and one sequence. Unsupported configurations keep the existing snapshot mode. Empty recurrent-layer selections do not enable replay.
- Validate imported pending counts before conversion or upload. Counts must be finite integers between 0 and P. Validate the existing serialized dimensions, masks, and row sizes as in 0070.
- Bound pending counts and rejected-token counts in the CUDA kernel before indexing. Invalid pending counts cause no pending-token reads. These checks do not alter valid counts.
- Include pending bytes in the constructor total. Existing backend-buffer accounting already includes the allocated pending tensors.

## Security review

These three patches add no network clients, outbound requests, telemetry, subprocess execution, or command evaluation. Checkpoint logs contain positions and sizes. They do not add conversation excerpts.

The imported state format is a local binary parser. The upstream float-to-integer conversion accepted invalid counts that could produce reads outside pending rows. The port rejects these counts. The distinct state tag prevents the existing SYCL payload from being interpreted as CUDA pending inputs. This is a bounded review. It is not a complete audit of the engine or all pre-existing state parsers.

## Static checks

`git diff --check` passed. Focused compiler syntax checks passed for:

- `ggml/src/ggml.c`
- `src/llama-memory-recurrent.cpp`
- `src/llama-graph.cpp`
- `src/models/delta-net-base.cpp`
- `src/models/qwen35.cpp`
- `ggml/src/ggml-cpu/ops.cpp`

The syntax checks used the existing native compile commands with paths changed to this worktree. They used `-fsyntax-only` and removed object-output arguments. They ran serially through heavy-management with a 4 GiB memory cap, no swap, and a five-minute time cap. No binaries or model servers were started. CUDA compilation and inference remain to be checked by the parent task.

The two new MMA source files have intent-to-add entries, so `git diff` includes them. No changes were made to manifests, launchers, presets, or model files. The original bonsai-native worktree was unchanged.

## Required runtime checks

1. Check greedy output with replay disabled and enabled at ubatch 64. Exercise MTP depth 2, chaining, rejected drafts, a 16K draft window, and a prompt beyond that window.
2. Check ubatch 128 with MMA disabled and enabled. Confirm which kernel actually runs. With depth 2 and ubatch 64, 64 - (P + 1) is 61, below the 64-token MMA threshold.
3. Change text inside a long message and confirm that the nearest preceding checkpoint is restored. Appending text alone does not test this behavior. Existing minimum spacing and batch boundaries make the checkpoint interval approximate.
4. Check cancellation, the following request, checkpoint restore, slot save/restore, and OMP resume/compaction.
5. Check plain-state restore into replay mode. Reject replay layout 3 with replay disabled. Reject layout 2 on the CUDA replay path. Check corrupted pending counts if a slot-file fixture is available.
6. Compare GPU memory, host memory, prefill speed, decode speed, and draft acceptance. Preserve the existing binary until these checks pass.

Source was frozen after these checks. No build or inference result is claimed.
