# Bonsai native candidate

Status: source prepared. No configure, build, inference, or benchmark has run.
The stock runtime remains the default.

This candidate uses our existing Prism revision `e3de715222b2667c54a267441159a5729f11155e`.
It retains that revision's MTP chaining and `xhigh` handling.
It uses selected native changes from the reviewed bundle revision
`f0a11b8b7ac5e82dcfdcd48015beb9eabf9501d2`.
It does not use the bundle's Python agent layer or its effort rewrites.

## Changes

- 0028 adds an explicit VRAM head and pinned RAM tail for the target KV cache.
- 0029 selects native quantized tensor-core attention for wide GQA at small query counts.
- The window part of 0031 limits the MTP draft cache. The target keeps its full context.
- 0041 adds host-tail prefetch and writes new rows through a VRAM alias.
- 0042 replaces a full grammar scan during cloning with a binary search.
- 0047 adds optional partial staging controls. The defaults stage the full host tail.
- 0053 uses the prefetch pipeline for graphs whose attention reads the host tail.

`native.patch` is the patch to apply to the exact base revision.
The files in `upstream/` record the original source patches.
Do not apply those files as a series. The combined patch contains the required adaptations.

The full 0031 patch would add a shared slot cap and regress this Prism revision's
independent lookup and MTP lengths. It also contains a generic stopping-rule hunk
that matches Eagle3 in this revision. The candidate omits both changes.
It does not add `--spec-draft-n-max-tail`.

## Check the prepared source

From the local-models directory:

```sh
python experiments/bonsai-native/prepare.py
python experiments/build-native.py bonsai-native
```

The first command checks the source and metadata hashes. The second prints the build plan.
Neither command builds or starts a model.

When the machine is available, use the explicit `--run` option on the generic builder.
The manifest uses the private CUDA 13 toolkit, Ada machine code, two build jobs,
and the repository's capped build workflow. CUDA VMM must remain enabled.
The manifest disables embedded UI downloads, subprocess tools, RPC, and FetchContent downloads.

The planned binary is `state/builds/bonsai-native/bin/llama-server`.
The build does not replace the installed Prism binary.

## Native controls

- `--spec-draft-window 16384` retains recent MTP draft rows. Zero retains the full draft history.
  Absolute positions remain unchanged. The target continues to attend to its full context.
- `--kv-vram-cells N` keeps the first N target KV cells per layer in VRAM.
  Later cells use pinned RAM. Zero disables explicit tiering.
  The runtime rounds the boundary to its cache padding.
- `GGML_CUDA_FA_MMA_DECODE_MIN_KV=256` enables the new MMA route above that KV length.
  Zero disables the override.
- `--spec-type ngram-mod,draft-mtp --spec-ngram-mod-n-max 32 --spec-draft-n-max 2`
  uses separate lookup and MTP limits already present in our Prism revision.
- `LLAMA_MTP_CHAIN=1` retains the existing chained draft path.

Use one GPU and one slot. Use equal K and V cache types with flash attention.
Keep the current weights. This preparation did not fetch or change weights.
The wrapper must bind to loopback and retain the requested `xhigh` setting.

Explicit KV tiering differs from the RAM cache for saved conversation states.
The active attention cache reads its RAM tail over PCIe. It also needs VRAM staging space.
A smaller VRAM head does not remove all staging allocations.
Check the allocator log for actual VRAM and host bytes before relying on a selected boundary.

The optional upstream environment controls are `GGML_CUDA_KV_TIER_STAGING`,
`GGML_CUDA_KV_TIER_REDIRECT`, `GGML_CUDA_KV_TIER_PREFETCH`,
`GGML_CUDA_KV_TIER_STAGING_FRAC`, `GGML_CUDA_KV_TIER_STAGE_MIN_Q`,
and `GGML_CUDA_KV_TIER_GRAPHS`. Leave them unset for the initial comparison.
Do not infer performance from their presence.

## Reproduce the source

The manifest records the exact base archive URL, size, SHA-256, combined patch digest,
and hashes of all 3,587 source files. `changed-source-files.json` records both the
base and prepared hashes of the 17 changed source files.

Use a new directory to fetch the pinned base revision. Apply `native.patch` with
`git apply --check` followed by `git apply`. Do not apply it over an existing modified tree.
Run the source hash check before configuring the build.

See `source-audit.md` for the review scope and checks still required.
