# Source and runtime provenance

The source package includes local launchers, configuration, and reviewed patches.
It does not include model weights, native engine binaries, or installed package
trees. Fetch and setup commands retrieve those files using their recorded pins.
No general license for third-party code is implied by this package.

## Included helper source

- `libexec/capped` and `libexec/capped-record` are unchanged copies from
  `heavy-management` commit `96147e5de0fa6fa4471eddf317f92c4771115fa1`.
  Copyright 2026 Tharkun AB. The MIT terms are included in
  [`libexec/heavy-management.LICENSE`](../libexec/heavy-management.LICENSE).
- `libexec/bonsai2-models` is the existing local model-file helper. Packaging
  changes its source location and creates missing parent directories. Model
  data stays under `bonsai/models`.
- `libexec/graft/{gguf_raw,extract_head,merge}.py` are unchanged copies of the
  reviewed files from [sudoingX/bonsai2-small-gpu](https://github.com/sudoingX/bonsai2-small-gpu)
  at commit `1e3ff915cc919e182bbcfb7dc820222dc2a312a6`. No explicit repository
  license was found. Their exact hashes remain checked by `fetch_head.py`.
  The [head audit](../experiments/bonsai-next/head-audit.md) describes the scope.
- `collab/upstream` contains the selected OMP collaboration source and its
  upstream license files. `collab/upstream-revision.json` records its revision,
  selected paths, and local changes. See [the audit](../collab/AUDIT.md).
- `bin/start-remote-model.sh` is the supplied standalone script with renamed
  command references. Its [guide](runpod.md) records the attachment hash and
  the limits of packaging validation.

## Downloaded components

| Component | Pin and verification source |
| --- | --- |
| OMP 18.8.6 | `config/omp-package/bun.lock`; exact input/output bundle SHA256 in `config/omp-runtime.json` |
| Optional OpenCode goal plugin | Archive and file hashes in `config/opencode-goal-plugin-manifest.json` |
| SearXNG | Commit in `bin/setup-searxng`; upstream requirements; installed versions recorded in `search/installed-requirements.txt` |
| Bonsai runtimes | `config/runtime.json`; `experiments/bonsai-next/manifest.json`, combined patch, and complete source-file hashes |
| Bonsai compiler | `config/cuda-build-requirements.txt`; CMake/Ninja in `config/native-build-requirements.txt` |
| Strata control and DeepSeek engine | `experiments/deepseek-strata/manifest.json`; source receipt, signed feed, archive and binary hashes |
| Qwen engine and model preparation | `experiments/qwen-strata/manifest.json`; signed feed, source and artifact hashes |
| Model files | `config/models.json` and the Strata manifests: pinned revisions, byte counts, and SHA256 |

Native manifests retain the original recorded paths because installed build
receipts hash the complete manifest. `project_paths.read_manifest` resolves
that recorded root to the current checkout after parsing. It does not change
the raw manifest or source-file hashes. Fresh prepared packs use the new path.
This does not make already generated packs or file-identity receipts portable.

OMP's patched bundle must match its expected digest before it runs. This check
does not audit every transitive dependency. Strata verifies its signed native
artifacts and starts behind an outgoing-connection filter. A valid signature
does not make a native binary's implementation inspectable.

Keep the configuration pins when reproducing a result. Review an upgrade before
changing a package version, source revision, or expected digest.
