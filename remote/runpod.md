# Run the remote launcher on RunPod

`remotemodel` manages a model server on the rental GPU host. Its commands
mirror `localmodel`: `start`, `fetch`, `list`, `status`, and `stop`. Install
the root `start-remote.sh` bootstrap as `remotemodel` to fetch the latest
published code on each invocation. A source checkout also provides
`remote/bin/remotemodel` for a fixed version without bootstrap updates.

The CLI reuses `remote/start-remote-model.sh` for GPU selection, preparation,
build caching, and serving. That standalone backend remains available with
its existing flags. Neither interface requires systemd or `heavy-management`
on the rental host. The Strata TUI and its dependency lock are shared with
the local harness; model profiles remain specific to the host.

Automatic selection serves **DeepSeek V4.1 Flash 1017** on supported Strata
GPUs. One B300 selects **Mistral Large 4 UD-IQ1_S** with llama.cpp, DSpark,
and 512K context. Other models require an explicit profile. The legacy `qwen`
profile serves Qwen3.8-27B; the local harness uses a separate
Qwen3.8-Flash-Next profile. `remotemodel` does not connect to a Pod over SSH.
Run it on the GPU host; run `omp-remote` on your workstation.

All remote DeepSeek profiles now use the unmodified official Strata source
`d2f096475817d4666c364c820b0ca830f751cfc5` (0.2.13), Control 0.1.59, and signed
CUDA 13 engine 0.1.59. This includes the official API server and GGUF packer.
See the [update review](../local/docs/strata-0.2.13-review.md) for checks and known parser behavior.
The standalone file embeds the artifact pins, signature
verification, and the same outgoing-network restriction as the local harness.
Qwen, Flint, Ornith, and MiniMax keep their explicit llama.cpp paths.

The original `start-qwen.sh` attachment has SHA-256
`955915e9fc13bc9ba44a580285cd21a70a0c183c2fafb514e3e1c274326b8d3a`.
The current embedded backend version is `2026-11-13-vllm-review-1`.
Packaging checks do not establish remote GPU inference or throughput.

## Migrate an existing DeepSeek Pod

On four A40 cards, `--profile auto` now selects `deepseek-a40-quad`:
**Unsloth Dynamic V3.1 UD-IQ1_M, 1M context, DSpark, and split 10,12,9,9**.
The weights are pinned to `8a6ec9cf118e785a362ea94ee123c61c70553be6`.
This preset has a checked capacity estimate. It has not been benchmarked here.
It replaces the automatic GSQ choice; the explicit GSQ profile remains available.

Run the bootstrap again to fetch the latest published payload. Keep the same
`--root` to reuse the
existing verified DSpark draft. Stop the previous server before a runtime
upgrade. Prepare and start the new profile:

```sh
bash /workspace/start-remote.sh --profile deepseek-a40-quad \
  --root /opt/qwen --no-ssh --prepare-only --install-deps
bash /workspace/start-remote.sh --profile deepseek-a40-quad \
  --root /opt/qwen --no-ssh
```

The new five-file checkpoint needs 250.4 GB of disk space. Preparation verifies
every file against its pinned SHA-256. It does not remove the old GSQ weights
or restart a server. The script uses the official 0.2.13 parser and retains the
outgoing-network restriction. Mistral on one B300 still defaults to 512K.
If the full 0.2.13 runtime is already prepared, use
`--prepare-only --prepare-assets-only` instead of `--prepare-only --install-deps`
to stage the new weights while the old server runs. Stop it before the launch step.
The default new weight directory is
`/opt/qwen/deepseek-v4.1-1017-a40-quad-v31-iq1m/models/8a6ec9cf118e785a362ea94ee123c61c70553be6`.

### Update an existing Strata installation

Run the bootstrap again to fetch the latest published payload. To install
Strata 0.2.13, stop the old
server and run `--prepare-only --install-deps` with the same profile and root.
Then start it with the same command without those two preparation flags.
`--prepare-assets-only` does not update the runtime.

The source revision and native version select a separate runtime directory.
Preparation verifies the official source, signed native files, and Python packages.
It reuses existing verified GGUF files and the DSpark draft for the same profile
and root. It creates a small pack index in a directory named for the new source
revision. It does not download the weights again, requantize them, or modify the
old pack's receipt. The previous runtime and pack remain on disk.

Run preparation once after this update. The old 0.2.12 runtime does not satisfy
the new source and native pins. Serving requires the new runtime and pack.
This update does not deploy or restart an existing Pod automatically.

### Previous GSQ profile

The explicit `deepseek-gsq-a40-quad` profile defaults to
**IQ2_XS, 1M context, DSpark on, and layer counts 9,11,10,10**. It still selects
six experts per token. On engine 0.1.52, the first 13 normal requests generated
5,302 tokens in 95.468 decode seconds: 55.5 tokens/s. The previous ordinary
IQ2_XXS run averaged 37.1 tokens/s. These requests used different prompts and
context lengths, so this is an observation, not a controlled comparison.
The new requests used 11K–38K tokens. They did not fill the 1M capacity.

For an existing prepared runtime under `/opt/qwen`, stage the new assets while
the current server continues to run:

```sh
bash /workspace/start-remote.sh --profile deepseek-gsq-a40-quad \
  --root /opt/qwen --no-ssh --prepare-only --prepare-assets-only --dry-run
bash /workspace/start-remote.sh --profile deepseek-gsq-a40-quad \
  --root /opt/qwen --no-ssh --prepare-only --prepare-assets-only
```

`--prepare-assets-only` requires `--prepare-only`. It verifies the prepared
runtime offline, downloads missing selected weights and the pinned draft,
and creates the separate DSpark pack. It does not update runtime packages,
rewrite the network guard, change the active API configuration or launch
record, load a GPU model, or stop a server. It uses separate preparation locks
and does not require a free API port or idle GPUs. Missing runtime dependencies
cause an error; this mode cannot combine with `--install-deps`.

For a fresh Pod, first prepare with `--prepare-only --install-deps`, without
`--prepare-assets-only`. That mode can install the pinned runtime and packages.

The selected GSQ shards and shared Engram remain under
`/opt/qwen/gsq-models/cb02422c900df489cea5833e8a92171224059b3c`. Existing IQ2_XXS
files stay there when IQ2_XS is added. The draft is separate:
`/opt/qwen/strata-drafts/f924b94f19c1fea02c06a65c8475b0a45e2b88b1/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`.
It is pinned to 7,976,914,240 bytes and SHA-256
`6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a`.

The stock `tools/dsv41/pack_gguf.py` creates
`strata-pack-iq2_xs-dspark-<source-revision>` with `--dspark-gguf`, noise token 128799, and Markov
rank 256. The pack receipt includes every generated index/tokenizer asset and
the draft identity. Unknown pack entries are rejected. The pack references
verified GGUF files; it does not copy or requantize weights. The existing
ordinary pack remains unchanged.

After you stop the previous server, start the prepared profile:

```sh
bash /workspace/start-remote.sh --profile deepseek-gsq-a40-quad \
  --root /opt/qwen --no-ssh
```

To keep existing compatible IQ2_XXS weights, add `--reuse` to preparation and
launch. On native GSQ profiles, this selects IQ2_XXS when all four shards and
the shared Engram exist at their exact sizes. Otherwise it selects IQ2_XS.
Preview labels this as a size-based candidate selection. Preparation and
serving require matching SHA-verification receipts or compute the hashes.
A selected file with an invalid checksum causes an error. The file is retained.
Explicit `--quant` takes precedence over `--reuse`. Other model families reject
`--reuse`.

For the previous ordinary IQ2_XXS configuration, use:

```sh
bash /workspace/start-remote.sh --profile deepseek-gsq-a40-quad \
  --root /opt/qwen --no-ssh --quant IQ2_XXS --ctxt 1M \
  --spec none --layer-split auto
```

Serving verifies the prepared source, signed native engine, dependencies,
GGUFs, and pack. It does not install or download updates. Missing preparation
causes an error. An existing server on port 8897 is never stopped automatically.
The upstream release catalog can change. Preparation verifies its publisher
signature and requires the pinned engine version, archive, size, and manifest.
Serving verifies the cached catalog offline. A catalog update cannot select a
different engine. If an older script reports that `strata-feed.json` differs
from its pin, rerun the bootstrap and repeat preparation. No data removal
is required.
The existing `/opt/qwen` data is usually on the container disk. Preserve it
before a Pod operation that clears that disk, or prepare a persistent root.

Remove `--cpu-moe-layers 2` from the old command. Strata places experts
automatically across all four selected GPUs. It maps their UUIDs to native
indices `0,1,2,3`. The following controls replace llama.cpp placement controls:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--layer-split` | Planned contiguous counts; `10,12,9,9` for Unsloth on quad A40 | Layer counts per GPU; sum must be 40; `auto` delegates placement to Strata |
| `--spec` | `dspark` when it fits; `none` otherwise | Native DSpark or ordinary decoding |
| `--dspark N` | No block override | Positive N enables DSpark and limits drafts per round; 0 disables |
| `--dspark-conf` | `0.65` when DSpark is enabled | Draft confidence threshold, 0–1 |
| `--expert-ram-gb` | Up to `36`, after host headroom | Soft expert RAM cache in GiB; an explicit value overrides automatic sizing |
| `--gpu-expert-fraction` | `1` | Fraction of the available native GPU expert budget |
| `--vram-reserve-mib` | `2048` | VRAM reserve per GPU in MiB |
| `--prefill-chunk` | `4096` | Prompt chunk size in tokens |
| `--cpu-threads` | Up to 32 host CPUs | CPU expert threads; accepted range 1–64 |
| `--engram-cache-mib` | `256` | Engram row cache in MiB |
| `--engram-threads` | `16` | Engram SSD read threads |
| `--thinking-budget` | `32768` | Maximum reasoning tokens per request |
| `--reasoning-effort` | `high` | Server default: none, low, medium, or high |

API requests can send `xhigh` or `max`. The native server maps those request
values. Numeric llama.cpp effort defaults are rejected. Stock Strata detects
older GGUF tool templates and selects its bundled DeepSeek V4.1 template.
This emits the spaced DSML tags that its tool parser expects. No custom
template or parser patch is applied.

`--expert-ram-gb` converts GiB to the native engine's decimal-GB `--ssd-cache`
unit. It is **not a hard process RAM cap**. The launcher checks available RAM,
including cgroup v1/v2 limits, and requires the cache plus the Engram row cache
plus 24 GiB headroom. It does not create a systemd scope or a new cgroup inside
the Pod. Without an explicit cache size, the launcher chooses up to 36 GiB
after reserving 24 GiB and the Engram row cache. Low-memory hosts can therefore
use a smaller expert cache. A 200 GB RAM allocation has about 186 GiB before
other use. Inspect memory use before increasing the cache.

DeepSeek rejects `--cpu-moe-layers`, `--tensor-split`, `--gsq-engram`,
`--engram-ram`, llama.cpp KV/prompt-cache/batch flags, and build-archive flags.
Use a dedicated `deepseek-gsq-*` profile for GSQ; `--ds-source gsq-rco` does not
substitute a different quant into an Unsloth profile. Existing Unsloth quant
choices and GSQ IQ1_M/IQ2_XXS/IQ2_XS choices remain explicit. Remote Strata uses
six experts per token. `--spec dspark` enables the pinned native draft;
`--spec none` does not load it. MTP and DFlash remain unsupported.
`--dspark N` accepts 1–64 as a native block limit; omitting it leaves the
native limit unchanged. It conflicts with an incompatible explicit `--spec`.

`--layer-split` takes one positive layer count per selected GPU, summing to 40.
The launcher sets `DS4_SPLIT`; the native engine ignores the generic
`--layer-split` argument. Use `--layer-split auto` to leave native placement
automatic. The draft stays on the first selected GPU. The Unsloth quad-A40
profile gives that GPU ten trunk layers. The explicit GSQ profile gives it nine.
The counts come from
weight-size estimates; they do not prove that 1M runtime buffers fit.
`--layer-split 10,10,10,10` remains an explicit override.
The native packer can reject an unsupported tensor layout; it does not change
the quant or download another model. Remove unrelated drafter/vision assets
from the packer's automatic-discovery locations before preparing a plain pack.

The server and native children run under an inherited libseccomp policy.
It blocks outgoing connections, UDP, addressed sends, descriptor transfers,
and alternate asynchronous I/O paths. It closes inherited descriptors above
standard input/output/error. Local API requests and replies still work.
Missing libseccomp or a policy-load error prevents server startup. File access
and listening sockets are not restricted. Downloads happen only during the
explicit preparation step, outside the serving restriction.
The policy permits local Unix sequenced-packet sockets for Linux CUDA's UVM
listener. It still blocks outgoing connections and addressed sends.
Older guards can cause `no CUDA device` even when `nvidia-smi` lists the GPUs.
Rerun the bootstrap with `--prepare-only` to update the guard.

The serving environment includes upstream's pinned `psutil` for disk and
process statistics. If the disk card says `needs psutil`, stop the server,
copy the current script, repeat `--prepare-only`, and start the server again.
Refresh the browser after startup. The disk rate covers all reads visible
to the operating system or container; it is not a per-process measurement.

## Mistral on one B300

`--profile auto` selects `mistral-b300` for one full B300. It uses the pinned
Unsloth UD-IQ1_S weights and DSpark draft with a pinned upstream llama.cpp build.
The defaults are **512K context, q8_0 KV, three draft tokens, and GPU weights**.
The draft has no confidence head, so its confidence cutoff is zero.
Use a CUDA development image with toolkit 12.9 or later. Allocate at least
300 GiB of disk space for weights, the draft, source, and build files.

Prepare once, then serve:

```sh
bash /workspace/start-remote.sh --profile mistral-b300 \
  --root /workspace/remote-models --no-ssh --install-deps --prepare-only
bash /workspace/start-remote.sh --profile mistral-b300 \
  --root /workspace/remote-models --no-ssh
```

Preparation downloads and verifies the artifacts, then compiles llama.cpp.
Serving checks the prepared binary, weights, and network guard. It does not
download or compile. The server uses loopback port 8897 and the same inherited
outgoing-connection restriction as Strata. The SSH tunnel and `omp-remote`
commands are unchanged. OMP discovers the served 512K capacity and maps its
`xhigh` setting to Mistral's `high` effort.

The planner estimates main KV and index-cache memory from the GGUF geometry.
It includes the draft and an 8 GiB runtime allowance. It checks actual reported
VRAM, not the advertised 288 GB. This is a capacity estimate, not a B300 inference
test. Use `--ctxt 256K` to reduce capacity explicitly, or `--ctxt 1M` to request
more. `--kv f16|q8_0|q4_0`, `--draft-depth N`, and `--spec none` are explicit
controls. An allocation failure does not select smaller weights or context.
The official NVFP4 checkpoint does not fit the advertised single-B300 GPU and
host memory allocation. This profile therefore uses GGUF with llama.cpp.

## vLLM compatibility watch

Reviewed on November 13, 2026. The daily research watch covers the remote
launcher and official vLLM releases. The launcher does not yet have an enabled
vLLM backend. Its current Strata and llama.cpp profiles remain available.

The latest published release is named **v0.33.0**, with GitHub release tag
`v0.33.0rc3`, published on November 2. GitHub marks that release as non-prerelease.
The separate `v0.33.0` tag resolves to the same commit. Keep the artifact tag
and display name distinct when checking an update.
See the [official release](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3).

For `nvidia/Qwen3.8-Flash-Next-FP4-Local`, upstream
[CSF scale decoding](https://github.com/vllm-project/vllm/pull/64312) merged
on November 3. The checkpoint's shards are under `tensors/`. Loading that
layout still requires the unmerged [shard-loader fix](https://github.com/vllm-project/vllm/pull/64739).
Neither change is in the November 2 release. The `flash-next-fp4` profile
therefore remains blocked. The older NVIDIA `NVFP4` checkpoint is a different
artifact and must not be substituted without a separate profile review.

[Mistral Large 4 support](https://github.com/vllm-project/vllm/pull/64534)
merged on November 13. This is newer than the published release. The
[official recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/mistralai/Mistral-Large-4-1T-A52B.yaml)
still names the dedicated `mistral-large-4` image and lists multi-GPU layouts.
It does not validate the launcher's single-B300 GGUF configuration.

For DeepSeek, track the new [shared CPU KV change](https://github.com/vllm-project/vllm/pull/65373)
and the [official deployment recipes](https://github.com/vllm-project/vllm/blob/main/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml).
The CPU KV improvement was measured with eight H100 GPUs and concurrent
LongBench requests. It does not establish the same gain for a single coding
conversation or the launcher's smaller allocations.

Before enabling vLLM, pin a complete official runtime and checkpoint, check
capacity, and test tools, reasoning, cache reuse, and shutdown. Validate a
serving isolation path that permits its internal IPC while blocking external
connections. The current guard denies `connect`, `sendmsg`, and `recvmsg`,
including local sockets. It cannot be reused unchanged without those checks.

## Choose the Pod

Use a Linux x86_64 GPU Pod with Python 3.10 or later. DeepSeek Strata uses a
prebuilt CUDA 13 engine and needs a compatible NVIDIA host driver plus
`libseccomp.so.2`. It does not require `nvcc` or a CUDA development image.
Its `--install-deps --prepare-only` installs Python venv support, certificates,
libseccomp, and filesystem utilities on Debian/Ubuntu.

The pinned native executable contains SM86, SM89, and SM120 kernels, plus
SM120 PTX. The launcher accepts those three architectures. A newer driver
does not supply missing engine kernels. H200 (SM90), B200 (SM100), and B300
(SM103) cannot use Strata. This restriction also
applies to their legacy explicit DeepSeek profile names. No source build or
alternate DeepSeek engine is selected automatically. A single B300 uses the
separate Mistral profile described below. Multiple B300s and mixed B300
allocations have no automatic profile.

For Mistral, Qwen, Flint, Ornith, and MiniMax profiles, select a **CUDA development** image.
Their launcher path compiles llama.cpp on the Pod. It requires `nvcc`; a CUDA runtime
image is insufficient. `--install-deps` installs Debian/Ubuntu build tools but
does not install CUDA or change the NVIDIA driver. For Blackwell, the launcher
requires CUDA toolkit 12.8 or later; B300 requires 12.9 or later. Check
`nvcc --version`, not the CUDA version printed by `nvidia-smi`.

Select the GPU type, GPU count, host RAM, and disk capacity separately. Each
multi-GPU profile requires GPUs on one host. Separate Pods cannot combine their
VRAM. The launcher rejects MIG/vGPU allocations. Use RunPod's CUDA compatibility
filter for the selected image. See [Choose a Pod](https://docs.runpod.io/pods/choose-a-pod)
and [Manage Pods](https://docs.runpod.io/pods/manage-pods).

`--profile auto` selects DeepSeek on supported Strata allocations and Mistral
on a single B300. It uses the
detected total VRAM to make the choice stable while another server runs.
Startup checks current free memory separately. Restrict `CUDA_VISIBLE_DEVICES`
or use `--gpu-count` to select fewer GPUs deliberately.

The planner checks exact packed layer sizes from the pinned GGUF headers.
It assigns contiguous layers to each GPU, the output head to the last GPU,
and the DSpark draft to the first GPU. Input embedding rows come from host
memory. Native serving uses `--tp 1` so the placement matches this calculation.
It allows 3.5 GiB per GPU for native allocations at 1M context, plus the
configured reserve, which defaults to 2 GiB. The 3.5 GiB allowance comes from
the quad-A40 observations. It is not a measured bound for every GPU or flag.

It selects the highest precision in this pinned catalog that fits those
per-GPU budgets: Unsloth Q8_K_XL, Q4_K_XL, Q3_K_XL, IQ3_S, IQ3_XXS, then
GSQ IQ2_XS, IQ2_XXS, and IQ1_M. It adds DSpark only if the selected weights
and draft fit together. It does not reduce precision just to add a draft.
If no candidate fits, it selects GSQ IQ1_M with CPU/RAM/SSD expert streaming.
That choice minimizes weight traffic; it is not a speed benchmark.
Explicit `--quant` and `--spec` can override the choice.
Four A40 cards use the separate Dynamic V3.1 IQ1_M profile above.
The Q2_K_XL candidate is excluded because its Q5_K Engram is not supported
by the pinned Strata engine. The new Unsloth IQ2_XXS has the same restriction.

These examples use 46,068 MiB per A40 and 97,887 MiB per PRO 6000. Actual
reported capacity can change a selection. DeepSeek defaults to **1M** context.
Mistral on B300 defaults to **512K**.

| Visible GPUs | Automatic weights | DSpark | Capacity status |
| --- | --- | --- | --- |
| 1 RTX 4090 or RTX 5090 | GSQ IQ1_M | Off | CPU/SSD experts required |
| 1 supported 48 GB card | GSQ IQ1_M | Off | CPU/SSD experts required |
| 2 A40 | GSQ IQ1_M | Off | CPU/SSD experts required |
| 3 A40 | GSQ IQ1_M | Off | Estimated GPU fit |
| 4 A40 | Unsloth V3.1 UD-IQ1_M | On | Estimated GPU fit, split 10,12,9,9; not benchmarked |
| 5 A40 | GSQ IQ2_XS | On | Estimated GPU fit; Q5_K Engram candidate excluded |
| 6 A40 | Unsloth UD-IQ3_XXS | On | Estimated GPU fit |
| 7 A40 | Unsloth UD-IQ3_S | On | Estimated GPU fit |
| 8 A40 | Unsloth UD-Q8_K_XL | Off | Estimated GPU fit; draft exceeds a stage budget |
| 1 RTX PRO 6000 Blackwell | GSQ IQ1_M | Off | CPU/SSD experts required |
| 2 RTX PRO 6000 Blackwell | GSQ IQ2_XS | On | Estimated GPU fit |
| 3 RTX PRO 6000 Blackwell | Unsloth UD-Q3_K_XL | Off | Estimated GPU fit; draft exceeds a stage budget |
| 4 RTX PRO 6000 Blackwell | Unsloth UD-Q8_K_XL | On | Estimated GPU fit |
| 1 B300 | Mistral UD-IQ1_S | On, 3 draft tokens | Estimated GPU fit at 512K; separate llama.cpp runtime |
| H200 or B200 | None | None | Missing kernels in the pinned Strata engine |

These are capacity candidates, not a performance ranking. Quants from different
publishers do not form a measured quality ranking merely from their names.
The [GSQ model card](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
reports its benchmark results. The [Unsloth analysis](https://unsloth.ai/docs/models/deepseek-v4.1-flash#quantization-analysis)
reports probability agreement with the base checkpoint. Its Q8 label retains
native MXFP4 experts and uses BF16 for other weights; it is not uniformly 8-bit.

Here K and M use powers of 1024. The smallest GSQ download is about 231 GB,
including the shared Engram. The largest listed Unsloth download is about
508 GB. DSpark adds about 8 GB. Allow at least 20 GiB of free disk reserve,
plus space for the runtime and temporary downloads.

Other model families remain explicit choices. For example, `--profile qwen`
selects Qwen3.8-27B and its original context and MTP defaults. Automatic
selection never changes to another model family after a compatibility or
capacity failure.

Host RAM is separate from VRAM. Qwen/Flint require 16 GiB available host RAM
and Ornith requires 32 GiB with their default prompt cache. MiniMax calculates
additional host requirements from CPU expert placement. Leave room for the OS
and builds. For DeepSeek, use the native cache and headroom rules above.
Large Engram tables can make random disk reads dominate inference when RAM is
limited. Preview and GPU capacity checks do not establish throughput.

MiniMax remains an explicit text-only experiment and requires
`--accept-experimental`. The official Strata DeepSeek path does not require
that old fork-acceptance flag. The parser retains it for older command lines.

## Allocate storage

Use a volume disk mounted at `/workspace` for the working model files when
possible. A volume disk survives a stop but is deleted when the Pod is
terminated. A network volume persists independently of the Pod and replaces
the volume disk at `/workspace`. The container disk is cleared when the Pod
stops. See RunPod's [storage options](https://docs.runpod.io/pods/storage/types).

`remotemodel` defaults to `/workspace/remote-models` for weights, runtimes,
profile records, and logs. Build archives default to `/workspace/archive`.
Set `--root` or `REMOTEMODEL_ROOT` to change the data root; the existing
`QWEN_ROOT` environment variable is also supported. Keep the same root to
reuse prepared files. The legacy option-only script interface retains its
`/opt/qwen` default, which is usually on the disposable container disk.
For an initial DeepSeek run, budget at least 300 GB of working storage. Higher precision
profiles need their model-file total plus build space, drafts, and at least the
launcher's 20 GiB free-space reserve. Quantization variants occupy separate
files and can require hundreds of additional GB. Also allocate container disk
space for the development image and system packages.

The launcher rejects detected network filesystems by default because models
use memory mapping. A RunPod network volume can trigger that check. Either
keep the active model directory on local SSD/NVMe with `--model-dir`, or accept
the storage trade-off explicitly with `--allow-network-storage` on both the
preview and launch commands. Network-backed Engram random reads can be slow.
A successful preview checks filesystem type; it does not measure storage
throughput or perform the later manifest-based free-disk check.

## Install remotemodel once and preview

Add your public SSH key in RunPod and select a template with full SSH access.
Use the **SSH over exposed TCP** command from the Pod's Connect tab. It gives
the current public IP and mapped SSH port. Basic `ssh.runpod.io` access does
not provide the connection needed for SCP and this TCP tunnel. Official RunPod
templates can provide full SSH already. Custom templates need an SSH daemon
and exposed TCP port 22. See [RunPod SSH](https://docs.runpod.io/pods/configuration/use-ssh).

On your computer, replace `PORT` and `IP` with those current values. Run these
commands from the repository root:

```sh
scp -P PORT -i ~/.ssh/id_ed25519 start-remote.sh root@IP:/workspace/start-remote.sh
ssh -p PORT -i ~/.ssh/id_ed25519 root@IP
```

On the Pod, keep the bootstrap on persistent storage and link the command:

```sh
chmod +x /workspace/start-remote.sh
mkdir -p ~/.local/bin
ln -s /workspace/start-remote.sh ~/.local/bin/remotemodel
export PATH="$HOME/.local/bin:$PATH"
remotemodel --help
remotemodel list
remotemodel start deepseek --dry-run
```

Add the `PATH` setting to your shell configuration. A root user can link
`/usr/local/bin/remotemodel` instead. If the container is replaced, recreate
the link to the retained `/workspace/start-remote.sh` file.

Each invocation resolves `TharkunAB/localmodel`'s latest published `main`
commit. The bootstrap verifies the Git blob hash of each required remote
and shared file at that exact commit. Code is cached under
`/workspace/remote-models/launcher/<commit>`; `REMOTEMODEL_CODE_DIR` can select
a different code-cache directory. Fetch or verification failure stops the
bootstrap. It does not silently run an older cached release.

Changes must be published to `main` before this bootstrap can fetch them.
A local checkout or branch does not change the published code. Future updates
need no new SCP command. Keep weights and runtime directories across updates;
the new code reuses prepared files that still match its pins.

The Pod needs Bash, Python 3.10 or later, `/dev/fd`, and outbound HTTPS access
for the bootstrap. Every bootstrap invocation checks GitHub, including help,
status, stop, preview, and `--no-fetch`. These commands can run without that
check through the fixed source checkout described below.

`remotemodel start deepseek --dry-run` reads GPU inventory and prints the plan
and commands. It does not prepare files, build, or start a server. With the
bootstrap, only the code fetch/cache occurs first. Inspect the profile,
model revision, GPU placement, context, expert cache, root, and endpoint.
A plan can be inspected while the GPU or API port is busy. A low-RAM preview
reports `ready: false`; preparation and serving enforce their RAM requirements.
A successful preview is not a remote GPU inference or throughput test.

## Start, fetch, inspect, and stop

Use `tmux` so SSH disconnects do not interrupt preparation or inference. Install
`tmux` separately if the image lacks it; `--install-deps` does not install it.

```sh
# Optional: prepare weights, runtime, and monitor before loading a model.
remotemodel fetch deepseek --tui --install-deps

tmux new -s remote-model
remotemodel start deepseek --tui
```

`start` runs preparation before serving. It reuses verified weights, runtimes,
and builds; missing files are fetched automatically. Preparation can install
pinned runtime and monitor dependencies. OS package installation needs the
explicit `--install-deps` flag. `fetch` performs preparation and exits without
loading a model. An existing server must stop before its runtime is updated.

The serving process and its children run under the existing outgoing-network
restriction after preparation finishes. Serving does not download or build.
Use `--no-fetch` to skip preparation and require prepared files:

```sh
remotemodel start deepseek --tui --no-fetch
```

This flag does not disable the bootstrap's GitHub request. Use a fixed source
checkout to start fully offline. The monitor must already be prepared when
combining `--tui` with `--no-fetch`.
The shared TUI dependencies require Python 3.11 or later with `venv` support.
The launcher checks for that interpreter before preparing model files.

The Strata monitor is read-only. Press `t` to cycle terminal, dark, and light
colors. Press `q` to close the display while the server stays in the foreground.
Press Ctrl-C to stop the server. `--tui` is rejected for non-Strata profiles.
Detach from tmux with Ctrl-b then d; reconnect with `tmux attach -t remote-model`.

In another terminal on the Pod:

```sh
remotemodel status
remotemodel stop
```

These commands inspect or stop only a server owned by `remotemodel`. Use the
same `--root` and `--port` when the server uses non-default values. `stop`
checks the saved process identity before sending a signal. It does not stop
an unrelated process that happens to use the port.

Model names select the existing backend profiles:

| Model name | Selection |
| --- | --- |
| `auto` | Existing hardware detector: DeepSeek/Strata on supported GPUs, Mistral on one B300. |
| `deepseek` | DeepSeek-only hardware detection. Unsupported Strata GPUs fail. |
| `mistral` | Mistral Large 4 on one B300. |

```sh
# No model name also selects auto.
remotemodel start
remotemodel start auto
remotemodel start mistral --context 512K
# Keep a specific DeepSeek profile and existing prepared files.
remotemodel start deepseek --profile deepseek-a40-quad --root /opt/qwen --tui
```

The API binds to `127.0.0.1:8897`. `--port` changes the port; there is no host
flag. Existing controls such as `--profile`, `--quant`, `--context`,
`--spec`, `--dspark`, `--layer-split`, `--expert-ram-gb`, `--vram-reserve-mib`,
and build-archive flags pass to the backend. Their existing profile restrictions
still apply. Use `remotemodel --backend-help` for the complete backend flags.
Build identities and archives retain their existing verification rules.
No context, precision, or model is silently reduced after a failed start.

The new CLI does not set up SSH by default. RunPod's SSH service remains the
connection used by these examples. The legacy backend still exposes
`--setup-ssh` and `--ssh-only` for explicit SSH administration. Those options
require root and existing or supplied public keys. They can create host keys,
append public keys from `SSH_PUBLIC_KEY` or `PUBLIC_KEY`, and start an installed
key-only SSH daemon with TCP forwarding. They retain existing configuration
and keys; they do not install the daemon.

## Use a fixed payload or start offline

A source checkout provides the new CLI without bootstrap updates. Select and
retain a reviewed commit, then invoke its command directly on the Pod:

```sh
cd /workspace/localmodel
remote/bin/remotemodel fetch deepseek --tui --install-deps
# Later, after preparation, with no bootstrap request:
remote/bin/remotemodel start deepseek --tui --no-fetch
remote/bin/remotemodel status
remote/bin/remotemodel stop
```

Retain both `remote/` and the shared files under `local/` from the same commit.
The source checkout does not need local model weights, workstation runtimes,
OMP, or systemd. It needs the shared monitor module and dependency lock when
`--tui` is used. Runtime data still defaults to `/workspace/remote-models`.

The old standalone payload also remains usable. Copy it directly when you
need a single file with the exact backend version from a checkout:

```sh
# On the workstation, from the selected checkout's repository root:
scp -P PORT -i ~/.ssh/id_ed25519 remote/start-remote-model.sh \
  root@IP:/workspace/start-remote-model.sh
```

Prepare with that file while internet access is available, then serve:

```sh
# On the Pod:
bash /workspace/start-remote-model.sh --profile auto \
  --root /workspace/remote-models --no-ssh --install-deps --prepare-only
bash /workspace/start-remote-model.sh --profile auto \
  --root /workspace/remote-models --no-ssh
```

The direct file does not update itself from GitHub. DeepSeek serving verifies
the prepared runtime and weights offline. Missing preparation causes an error.
The standalone file does not provide the new `--tui` integration. Keep it with
the preparation records when reproducing a run. To update it, copy the desired
payload explicitly and prepare it before restarting the server.

Existing option-only `bash /workspace/start-remote.sh --profile ...` commands
still select the legacy interface. Their defaults and preparation flags remain
unchanged. The examples earlier in this guide retain that form where they
document a specific historical deployment or an advanced backend operation.

For legacy commands, do not combine preview flags with `--archive-build-only`,
`--smoke-test`, or `--smoke-full-context`. Those actions can take precedence over
preview and perform builds or inference. Run them separately from a preview.

## Connect the client

In another terminal on your computer, keep this tunnel running:

```sh
ssh -p PORT -i ~/.ssh/id_ed25519 -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8898:127.0.0.1:8897 root@IP
```

This is `-L 8898:127.0.0.1:8897` with an explicit local loopback bind.
The client base URL is `http://127.0.0.1:8898/v1`. Both the remote server and the
local tunnel listen on loopback. There is no need to expose port 8897 through
RunPod's HTTP proxy. Rediscover the IP and mapped SSH port after Pod recreation.

On the Pod, check readiness and the model ID:

```sh
curl --fail http://127.0.0.1:8897/health
curl --fail http://127.0.0.1:8897/v1/models
```

On your workstation, use port 8898 for the same checks. Run `searxng-local`
in a separate terminal, then run `omp-remote --mem 32` from your project.
Both `omp-remote` and `omp-local` belong to the workstation installation under
`local/`; the Pod runs only the payload and model server. See the
[local harness guide](../local/docs/harness.md#start-the-components).

After the server is ready, the optional command below runs on the Pod. It
sends real generation and tool-call smoke requests:

```sh
bash /workspace/start-remote.sh --port 8897 --smoke-test
```

`--smoke-full-context` additionally fills about 95% of the running context and
runs a soak test. Its default time limit is three hours. Use it only when you
intend to spend that GPU time. A configured context is not evidence of usable
full-context throughput.

## Files and restart behavior

With `--root /workspace/remote-models`, the script uses:

| Path | Contents |
| --- | --- |
| `start.log` under the root | Launcher log |
| `<model-id>/server.log` | Server output |
| `<model-id>/last-launch.json` | Resolved plan and server command |
| `<model-id>/models/<revision>/` | Model and draft files, unless `--model-dir` overrides it |
| `gsq-models/<revision>/` | Shared GSQ model files |
| `runtimes/` | Pinned native Strata files or llama.cpp GPU-specific builds |
| `<model-id>/strata-engine.log`, `strata-server.json` | Native DeepSeek log and API configuration |
| `<model-directory>/strata-pack-<quant>[-dspark]-<source-revision>/` | Native DeepSeek index, tokenizer, and pack receipt |
| `download-venv-1.0.1/`, `hf-home/` | Download environment and metadata |
| `/workspace/archive/` | Default compiled executable archives; configurable with `--archive-dir` |

For the Qwen example, inspect
`/workspace/remote-models/qwen3.8-27b/server.log` and `last-launch.json`.
A restart reuses downloaded files and matching builds. Re-run the preparation
command after a Pod reset if system build tools were lost. Keep the CUDA image
consistent when reusing builds. Ctrl-c stops this launcher's server and retains
model downloads. The script does not stop an unrelated server when a port is
busy.

Runtime commits are fixed in the script. Most checkpoint revisions are also
fixed. Flint resolves the publisher revision once and records
`flint-source-pin.json`. To make a new Flint deployment reproducible from the
first run, set `FLINT_REVISION` to a verified full publisher commit before
preparation. Existing source pins must agree with that value. Keep the source
pin and manifests in the persistent profile root, together with the model files.

A custom Docker image is optional. The standalone launcher runs inside
a RunPod GPU container. Use a template with working SSH and a host driver that
supports the selected engine. Qwen/Flint/Ornith/MiniMax also need CUDA development
tools. The local systemd/cgroup harness is not a
container entrypoint, and a plain Docker port mapping cannot expose this
launcher's loopback-only API.
