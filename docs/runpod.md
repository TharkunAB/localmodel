# Run the remote launcher on RunPod

`bin/start-remote-model.sh` is the supplied standalone rental-GPU launcher.
It embeds Python and does not require the local harness, systemd, or
`heavy-management`. Automatic selection serves **DeepSeek V4.1 Flash 1017**.
Other models require an explicit profile. The legacy `qwen` profile serves
Qwen3.8-27B; the local harness uses a separate Qwen3.8-Flash-Next profile.

All remote DeepSeek profiles now use the unmodified official Strata source
`46c152a1310f77cd2f0ce2c7cd213962df14fe83` (0.2.7.1), Control 0.1.53, and signed
CUDA 13 engine 0.1.52. The standalone file embeds the artifact pins, signature
verification, and the same outgoing-network restriction as the local harness.
Qwen, Flint, Ornith, and MiniMax keep their explicit llama.cpp paths.

The original `start-qwen.sh` attachment has SHA-256
`955915e9fc13bc9ba44a580285cd21a70a0c183c2fafb514e3e1c274326b8d3a`.
The current embedded version is `2026-10-29-deepseek-strata-auto-1`.
Packaging checks do not establish remote GPU inference or throughput.

## Migrate an existing DeepSeek Pod

Copy the updated script onto the Pod. The quad-A40 profile now defaults to
**IQ2_XS, 1M context, DSpark on, and layer counts 9,11,10,10**. It still selects
six experts per token. After restart, the first 13 normal requests generated
5,302 tokens in 95.468 decode seconds: 55.5 tokens/s. The previous ordinary
IQ2_XXS run averaged 37.1 tokens/s. These requests used different prompts and
context lengths, so this is an observation, not a controlled comparison.
The new requests used 11K–38K tokens. They did not fill the 1M capacity.

For an existing prepared runtime under `/opt/qwen`, stage the new assets while
the current server continues to run:

```sh
bash /workspace/start-remote-model.sh --profile deepseek-gsq-a40-quad \
  --root /opt/qwen --no-ssh --prepare-only --prepare-assets-only --dry-run
bash /workspace/start-remote-model.sh --profile deepseek-gsq-a40-quad \
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
`strata-pack-iq2_xs-dspark` with `--dspark-gguf`, noise token 128799, and Markov
rank 256. The pack receipt includes every generated index/tokenizer asset and
the draft identity. Unknown pack entries are rejected. The pack references
verified GGUF files; it does not copy or requantize weights. The existing
ordinary pack remains available for rollback.

After you stop the previous server, start the prepared profile:

```sh
bash /workspace/start-remote-model.sh --profile deepseek-gsq-a40-quad \
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
bash /workspace/start-remote-model.sh --profile deepseek-gsq-a40-quad \
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
from its pin, copy the current script and repeat preparation. No data removal
is required.
The existing `/opt/qwen` data is usually on the container disk. Preserve it
before a Pod operation that clears that disk, or prepare a persistent root.

Remove `--cpu-moe-layers 2` from the old command. Strata places experts
automatically across all four selected GPUs. It maps their UUIDs to native
indices `0,1,2,3`. The following controls replace llama.cpp placement controls:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--layer-split` | Planned contiguous counts; `9,11,10,10` on quad A40 | Layer counts per GPU; sum must be 40; `auto` delegates placement to Strata |
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
automatic. The draft stays on the first selected GPU. The quad-A40 default gives that
GPU nine trunk layers to leave space for the draft. The counts come from
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
Copy the current script and repeat `--prepare-only` to update the guard.

The serving environment includes upstream's pinned `psutil` for disk and
process statistics. If the disk card says `needs psutil`, stop the server,
copy the current script, repeat `--prepare-only`, and start the server again.
Refresh the browser after startup. The disk rate covers all reads visible
to the operating system or container; it is not a per-process measurement.

## Choose the Pod

Use a Linux x86_64 GPU Pod with Python 3.10 or later. DeepSeek Strata uses a
prebuilt CUDA 13 engine and needs a compatible NVIDIA host driver plus
`libseccomp.so.2`. It does not require `nvcc` or a CUDA development image.
Its `--install-deps --prepare-only` installs Python venv support, certificates,
libseccomp, and filesystem utilities on Debian/Ubuntu.

The pinned native executable contains SM86, SM89, and SM120 kernels, plus
SM120 PTX. The launcher accepts those three architectures. A newer driver
does not supply missing engine kernels. H200 (SM90), B200 (SM100), and B300
(SM103) therefore fail before preparation or serving. This restriction also
applies to their legacy explicit DeepSeek profile names. No source build or
alternate engine is selected automatically.

For explicit Qwen, Flint, Ornith, and MiniMax profiles, select a **CUDA development** image.
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

`--profile auto` selects DeepSeek on every supported allocation. It uses the
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
per-GPU budgets: Unsloth Q8_K_XL, Q4_K_XL, Q3_K_XL, IQ3_S, IQ3_XXS, Q2_K_XL, then
GSQ IQ2_XS, IQ2_XXS, and IQ1_M. It adds DSpark only if the selected weights
and draft fit together. It does not reduce precision just to add a draft.
If no candidate fits, it selects GSQ IQ1_M with CPU/RAM/SSD expert streaming.
That choice minimizes weight traffic; it is not a speed benchmark.
Explicit `--quant` and `--spec` can override the choice.

These examples use 46,068 MiB per A40 and 97,887 MiB per PRO 6000. Actual
reported capacity can change a selection. Context defaults to **1M** throughout.

| Visible GPUs | Automatic weights | DSpark | Capacity status |
| --- | --- | --- | --- |
| 1 RTX 4090 or RTX 5090 | GSQ IQ1_M | Off | CPU/SSD experts required |
| 1 supported 48 GB card | GSQ IQ1_M | Off | CPU/SSD experts required |
| 2 A40 | GSQ IQ1_M | Off | CPU/SSD experts required |
| 3 A40 | GSQ IQ1_M | Off | Estimated GPU fit |
| 4 A40 | GSQ IQ2_XS | On | Measured working preset, split 9,11,10,10 |
| 5 A40 | Unsloth UD-Q2_K_XL | Off | Estimated GPU fit |
| 6 A40 | Unsloth UD-IQ3_XXS | On | Estimated GPU fit |
| 7 A40 | Unsloth UD-IQ3_S | On | Estimated GPU fit |
| 8 A40 | Unsloth UD-Q8_K_XL | Off | Estimated GPU fit; draft exceeds a stage budget |
| 1 RTX PRO 6000 Blackwell | GSQ IQ1_M | Off | CPU/SSD experts required |
| 2 RTX PRO 6000 Blackwell | GSQ IQ2_XS | On | Estimated GPU fit |
| 3 RTX PRO 6000 Blackwell | Unsloth UD-Q3_K_XL | Off | Estimated GPU fit; draft exceeds a stage budget |
| 4 RTX PRO 6000 Blackwell | Unsloth UD-Q8_K_XL | On | Estimated GPU fit |
| H200, B200, or B300 | None | None | Missing kernels in the pinned engine |

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

Pass `--root /workspace/remote-models`. The unchanged script default is
`/opt/qwen`, which is usually on the disposable container disk. For an initial
DeepSeek run, budget at least 300 GB of working storage. Higher precision
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

## Copy and preview

Add your public SSH key in RunPod and select a template with full SSH access.
Use the **SSH over exposed TCP** command from the Pod's Connect tab. It gives
the current public IP and mapped SSH port. Basic `ssh.runpod.io` access does
not provide the connection needed for SCP and this TCP tunnel. Official RunPod
templates can provide full SSH already. Custom templates need an SSH daemon
and exposed TCP port 22. See [RunPod SSH](https://docs.runpod.io/pods/configuration/use-ssh).

On your computer, replace `PORT` and `IP` with those current values. Run these
commands from the repository root:

```sh
scp -P PORT bin/start-remote-model.sh root@IP:/workspace/start-remote-model.sh
ssh -p PORT root@IP
```

On the Pod, inspect the help and the selected DeepSeek plan:

```sh
bash /workspace/start-remote-model.sh --help
bash /workspace/start-remote-model.sh \
  --profile auto --root /workspace/remote-models --no-ssh --show-config
```

`--dry-run` is an alias for `--show-config`. A native DeepSeek preview reads GPU
inventory, checks storage, and reports the required and available host RAM.
It can inspect a plan while the GPU or API port is busy. A low-RAM preview
reports `ready: false`; preparation and serving enforce their RAM requirements.
Preview does not download, build, or start services. Inspect the printed profile,
model revision, GPU placement, context, cache settings, root, and endpoint.

Do not combine preview flags with `--archive-build-only`, `--smoke-test`, or
`--smoke-full-context`. Those actions take precedence in the supplied script:
an archive action can build and write files, and smoke actions send inference
requests. Preview also rejects `--setup-ssh` and `--ssh-only`.

## Prepare and run

Use `tmux` so SSH disconnects do not interrupt preparation or inference. Install
`tmux` separately if the image lacks it; `--install-deps` does not install it.

```sh
tmux new -s remote-model
bash /workspace/start-remote-model.sh \
  --profile auto --root /workspace/remote-models --no-ssh \
  --install-deps --prepare-only
bash /workspace/start-remote-model.sh \
  --profile auto --root /workspace/remote-models --no-ssh
```

Preparation installs the requested system dependencies, verifies the signed
Strata engine, downloads and verifies the selected weights, and creates the
native pack. `--prepare-only` exits without starting the server. DeepSeek
requires `--prepare-only` with `--install-deps`; serving never installs updates.
Use the same profile and settings for preview, preparation, and launch. Add
`--accept-experimental` only for explicit MiniMax profiles.
Detach with Ctrl-b then d. Reconnect with `tmux attach -t remote-model`.

The API binds to `127.0.0.1:8897`. `--port` changes the port; there is no host
flag. The examples use `--no-ssh` because the existing Pod SSH service is enough.
By default the launcher only checks for an SSH listener. `--setup-ssh` or
`--ssh-only` can create host keys, append public keys from `SSH_PUBLIC_KEY` or
`PUBLIC_KEY`, and start an installed key-only SSH daemon with TCP forwarding.
Those options require root and existing or supplied public keys. The launcher
retains existing SSH configuration and keys. It does not install the daemon.

## Connect the client

In another terminal on your computer, keep this tunnel running:

```sh
ssh -p PORT -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8898:127.0.0.1:8897 root@IP
```

The client base URL is `http://127.0.0.1:8898/v1`. Both the remote server and the
local tunnel listen on loopback. There is no need to expose port 8897 through
RunPod's HTTP proxy. Rediscover the IP and mapped SSH port after Pod recreation.

On the Pod, check readiness and the model ID:

```sh
curl --fail http://127.0.0.1:8897/health
curl --fail http://127.0.0.1:8897/v1/models
```

On your computer, use port 8898 for the same checks. After the server is ready,
the optional command below sends real generation and tool-call smoke requests:

```sh
bash /workspace/start-remote-model.sh --port 8897 --smoke-test
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
| `<model-directory>/strata-pack-<quant>/` | Native DeepSeek index, tokenizer, and pack receipt |
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
