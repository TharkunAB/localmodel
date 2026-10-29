# Run the remote launcher on RunPod

`bin/start-remote-model.sh` is the supplied standalone rental-GPU launcher.
It embeds Python and does not require the local harness, systemd, or
`heavy-management`. Its `qwen` profile serves **Qwen3.8-27B**. The local
harness uses a separate **Qwen3.8-Flash-Next** profile.

All remote DeepSeek profiles now use the unmodified official Strata source
`46c152a1310f77cd2f0ce2c7cd213962df14fe83` (0.2.7.1), Control 0.1.53, and signed
CUDA 13 engine 0.1.52. The standalone file embeds the artifact pins, signature
verification, and the same outgoing-network restriction as the local harness.
Qwen, Flint, Ornith, and MiniMax keep their existing llama.cpp paths.

The original `start-qwen.sh` attachment has SHA-256
`955915e9fc13bc9ba44a580285cd21a70a0c183c2fafb514e3e1c274326b8d3a`.
The current embedded version is `2026-10-29-deepseek-official-strata-2`.
Packaging checks do not establish remote GPU inference or throughput.

## Migrate an existing DeepSeek Pod

Copy the updated script onto the Pod. Keep the existing model root and quant.
For four A40 cards with the existing GSQ IQ2_XXS files under `/opt/qwen`, use:

```sh
bash /workspace/start-remote-model.sh --profile deepseek-gsq-a40-quad \
  --ctxt 256K --root /opt/qwen --no-ssh --prepare-only --dry-run
bash /workspace/start-remote-model.sh --profile deepseek-gsq-a40-quad \
  --ctxt 256K --root /opt/qwen --no-ssh --prepare-only --install-deps
```

Preparation reuses the four pinned IQ2_XXS GGUF shards and shared IQ4_NL Engram
in `/opt/qwen/gsq-models/cb02422c900df489cea5833e8a92171224059b3c` after size and
SHA-256 checks. It downloads missing runtime files and missing model files only.
It runs the stock `tools/dsv41/pack_gguf.py` with the plain recipe. This creates
a small index and tokenizer files; it does not copy or requantize the weights,
load a GPU model, or start a server. No llama.cpp fork runs for DeepSeek.
Preparation does not require a free API port or unused GPUs.

After you stop the previous server, start the same prepared profile:

```sh
bash /workspace/start-remote-model.sh --profile deepseek-gsq-a40-quad \
  --ctxt 256K --root /opt/qwen --no-ssh
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
| `--expert-ram-gb` | `36` | Soft expert RAM cache in GiB |
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
the Pod. A 200 GB RAM allocation has about 186 GiB before other use. Start with
the default cache and inspect memory use before increasing it.

DeepSeek rejects `--cpu-moe-layers`, `--tensor-split`, `--gsq-engram`,
`--engram-ram`, llama.cpp KV/prompt-cache/batch flags, and build-archive flags.
Use a dedicated `deepseek-gsq-*` profile for GSQ; `--ds-source gsq-rco` does not
substitute a different quant into an Unsloth profile. Existing Unsloth quant
choices and GSQ IQ1_M/IQ2_XXS/IQ2_XS choices remain explicit. Remote Strata uses
six experts per token and ordinary decoding. DSpark/MTP/DFlash are rejected.
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

## Choose the Pod

Use a Linux x86_64 GPU Pod with Python 3.10 or later. DeepSeek Strata uses a
prebuilt CUDA 13 engine and needs a compatible NVIDIA host driver plus
`libseccomp.so.2`. It does not require `nvcc` or a CUDA development image.
Its `--install-deps --prepare-only` installs Python venv support, certificates,
libseccomp, and filesystem utilities on Debian/Ubuntu.

For Qwen, Flint, Ornith, and MiniMax, select a **CUDA development** image.
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

Use `--profile` explicitly. These are the supplied auto-selection defaults for
homogeneous allocations; capacities are card labels, not measured free VRAM.

| Visible GPUs | Default profile | Default context | Approximate model files |
| --- | --- | --- | --- |
| 1 RTX 4090, 24 GB | `qwen`, Q4_K_M + MTP | 128K | 19 GB + 1.7 GB draft |
| 1 RTX 5090, 32 GB | `flint`, NVFP4 | 192K | 23 GB |
| 1 supported 48 GB card, including A40/A6000 | `flint`, NVFP4 | 256K | 23 GB |
| 2 A40/RTX A6000, 48 GB each | `ornith`, UD-Q4_K_XL | 256K | 80 GB |
| 3–7 A40/RTX A6000 | `flint` on one selected GPU | 256K | 23 GB |
| 8 or more A40/RTX A6000 | `deepseek`, UD-Q8_K_XL | 256K | 508 GB |
| 4 RTX PRO 6000 Blackwell, 96 GB each | `deepseek-pro6000-quad-q8` | 256K | 508 GB |
| 1 B300, 288 GB | `deepseek-b300-iq3` | 1M | 395 GB |
| 1 B200, 180 GB | `deepseek-b200-iq2` | 256K | 301 GB |
| 1 H200, 141 GB | `deepseek-h200-iq2` | 256K | 301 GB |
| 2 H200, 141 GB each | `deepseek-h200-dual-iq3` | 256K | 395 GB |

Here K and M use powers of 1024. Model-file sizes use decimal GB. They exclude
builds, caches, temporary files, and optional drafts. These are planning values.
The launcher checks free VRAM, available host RAM, and storage before launch.

One to three RTX PRO 6000 Blackwell cards select a blocked FP4-Local path in
`auto` mode. Select `--profile qwen` or another supported profile deliberately.
More than four PRO cards require explicit `--gpu-count 4` or restricted CUDA
visibility. The Qwen profile uses one GPU: Q4_K_M with q4_0 KV on 4090/5090,
and Q8_0 with q8_0 KV on supported 48/96 GB cards. Its default context is 128K
on 4090 and 256K otherwise. It enables MTP at depth 3.

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
Qwen or Flint run, budget at least 80 GB of persistent working storage. Larger
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

On the Pod, inspect the help and the selected plan. This example deliberately
selects Qwen3.8-27B on a supported single GPU:

```sh
bash /workspace/start-remote-model.sh --help
bash /workspace/start-remote-model.sh \
  --profile qwen --root /workspace/remote-models --no-ssh --show-config
```

`--dry-run` is an alias for `--show-config`. A plain preview reads GPU
inventory and checks memory, filesystem type, and the API port. It does not
download, build, or start services. It can fail if another server already uses
8897. Inspect the printed profile, model revision, GPU selection, context,
cache settings, root, and endpoint before continuing.

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
  --profile qwen --root /workspace/remote-models --no-ssh \
  --install-deps --prepare-only
bash /workspace/start-remote-model.sh \
  --profile qwen --root /workspace/remote-models --no-ssh
```

For the Qwen example, preparation installs build tools when requested, builds or restores the pinned
runtime, creates a download venv, downloads and verifies the model files, and
writes the planned launch. `--prepare-only` then exits without starting the
server. `--install-deps` alone continues through preparation into serving.
Use the same profile and settings for preview, preparation, and launch. Add
`--accept-experimental` for MiniMax. Use the separate DeepSeek preparation flow above.
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
