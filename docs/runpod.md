# Run the remote launcher on RunPod

`bin/start-remote-model.sh` is the supplied standalone rental-GPU launcher.
It embeds Python and does not require the local harness, systemd, or
`heavy-management`. Its `qwen` profile serves **Qwen3.8-27B**. The local
harness uses a separate **Qwen3.8-Flash-Next** profile.

The packaged script keeps the supplied runtime revisions, checkpoint selection,
and experimental gates. Only its command name, help examples, and log prefix
changed, including the smoke-report filename prefix and the embedded-code
marker. The original `start-qwen.sh` attachment has SHA-256
`955915e9fc13bc9ba44a580285cd21a70a0c183c2fafb514e3e1c274326b8d3a`.
The embedded version remains `2026-10-24-deepseek1017-gsq-profiles-3`.
Packaging checks do not establish GPU inference or throughput.
RunPod documentation links below were checked on 2026-10-28.

## Choose the Pod

Use a Linux GPU Pod with Python 3.10 or later and a **CUDA development** image.
The launcher compiles llama.cpp on the Pod. It requires `nvcc`; a CUDA runtime
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
| 1 H200, 141 GB | `deepseek-h200-iq2`, 18 CPU MoE layers | 256K | 301 GB |
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

Host RAM is separate from VRAM. With the default 8192 MiB prompt cache, the
launcher requires at least 16 GiB available host RAM for Qwen/Flint, 32 GiB for
Ornith, 64 GiB for B200/B300 or multi-Ampere DeepSeek, and 96 GiB for the H200
and quad-PRO DeepSeek profiles. These are minimum available-memory checks, not
recommended Pod RAM sizes. Leave room for the OS and build. DeepSeek also uses
large CPU/SSD Engram tables. Limited RAM can make random disk reads dominate
inference. `--engram-ram` requires the full table plus the calculated host
working set. MiniMax and GSQ calculate additional host requirements from their
CPU expert placement.

All DeepSeek, GSQ, and MiniMax launches require `--accept-experimental`.
Dedicated `deepseek-gsq-*` profiles use the pinned GSQ loader adapter;
`--ds-source gsq-rco` on an ordinary DeepSeek profile remains blocked.
MiniMax profiles are explicit, text-only experiments. Smaller allocations can
require explicit `--cpu-moe-layers`. Their GPU fit, runtime, and throughput
claims remain limited to the checks stated in `--help`. Do not infer successful
inference from a dry run.

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

Preparation installs build tools when requested, builds or restores the pinned
runtime, creates a download venv, downloads and verifies the model files, and
writes the planned launch. `--prepare-only` then exits without starting the
server. `--install-deps` alone continues through preparation into serving.
Use the same profile and settings for preview, preparation, and launch. Add
`--accept-experimental` to preparation and launch for an experimental profile.
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
| `runtimes/` | Pinned runtime sources and GPU-specific builds |
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

A custom Docker image is optional. The standalone launcher already runs inside
a RunPod CUDA development container. An official development template with
working SSH is the shortest setup. The local systemd/cgroup harness is not a
container entrypoint, and a plain Docker port mapping cannot expose this
launcher's loopback-only API.
