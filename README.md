# Local and remote model harness

Run a coding agent on your workstation with a local model or a model on a
rented GPU. Both `omp-local` and `omp-remote` run on your workstation. They
discover the served model and context, use local SearXNG for search, and apply
a shared memory cap to the agent and its child processes.

```text
README.md         Installation and launch overview
start-remote.sh   Install once as remotemodel; fetch the latest main code
local/            Workstation launchers, configuration, docs, and experiments
remote/           remotemodel CLI, rental-GPU backend, and RunPod guide
```

Source, dependency pins, and reviewed patches are tracked. Weights, installed
runtimes, conversations, credentials, and private certificates are excluded.

- [Local setup and model guide](local/docs/harness.md): dependencies, Bonsai,
  Qwen, DeepSeek, profiles, memory caps, OpenCode, and optional collaboration.
- [RunPod guide](remote/runpod.md): GPU selection, storage, preparation,
  SSH forwarding, fixed payloads, and troubleshooting.
- [Measurements and reference](local/docs/reference.md).
- [Source provenance](local/docs/provenance.md).

## Install the workstation harness

The local setup targets Ubuntu 24.04 on Linux x86-64, including WSL2, with
systemd user scopes and cgroup v2. Local model serving also needs a compatible
NVIDIA GPU and driver. Install the [system dependencies](local/docs/harness.md#install-from-a-fresh-checkout),
Python 3.12, uv, and Bun first. Keep a WSL checkout on the Linux filesystem.

From the repository root:

```bash
cd ~/Code/local-models
local/bin/setup-localmodel --harness --search --dry-run
local/bin/setup-localmodel --harness --search --run
export PATH="$HOME/.local/bin:$PATH"
```

Setup links launchers into `~/.local/bin` and installs dependencies under
`local/`. It does not fetch model weights or start services. It refuses to
replace unrelated files or links. Put the `PATH` line in your shell config.
Select a backend before fetching a local model: `--bonsai` builds the reviewed
Bonsai runtime, and `--strata` installs the pinned Qwen/DeepSeek backend.
See the [model setup steps](local/docs/harness.md#set-up-bonsai-2-27b).

## Use a local model

Run search, one model, and the agent in separate terminals:

```bash
# Terminal 1: search.
searxng-local

# Terminal 2: fetch once, then start the default Bonsai model.
# Install its backend first: local/bin/setup-localmodel --bonsai --run
localmodel fetch 27b-256k-optimized
localmodel start

# Terminal 3: check services, then open the agent in your project.
curl --fail http://127.0.0.1:8888/healthz
localmodel status
cd /path/to/your/project
omp-local --mem 32
```

The model listens on `127.0.0.1:8899`. `--mem 32` caps OMP and its children
at 32 GiB with no swap; model and search processes have separate limits.
Stop the model with Ctrl-C or `localmodel stop`. Stop search with Ctrl-C.
The [local guide](local/docs/harness.md) covers Qwen, DeepSeek, settings, and
measured limits.

## Use a model on a rental GPU

`remotemodel` runs on the rental GPU host. `omp-remote` runs on your workstation.
Choose a Pod with direct SSH over TCP and a compatible CUDA image. The
[RunPod guide](remote/runpod.md) lists supported GPUs and storage requirements.
Automatic selection uses DeepSeek on supported Strata GPUs or Mistral on a
single B300. The existing quantization, DSpark, GPU-placement, and build-cache
settings remain in the remote backend.

Copy the bootstrap once. Use the IP and mapped SSH port from the Pod's
Connect panel:

```bash
scp -P PORT -i ~/.ssh/id_ed25519 start-remote.sh root@IP:/workspace/start-remote.sh
ssh -p PORT -i ~/.ssh/id_ed25519 root@IP
```

On the Pod, install the command and start DeepSeek:

```bash
chmod +x /workspace/start-remote.sh
mkdir -p ~/.local/bin
ln -s /workspace/start-remote.sh ~/.local/bin/remotemodel
export PATH="$HOME/.local/bin:$PATH"

remotemodel list
remotemodel start deepseek --dry-run
# First preparation can install missing OS dependencies when explicitly requested.
remotemodel fetch deepseek --tui --install-deps

tmux new -s model
remotemodel start deepseek --tui
```

`start` prepares missing runtime files and weights, then starts the server.
It reuses verified downloads and builds. Data defaults to
`/workspace/remote-models`; build archives stay under `/workspace/archive`.
Use `start auto` for hardware-based model selection, including Mistral on one
B300. `--tui` is available for Strata: `t` cycles terminal, dark, and light
colors; `q` closes the display; Ctrl-C stops the server. The display shares
its code and dependency lock with the local Strata monitor.

```bash
# These commands run on the Pod, in another terminal if needed.
remotemodel status
remotemodel stop
# Require prepared files; skip model/runtime preparation.
remotemodel start deepseek --no-fetch
```

Each bootstrap invocation resolves the latest published `main` commit from
`TharkunAB/localmodel`. It verifies and caches the required remote and shared
files from that exact commit. Changes must be published to `main` before the
bootstrap can use them. Future updates need no new upload.

The bootstrap needs internet access even for help, status, previews, and
`--no-fetch`. Preparation can download files; the inference process and its
children run with outgoing connections blocked. To use checked-out code
without bootstrap updates, run `remote/bin/remotemodel` from a source checkout.
The [fixed-version workflow](remote/runpod.md#use-a-fixed-payload-or-start-offline)
also retains the standalone `remote/start-remote-model.sh` interface.

In another workstation terminal, keep the tunnel running:

```bash
ssh -p PORT -i ~/.ssh/id_ed25519 -N \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:8898:127.0.0.1:8897 root@IP
```

This is `-L 8898:127.0.0.1:8897` with an explicit local loopback bind.
Start `searxng-local` on the workstation, then open the agent:

```bash
curl --fail http://127.0.0.1:8898/v1/models
cd /path/to/your/project
omp-remote --mem 32
```

`omp-remote` uses the tunnel on port 8898. Search uses port 8888. An optional
local model on port 8899 serves subagents. New conversations use `router/auto`,
which can route requests through idle local Qwen3.8-Flash-Next. Select
`/model model/<served-id>` to keep the main conversation remote. See the
[routing and context rules](local/docs/harness.md#start-the-components).
