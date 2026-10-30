# Local model harness

Run a coding agent against a local model or a model on a rented GPU.
`localmodel` fetches, starts, inspects, and removes model files.
`omp-local` and `omp-remote` discover the model ID and context from the server.
Both use local SearXNG for search and share a process memory limiter.

This package contains launchers, reviewed runtime patches, dependency pins,
model manifests, and build helpers. It excludes weights, installed runtimes,
conversation history, credentials, and private collaboration certificates.

- [Local harness details](docs/harness.md): profiles, subagents, memory limits,
  settings, OpenCode, search, and optional private collaboration.
- [RunPod guide](docs/runpod.md): GPU selection, storage, remote script, SCP,
  SSH forwarding, and troubleshooting.
- [Earlier measurements and reference](docs/reference.md).
- [Source provenance](docs/provenance.md).

## What runs where

| Component | Address | Purpose |
| --- | --- | --- |
| Local model | `127.0.0.1:8899` | Main model for `omp-local`; subagents for `omp-remote` |
| SearXNG | `127.0.0.1:8888` | DuckDuckGo search for both harness profiles |
| Remote SSH tunnel | `127.0.0.1:8898` | Main model for `omp-remote` |
| Model inside the remote Pod | `127.0.0.1:8897` | Target of the SSH tunnel |
| Optional collaboration relay | Tailnet address, port `7467` | Private browser collaboration |

The model APIs use `/v1`. Run one managed local model at a time. Start search,
the model, and the agent separately. The agent does not start these services.

## Install on Ubuntu or WSL2

The packaged local runtimes target **Ubuntu 24.04, Linux x86-64, and NVIDIA**.
The optimized Bonsai build targets **SM 8.9**, as used by the RTX 4070 SUPER.
It is not a portable binary for every GPU. The local CUDA 13 engines require
a compatible NVIDIA driver. Apple Silicon is outside this setup.

On WSL2, install the NVIDIA driver on Windows. Do not install a Linux display
driver inside WSL. Enable systemd in `/etc/wsl.conf` if it is not already on:

```ini
[boot]
systemd=true
```

Apply that WSL change with `wsl --shutdown` from Windows after stopping work.
The RAM limits need a working systemd user manager and cgroup v2 memory control.
See [Microsoft's systemd instructions](https://learn.microsoft.com/windows/wsl/systemd)
and [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

Install the system dependencies:

```bash
sudo apt update
sudo apt install -y build-essential git curl ca-certificates python3.12 \
  python3.12-venv python3.12-dev pkg-config libseccomp2 openssl \
  libffi-dev libssl-dev libxml2-dev libxslt1-dev zlib1g-dev unzip tmux
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Bun](https://bun.com/docs/installation/) using their official instructions.
OMP requires Bun **1.3.14 or later**; the dependency lock was checked with 1.4.2.
Verify the host before setup:

```bash
uv --version
bun --version
nvidia-smi
systemctl --user show-environment >/dev/null
cat /sys/fs/cgroup/cgroup.controllers
```

The last output must include `memory`. Put the checkout on the Linux filesystem,
for example `~/Code/local-models`, rather than `/mnt/c` or `/mnt/d` on WSL.
The Strata SSD workload needs fast local storage.

From the checkout, install the launchers, pinned OMP, and SearXNG:

```bash
cd ~/Code/local-models
bin/setup-localmodel --harness --search --dry-run
bin/setup-localmodel --harness --search --run
export PATH="$HOME/.local/bin:$PATH"
```

Put the `PATH` line in your shell configuration. Setup creates links in
`~/.local/bin`; `--bin-dir DIRECTORY` changes that location. It refuses to
replace unrelated files or links. It creates a project Python environment,
installs OMP in `state/omp-package/`, and installs SearXNG in `search/`.
It does not download models or start services. It does not require a separate
`heavy-management` checkout: the two memory-cap scripts are included.

Install the model backend next. Select only the model families you need.
`--all` installs the Bonsai, Strata, OMP, and search dependencies together.
Setup prints a plan unless you add `--run`.

## Set up Bonsai 2 27B

This is the default local model. The selected preset is
`27b-256k-optimized`: official PTQ1 target weights, the trained ProCreations
MTP head, two draft tokens, chaining, a 16K draft window, Q4 KV, 8192-token
checkpoints, recurrent replay, and chunked prefill. Context capacity is 256K.
The prompt snapshot cache can use 32 GiB of RAM; the default process cap is
40 GiB. The main weights and active attention run on the GPU.

Install the compiler dependencies and build the pinned patched runtime:

```bash
bin/setup-localmodel --bonsai --dry-run
bin/setup-localmodel --bonsai --run
```

This downloads and verifies the source archive, applies the reviewed patch,
checks every recorded source file, and builds with two compiler jobs under
a 16 GiB cap. Builds can take several minutes. Stop managed model jobs before
building. The compiler uses six pinned NVIDIA wheels. For compatibility with
the existing build receipts, that small toolkit lives under `dory/.venv`;
this step does not install vLLM or a Dory model.

Fetch the target and the trained MTP head, then start:

```bash
localmodel fetch 27b-256k-optimized
localmodel start --dry-run
localmodel start
```

The trained-head file preserves the target tensor bytes. The download helper
verifies the required donor ranges and the complete generated file.
See the [head audit](experiments/bonsai-next/head-audit.md).
Allow roughly **15–20 GiB** of model/build space plus room for the CUDA toolkit
and package caches. The retained target and generated model alone use 11.5 GiB.

For the stock runtime without the optimized build or MTP head:

```bash
localmodel fetch bonsai2-27b
localmodel start bonsai2-27b
```

This is a separate configuration; it does not reproduce the optimized timings.

## Set up Qwen3.8 Flash-Next

`qwen3.8` selects **ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ2_XS**, with the
prepared MTP layer. It is not Qwen3.8-27B, which the remote script also supports.

```bash
bin/setup-localmodel --strata --dry-run
bin/setup-localmodel --strata --run
localmodel fetch qwen3.8
localmodel start qwen3.8 --dry-run
localmodel start qwen3.8
```

Fetch also prepares the Qwen engine and MTP artifacts. The defaults are
**256K context, a 64 GiB process cap, 32 GiB resident expert budget, INT8 KV,
three draft tokens, and a 700 MiB VRAM reserve**. Strata's native speculation
value includes the accepted token, so the launcher maps three drafts to four.
Vision, suffix drafting, and routing priors are disabled.

The prepared model uses about **71.25 GiB**. Allow **at least 90 GiB free**
for an initial fetch, preparation, and runtimes. The RAM cap includes more than
the expert budget: cache, page cache charged to the process group, and runtime
buffers also count. Leave RAM for the OS and the coding agent.

Example overrides:

```bash
localmodel start qwen3.8 --context 128k --mem 48G --expert-ram-gb 24
localmodel start qwen3.8 --context 256k --draft-tokens 2 --prefill-chunk 4096
```

Stop the first server before starting another. These overrides are tuning
examples, not claims that they outperform the defaults.

## Set up DeepSeek V4.1 Flash 1017

`deepseek` selects Unsloth **Dynamic V3.1 UD-IQ1_S** with an **IQ4_NL Engram**
table, pinned to revision `8a6ec9cf118e785a362ea94ee123c61c70553be6`.
Strata keeps selected experts in RAM and reads other
weights from SSD. This is how the larger model runs on the 12 GiB card.

```bash
bin/setup-localmodel --strata --run
localmodel fetch deepseek
localmodel start deepseek --dry-run
localmodel start deepseek
```

The defaults are **256K context, a 64 GiB process cap, a 36 GiB resident expert
budget, 4096-token prefill chunks, four CPU expert threads, and all six active
experts**. DSpark is optional. Allow **at least 250 GiB free**
on a fast Linux SSD: the weights and prepared pack use about 217 GiB.
The complete three-model collection uses about 300 GiB before toolchains,
runtimes, caches, and spare working space.

Example with less RAM and context:

```bash
localmodel start deepseek --context 64k --mem 48G --expert-ram-gb 36
```

Fetch and enable the optional DSpark drafter:

```bash
localmodel fetch deepseek --speculative dspark
localmodel start deepseek --speculative dspark --draft-tokens 5
```

The drafter adds about **7.43 GiB** on disk. Strata can keep draft experts
in RAM when they do not fit in VRAM. This can reduce the main model's GPU
cache and increase SSD reads. A speed gain is not guaranteed.
Depth 5 is the default when DSpark is selected. A 16K OMP comparison reached
4.60 tok/s with DSpark and 4.98 without it. Both runs passed the repair,
100 assertions, recall, and resume checks. DSpark remains optional.
See the [measurements and limits](experiments/deepseek-strata/README.md#16k-omp-comparison-2026-10-30).
Use `--speculative none` to unload the drafter.
Use `localmodel remove deepseek --speculative dspark` to remove only the
draft weights and their pack. This keeps the main model.

Both Strata profiles use pinned control source **0.2.7.1** and signed native
engines **0.1.54** from Strata 0.2.8. The server remains on 0.2.7.1 because
the 0.2.8 parser can treat a quoted tool example as a tool call.
See the [update review](docs/strata-0.2.8-review.md).
The DeepSeek packer uses the same pinned 0.2.7.1 source.
No native compilation is required for Strata.
Its server and children cannot open outgoing connections; fetch and installation
remain network-enabled. The local profile defaults to **262144** context
tokens. A remote model's advertised 1M capacity is handled separately.

## Start the whole local setup

Use three terminals or panes:

```bash
# Terminal 1: search.
searxng-local

# Terminal 2: choose one model and wait for it to become ready.
localmodel start              # Bonsai 2 27B
# localmodel start qwen3.8
# localmodel start deepseek

# Terminal 3: check services, then open the agent in your project.
curl --fail http://127.0.0.1:8888/healthz
localmodel status
cd /path/to/your/project
omp-local --mem 32
```

`--mem 32` caps OMP and its child processes together at 32 GiB, with swap
disabled. It does not cap the model or SearXNG processes started in the other
terminals; those have their own limits. Keep `--mem` before OMP's other flags.
Exceeding the cap can cause the kernel to kill a process in the group.
It is not a catchable application error for every command.

OMP defaults to `xhigh` reasoning. Search must return `200 OK` from `/healthz`
or the launcher refuses to start. Web fetch uses OMP's native reader.
Settings made in OMP remain in each profile's private settings file.
See the [profile, settings, and skills guide](docs/harness.md).

Stop a model with `Ctrl-C` in its terminal or `localmodel stop`.
Stop SearXNG with `Ctrl-C`. The wrappers do not start models in the background.

## Use a RunPod model

The supplied `start-qwen.sh` is tracked as
[`bin/start-remote-model.sh`](bin/start-remote-model.sh). Automatic selection uses
DeepSeek V4.1 Flash 1017 on pinned official Strata for supported GPUs.
A single B300 selects **Mistral Large 4 UD-IQ1_S**, llama.cpp, DSpark,
and **512K context**. Both serving paths block outgoing connections.
DeepSeek selects weights and DSpark from the GPU capacity and defaults to
1M context. Qwen, Flint, Ornith, and MiniMax require explicit profile selection.
All profiles serve on Pod loopback port **8897**.

Choose a template with direct SSH over TCP. DeepSeek needs Python and a driver
compatible with CUDA 13; its prebuilt engine does not need a compiler.
The other profiles need a CUDA **development** image. The basic
`ssh.runpod.io` gateway is unsuitable for this SCP/forwarding workflow.
The pinned Strata engine supports SM86, SM89, and SM120. These include A40,
RTX A6000, RTX 4090, RTX 5090, and RTX PRO 6000 Blackwell. H200, B200, and B300
cannot run this Strata engine. H200 and B200 are rejected. A single B300 uses
the separate Mistral profile and requires a CUDA 12.9 or later development image.
Its capacity estimate and launch flags are checked; GPU inference is not yet tested.
GPU VRAM alone is insufficient: see the [RunPod selection guide](docs/runpod.md).
For an existing DeepSeek Pod, follow its [Strata migration steps](docs/runpod.md#migrate-an-existing-deepseek-pod).

Use the IP, mapped SSH port, and key shown in the Pod's Connect panel:

```bash
# On your local machine, from this checkout:
scp -P PORT -i ~/.ssh/id_ed25519 bin/start-remote-model.sh \
  root@IP:/workspace/start-remote-model.sh
ssh -p PORT -i ~/.ssh/id_ed25519 root@IP

# On the Pod:
tmux new -s model
bash /workspace/start-remote-model.sh --profile auto \
  --root /workspace/remote-models --no-ssh --show-config
bash /workspace/start-remote-model.sh --profile auto \
  --root /workspace/remote-models --no-ssh --install-deps --prepare-only
bash /workspace/start-remote-model.sh --profile auto \
  --root /workspace/remote-models --no-ssh
```

Use `--root /workspace/remote-models` because the script's default `/opt/qwen`
can be on ephemeral container storage. A RunPod network volume also needs
`--allow-network-storage`, and random reads can be slower. The guide explains
volume disks, network volumes, sizing, and experimental launch flags.
`--no-ssh` leaves the Pod template's SSH service in control.

In another local terminal, create the requested tunnel:

```bash
ssh -p PORT -i ~/.ssh/id_ed25519 -N \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:8898:127.0.0.1:8897 root@IP
```

This is `-L 8898:127.0.0.1:8897` with an explicit local loopback bind.
Keep the tunnel running. Then check discovery and open OMP:

```bash
curl --fail http://127.0.0.1:8898/v1/models
omp-remote --mem 32
```

`omp-remote` also requires SearXNG on **8888** and a local subagent model on
**8899**. It discovers that local model dynamically, including our Qwen
Flash-Next profile. There is no cached-model fallback. Its legacy `bonsai`
provider name does not restrict which model can be served there.
The main remote model can advertise 1M context; OMP uses the discovered capacity.
Output reserves at most one quarter of that context. At 256K, output is limited
to 65536 tokens and automatic compaction starts at 180224 tokens.
Relaunch OMP after switching a remote server profile.

## Inspect, tune, remove, and package

```bash
localmodel list
localmodel presets
localmodel --help
localmodel start --help
localmodel --stats
localmodel start qwen3.8 --context 128k --dry-run

# Inspect the exact removal first. Stop the model before removal.
localmodel remove qwen3.8 --dry-run
localmodel remove qwen3.8
localmodel remove deepseek --dry-run
localmodel remove 27b-256k-optimized --dry-run
```

`remove` deletes the selected weights and download data. It retains launchers,
runtime installations, and configuration. Draft variants can require separate
removal; inspect each preview. Measurements live under `--stats`, not `--help`.
They are observations from particular workloads, not throughput guarantees.

Source files live in Git. Runtime and personal data live in ignored directories:
`models/`, `bonsai/models/`, `state/`, `search/`, `.venv/`, and `results/`.
Back up OMP settings and sessions separately if you need them; do not add them
to a source archive. Moving already prepared Strata weights can invalidate
absolute-path packs and file-identity receipts. Use a fresh fetch/preparation
at the destination rather than copying only the generated receipt files.

Create a source-only archive from the committed checkout:

```bash
mkdir -p dist
git archive --format=tar.gz --prefix=local-models/ \
  --output=dist/local-models.tar.gz HEAD
```

The archive contains committed files only. It has no models or conversations.
Optional OpenCode and Tailnet collaboration setup are in the
[harness guide](docs/harness.md). Older Dory and legacy Bonsai profiles require
their separate historical installations; the fresh-install path above covers
the current Bonsai 2 27B, Qwen Flash-Next, and DeepSeek configurations.

Docker is not required. The local wrappers rely on host systemd scopes to cap
agent children. RunPod already supplies the remote container; use a CUDA 13
compatible template with SSH for Strata. A single Dockerfile would not reproduce both
the local process limits and the remote SSH workflow.
