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
three draft tokens, and a 768 MiB VRAM reserve**. Strata's native speculation
value includes the accepted token, so the launcher maps three drafts to four.
Vision, suffix drafting, and routing priors are disabled.
The stock 0.2.9 API enables the Linux CUDA GDN defaults through
`STRATA_GDN_TC=1` and `STRATA_GDN_CONVL2=2`. The tensor-core path needs about
57.5 MiB of extra GPU scratch space. See the
[kernel review](docs/strata-0.2.9-review.md#qwen-linux-cuda-gdn-defaults).

The 2026-11-05 matched retest found 2.25 to 2.82 times faster prefill on 0.2.11
than 0.2.9, with approximately unchanged decode. A longer default run reached
53.10 decode and 1548.20 new-suffix prefill tokens/s at about 128K actual input.
All coding, recall, and strict JSON resume checks passed.

An optional 64K GPU KV window averaged 70.05 decode tokens/s across 16K to 128K
input. It keeps full 256K capacity and stores the larger KV cache in RAM:

```bash
localmodel start qwen3.8 --kv-vram-cells 65536
```

That run passed coding and recall, but its four resume answers used Markdown
fences around otherwise correct JSON. The strict format checks failed. Defaults
remain unchanged. See the [measurements and controls](docs/qwen-0.2.11-tuning.md)
for memory use, MTP depth, suffix lookup, CPU workers, and benchmark limits.
The [older 0.2.9 results](docs/strata-0.2.9-review.md#local-measurements) remain available.

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

`deepseek` and `deepseek-coder` select **ISTA GSQ-RCO Coder IQ2_XXS**,
pinned to revision `54229d38d588a66d1c6cff08af7b281b9326c656`.
It retains 192 of the original 384 routed experts per layer. Strata keeps
selected experts in VRAM and RAM and reads other weights from SSD.

```bash
bin/setup-localmodel --strata --run
localmodel fetch deepseek
localmodel start deepseek --dry-run
localmodel start deepseek
```

The defaults are **256K context, a 64 GiB process cap, zero private expert cache,
4096-token prefill chunks, twelve CPU workers, and all six selected experts**.
CPU experts remain file-backed. Linux uses spare RAM to cache their pages and
can reclaim those pages for other programs. GPU allocations stay in VRAM.
Engram remains SSD-backed. `--expert-ram-gb 32` selects a dedicated 32 GiB
expert cache. It is separate from the total process cap.
Conversation checkpoints are retained. Coder currently runs without DSpark.
`omp-local` discovers its identity and context from the running server.

Allow **200 GiB free** on a fast Linux SSD for a new installation. Coder uses
about 170.2 GiB, including the 103 GiB Engram file. If the full profile already
has that verified Engram file, fetch reuses it through a hard link. The new
transformer files need 67.2 GiB plus the downloader's 10 GiB reserve.

The 2026-11-05 comparison uses these defaults and growing 16K/32K/64K OMP input.
Strata 0.2.11 averaged **12.46 decode tokens/s**, versus **12.50** on 0.2.10.
Repair and resume took **372 seconds** in each run. All quality checks passed.
Both reached the 64 GiB cap without OOM or model swap. This single pair shows
no material throughput change. See the [report](docs/strata-0.2.11-coder.md).

The earlier Strata 0.2.10 resident repeat at 64 GiB averaged **13.80 decode tokens/s**, versus
**13.25** on 0.2.9. Prefill was **483/618/757 tokens/s** at growing 16K/32K/64K
input. The 70 GiB follow-up averaged **14.04 tokens/s** and kept all routed
experts in RAM or VRAM. That was only a 1.7% decode gain over 64 GiB.
Use `--expert-memory resident --mem 70G` to select that full-residency configuration.
The file-backed test at 70 GiB averaged **12.80 tokens/s** when quiet and
**6.40** under controlled memory pressure. A 32 GiB private cache averaged
**9.08** and **9.14**, respectively. The default keeps the existing 64 GiB cap;
these sharing measurements used 70 GiB. See the [memory-sharing comparison](docs/coder-elastic-memory.md).

The local [Strata release watch](docs/strata-watch.md) checks for stable releases
every six hours. It queues updates and guarded Coder measurements when resources
are available. It does not interrupt a running model or deploy to a remote server.
A daily Grok search checks public X reports for useful optimizations. Local
configuration and measurements stay out of the search request.

Every final code, recall, and resume check passed. Output lengths and recovered
tool errors differed, so this does not establish faster completed coding tasks.
No run used swap or hit OOM. These tests do not fill 256K or test automatic
compaction. See the [earlier resident comparison](docs/strata-0.2.10-coder.md).

The original matched comparison used native 0.1.55 and the same 64 GiB cap
and 48 GiB demand cache. Coder averaged **11.24 decode tokens/s** against
**8.16** for full Unsloth UD-IQ1_S. See the
[model comparison and quality limits](docs/deepseek-coder-comparison.md).

The larger Unsloth **Dynamic V3.1 UD-IQ1_S** profile remains available as
`deepseek-full` or `deepseek-iq1-s`. Its default cap is 70 GiB and its expert
cache is 48 GiB. Its transformer shards need about 114 GiB in addition to
the shared Engram. Stop the running server before switching.

```bash
localmodel fetch deepseek-full
localmodel start deepseek-full
```

If there is not enough disk space, first use `localmodel remove deepseek-coder`.
This preserves an Engram link in the full profile when that link exists.
The downloader requires the remaining download size plus 10 GiB free.
The optional full-profile DSpark files can be removed separately to recover
another 7.43 GiB. Original tuning results remain in the
[full-profile report](docs/deepseek-029-tuning.md).

Strata keeps one saved conversation in RAM. It uses a 2 GiB disk-cache budget
per weights/runtime/context configuration under `state/deepseek-slots/`.
The newest slot can exceed that disk budget. Cache directories are private.
The cache contains local inference state, including prompt tokens.
It survives a server restart.
Use `--no-strata-slots` to disable retention. This does not delete saved files.
Use `--cpu-threads 8` to leave more CPU capacity for other programs.

`--mem` sets the hard process-tree cap and disables swap.
`--expert-memory resident` uses bounded copies of variable-size experts.
`--expert-memory cache` uses the separate `--expert-ram-gb` demand-cache budget.
The cache-mode default is zero for Coder and 48 GiB for `deepseek-full`.
Omit `--expert-ram-gb` in resident mode.
Use `--expert-memory cache --expert-ram-gb 0` to let Linux cache and reclaim
CPU expert file pages. The [memory-sharing test](docs/coder-elastic-memory.md)
measures the throughput cost, reclaim under pressure, and recovery.

Example with less RAM and context:

```bash
localmodel start deepseek --context 64k --mem 48G --expert-memory cache --expert-ram-gb 36
```

Fetch and enable the optional DSpark drafter for the full profile:

```bash
localmodel fetch deepseek-full --speculative dspark
localmodel start deepseek-full --speculative dspark --draft-tokens 5
```

The drafter adds about **7.43 GiB** on disk. Strata can keep draft experts
in RAM when they do not fit in VRAM. This can reduce the main model's GPU
cache and increase SSD reads. A speed gain is not guaranteed.
Depth 5 is the default when DSpark is selected. A 16K OMP comparison reached
4.60 tok/s with DSpark and 4.98 without it. Both runs passed the repair,
100 assertions, recall, and resume checks. DSpark remains optional.
See the [measurements and limits](experiments/deepseek-strata/README.md#16k-omp-comparison-2026-10-30).
Use `--speculative none` to unload the drafter.
Use `localmodel remove deepseek-full --speculative dspark` to remove only the
draft weights and their pack. This keeps the main model.

Local DeepSeek, Qwen, and remote DeepSeek use pinned source, API, and packer
**0.2.11**, Control **0.1.57**, and signed native engine **0.1.57**.
Qwen keeps MTP with three draft tokens. Coder keeps zero private expert cache.
See the [runtime verification](docs/strata-0.2.11-review.md).
A crafted parser example can still turn quoted text into a tool call.
This has not been observed in the OMP workloads. The client must still
enforce tool permissions. See the [Coder update and comparison](docs/strata-0.2.11-coder.md).
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
