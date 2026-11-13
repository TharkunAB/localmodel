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
| Local model | `127.0.0.1:8899` | Main model for `omp-local`; optional Qwen routing and subagents for `omp-remote` |
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
The [DSpine comparison](docs/bonsai-dspine-20261108.md) covers the separate
experimental drafter. Its long OMP results keep MTP2 as the default.
Allow roughly **15–20 GiB** of model/build space plus room for the CUDA toolkit
and package caches. The retained target and generated model alone use 11.5 GiB.

For the stock runtime without the optimized build or MTP head:

```bash
localmodel fetch bonsai2-27b
localmodel start bonsai2-27b
```

This is a separate configuration; it does not reproduce the optimized timings.

## Set up Qwen3.8 Flash-Next

`qwen3.8` selects **Unsloth Qwen3.8-Flash-Next UD-IQ1_M Dynamic v3.1**, with the
official Qwen MTP layer. The separate `qwen3.8-27b` option uses the dense model.

```bash
bin/setup-localmodel --strata --dry-run
bin/setup-localmodel --strata --run
localmodel fetch qwen3.8
localmodel start qwen3.8 --dry-run
localmodel start qwen3.8
```

Fetch also prepares the Qwen engine and MTP artifacts. The defaults are
**192K context, a 64K GPU KV window, a 64 GiB process cap, a 36 GiB resident
expert budget, INT8 KV, four draft tokens, and a 1024 MiB VRAM reserve**.
Adaptive fetch is off with resident experts.
The rest of the KV cache uses RAM. Strata's native speculation value includes
the accepted token, so the launcher maps four drafts to five.
Vision, suffix drafting, routing priors, and expert tail skipping are disabled.
The PLE embedding table uses direct file reads.
The pinned Strata 0.2.13 native engine enables the Linux CUDA GDN defaults
equivalent to `STRATA_GDN_TC=1` and `STRATA_GDN_CONVL2=2`. The tensor-core path
needs about 57.5 MiB of extra GPU scratch space. See the
[kernel review](docs/strata-0.2.9-review.md#qwen-linux-cuda-gdn-defaults).

Add `--tui` to a Qwen or DeepSeek start command to show Strata telemetry:

```bash
localmodel start qwen3.8 --tui
# Or, after stopping Qwen:
localmodel start deepseek --tui
```

The dashboard opens when the server is ready. Press `q` to close the dashboard
and keep serving in the same terminal. Press Ctrl-C to stop the server.
`localmodel stop` also closes the dashboard and stops the server.
Startup output goes to `state/qwen-console.log` or `state/deepseek-console.log`.
Use at least 128 columns and 36 rows for the full dashboard.
The first use installs pinned Textual dependencies in a separate environment.
The dashboard reads loopback telemetry only. The model's memory cap and outgoing
network block remain active. Upstream update, prompt-test, and configuration
controls are not available from this dashboard.

The optional `--cache-reuse-fix` flag selects a separately built Qwen CUDA
backport. It limits speculative cache commits to the emitted response when
generation reaches its output limit or EOS. It does not change DeepSeek or fix
cache invalidation after edits to old conversation messages. Build and rollback
instructions are in the [backport record](experiments/strata-cache-reuse-20261113/README.md).
The official executable remains the default.

The Strata 0.2.12 OMP comparison with a 36 GiB resident expert budget reached
about **185K actual input tokens**.
MTP4 averaged **52.17 decode tokens/s**. Tuned DFlash2 at confidence 0.8
averaged **58.41 tokens/s**. Both passed all four code, recall, tool and strict
JSON checks. The DFlash2 run required extra prefill after OMP changed an old
tool result. It did not complete the whole workload faster. MTP4 remains the
normal default. See the [measurements and limits](docs/strata-0.2.12-measurements.md).

Use the optional tuned drafter with these commands:

```bash
localmodel fetch qwen3.8 --speculative dflash2
localmodel start qwen3.8 --speculative dflash2
```

This adds a verified 324 MiB draft pack. It keeps the target GGUF weights.
The DFlash2 defaults are five draft tokens and confidence 0.8. Use
`--draft-tokens` and `--draft-conf` to override them. The measured run still
produced low-VRAM warnings. Do not infer spare capacity for larger settings.
These DFlash2 measurements used `--expert-ram-gb 36`. The combination of
file-backed experts and DFlash2 has not been benchmarked here.
`localmodel remove qwen3.8 --speculative dflash2 --dry-run` previews removal
of only this optional pack.

Use a settings profile to select the memory layout:

```bash
localmodel start qwen3.8          # Faster resident expert configuration.
localmodel start qwen3.8 lowram   # File-backed experts and adaptive fetch 1.
```

Start only one server at a time. `localmodel start qwen3.8 default` is the
same as omitting the profile. `localmodel list` shows the available profiles.
Explicit flags override profile settings. For example,
`localmodel start qwen3.8 lowram --adaptive-fetch 0` keeps mapped experts
and disables adaptive admission. `localmodel status --json` reports `settings_profile`.
Profiles use the same weights. They do not require a separate fetch.

The `lowram` profile sets the dedicated expert RAM budget to zero.
Linux can reclaim cached expert file pages when compilation or other work
needs RAM. The process can still use idle RAM for its file cache.
Both profiles keep the 64 GiB cap. This cap is a limit, not a reservation.
Use `--mem` to change it. In the 16K/32K/64K comparison, `lowram` averaged
44.46 decode tokens/s, against 53.63 with resident experts. Adaptive fetch
itself added no useful speed gain. Mapped experts were tested through about
66K actual input at 192K capacity. The larger input test above used resident experts.
Use `--expert-ram-gb` to override the expert budget. Adaptive fetch defaults
to 1 with a zero budget and 0 with a resident budget. Explicit modes 1 and 2
require a zero expert RAM budget.

The [earlier reserve comparison](docs/qwen-unsloth-192k.md) and
[Qwen and DeepSeek comparison](docs/local-192k-comparison.md) retain the
Strata 0.2.11 results.

OMP discovers the 192K capacity and normally compacts near 157K input.
The benchmark disabled automatic compaction to test the larger input directly.
The [previous ISTA measurements](docs/qwen-0.2.11-tuning.md) remain available.
They are not a matched quantization comparison after the VM move and changes
in desktop VRAM use.

The Unsloth shards total **68.01 GB** before the prepared index and MTP files.
Preparation uses upstream `--compat-bf16` for small projections. The CUDA engine
reads native Q8 hyper-connection projections with `STRATA_HC_Q8=1`. The main
GGUF files stay unchanged and are verified against pinned SHA256 values.
Allow **at least 90 GiB free**
for an initial fetch, preparation, and runtimes. The RAM cap includes more than
the expert budget: cache, page cache charged to the process group, and runtime
buffers also count. Leave RAM for the OS and the coding agent.

Example overrides:

```bash
localmodel start qwen3.8 --context 128k --mem 48G --expert-ram-gb 24
localmodel start qwen3.8 --context 256k --draft-tokens 2 --prefill-chunk 4096
localmodel start qwen3.8 --ple-io mmap
```

Stop the first server before starting another. These overrides are tuning
examples, not claims that they outperform the defaults.

## Set up Qwen3.8 27B

`qwen3.8-27b` selects the dense model with the November 11 Unsloth Dynamic 3.1
`UD-Q4_K_M` weights. The 15.43 GiB GGUF includes an MTP head. The download uses
a fixed repository revision and SHA256 check. Allow at least 18 GiB free disk
space. This option uses the existing pinned llama.cpp runtime, not Strata's
Flash-Next engine.

```bash
localmodel fetch qwen3.8-27b
localmodel start qwen3.8-27b --dry-run
localmodel start qwen3.8-27b
# Optional two-token MTP profile; uses the same downloaded file.
localmodel start qwen3.8-27b mtp
```

Start one server at a time. Defaults are 192K context capacity, Q4 KV,
automatic GPU layer fitting, a 48 GiB process cap, an 8 GiB prompt snapshot
cache, and xhigh reasoning. A 12 GiB GPU cannot hold these weights in full.
The runtime places some dense layers in RAM. Those layers run for every token.
The runtime chooses the layer split at startup and reports it in its log.

Use `--gpu-layers N` to select a fixed layer count, `--context 32k` for a
shorter context, or `--mem 64` to change the process cap. A fixed layer count
disables automatic fitting. `--no-offload` requests all layers on the GPU and
can fail if VRAM is insufficient. `--cache-ram` controls snapshots; it does not
set the RAM weight budget. The `mtp` profile accepts `--draft-tokens N` and
`--speculative none` overrides. The November 11 comparison measured 5.37 decode
tokens/s with MTP and 2.99 without it on the 12 GiB RTX 4070 SUPER. Both passed
the constructed 16K OMP repair and resume checks. Initial prefill was slower
with MTP. These runs used 192K capacity; they did not fill that capacity.
See the [27B comparison](docs/qwen27-20261111.md) for timings and limitations.

`omp-local` discovers the model and context automatically. `omp-remote`
subagents also discover it at launch. The automatic main-request router still
requires Flash-Next; it does not classify requests with dense 27B.

Preview removal with `localmodel remove qwen3.8-27b --dry-run`. The plain and
MTP options use the same file, so removal of either deletes that file.

This is a quantization update, not a new base model. Unsloth reports improved
benchmark accuracy and leaves its older 3-bit files unchanged. Dense 27B uses
less total storage than Flash-Next but activates more parameters per token.
Its partial CPU execution can reduce throughput. Published base-model scores
do not establish the quality of these local quantizations. See the
[Unsloth update](https://unsloth.ai/docs/models/qwen3.8) and
[Qwen comparison](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## Set up DeepSeek V4.1 Flash 1017

`deepseek`, `deepseek-full`, `deepseek-iq1-s`, and `strata` select
**Unsloth Dynamic V3.1 UD-IQ1_S**, pinned to revision
`8a6ec9cf118e785a362ea94ee123c61c70553be6`. It keeps the full expert pool.

```bash
bin/setup-localmodel --strata --run
localmodel fetch deepseek
localmodel start deepseek --dry-run
localmodel start deepseek
```

The defaults are **192K context, a 64 GiB process cap, a 48 GiB expert cache,
4096-token prefill chunks, twelve CPU workers, all six selected experts,
and no drafter**. The expert cache is included in the process cap.
Strata keeps selected experts in VRAM and RAM and reads other weights from SSD.
Its transformer shards need about 114 GiB in addition to the 103 GiB Engram
file. The downloader requires the remaining download size plus 10 GiB free.
Stop the running server before switching. `omp-local` discovers the served
model and context; no harness configuration change is needed.

The Strata 0.2.12 repeat reached **184,758 actual input tokens**, with
**8.99 decode tokens/s** and **373.09 new-prefill tokens/s**. All four repair,
recall and strict resume checkpoints passed. The matched 0.2.11 run measured
9.55 decode and 355.32 new-prefill tokens/s. The new run generated fewer tokens;
its shorter wall time does not establish a fixed-work speedup. See the
[full-model 0.2.12 comparison](docs/deepseek-full-0.2.12.md).

The later [nine-run tuning sweep](docs/deepseek-full-0.2.12-tuning.md) kept
these defaults. The two 16K controls measured 9.19 and 9.16 decode tokens/s.
A larger cache gave no established gain. Smaller prefill chunks reduced prompt
speed, and both tested DSpark depths decoded more slowly than the controls.

`deepseek-coder` remains an optional **ISTA GSQ-RCO Coder IQ2_XXS** profile,
pinned to revision `54229d38d588a66d1c6cff08af7b281b9326c656`.
It retains 192 of the original 384 routed experts per layer.
Use `localmodel fetch deepseek-coder` and `localmodel start deepseek-coder`
to restore and run it. Coder defaults to zero private expert cache.
CPU experts remain file-backed. Linux uses spare RAM to cache their pages and
can reclaim those pages for other programs. GPU allocations stay in VRAM.
Engram remains SSD-backed. `--expert-ram-gb 32` selects a dedicated 32 GiB
expert cache. This budget is included in the total process cap.
Conversation checkpoints are retained. Coder currently runs without DSpark.
`omp-local` discovers its identity and context from the running server.

The 2026-11-07 growing OMP test reached 184,811 input tokens at **11.92 decode
tokens/s**. Full Unsloth reached 184,898 input tokens at **9.55 tokens/s**.
Both passed the four repair, recall, and resume checkpoints on Strata 0.2.11.
Full Unsloth is now the selected DeepSeek default. It also
produced one correct lower-effort repair, but its test workflow hit the
deadline. The small diagnostics do not establish a general quality ranking.
See the [complete comparison](docs/local-192k-comparison.md).

Allow **200 GiB free** on a fast Linux SSD for a new installation. Coder uses
about 170.2 GiB, including the 103 GiB Engram file. If the full profile already
has that verified Engram file, fetch reuses it through a hard link. The new
transformer files need 67.2 GiB plus the downloader's 10 GiB reserve.

The 2026-11-05 comparison used 256K capacity and growing 16K/32K/64K OMP input.
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
**9.08** and **9.14**, respectively. The Coder profile keeps the existing 64 GiB cap;
these sharing measurements used 70 GiB. See the [memory-sharing comparison](docs/coder-elastic-memory.md).

The local [Strata release watch](docs/strata-watch.md) checks for stable releases
every six hours. It queues updates and guarded full-Unsloth measurements when resources
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

If disk space is limited, inspect `localmodel remove deepseek-coder --dry-run`
before using `localmodel remove deepseek-coder`. This preserves an Engram link
in the full profile when that link exists. Keep the weights on the native
Linux filesystem; the Windows drive containing the WSL virtual disk can vary.
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
Depth 5 is the default when DSpark is selected. The 2026-10-30 16K OMP
comparison reached 4.60 tok/s with DSpark and 4.98 without it. Both runs passed the repair,
100 assertions, recall, and resume checks. DSpark remains optional.
See the [measurements and limits](experiments/deepseek-strata/README.md#16k-omp-comparison-2026-10-30).
Use `--speculative none` to unload the drafter.
Use `localmodel remove deepseek-full --speculative dspark` to remove only the
draft weights and their pack. This keeps the main model.

Local DeepSeek, Qwen, and remote DeepSeek use pinned source, API, and packer
**0.2.13**, Control **0.1.59**, and signed native engine **0.1.59**.
Qwen uses four MTP draft tokens. Coder keeps zero private expert cache.
See the [runtime verification and comparison](docs/strata-0.2.13-review.md).
A crafted parser example can still turn quoted text into a tool call.
This has not been observed in the OMP workloads. The client must still
enforce tool permissions. See the [Coder update and comparison](docs/strata-0.2.11-coder.md).
No native compilation is required for Strata.
Its server and children cannot open outgoing connections; fetch and installation
remain network-enabled. Each local profile has the context default listed
above. Remote context capacity is discovered separately.

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

`omp-remote` also requires SearXNG on **8888**. A local model on **8899** is
optional. New conversations select `router/auto`: idle Qwen3.8-Flash-Next
classifies each new request and chooses local or remote. It is instructed to
send complex or uncertain work remotely. Large conversations stay remote.
Use `/model router/auto` in an existing conversation, or select
`/model model/<served-id>` to stay remote. See the
[routing rules](docs/harness.md#start-the-components) for details.

Subagents use the local model discovered at launch, or the remote model when
none is ready. The legacy `bonsai` provider name does not restrict which model
can be served locally. Discovery does not use saved model metadata.
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
localmodel start qwen3.8 lowram --dry-run

# Inspect the exact removal first. Stop the model before removal.
localmodel remove qwen3.8 --dry-run
localmodel remove qwen3.8
localmodel remove deepseek --dry-run
localmodel remove deepseek-full --dry-run
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
