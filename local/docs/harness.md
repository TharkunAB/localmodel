# Local coding harness

The `local/` directory contains the launchers, profile defaults, runtime patch
manifest, dependency locks, local search plugin, and optional collaboration source.
Model weights, runtime installations, credentials, sessions, and private
certificates stay outside version control.

The packaged setup supports Ubuntu on Linux x86_64, including WSL2. A fresh
installation does not require the original user's home directory or a global
OMP installation.

Both `omp-local` and `omp-remote` run on your workstation. The remote
profile reaches the rental GPU through an SSH tunnel. The rental GPU
runs the separate [remote payload](../../remote/start-remote-model.sh).

Unless stated otherwise, paths in this guide are relative to `local/`.
Run its commands from `~/Code/local-models/local`. Model files, runtimes,
and private state are also stored under that directory.

## Install from a fresh checkout

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

From the local directory, install the launchers, pinned OMP, and SearXNG:

```bash
cd ~/Code/local-models/local
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

The OMP step runs this command. It can also run separately:

```bash
uv run --no-project --python .venv/bin/python python bin/setup-harness --dry-run
uv run --no-project --python .venv/bin/python python bin/setup-harness --run
```

`setup-harness` installs `@oh-my-pi/pi-coding-agent@18.8.6` and the Linux x86_64
native package from `config/omp-package/bun.lock`. It uses an isolated temporary
directory, disables package scripts, and checks the reviewed OMP bundle before
it activates the installation. It omits optional speech and transformer
packages. The native package is an explicit dependency, so it remains installed.
It does not install Bun, update a global package, start a service, or fetch a
model. A repeated install checks and keeps the existing managed package.

SearXNG has its own Python environment and pinned source checkout under
`search/`. Its installer records the installed Python versions there. Model
runtime installation is separate. Use `bin/setup-localmodel` with its `--bonsai`
or `--strata` option when the selected model requires that backend. The sections
below list model-specific dependencies and downloads.

Strata's disk monitor requires `psutil`. The local Strata environment includes
the version pinned by upstream. For an older installation, run
`bin/setup-localmodel --run --strata`, then restart the model server.
Refresh the browser after the restart. Disk activity includes all reads
visible to the operating system or container. It is not limited to Strata.

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
See the [head audit](../experiments/bonsai-next/head-audit.md).
The [DSpine comparison](bonsai-dspine-20261108.md) covers the separate
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
official Qwen MTP layer and the separate **Q8_0 Engram table**.
The separate `qwen3.8-27b` option uses the dense model.

The [Engram and context measurements](qwen-engram-20261113.md) cover both table
precisions, 192K and 256K capacities, filled conversations, and compaction.

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
The Q8_0 PLE embedding table uses direct file reads. Its 54.4 GB file does not
require 54.4 GB of RAM or VRAM. The default row cache holds about 170 MiB of
table data. Core weights and the MTP layer keep their existing precision.
The pinned Strata 0.2.13 native engine enables the Linux CUDA GDN defaults
equivalent to `STRATA_GDN_TC=1` and `STRATA_GDN_CONVL2=2`. The tensor-core path
needs about 57.5 MiB of extra GPU scratch space. See the
[kernel review](strata-0.2.9-review.md#qwen-linux-cuda-gdn-defaults).

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
instructions are in the [backport record](../experiments/strata-cache-reuse-20261113/README.md).
The official executable remains the default.

The Strata 0.2.12 OMP comparison with a 36 GiB resident expert budget reached
about **185K actual input tokens**.
MTP4 averaged **52.17 decode tokens/s**. Tuned DFlash2 at confidence 0.8
averaged **58.41 tokens/s**. Both passed all four code, recall, tool and strict
JSON checks. The DFlash2 run required extra prefill after OMP changed an old
tool result. It did not complete the whole workload faster. MTP4 remains the
normal default. See the [measurements and limits](strata-0.2.12-measurements.md).

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

The [earlier reserve comparison](qwen-unsloth-192k.md) and
[Qwen and DeepSeek comparison](local-192k-comparison.md) retain the
Strata 0.2.11 results.

OMP discovers the 192K capacity and normally compacts near 157K input.
The benchmark disabled automatic compaction to test the larger input directly.
The [previous ISTA measurements](qwen-0.2.11-tuning.md) remain available.
They are not a matched quantization comparison after the VM move and changes
in desktop VRAM use.

The original Unsloth shards total **68.01 GB** before the prepared index and MTP
files. The default Q8_0 table adds **54.4 GB**, for **122.41 GB** of downloaded
weights. Strata keeps all three original shards for dense tensor validation.
The separate table has its own SHA256 pin and verification receipt.
Preparation uses upstream `--compat-bf16` for small projections. The CUDA engine
reads native Q8 hyper-connection projections with `STRATA_HC_Q8=1`. The main
GGUF files stay unchanged and are verified against pinned SHA256 values.
Allow **at least 145 GiB free** for the default initial fetch, preparation, and
runtimes. The smaller table option needs at least 90 GiB. The RAM cap includes more than
the expert budget: cache, page cache charged to the process group, and runtime
buffers also count. Leave RAM for the OS and the coding agent.

Use the bundled IQ4_NL table to avoid the separate Q8_0 download:

```bash
localmodel fetch qwen3.8 --engram iq4_nl
localmodel start qwen3.8 --engram iq4_nl
```

Both commands use the same IQ1_M core weights and MTP layer. The explicit
`--engram default` value remains a legacy alias for `--engram iq4_nl`.
The positional `default` settings profile still uses Q8_0 when no Engram flag
is given. Fetch does not delete an existing table.

To reclaim the separate table's space, stop Qwen and preview
`localmodel remove qwen3.8 --engram q8_0 --dry-run`. Remove `--dry-run` to delete
only that table. Plain `localmodel remove qwen3.8`, or the same command with
`--engram iq4_nl`, removes the core model and its draft files. Both commands
keep the separate Q8_0 table.

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
See the [27B comparison](qwen27-20261111.md) for timings and limitations.

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
[full-model 0.2.12 comparison](deepseek-full-0.2.12.md).

The later [nine-run tuning sweep](deepseek-full-0.2.12-tuning.md) kept
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
See the [complete comparison](local-192k-comparison.md).

Allow **200 GiB free** on a fast Linux SSD for a new installation. Coder uses
about 170.2 GiB, including the 103 GiB Engram file. If the full profile already
has that verified Engram file, fetch reuses it through a hard link. The new
transformer files need 67.2 GiB plus the downloader's 10 GiB reserve.

The 2026-11-05 comparison used 256K capacity and growing 16K/32K/64K OMP input.
Strata 0.2.11 averaged **12.46 decode tokens/s**, versus **12.50** on 0.2.10.
Repair and resume took **372 seconds** in each run. All quality checks passed.
Both reached the 64 GiB cap without OOM or model swap. This single pair shows
no material throughput change. See the [report](strata-0.2.11-coder.md).

The earlier Strata 0.2.10 resident repeat at 64 GiB averaged **13.80 decode tokens/s**, versus
**13.25** on 0.2.9. Prefill was **483/618/757 tokens/s** at growing 16K/32K/64K
input. The 70 GiB follow-up averaged **14.04 tokens/s** and kept all routed
experts in RAM or VRAM. That was only a 1.7% decode gain over 64 GiB.
Use `--expert-memory resident --mem 70G` to select that full-residency configuration.
The file-backed test at 70 GiB averaged **12.80 tokens/s** when quiet and
**6.40** under controlled memory pressure. A 32 GiB private cache averaged
**9.08** and **9.14**, respectively. The Coder profile keeps the existing 64 GiB cap;
these sharing measurements used 70 GiB. See the [memory-sharing comparison](coder-elastic-memory.md).

The local [Strata release watch](strata-watch.md) checks for stable releases
every six hours. It queues updates and guarded full-Unsloth measurements when resources
are available. It does not interrupt a running model or deploy to a remote server.
A daily Grok search checks public X reports for useful optimizations. Local
configuration and measurements stay out of the search request.

Every final code, recall, and resume check passed. Output lengths and recovered
tool errors differed, so this does not establish faster completed coding tasks.
No run used swap or hit OOM. These tests do not fill 256K or test automatic
compaction. See the [earlier resident comparison](strata-0.2.10-coder.md).

The original matched comparison used native 0.1.55 and the same 64 GiB cap
and 48 GiB demand cache. Coder averaged **11.24 decode tokens/s** against
**8.16** for full Unsloth UD-IQ1_S. See the
[model comparison and quality limits](deepseek-coder-comparison.md).

If disk space is limited, inspect `localmodel remove deepseek-coder --dry-run`
before using `localmodel remove deepseek-coder`. This preserves an Engram link
in the full profile when that link exists. Keep the weights on the native
Linux filesystem; the Windows drive containing the WSL virtual disk can vary.
The downloader requires the remaining download size plus 10 GiB free.
The optional full-profile DSpark files can be removed separately to recover
another 7.43 GiB. Original tuning results remain in the
[full-profile report](deepseek-029-tuning.md).

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
CPU expert file pages. The [memory-sharing test](coder-elastic-memory.md)
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
See the [measurements and limits](../experiments/deepseek-strata/README.md#16k-omp-comparison-2026-10-30).
Use `--speculative none` to unload the drafter.
Use `localmodel remove deepseek-full --speculative dspark` to remove only the
draft weights and their pack. This keeps the main model.

Local DeepSeek, Qwen, and remote DeepSeek use pinned source, API, and packer
**0.2.13**, Control **0.1.59**, and signed native engine **0.1.59**.
Qwen uses four MTP draft tokens. Coder keeps zero private expert cache.
See the [runtime verification and comparison](strata-0.2.13-review.md).
A crafted parser example can still turn quoted text into a tool call.
This has not been observed in the OMP workloads. The client must still
enforce tool permissions. See the [Coder update and comparison](strata-0.2.11-coder.md).
No native compilation is required for Strata.
Its server and children cannot open outgoing connections; fetch and installation
remain network-enabled. Each local profile has the context default listed
above. Remote context capacity is discovered separately.

## Start the components

Run search in one terminal:

```bash
searxng-local
```

Run the selected local model in another terminal. For example:

```bash
localmodel list
localmodel fetch qwen3.8
localmodel start qwen3.8
```

The fetch command downloads the selected model and its required runtime. Check
the model's documented disk, RAM, and GPU requirements first. The setup
installer does not perform this step.

`qwen3.8` is Flash-Next. For dense Qwen3.8 27B, use
`localmodel fetch qwen3.8-27b`, then `localmodel start qwen3.8-27b`.
The optional `localmodel start qwen3.8-27b mtp` profile uses the same weights.
See the [dense Qwen setup](#set-up-qwen38-27b) for memory controls.

Start the coding harness from the project you want to work on:

```bash
cd /path/to/project
omp-local --mem 32
```

`omp-local` discovers the active `localmodel` endpoint. All conversation and
task roles use that model. Search uses SearXNG at port 8888. URL reads fetch
the URL directly.

For a remote main model, run `remotemodel` on the rental GPU host. It uses
`/workspace/remote-models` by default and reuses prepared weights and runtimes.
`remotemodel start deepseek --tui` prepares missing files and starts Strata with
the same telemetry display as `localmodel start MODEL --tui`. `remotemodel fetch deepseek`
prepares without serving; `--no-fetch` on `start` requires prepared files.
`remotemodel status` and `remotemodel stop` control that host's managed server.
These commands do not SSH from the workstation to a Pod.

Expose the remote OpenAI-compatible endpoint on workstation port 8898.
Follow the [RunPod guide](../../remote/runpod.md) to install `remotemodel` and
start the SSH tunnel. A local model on port 8899 is optional:

```bash
omp-remote --mem 32
```

`omp-remote` discovers the main model on port 8898 through IPv4 or IPv6. Its
context limit defaults to 1048576 tokens and decreases when the server reports
a smaller limit. The output limit is one quarter of that context, up to 262144
tokens. This leaves room for the prompt and compaction. The plan, small,
slow, and title roles use this endpoint. Served IDs such as `deepseek-gsq-*`
are read directly from the endpoint; no fixed model ID is required.

New remote conversations select `router/auto`. The router sends simple requests
to a ready Qwen3.8-Flash-Next on port 8899. It keeps complex requests on the
remote model. It checks local availability for each request, so Qwen can start
after OMP. It does not start a model server.

Local Qwen decides the route with a separate classification request. It receives
the new user request and excerpts from the two preceding user or reply messages.
Each excerpt contains at most 1000 characters. It can choose local for routine
work, including straightforward code changes. It should choose remote for
difficult reasoning, broad changes, or uncertain scope. There is no keyword rule.
The classifier has no tools. It uses temperature 0, disables thinking, and can
produce at most 64 tokens. It must return a JSON routing decision within 10 seconds.
An invalid answer, failed request, or timeout keeps the turn remote. Requests
larger than 12000 UTF-8 bytes also stay remote to bound classification work.
The classification result does not enter the conversation. Normal replies keep
the selected reasoning effort. Classification adds local inference work and
can replace a cached prompt on the local server.

The router checks capacity and load before it calls the classifier. A busy
local server keeps a new turn remote. The load limits are 50% GPU utilization
and a one-minute CPU load of 0.75 per available CPU. Missing GPU telemetry
keeps the turn remote. Tool follow-ups retain the selected model while it is
available and the conversation fits. They do not call the classifier again.
Large conversations stay remote instead of being compacted to fit Qwen.
Compaction uses the remote model.

The local context check counts serialized UTF-8 bytes conservatively and
reserves room for output and template framing. It can therefore keep a
conversation remote even when the exact token count would fit locally.
If a local request fails before it produces output, the router retries once
remotely. It does not replay a response after partial output or cancellation.
Each new turn prints the selected endpoint and the reason. Switching models
can require a new prefill because the two servers do not share a prompt cache.

Restart `omp-remote` to load the extension. After resuming an existing
conversation, use `/model router/auto` to enable routing.
Use `/model model/<served-id>` to keep the conversation remote. These selections
also work with `--model`. The routing extension is
[`config/omp-router.ts`](../config/omp-router.ts). It uses the pinned OMP
custom-provider API. It does not require a runtime upgrade or a proxy service.

| Reported context | Output limit | OMP compaction threshold |
| --- | --- | --- |
| 8192 | 2048 | 5120 |
| 131072 | 32768 | 81920 |
| 262144 | 65536 | 180224 |
| 524288 | 131072 | 376832 |
| 1048576 | 262144 | 770048 |

The task, scout, reviewer, security-reviewer, and sonic agents use the live
model on port 8899 when it is ready at launch. This includes Qwen3.8. Otherwise,
they use the remote model. The provider name `bonsai` remains for compatibility;
it does not require a Bonsai model. Discovery does not use saved model metadata.
Relaunch to change these task roles after starting or replacing the local model.
The main router discovers Qwen independently. If the remote endpoint is
unavailable at launch, `omp-remote` exits.

Remote sessions saved with `deepseek-v4.1` or a `deepseek-v4.1-*` profile can
resume with another served profile in that family. The current server must
report one matching model. Other model families retain OMP's normal checks.

Stop the local model with Ctrl-C in its terminal or `localmodel stop`. Stop
SearXNG with Ctrl-C. The wrappers do not start these services in the background.

## Dependencies and state

| Component | Runtime dependencies | Local installation or state |
| --- | --- | --- |
| Python launchers | Python 3.12; local `.venv` | Scripts and `.venv/` under `local/` |
| OMP | Bun >=1.3.14; locked OMP and native packages | `state/omp-package/18.8.6/` |
| Reviewed OMP patches | Input and output SHA256 in `config/omp-runtime.json` | `state/omp-runtime/18.8.6/managed/` |
| Local OMP profile | Ready model; healthy SearXNG | `state/omp-local/` |
| Remote OMP profile | Port 8898; optional local model on 8899; healthy SearXNG | `state/omp-remote/` |
| SearXNG | Pinned source; its Python requirements; Granian | `search/`, `state/searxng/` |
| Memory limits | Linux cgroup v2; systemd user manager; Python 3 | `libexec/capped*`, profile peak logs |
| OpenCode, optional | External OpenCode v2.0.26 binary; reviewed goal plugin | `~/.opencode/bin/opencode`, `state/opencode*/` |
| OpenCode goal plugin | Plugin 0.11.1; Zod 4.6.5 | `state/plugins/opencode-goal-plugin-0.11.1/` |
| OpenCode search plugin | OpenCode plugin API; no npm dependencies | `config/searxng-plugin/` |
| Collaboration, optional | Bun; checked-in browser source and dependency lock; private TLS certificate | `collab/`, `state/collab/` |

OMP first selects the managed package. If that directory is absent, it can
use `omp` from `PATH`. Either source must match the exact reviewed version and
input checksum. A partial or changed managed installation causes an error.
It does not cause a fallback to the global package. The launcher checks the
patched output checksum before execution and sets `PI_PACKAGE_DIR` to the
selected package. It leaves the global package unchanged.

The local and remote profiles have separate sessions, authentication files,
caches, and preferences. The launcher seeds a missing `agent/config.yml`
from `config/omp.json`. It preserves an existing `config.yml` or `config.yaml`.
Settings changed through OMP therefore persist. The generated `launcher.yml`
overlay controls model routing, context thresholds, search, and network
settings. Edit persistent defaults or profile settings, rather than this
generated overlay. The launchers reserve `--profile` for their isolated state.

OMP defaults to `xhigh` reasoning. For DeepSeek V4.1 and `deepseek-gsq-*` IDs,
the launcher sends `max` when OMP selects `xhigh`. Pinned Strata maps both
values to the template's effort 100. The template accepts `max` directly.
The `minimal` selection sends `low`, which keeps thinking enabled. Other
DeepSeek efforts retain their normal mapping. For Mistral Large 4, `xhigh`
and `max` send `high`; `minimal` sends `low`. These are supported Mistral
effort values. Mistral requests use temperature 1.0 and top-p 0.95.
Remote discovery reads the served context capacity, including the B300
profile's 512K default. It reserves one quarter of context for output.

Compaction uses the conversation model, runs synchronously, and keeps 20000
recent tokens. It starts at the lower of 80% of context or context minus the
output limit and a reserve. The reserve is the smaller of 16384 tokens or one
eighth of context. The table above shows the resulting remote profile limits.

OMP supports its native `/goal` and `/skill:NAME` commands. Skills load from
the user and project `.agents/skills` directories. A profile links the user's
`~/AGENTS.md` when that file exists. Project instructions continue to apply.

## Search and memory checks

Every interactive OMP launch checks the configured SearXNG `/healthz` endpoint
without a proxy or redirect. It requires HTTP 200 and `OK`. This check sends
no search query. Help, version, config, agent-listing, and completion commands
work without search or a model. The required OMP package and Bun must still
be installed.

`--mem` must be the first launcher option. Bare sizes use GiB. For example,
`--mem 32` sets a shared 32 GiB limit with no swap for the foreground harness
and its ordinary child processes. The model and search servers have separate
limits. The bundled memory recorder removes the dependency on a separate
`heavy-management` checkout. Peak records remain under the profile's state
directory. An external OpenCode `--server` cannot receive this limit.

OMP's shared LSP broker and background-service launcher are disabled by the
managed overlay. Optional upstream daemons can use separate scopes. In
particular, the text-prediction daemon has previously been observed outside
the foreground scope. Do not treat `--mem` as a limit for every process in the
user account.

The runtime disables automatic xAI search, hosted URL-reader fallbacks,
session uploads, and collaboration destinations outside the configured list.
SearXNG itself contacts the search engines in `config/searxng.yml`; direct URL
reads contact the requested website. These settings do not provide a network
firewall for project commands or explicit user configuration.

## Optional OpenCode

The supplied OpenCode profiles target v2.0.26 at `~/.opencode/bin/opencode`.
This repository does not contain a pinned OpenCode binary installer. Install
that compatible version separately before using these launchers. A current
upstream release can have a different plugin API or profile schema.

Install the separately reviewed goal plugin and Zod dependency:

```bash
uv run --no-project --python .venv/bin/python python bin/setup-harness --dry-run --with-opencode-goal
uv run --no-project --python .venv/bin/python python bin/setup-harness --run --with-opencode-goal
```

Link the optional OpenCode launchers with `bin/setup-localmodel --opencode --run`.
If an existing launcher is a wrapper file instead of a matching symbolic link,
setup preserves it and stops. You can keep using that wrapper or select a
separate `--bin-dir`. The base setup does not replace OpenCode launchers.

The installer checks both archive SHA256 values and every extracted file
against `config/opencode-goal-plugin-manifest.json`. It executes no npm
scripts. Existing installations must match that manifest. The local SearXNG
plugin ships in the checkout and needs no package installation.

```bash
opencode-local --mem 32
opencode-remote --mem 32
opencode-local service status
opencode-remote service status
```

OpenCode uses separate profile state. Service commands work without a ready
model. They keep a saved model configuration when discovery fails. A fresh
service configuration starts with no allowed model providers. The OpenCode
main routes follow the local endpoint or port 8898. Its optional general and
explore agents retain their existing Bonsai route and cached-metadata behavior.
OMP remote task discovery uses a live port 8899 when it is available at launch.

To start an OpenCode service with a memory limit, put the limit before
`service start`. Restarting a service interrupts its work. A service already
running outside the requested scope needs an explicit `service restart` to
receive the limit. Setup does not start or restart these services.

## Optional private collaboration

Collaboration is disabled when `config/omp-collab.json` is absent. Each machine
needs its own configuration and TLS certificate. The example files use a
reserved hostname and loopback addresses. They are templates, not a working
service configuration.

To configure a new private relay, copy the examples only when the destination
files do not already exist:

```bash
cp -n config/omp-collab.example.json config/omp-collab.json
cp -n config/omp-collab-service.example.json config/omp-collab-service.json
```

Set the real private hostname, listen address, and permitted client CIDRs.
Keep `relayUrl`, `webUrl`, `origin`, and `allowedOrigins` consistent. Client
origins must use HTTPS and WSS. Generate a certificate with your private CA
for that hostname. Store the server certificate, private key, and CA file at
the configured paths under `state/collab/tls/`. Restrict the key to its owner.
Trust that CA in the client browser. OMP reads the configured CA file only for
its collaboration connection; setup does not alter the system trust store.

Build the supplied browser client:

```bash
bun install --cwd collab --frozen-lockfile --ignore-scripts
bun run --cwd collab build
```

The upstream source revision and local changes are recorded in
`collab/upstream-revision.json` and `collab/AUDIT.md`. The source and dependency
locks ship in this checkout. The generated `collab/dist/` directory does not.

For a user service, write a machine-specific `config/omp-collab.service`.
Replace the example paths with the paths on this machine:

```ini
[Unit]
Description=Private OMP collaboration
After=network.target

[Service]
Type=simple
WorkingDirectory=/absolute/path/to/local-models/local
ExecStart=/absolute/path/to/bun /absolute/path/to/local-models/local/collab/server.ts --config /absolute/path/to/local-models/local/config/omp-collab-service.json
Restart=on-failure
RestartSec=3
UMask=0077
MemoryMax=512M
MemoryHigh=384M
MemorySwapMax=0
TasksMax=64
NoNewPrivileges=yes
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX

[Install]
WantedBy=default.target
```

Review the addresses, certificate, and unit before enabling the service:

```bash
omp-collab install
omp-collab start
omp-collab info
```

The install command refuses to replace another unit. It enables the unit for
future user-manager starts. Core setup leaves collaboration files and services
unchanged. Keep machine configuration and certificates out of version control.

## Updates and recovery

OMP updates require a new reviewed input bundle, patches, output checksum,
package manifest, and dependency lock. Updating only a global `omp` executable
does not update the managed runtime. Never bypass a checksum failure to make
an unreviewed package launch.

If setup reports an existing directory with different files, inspect that
directory before moving or replacing it. The installer does not delete an
existing installation or profile. Keep profile state when changing runtime
packages. Model, search, and harness processes must be stopped separately
when they are no longer needed.

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

Create a source-only archive from the committed repository root:

```bash
cd ~/Code/local-models
mkdir -p local/dist
git archive --format=tar.gz --prefix=local-models/ \
  --output=local/dist/local-models.tar.gz HEAD
```

The archive contains committed files only. It has no models or conversations.
CMake build caches can retain absolute paths after a directory move. Reconfigure
the build cache before rebuilding. Keep the verified binaries and build receipts.
See the [OpenCode](#optional-opencode) and
[private collaboration](#optional-private-collaboration) setup sections above.
Older Dory and legacy Bonsai profiles require
their separate historical installations; the fresh-install path above covers
the current Bonsai 2 27B, Qwen Flash-Next, and DeepSeek configurations.

Docker is not required. The local wrappers rely on host systemd scopes to cap
agent children. RunPod already supplies the remote container; use a CUDA 13
compatible template with SSH for Strata. A single Dockerfile would not reproduce both
the local process limits and the remote SSH workflow.
