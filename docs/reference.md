<!-- Preserved when the installation guide was added on 2026-10-28. -->

# Historical reference and measurements

For current installation instructions, use the [README](../README.md) and
[harness guide](harness.md). This document preserves the earlier setup notes
and measurements. Some paths refer to local artifacts that are excluded from
the source package. Use `localmodel --stats` for the packaged measurements.

Run local models on the GPU, with optional CPU expert weights. Fetch pinned model files, start one model,
and configure OpenCode from the live server.

The [long-context experiments](../experiments/README.md) contain isolated Bonsai
native patches and a Dory Q5 GGUF backend. Both runtimes have built. The
completed comparison covers nine 27B configurations. Dory tests stopped at Matti's request.
See the [final results](../results/native-long-context-20261019/summary.md).
The default selects the measured native 16K MTP draft window configuration.

## Start the current setup

Use three terminals or terminal panes. The launchers are installed in
`~/.local/bin`. If a command is not found, add that directory to the current
shell's path:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

1. Start local search in the first terminal:

   ```sh
   searxng-local
   ```

   Keep this terminal open. SearXNG runs without Docker at
   `http://127.0.0.1:8888`. The current configuration uses DuckDuckGo only.
   Wikipedia and arXiv have not been enabled.

2. Start the model in the second terminal:

   ```sh
   localmodel start
   ```

   Keep this terminal open and wait for the server to finish loading. This
   command selects Bonsai 2 27B, 262144 context tokens, MTP with two draft tokens,
   chained drafting, a 16K MTP draft window, Q4 KV cache, and a 32 GiB RAM
   prompt snapshot cache limit. It uses the trained ProCreations MTP head,
   8192-token checkpoints, recurrent replay, and 128-token chunked prefill.
   Transformer layers and active attention computation use the GPU. The RAM cache stores reusable
   prompt state. The OpenAI-compatible API is at `http://127.0.0.1:8899/v1`.
   Reasoning effort defaults to `xhigh`. For shorter reasoning, use
   `localmodel start --reasoning-effort medium` instead. An omitted model name
   selects `27b-256k-optimized`. Use `--draft-tokens N` to change the draft depth and
   `--no-mtp-chain` to disable chaining.

3. Check readiness in the third terminal, then start OpenCode from the project
   directory where you want to work:

   ```sh
   localmodel status
   curl --fail --silent --show-error --output /dev/null http://127.0.0.1:8888/config
   opencode-local
   ```

   `localmodel status` must report the loaded model and context. The search
   check exits successfully without output when SearXNG is ready. If either
   check fails, inspect its server terminal before starting OpenCode.
   `opencode-local` reads the live model ID and context, loads the local search
   plugin, and uses its private OpenCode state. It starts a standalone OpenCode
   server for the terminal UI. No separate `service start` command is required.

If model files are missing, run `localmodel fetch 27b-256k-optimized`
before step 2. If the native runtime is missing, use the
[native build command](../experiments/README.md#build-or-repeat-explicitly).
If SearXNG's
environment is missing, run `~/Code/local-models/bin/setup-searxng` before
step 1. These dependencies are already installed on the current machine.

To select another measured configuration, replace the model command in step 2:

| Command | Use |
| --- | --- |
| `localmodel start 27b-256k --no-offload` | Stock 27B runtime with full MTP draft history; previous default |
| `localmodel start 27b-192k --no-offload` | Bonsai 27B, 192K context, DFlash; faster measured generation at a filled context |
| `localmodel start 27b-256k-native --no-offload` | Bonsai 27B, 256K target context, MTP2, 16K draft window; measured long OpenCode improvement |
| `localmodel start 27b-256k-native-mma --no-offload` | The native window configuration with MMA threshold 256; no established additional speed benefit |
| `localmodel start 27b-256k-native-lookup --no-offload` | The same native configuration with prompt lookup; best suited to repeated text |
| `localmodel start 27b-256k-native-spill --no-offload` | Native 27B with a RAM tail for active KV; more GPU headroom and slower long-history decode |
| `localmodel start 27b-256k-medium --no-offload` | Stock 27B with medium reasoning effort; the default remains xhigh |

Run one model server at a time. To switch, exit OpenCode, stop the model with
Ctrl-C in its terminal or `localmodel stop`, then start the new preset and
relaunch `opencode-local`. Search can stay running. In a resumed conversation,
use OpenCode's model picker if it still shows the previous model.

To stop the whole setup, exit OpenCode, stop the model, and use Ctrl-C in the
SearXNG terminal.
If you also started a background OpenCode service, manage it separately:

```sh
opencode-local service status
opencode-local service stop
```

Service commands work without a loaded model. They do not stop SearXNG or the
model server. Persistent search-provider and web-permission settings are in
`config/opencode-web.json`. Engine settings are in `config/searxng.yml`.
See [Local web search](#local-web-search) for configuration and provider examples.

## Groot and LFM3 candidates (2026-10-28)

The [2026-10-28 comparison](../results/groot-lfm-20261028/summary.md) tests these
profiles through OMP at about 16K and 192K actual input tokens. Both retain
256K capacity. A full 256K prompt has not been tested.
The default `localmodel start` profile remains Bonsai.

LFM3 measured 160.7 tokens/s at 16K and 95.7 tokens/s at 192K, before
compaction. Its code repairs passed independent checks, but exact recall and
tool instructions failed. Groot measured 34.0 and 26.2 tokens/s respectively.
Groot passed all checks in the 192K workflow, including compaction. Its initial
192K prompt took 259 seconds to process, against 28 seconds for LFM3.
These are single runs of constructed coding history. Run `localmodel --stats`
for the measured settings and limits.

| Profile | Weights | Context | CPU expert layers | RAM cap | RAM prompt cache |
| --- | --- | --- | --- | --- | --- |
| `groot` / `groot-19b` | Official Groot-19B-A3B-261023 Q4_K_M, 11.66 GB | 256K | First 24 of 40 | 32 GiB | 8 GiB |
| `lfm` / `lfm3` / `lfm3-1.7b` | Official LFM3-1.7B Q8_0, 1.85 GB | 256K | None | 16 GiB | 4 GiB |

Both use Q8 KV, one slot, a 2048-token batch, and a 256-token physical batch.
The isolated llama.cpp b11946 runtime includes LFM3 and its tool parser.
Weights and runtime downloads use pinned revisions and SHA256 checks.
The weight downloads contain GGUF data. They do not execute Hugging Face Python code.

Fetch and preview without loading a model:

```sh
localmodel fetch groot
localmodel fetch lfm
localmodel start groot --dry-run
localmodel start lfm --dry-run
```

After the GPU is free, start one candidate and then launch `omp-local` or
`opencode-local`. Both harnesses discover its model ID and context.
They also use the profile temperature and top-p. OMP requires SearXNG.

```sh
localmodel start groot
# Or, after stopping the first server:
localmodel start lfm
```

Use `--context 128k` for an initial shorter-context test. Groot keeps attention
and active KV on the GPU. `--cpu-moe-layers 32` puts more experts in RAM.
`--no-offload` sets that count to zero; it does not guarantee a fit.
`--moe-cache-mib N` reserves GPU memory for CPU expert weights. Test it only
after checking free VRAM. `--mem` and `--cache-ram` remain separate controls.
Groot uses temperature 0.6, top-p 0.95, top-k 20, and repetition penalty 1.0.
LFM3 uses temperature 0.1, top-p 1.0, top-k 50, and repetition penalty 1.1.
MTP and DFlash are not configured for either candidate. Groot's base checkpoint
has an MTP layer, but its official 19B GGUF release has no separate MTP file.

Removal uses the same managed file checks as the existing GGUF profiles:

```sh
localmodel remove groot --dry-run
localmodel remove lfm --dry-run
```

### Why these candidates

[Groot 19B](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023)
is the first coding candidate. It activates about 3B parameters per token.
Its publisher reports Terminal-Bench 2.1 at 52.10 and LongBench v2 at 62.03.
These are publisher measurements, not results from this machine.

[Groot 65B](https://huggingface.co/GrootTeam/Groot-65B-A6B-261023)
is a possible later experiment. It includes an extra 51B-parameter lookup table.
The official Q4_K_M files total 78.75 GB, plus a 5.25 GB MTP file.
It is not a small download relative to the current Qwen IQ2_XS setup.
Its published Terminal-Bench 2.1 score is 63.37. Its aggregate score is below
its Qwen3.8-Flash-Next parent. Compatibility with our Strata pack and cache
implementation is unverified. The 238B and 280B releases are larger again.

[LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)
is a candidate for short tool tasks, structured extraction, and routing.
Its publisher reports BFCL V4 at 71.39, SWE-bench Verified in OpenCode at
43.40, and Terminal-Bench 2.1 in OpenCode at 12.92. The publisher does not
recommend it for agentic coding or complex multi-step reasoning.
Its RULER 128K score is 65.56; a 256K capacity claim does not imply reliable
recall throughout that window. Use it as a helper candidate, not as a confirmed
replacement for Bonsai or Qwen. Scores across publishers use different harnesses.

## Qwen3.8 Flash-Next IQ2_XS with MTP

Use the pinned Strata 0.2.7.1 source with Control 0.1.53 and CUDA engine 0.1.54.
The engine comes from 0.2.8. The older server avoids a tool-parser regression.
See the [update review](strata-0.2.8-review.md).
It uses the original ISTA IQ2_XS weights and their MTP draft layer.
The download includes both model shards. Preparation
keeps the native expert weights in their original shard.

```sh
localmodel fetch qwen3.8
localmodel start qwen3.8 --dry-run
localmodel start qwen3.8
```

Stop the current model before starting Qwen. `serve` is an alias for `start`.
The default uses 256K context, three MTP draft tokens, int8 KV, a 32 GiB resident
expert budget, and a 64 GiB process RAM limit with no swap. The resident expert
budget is part of the process limit. Model files and preparation data stay in
`models/qwen38-ista-iq2-xs` on the native Linux disk.

The API listens on `127.0.0.1:8899`. Run `omp-local` or `opencode-local` after
it is ready. They discover the model and context size. Keep SearXNG running
for OMP. Reasoning effort defaults to `xhigh`. Vision is off. The server and
its children use the same outgoing-connection guard as DeepSeek.

```sh
localmodel start qwen3.8 --context 256k --mem 64
localmodel start qwen3.8 --draft-tokens 2 --expert-ram-gb 32
localmodel remove qwen3.8 --dry-run
localmodel remove qwen3.8
```

Use `--prefill-chunk N` to set the prefill chunk size and `--vram-reserve-mib N`
to change the GPU memory reserve. This profile uses MTP with 1 to 7
draft tokens. Strata adds one accepted token to each verification window.
Prompt lookup is off. `--speculative none` and `--no-offload` are not supported
by this profile. The launcher forces `STRATA_ROUTE_PRIOR=0`.
Upstream 0.2.8 also disables this bias by default on 12 GiB NVIDIA cards.
Bonsai remains the default for `localmodel start` without a model name.

With engine 0.1.52, the 256K / 64 GiB repeat passed the 16K OMP benchmark at 520.1 prompt and
35.49 decode tokens/s. Code, 100 assertions, recall, and strict JSON resume
passed. Peak model RAM was 64 GiB, with no swap or OOM. This run omitted
Windows memory queries and kept the Linux floor and cgroup cap. Its GPU
expert cache was 0.43 GiB. See the
[repeat report](../results/strata-0271-linux-guard-20261028/summary.md).

The previous 128K capacity / 48 GiB cap completed a filled 16K OMP workflow at 1396 prompt
tokens/s and 61.7 decode tokens/s. Code repair, 100 assertions, and recall passed. The
resumed answer had the correct value but used Markdown fences, so the strict
raw-JSON check failed. The smaller 40 GiB cap / 16 GiB expert budget passed
all checks at 609 prompt and 46.3 decode tokens/s. Both are single runs.
See the [complete report](../results/strata-027-20261027/summary.md).
The 128K and 256K settings are capacities, not filled workload validations.

## DeepSeek 1017 with SSD experts

`deepseek` selects Unsloth Dynamic V3.1 UD-IQ1_S and official
[Strata](https://github.com/Niko1221/Strata). The API source is pinned to
the 0.2.7.1 tag (Control 0.1.53). The signed CUDA engine is `strata-dsv41` 0.1.54
from Strata 0.2.8. The older server avoids a tool-parser regression.
The source CLI still reports product version 0.2.7 and build 0.1.53.
See the [update review](strata-0.2.8-review.md).
The new weights are pinned to `8a6ec9cf118e785a362ea94ee123c61c70553be6`.
The following measurements used the previous ISTA GSQ-RCO IQ1_M weights.
They do not measure the new Unsloth weights.
Both Strata profiles use a 64 GiB process cap after the WSL memory increase.
With engine 0.1.52, the 64 GiB DeepSeek repeat passed the 16K OMP checks at 264.7 prompt and
5.15 decode tokens/s, with a 47.58 GiB peak. It did not show a speed gain.
See the [repeat report](../results/strata-0271-20261028/summary.md).
The [update checks](../results/strata-official/20261028-update/summary.md) cover
the source, signatures, model receipts, launch settings, and effective cap.
The 0.2.7 update passed a filled 16K OMP repair, recall, and resume run. Vision stays off.
With a 48 GiB cap and 36 GiB expert cache, prefill measured 284 tokens/s and
decode 5.31 tokens/s. This is one sampled run. See the
[comparison](../results/strata-027-20261027/comparison.json).
The installation uses four pinned Python packages. It does not use Abyss
patches, a compiler, or the private CUDA toolkit from the Dory setup.
The five Unsloth GGUF files use about 217 GiB on native WSL storage on C:.
The model directory is `models/deepseek41-1017-unsloth-iq1-s`.
Bonsai remains the default when you omit the model name.

Stop the current model before switching. Keep SearXNG running. Use these
commands in the model terminal:

```sh
localmodel fetch deepseek
localmodel start deepseek
```

`fetch` installs and verifies the pinned official runtime, resumes missing
weight downloads, and creates an index of the original GGUF files.
`start` uses 256K context capacity, a hard 64 GiB RAM cap, a 36 GiB expert
RAM cache, 4096-token prefill chunks, four CPU expert threads, and zero swap
allowance. Dense layers and attention run on the GPU. Expert weights use GPU and RAM caches with
SSD reads. OMP and OpenCode send `xhigh` reasoning requests. No DSpark
drafter is loaded. The launcher passes `--experts 6` to keep all six routed
experts. It does not select the new five-expert Faster mode.

DeepSeek blocks outgoing connections for the server and its native children.
The local API and file access use their normal paths. `fetch` uses the network
to download pinned files before serving. See the
[security review](../results/strata-official/20261025-security/audit.md).

In the harness terminal, run:

```sh
localmodel status
omp-local --mem 8
```

OMP reads the live model ID and 262144-token context from port 8899.
Its RAM cap is separate from the model's cap. `opencode-local` can use the
same server. OMP still requires local SearXNG to be ready.

The smaller expert cache leaves RAM for other work. On a quiet machine,
you can request a larger cache. You can also select a smaller context:

```sh
localmodel start deepseek --expert-ram-gb 20
localmodel start deepseek --expert-ram-gb 36
localmodel start deepseek --context 16k
```

Keep the expert cache below the hard cap. The process also needs RAM for
dense weights, buffers, and file cache. The cap includes child processes.
The normal server does not monitor free Windows RAM.
The initial stock-runtime tests with 36 GiB and 20 GiB caches crossed the
8 GiB Windows headroom floor while another job used about 37 GiB of RAM.
The test guard stopped those runs. Neither run recorded an OOM event.

The stock API listens only on `127.0.0.1`. Offline mode disables remote
image fetching and server-side MCP tools. The API request monitor and
disk-backed conversation cache are off. Matching prompt prefixes can still
be reused in RAM. The official packer indexes existing tensor bytes and
does not convert the model weights. The native `expert-profile.bin` file
stores expert-routing counts, not conversation text.

Stock Strata accepts `xhigh` in API requests and maps it to DeepSeek `max`
(100). Its server configuration rejects that value. Bare API requests use
the model's default `high` (75). The harnesses send their explicit `xhigh`
setting. `--thinking-budget` sets the separate reasoning-token cap.

The earlier Strata 0.2.4.2 OMP check passed with the normal core prompt, a real file
read, and `xhigh` requests. It processed 2801 input tokens at 55.2 tokens/s
and decoded at 3.6 to 4.0 tokens/s. The complete roundtrip took 73.8 seconds.
Peak server RAM was 16.79 GiB. Windows free RAM stayed above 16.68 GiB.
These are smoke-check figures from a busy machine, not a comparison with
the earlier 16K test. The server was stopped after validation.
See the [stock Strata report](../results/strata-official/20261024-migration/summary.md).

The Strata 0.2.6 benchmarks filled about 16K tokens through OMP. Both completed
settings passed the repair, 100 assertions, three recalled facts, and resume.
The 48 GiB cap with a 36 GiB expert cache gave 257.7 prompt tokens/s and
5.4 decode tokens/s. It used 40.3 GiB at its peak. The 64 GiB cap with a
48 GiB expert cache gave 262.0 prompt tokens/s and 5.6 decode tokens/s.
It used 59.0 GiB at its peak. Windows kept at least 14.9 GiB and 10.1 GiB
free in those completed runs. Both used all six experts and 4096-token chunks.

Use these explicit settings to reproduce either RAM budget:

```sh
localmodel start deepseek --mem 48 --expert-ram-gb 36 --prefill-chunk 4096
localmodel start deepseek --mem 64 --expert-ram-gb 48 --prefill-chunk 4096
```

Run one command at a time. The default now uses 64 GiB, a 36 GiB expert cache,
and 4096-token chunks. The 48 GiB expert cache did not give a clear speed gain
over the 36 GiB cache in these single passes. The 64 GiB run generated more tokens and recovered from
two tool errors, so its longer task time does not measure a RAM penalty.
An earlier 64 GiB attempt crossed the 8 GiB Windows floor and was stopped.
The successful retry ran after a separate job finished and after a targeted
release of clean model-file cache. The routing profile was warmer too.
The tests used an independent memory guard. The normal server does not
monitor Windows free RAM. Filled 256K context remains unverified.
See the [48/64 GiB benchmark report](../results/strata-official/20261025-update-bench/summary.md).

The earlier Abyss runtime passed short-prompt capacity checks at 64K, 128K,
and 256K. Its filled 16K OMP task passed the code repair, 100 assertions,
and three recalled facts. Cold prefill took 37.2 minutes; decode averaged
6.1 tokens/s. These are historical Abyss results. They do not measure the
new stock Strata runtime. Filled 256K context remains unverified.
See the [historical 16K report](../results/deepseek-filled-omp/20261024-0352/summary.md)
and [runtime notes](../experiments/deepseek-strata/README.md).

Use `localmodel start deepseek --dry-run` to inspect the command and settings.
`--context`, `--expert-ram-gb`, `--cpu-threads`, `--gpu-expert-fraction`,
`--vram-reserve-mib`, `--engram-cache-mib`, `--engram-threads`,
`--prefill-chunk`, and `--thinking-budget` control this backend.
`--gpu-expert-fraction` uses a fraction from 0 to 1. The old
`--gpu-expert-cache-gb` flag is rejected for this runtime.
`--no-offload` cannot fit this model on the 12 GiB GPU.
Run `localmodel start --help` for flag descriptions.
Run `localmodel --stats` for measurements.

To reclaim the model's disk space, inspect the deletion first:

```sh
localmodel stop
localmodel remove deepseek --dry-run
localmodel remove deepseek
```

Removal deletes this model's five GGUF files and its generated index. It keeps
the runtime, configuration, and benchmark results. Run `localmodel fetch deepseek`
to restore the files.

## Flint

The pinned model is Unsloth `flint-UD-Q3_K_XL.gguf`, revision
`20468edfc38ab43d6914cc88b08cbecb05d0fc62`. It uses a separate official
llama.cpp `b11803` CUDA runtime. The model and runtime downloads have SHA-256
checks. This setup keeps the existing Bonsai default.

```sh
localmodel fetch flint
localmodel start flint
# The equivalent named preset:
localmodel start flint-256k
```

The default is 262144 context tokens, one slot, Q8 K/V cache, batch 2048,
physical batch 1024, and a 32 GiB RAM prompt cache limit. The API uses port
8899. The first 33 transformer layers place their routed MoE expert weights
in RAM. Attention, active KV, and the other GPU tensors stay on the GPU.
The weights are 18.5 GB and do not fit entirely in 12 GiB of VRAM.
The server process has a 64 GiB RAM cap with the default prompt cache and no swap.
The prompt cache budget adds to the RAM used for expert weights.

The [posted RTX 3060 recipe](https://x.com/DogukanUrker/status/2113060329038970892)
uses the preview checkpoint and disables the prompt cache. To select those
runtime settings for the linked final checkpoint, use:

```sh
localmodel start flint --cache-ram 0
```

Useful controls:

```sh
localmodel start flint --context 128k
localmodel start flint --cpu-moe-layers 36
localmodel start flint --context 192k --batch-tokens 1024 --ubatch-tokens 256
localmodel start flint --dry-run
```

Increase `--cpu-moe-layers` to move more routed experts to RAM and leave
more VRAM free. The range is 0 to 49. `--no-offload` selects zero CPU MoE
layers. That setting cannot fit this quantization on a 12 GiB GPU.
The other context and placement options have not been benchmarked here.

For the first speculation experiment, keep the default expert placement:

```sh
# Ordinary decode baseline:
localmodel start flint --no-prompt-lookup --moe-cache-mib 0
# N-gram candidate. No draft model is required:
localmodel start flint --prompt-lookup --lookup-type ngram-mod \
  --lookup-tokens 16 --lookup-min-tokens 8 --moe-cache-mib 0
```

The installed runtime also supports a GPU cache for expert weights in RAM.
Test it separately with the same expert placement in both runs:

```sh
localmodel start flint --cpu-moe-layers 34 --moe-cache-mib 0 --no-prompt-lookup
localmodel start flint --cpu-moe-layers 34 --moe-cache-mib 256 --no-prompt-lookup
```

Stop the previous server before each start. These are unbenchmarked candidates.
Cutoff 34 moves one more layer's routed experts to RAM than the default.
It can make room for the cache, but this configuration has not passed a GPU
capacity check. The launcher does not adjust placement to fit the cache.
A small cache can reduce speed if expert transfers cost more than CPU work.
After separate measurements, add the n-gram options to the cache candidate
to test both features together.

`--lookup-tokens` sets the ngram-mod maximum. `--lookup-min-tokens` sets the
minimum accepted draft length. Flint uses `min(8, maximum)` when the minimum
is omitted. `--draft-tokens` controls neural drafting and does not set these
n-gram limits. The runtime's expert cache handles batches of at most 32 tokens.
The 16-token draft limit leaves room for verification within that limit.
This limit does not guarantee that the cache can serve every expert request.

Explicit `--prompt-lookup` or `--no-prompt-lookup` overrides
`LLAMA_ARG_SPEC_TYPE`. Explicit `--moe-cache-mib` overrides
`LLAMA_ARG_MOE_CACHE_MIB`. Omit these options to retain the native environment
settings. Add `--dry-run` to preview a command. Use `localmodel start --help`
for the option descriptions. No new optimization is enabled by default.

After the server is ready, start `omp-local --mem 32` or `opencode-local --mem 32`
from your project. They discover Flint and its live context capacity.
The advertised output budget is 32768 tokens. Prompt and output share the
context window. The server uses temperature 1.0, top-p 0.95, and top-k 20.
It preserves reasoning in history. Flint's template has no reasoning-effort
control; an `xhigh` request does not select a trained effort level.
This checkpoint has no configured MTP or DFlash decoder.

The setup check loaded the 256K window and completed a synthetic tool-call
and tool-result round trip. It did not fill the 256K window or benchmark
long coding conversations. See `results/flint-setup-20261022/` and the
[source review](../experiments/flint/AUDIT.md). The linked model download uses
about 17.2 GiB of disk space. Check free space before adding more weights.

## Remote model on port 8898

Start the remote OpenAI-compatible server, or forward its port to
`127.0.0.1:8898`. Then run this command from your project directory:

```sh
opencode-remote
```

The launcher reads the model ID from `http://127.0.0.1:8898/v1/models`.
It also checks IPv6 loopback at `http://[::1]:8898/v1/models` if IPv4 fails.
The endpoint must report one model. It does not require a localmodel preset.
For DeepSeek-V4.1-Flash-1017, the profile allows 1,048,576 total context tokens
and up to 262,144 output tokens. Output shares the context budget with input.
If the endpoint reports a smaller context capacity, the launcher uses that
capacity and prints a message. If capacity metadata is absent, it uses the
1M limit. Configure the model server to support that capacity.
Sampling uses temperature 1.0 and top_p 1.0, as specified in the
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference).
Reasoning mode and effort use the remote server defaults.

This profile uses separate configuration, sessions, goals, and service state
in `state/opencode-remote`. It uses the reviewed goal plugin and the same
SearXNG settings as `opencode-local`. Start `searxng-local` to use web search.
The memory limit also applies to processes that this launcher starts:

```sh
opencode-remote --mem 32
opencode-remote --mem 32 service start
opencode-remote service status
opencode-remote service stop
```

Service commands work when port 8898 is unavailable. The RAM cap does not
limit the separately started remote model server.

## Bonsai subagents for OpenCode

Both `opencode-local` and `opencode-remote` route the built-in `general` and
`explore` subagents to the Bonsai 2 27B server on `127.0.0.1:8899`.
The primary conversation keeps its selected local or remote model.
For example, ask the primary agent to use the `general` subagent to implement a
specified change, or the `explore` subagent to search the project.

Start Bonsai with `localmodel start`, then launch either OpenCode command.
The launcher reads the served model ID and context capacity from the endpoint.
It uses 262144 context tokens if capacity metadata is absent. It saves the
metadata in each profile's `bonsai-subagent.json` file for offline starts.
If no saved model ID exists, start Bonsai once before using subagents.
The profile uses `xhigh` reasoning and up to 16000 output tokens for Bonsai.
The `general` agent can change files. The `explore` agent keeps its built-in
read-only behavior. Neither agent receives a replacement system prompt.

Relaunch the OpenCode command to load these settings. For a background service,
use `opencode-local service restart` or `opencode-remote service restart` when
its current work has finished. The memory limiter covers these subagents and
their child processes within the same OpenCode service. The model server keeps
its separate memory limit.

## Limit OpenCode RAM

Put `--mem` before the OpenCode arguments:

```sh
opencode-local --mem 32
opencode-local --mem 32 run "Fix the parser and run its tests"
```

Bare numbers use GiB. Sizes such as `512M` and `32G` are also accepted.
Each invocation with `--mem` gets a systemd user scope. The limit applies to the combined
RAM used by OpenCode, its private server, and child processes. Shell commands,
build tools, and their child processes share that limit. Swap is disabled for
the scope. The limit does not reserve RAM in advance.

For the background service, apply the limit when you start it:

```sh
opencode-local --mem 32 service start
opencode-local service status
```

The detached service and commands it launches keep the cap after the startup
command exits. All clients of that service share the same budget. If the
service already runs outside the requested cap, startup returns an error.
To replace it with a capped service, use:

```sh
opencode-local --mem 32 service restart
```

Restart interrupts that service's jobs. Pass `--mem` again on later restarts.
Service commands still work without a loaded model. An external `--server`
cannot be capped by the client; cap that server when you start it.

When the group exhausts its budget, Linux can kill a process in the group.
OpenCode reports a killed job as a failure. `OOMPolicy=continue` prevents
systemd from automatically stopping the rest of the group after that failure.
Linux can still select OpenCode itself as the memory victim. This is a RAM
limit, not a promise that allocations return a recoverable exception.

The separately started model server, its RAM snapshots, and SearXNG have
separate budgets. Work submitted to another daemon, such as Docker or pueue,
also needs a limit at that daemon. A new `systemd-run` scope needs its own cap.
The OpenCode scope covers processes that stay in its cgroup.

This uses systemd cgroup v2 resource controls and the existing heavy-management
peak recorder. Inspect scopes with
`systemctl --user status 'opencode-local-*.scope'`. The launcher records peaks
through command exit in `state/opencode/memory-peaks`. A service continues
after that record; the record is not its lifetime peak.
See [systemd resource controls](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html)
and the [local validation](../state/opencode-memory-tests/integration/summary.json).

## OpenCode goals

`opencode-local` loads the reviewed local copy of `opencode-goal-plugin@0.11.1`.
Restart OpenCode to load the new configuration. Start a goal with a clear
completion condition:

```text
/goal fix the failing tests and verify the suite passes
/goal status
/goal pause
/goal resume
/goal clear
```

The plugin sends another prompt when the current response stops before the
goal is complete. It retains the goal through compaction. A new human message
pauses automatic continuation. Recovered goals also start paused.
`/goal resume` starts a fresh budget window.

The defaults allow 10 automatic continuations and one hour per budget window.
The context budget is 262144 tokens. Wrap-up starts at 98 percent of that
budget. The token meter records the largest context since the last compaction.
It does not sum repeated input tokens. Limits are checked between responses.
Set a limit for one goal with `--max-turns`, `--max-minutes`, or `--max-tokens`.
For example, `/goal fix the parser --max-turns 4 --max-minutes 30`.
Edit [config/opencode-goal.json](../config/opencode-goal.json) to change defaults.

Only human slash commands can create or resume goals in this configuration.
Agent-facing goal tools and extra agents are disabled. Completion uses the
model's adjacent evidence and completion markers. The separate completion
verifier is disabled. An evidence claim does not prove that tests passed.
Goal control commands use a model response to display their result.

Goal state and lifecycle logs contain plain text. They use the private
`state/opencode/goals/` directory, outside project checkouts. Each session uses
`state.json.sessions/<SHA256-of-session-ID>/`. The base directory has mode 0700.
Setting, changing, or clearing a goal can change the system prompt and cause
a new prompt prefill. Ordinary continuations keep the goal block stable.

The package and its Zod dependency use pinned local files. They do not update
automatically. See the [audit report](../results/opencode-goal-audit-20261019.md)
and [file manifest](../config/opencode-goal-plugin-manifest.json).

## OMP local and remote

Use `omp-local` or `omp-remote` to start OMP with this setup. Both commands
are linked in `~/.local/bin`. Start SearXNG and the model in separate terminals:

```bash
searxng-local
localmodel start
```

Both launchers check the configured SearXNG `/healthz` endpoint before they
start an OMP session. If search is unavailable, they exit and ask you to start
`searxng-local`. This check sends no search query. Help, version, configuration,
agent listing, and completion commands remain available without search.

Then start OMP in the project directory:

```bash
cd ~/Code/your-project
omp-local --mem 32
# Or use the model exposed on localhost:8898:
omp-remote --mem 32
```

`omp-local` reads the current localmodel endpoint and its context capacity.
`omp-remote` discovers the model on port 8898 through IPv4 or IPv6. The remote
context defaults to 1048576 tokens and decreases if the server reports a
smaller capacity. Its output limit is up to 262144 tokens within that budget.
The remote task agents use the live model on port 8899. The launcher reads
its model ID and context capacity, then sets its output budget and sampling.
This includes Qwen3.8-Flash-Next, started with `localmodel start qwen3.8`,
and other models served by `localmodel`. The bundled `task`, `scout`,
`reviewer`, `security-reviewer`, and `sonic` agents use this task route.
The main conversation, `plan`, `smol`, `slow`, and `title` roles keep the
remote model on port 8898.

The provider name `bonsai` remains for compatibility. It does not restrict
the task model: Qwen uses `bonsai/qwen3.8`. There is no cached fallback.
If port 8899 has no ready model, the launcher exits. Restart `omp-remote`
after changing the task model or its context capacity.

`omp-remote` can resume a session saved with another `deepseek-v4.1` profile.
The saved ID must be `deepseek-v4.1` or start with `deepseek-v4.1-`.
The current remote model must match the same rule. This includes
`deepseek-v4.1-flash-1017-b300-iq3` and
`deepseek-v4.1-1017-dual-h200-iq3`. Resume uses the current model's ID and
context limit. It requires one matching model in the main `model` provider.
Other model families, task agents, and `omp-local` keep their normal
resume checks. Restart `omp-remote` once to load this runtime change.
Restart it after a server profile switch so it can discover the new model ID.

Both profiles default to `xhigh` reasoning. Compaction uses the same model
as the conversation. It starts at the lower of 80% of context or context
minus the output limit and 16384 tokens. Compaction runs synchronously and
keeps 20000 recent tokens. Each model has a separate threshold.

Changes in OMP's settings panel persist across launches. Local and remote
profiles keep separate preferences. OMP saves them in
`state/omp-local/agent/config.yml` and `state/omp-remote/agent/config.yml`.
The launcher seeds a missing file from `config/omp.json`. It does not replace
an existing settings file. The alternative `config.yaml` name also works.

The launcher updates `state/omp-local/launcher.yml` or
`state/omp-remote/launcher.yml` as a native OMP config overlay. This file
controls model routing, context thresholds, local search, network restrictions,
and process containment settings. It does not override display, composer,
reasoning, approval, or other ordinary preferences. Model metadata remains in
the generated `agent/models.yml` file. Restart OMP once to load this change.

The built-in web search tool uses SearXNG on port 8888. SearXNG uses the
engines in `config/searxng.yml`. The built-in `read` tool can fetch a URL and
convert HTML locally. Search failures return an error. URL fetch failures
do not use a hosted reader. Sites that require JavaScript or block direct
requests can still fail.

Use these commands inside OMP:

```text
/goal Implement the requested change and verify it
/goal show
/goal pause
/goal resume
/goal budget 500000
/goal drop
/skill:NAME
```

The goal budget counts uncached input, cache writes, and output tokens.
It is a soft stop, separate from the RAM cap. Goals can continue between
turns in the interactive terminal. Skills load from `~/.agents/skills` and
the project's `.agents/skills`. The profile also loads `~/AGENTS.md` and
project instructions. OpenCode plugins and conversation history do not
transfer to OMP.

Put `--mem` first. `--mem 32` limits OMP and its child processes together
to 32 GiB, with no swap. This includes shell commands and task agents.
The model server and SearXNG keep their separate memory limits. LSP servers
run privately. The shared LSP broker and named background-service launcher
are disabled so they cannot start workers outside this limit. Ordinary
shell child processes remain available.

Edit `config/omp.json` for persistent settings. The launcher writes the
model metadata and settings under `state/omp-local/agent` or
`state/omp-remote/agent` at each normal launch. Sessions, caches, and auth
files stay separate from the global OMP profile. `--profile` is reserved by
the launcher. `--config FILE` can supply an explicit per-run overlay.
The profile asks before executing commands or delegating tasks. Project
settings and explicit overlays can change OMP behavior.
Automatic foreign integration discovery, project MCP configuration, and
unrelated model providers are disabled. Add MCP servers explicitly to the
profile's `mcp.json` file if needed. OMP keeps the built-in tools enabled.

These commands use a private OMP 18.8.6 bundle. The checksum manifest and
seven patches are in `config/omp-runtime.json`. The patches prevent automatic
xAI search and hosted URL-reader fallbacks. They also restrict collaboration
destinations and disable session uploads.
They do not change the globally installed `omp`. If that installation
changes, the launcher stops until the new runtime is reviewed.

Validation used live endpoint metadata and isolated mock completions. The
checks confirmed model routing, context limits, `xhigh` requests, SearXNG
success and error handling, direct URL reads, and the absence of unrelated
startup requests. The memory check confirmed the same 1 GiB limit for OMP,
a task worker, shell commands, and their children. See
`state/omp-checks/web-summary.json` and the memory check summaries under
`state/omp-checks`. These checks did not run inference on either real model.

### Private collaboration through Tailscale

The browser address is <https://YOUR_NODE.YOUR_TAILNET.ts.net:7467>.
The direct IP address is <https://100.64.0.10:7467>.
Connect the client device to Tailscale first. Tailnet access rules apply.
The server listens directly on the node's Tailscale IPv4 address, port 7467.
It uses a private CA certificate. Accept the browser certificate warning
for the address you use. Alternatively, import `state/collab/tls/ca.pem`
into the client trust store. `omp-collab info` prints its SHA-256 fingerprint.
The OMP wrappers trust this CA explicitly. They do not skip TLS verification.

The backend is an enabled user service. Control it without a running model:

```bash
omp-collab start
omp-collab status
omp-collab logs
omp-collab info
omp-collab stop
```

This setup does not use Tailscale Serve, Funnel, or Windows port forwarding.
It does not require an administrator command. Stop the service to close its
listener. The enabled user unit starts when the user service manager starts.
If Tailscale is not ready, the unit retries after three seconds.

Start a new `omp-local` or `omp-remote` process to load the private runtime.
Existing processes keep their loaded code. Neither profile starts a room
automatically. In OMP, use:

```text
/collab view
/collab start
/collab stop
```

Open the generated browser link on another Tailnet device. A view link
permits reading. A control link also permits prompts and approval responses.
Treat both links as credentials. A room can expose conversation history,
tool results, and task-agent content. Stopping a room ends access through
that room. It does not remove copies that a guest already received.

Both wrappers read `config/omp-collab.json`. They accept only the configured
Tailscale origin, including explicit relay arguments and project overrides.
The collaboration socket verifies TLS with the private CA file and bypasses
HTTP proxies. `/share` uploads, including GitHub gists, are disabled in this
private runtime. The global `omp` command keeps its original behavior.

The browser assets and relay run on this machine. The local browser build
removes the upstream analytics script and blocks external scripts, images,
fonts, and WebSocket destinations with its response policy. The relay stores
rooms in RAM. It does not write transcripts or log payloads. Journal logs
contain service starts, stops, and runtime errors. OMP still stores its own
sessions in each private profile.

The service has a 512 MiB RAM cap and no swap. It limits rooms, connections,
frame sizes, and socket buffers. The source review is in `collab/AUDIT.md`.
Protocol checks use synthetic sessions. Their summaries are under
`state/collab-checks` and `state/collab/client-checks`.
The final deployment checks are in `state/collab-checks/final-deployment.json`.

The listener binds only to `100.64.0.10`. It does not bind to the physical
LAN address or all interfaces. It accepts Tailnet peer addresses and the
configured HTTP Host and browser Origin. Change both collaboration configs
and the certificate if this node's Tailscale address changes.

## Commands

```sh
localmodel list
localmodel fetch bonsai2-2b
localmodel fetch bonsai2-4b
localmodel fetch bonsai2-9b
localmodel fetch 27b-256k
localmodel start 27b-256k
```

`fetch` checks the model size and SHA256. It also installs the pinned CUDA runtime
for the new Bonsai profiles. An interrupted model download can resume.
`pull` is an alias for `fetch`. `serve` is an alias for `start`.

Use `remove` to delete the weights and local download data selected by `fetch`.
It keeps configuration, presets, runtimes, and other weights. It refuses files
that a running process uses. Stop that process before removal.

```sh
localmodel remove flint --dry-run
localmodel remove flint
localmodel remove 27b-256k-native --dry-run
localmodel remove bonsai2-27b --speculative mtp
```

Base names select plain weights. Presets select their configured decoder.
`--speculative none` selects plain weights. `--speculative mtp` selects only
the MTP file. `--speculative dflash` selects plain weights and the draft file.
Some profiles share a weight file. Removal affects each profile that uses it.
For legacy `dory`, removal deletes its NVFP4 Hugging Face model cache.
Run `fetch` with the same name and flags to restore the removed files.

`start` defaults to `27b-256k-optimized` with MTP depth 2, chaining on, and a 16K
draft window. The target context remains 256K. It runs in the
foreground. Use Ctrl-C or `localmodel stop` to stop it before
you select another model. The stop command verifies the saved controller PID.
It does not stop legacy servers that were started outside the manager.
The start command refuses to replace a model that is already running.

New starts listen on `127.0.0.1:8899`. Select another port with `--port PORT`.
The defaults keep transformer layers and active KV cache on the GPU.
`--no-offload` selects the same behavior explicitly.
The new Bonsai profiles disable automatic memory fitting and place all
transformer layers on the GPU. The native loader also uses a 265 MiB
CPU-mapped token embedding buffer.
A failed runtime allocation stops startup.
Windows can still manage GPU memory outside the inference runtime.

```sh
localmodel start dory --rounds 1
localmodel start dory --rounds 2
localmodel start dory --rounds 4
localmodel start bonsai2-9b --context 128k
localmodel start bonsai2-27b --context 256k --no-offload
localmodel start bonsai2-27b --dry-run
localmodel stop
```

The Dory default is two rounds at 256K with FP8 last-loop KV cache.
`--loops` is an alias for `--rounds`.
Round selection requires a server restart. It applies to Dory.
The existing `dory sweep` command remains available.
CPU weight offload is available only for Dory through `--cpu-offload-gb N`.

## Measured presets

Use a preset to select the model, context, decoder, cache types, and host cache
budget together. Base model names retain their plain-decoding defaults.

```sh
localmodel --stats
localmodel presets
localmodel fetch 27b-192k
localmodel start 27b-192k
# In another terminal:
opencode-local
# Stop the server before you select another preset:
localmodel stop
localmodel start 27b-256k
```

`localmodel --help` lists commands and the default model.
`localmodel --stats` shows preset settings, filled-context measurements,
native application prompts, and actual OpenCode tool-task rates. It reads
saved results and does not start a model. `localmodel presets` and
`localmodel presets --json` remain available for the preset records.
`localmodel start --help` describes each flag. Explicit flags override preset
settings. For example, `localmodel start 27b-256k --cache-ram 0` disables host
snapshots for that run. The generated OpenCode model name includes the preset.
The context limit comes from the live backend.

See [the optimization measurements](../results/optimization-summary.md) for
context curves, application prompts, cache trials, and recorded failures.
The JSON files retain each command and its raw measurements. Filled-context
tests check capacity with a repetitive prompt. Application tests use separate
prompts. Neither test establishes general coding or long-context accuracy.

## Native runtime comparison

The 2026-10-19 27B comparison is complete. All nine configurations passed the
six OpenCode phases, filled-context capacity, and 107-schema replay. The
[report](../results/native-long-context-20261019/summary.md)
contains settings, actual context use, cold prefill, decode rates, compaction,
tool checks, capacity, and memory peaks. It preserves the raw request records.

The completed stock 27B control measured 54.4 tokens/s on long OpenCode history
and 90.6 after compaction. The native full-draft control measured 53.3 before
compaction. The native 16K MTP draft window measured 64.5 and 88.9 tokens/s.
All three completed before the reported CPU load. The initial input used about
213000 tokens. The target
context stayed at 262144 tokens. The smaller window restricts draft history.
It does not remove target history.

The stock cold first-token time was 402.4 seconds. The native window case took
386.8 seconds. Warm follow-ups, including compaction, took 75.7 and 78.2 seconds.
Generated output and summary lengths differ, so these wall times do not isolate
decode speed. The separate capacity hook used 262079 input tokens and 64 output
tokens in the 262144-token window.

Use these named configurations:

```sh
localmodel start                         # Optimized runtime, trained head, MTP2.
localmodel start 27b-256k-native          # Previous native configuration.
localmodel start 27b-256k                 # Stock runtime, full draft history.
localmodel start 27b-256k-native-mma      # Also select MMA threshold 256.
localmodel start 27b-256k-native-lookup   # Also enable ngram-mod lookup32.
localmodel start 27b-256k-native-spill    # Keep 128K active KV positions in VRAM.
localmodel start 27b-256k-medium          # Stock runtime with medium reasoning.
opencode-local                          # Discover the running configuration.
```

The native window preset sets the MMA override to zero. The combined window
and MMA-256 preset measured the same 64.5 tokens/s before compaction, 92.6
afterward, and 66.2 at filled capacity. Its warm follow-ups took 80.2 seconds.
The window-only case reached 65.4 tokens/s at filled capacity. These single
trials do not establish an additional MTP2 speed benefit from the MMA override.
Use `27b-256k-native` as the measured native option. Its run completed before
the reported host load. The separate MTP1 MMA comparison measured 13.3 tokens/s
with the override off and 56.0 with it on. Those runs overlap reported host load.

The lookup case measured 59.6 tokens/s on long OpenCode history and 88.1 after
compaction. It reached 199.7 tokens/s on the repetitive capacity prompt. This
prompt favors lookup. It does not represent general OpenCode throughput.
The lookup case overlaps reported host load. Its workflow times need that
qualification. The existing calibrated `27b-256k-lookup` preset remains available.

These configurations use GPU transformer layers and GPU attention computation,
Q4 target and draft KV, and a 32 GiB RAM snapshot limit. The native presets use
xhigh effort. The separate medium preset changes reasoning effort.
The spill preset moves 2304 MiB (2.25 GiB) of active KV to RAM. It passed all six
OpenCode phases, filled 256K capacity, and the 107-schema check. It measured 29.8 tokens/s
before compaction, 92.8 afterward, and 24.4 at filled capacity. Peak GPU memory
was 9704 MiB, compared with 11837 MiB for the native window preset. Its inference
scope peaked at 22923 MiB of RAM. Use it when GPU headroom takes priority over
long-history throughput. This run overlaps the reported host-load interval.
The [placement record](../results/native-long-context-20261019/kv-spill-runtime-validation.json)
confirms the host allocation and GPU prefetch pipeline. This active KV placement
is separate from the RAM snapshot limit. The current comparison excludes 9B.
Dory tests stopped at Matti's request. Its
[saved results](../results/native-long-context-20261019/dory-current/summary.md)
retain the completed checks and the canceled tool-schema check. On 2026-10-19,
the default changed to `27b-256k-native`. Stock, lookup, spill, medium, and MMA presets remain explicit choices.
The reported CPU-load interval remains open. Later trials carry that qualification.

## Trained MTP head and optimized default (2026-10-26)

The default is `27b-256k-optimized`, selected at Matti's request. The previous
`27b-256k-native` preset remains available. The optimized configurations use a
separate source build. The complete comparison is in the
[2026-10-26 report](../results/bonsai-next-20261026/summary.md).

```sh
localmodel fetch 27b-256k-optimized
localmodel start 27b-256k-optimized --mem 48G
# Alternative: use the original teacher MTP head with the same optimizations.
localmodel start 27b-256k-optimized-teacher --mem 48G
omp-local --mem 32
```

Start the model and OMP in separate terminals. SearXNG must already be ready.
OMP discovers the model and its full 262144-token capacity. Both configurations
keep MTP depth 2, chaining, xhigh effort, Q4 target and draft KV, a 16K draft
window, GPU layers and GPU attention. Both optimized presets use checkpoints
inside long messages at about 8192-token intervals, recurrent replay, and
128-token chunked prefill. `27b-256k-optimized` uses the trained ProCreations
head. The teacher alternative uses the original head.

Both passed the complete 64K OMP workflow. The trained configuration measured
84.5 tokens/s before compaction and 103.1 afterward, with a 63.7-second cold
first token. The previous configuration measured 82.2, 101.2, and 80.9 seconds.
Generated output lengths differ, so workflow wall times do not isolate speed.
At 192K, the trained configuration measured 65.3 and 96.6 tokens/s and passed
all repairs, recall, resume, and compaction checks. Its cancellation-recovery
response continued the canceled task and exhausted the 4096-token test limit.
The server recovered normally afterward. This single failure is not attributed
to the engine or head without a matching baseline. Matti requested promotion
with this limitation recorded. The 256K value is configured
capacity; this update filled up to 192K in actual OMP workflows.

Use `--mtp-head teacher` to select the previous head. Use `--checkpoint-every 0`
to disable checkpoints inside messages. Use `--replay` or `--no-replay` to
select rollback. These checkpoint and replay controls require
`--runtime bonsai-next`. `--gdn-mma` enables the chunked prefill path.
`--no-gdn-mma` disables it. MTP2 needs `--ubatch-tokens 128` or larger to use
that path. The `27b-256k-next` and `27b-256k-replay` presets retain batch64
for comparison. The former disables replay; the latter enables it.

`localmodel fetch bonsai2-27b --speculative mtp --mtp-head procreations` builds
the trained-head file from pinned donor ranges and the verified local target.
It preserves all 851 target tensors and the current tokenizer and chat template.
The ready-made community file changes the template, so this setup does not use
it. The trained head adds about 96 MiB. The [head audit](../experiments/bonsai-next/head-audit.md)
records the checks and the sparse-download verification limit.

`localmodel remove 27b-256k-next --dry-run` previews removal of the new weights.
The previous teacher weights and native runtime remain separate.
`--slot-save-path DIRECTORY` enables manual slot save and restore through the
local API. The directory must exist. Cache files contain prompt state. This
flag does not provide automatic persistence. Short save/restore and malformed
replay-state rejection passed. Long-history MTP draft-side disk persistence
has not been validated or added.

See the [source review](../experiments/bonsai-next/source-audit.md) and
[OMP benchmark instructions](../experiments/bonsai-next/README.md). Only public
source fixtures enter the benchmark logs. Each test model has a 48 GiB cap
and no swap. The monitor stops it if either host has less than 8 GiB free.

## 27B MTP depths at 256K

These measurements are from the full repeat, completed at 14:48 CEST.
Matti reported other machine work from 13:10 CEST during the first sweep.
That period overlaps depth 4/off and depth 1/on. The repeat ran all six cases
in reverse order. The [comparison](../results/27b-mtp-chain-256k-rerun1/comparison.md)
retains the first measurements and their time overlap. Launch defaults remain
MTP depth 2 with chaining on.

The 2026-10-19 OpenCode comparison uses the current `xhigh` effort and 32 GiB
RAM snapshot limit. It compares draft depths 1, 2, and 4 with chaining off and
on. All cases use the same pinned model and runtime, Q4 target and draft KV,
one slot, batch 256/64, and zero CPU weight offload.

Each private OpenCode conversation starts with about 213000 actual input tokens
in a 262144-token window. The constructed history contains 204 native messages
with 66 read-tool results from pinned public C++ source. The workflow checks
three related code repairs, fact recall, manual compaction, and continued
coding and recall after compaction. It does not use private conversations.
Post-compaction requests use 17622 to 19802 input tokens. Recall checks use
the summary and retained recent messages. The summaries alone do not contain
all three tracked facts. This tests OpenCode's normal manual compaction path.
It does not test the automatic compaction threshold.

| MTP depth | Chaining requested | Long history tok/s | After compaction tok/s | Compaction s | Warm follow-ups s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Off | 27.3 | 74.6 | 47.2 | 112.7 |
| 1 | On | 7.2 | 40.9 | 123.2 | 361.1 |
| 2 | Off | 58.4 | 92.4 | 62.2 | 109.3 |
| 2 | On | 53.8 | 87.9 | 25.4 | 72.1 |
| 4 | Off | 38.8 | 48.7 | 51.4 | 111.6 |
| 4 | On | 41.9 | 57.3 | 31.6 | 92.6 |

All six configurations passed all six workflow phases. All 81 recorded
requests returned HTTP 200 without recorded errors. Rates divide generated
tokens by native decode time. Validation confirms identical native history
messages, workload and recorder source, weights, runtime, and inference commands
between corresponding cases in both sweeps. Session directories and creation
times differ. Actual provider input counts are recorded for each request.
Warm follow-ups cover the five phases after the first coding turn, including
compaction. They exclude the initial cold prompt prefill. Output lengths and
tool-call counts differ between runs, despite the same seed and task.
These single runs do not establish a statistically significant difference
between the two depth-2 settings.
The depth-1 control measured 27.3/off and 7.2/on without any chain execution.
The first sweep measured 7.3/off and 21.0/on. This variation remains unexplained.
It cannot be attributed to chained execution. Depth 2 was faster in both sweeps.

The current default is depth 2 with chaining on, as requested after this
comparison. Depth 2/off had the highest native decode rate in this repeat.
Use `localmodel start --no-mtp-chain` for that selection. Depth 2/on had the
shortest warm workflow. Its compaction generated 948 tokens; depth 2/off
generated 2921. Their compaction decode rates were 63.4 and 64.4 tok/s.
The shorter summary accounts for much of the wall-time difference. These runs
do not establish a general chain benefit or a statistically significant winner
between the two depth-2 settings.

The default depth-2/on run reached its first token in 404.5 seconds from a cold
cache. Warm long-history follow-ups reached their first token in about
1.3 seconds. The first request after compaction processed the rewritten history
in 22.4 seconds before its first token. Later requests reused the new prefix.
The inference scope peaked at 20.1 GiB RAM and total GPU use at 11884 MiB.
The 32 GiB host cache is a limit, not RAM allocated at startup.

The repeat records Linux load, available RAM, GPU clocks, and bounded Windows
observations. Linux CPU medians were about 5 to 10 percent. Five cases have
Windows VM memory samples. Depth 4/on has no such sample. A separate Windows
engine sample found little non-VM activity among valid rows, but some rows were
invalid. These observations do not prove that the host was idle throughout.
Depth 4/on also had a slower cold prefill at 471.6 seconds. The load records
do not establish its cause.

Chained drafting defaults on for Prism MTP. `--no-mtp-chain` disables it.
`--mtp-chain` enables it explicitly. `LLAMA_MTP_CHAIN=0` or `1` can set the
selection when neither flag is present. Explicit flags take priority.
Depth 1 cannot form a multi-token chain, so its on/off pair is a control.
Logs confirm actual chain execution for depths 2 and 4 with the flag on.
The pinned CUDA source has an attention dispatch threshold between two and
three query rows. MTP depth 2 normally crosses this threshold. This can
explain a nonlinear speed change, but individual kernel timing was not
measured. See [the dispatch review](../results/27b-mtp-chain-256k/depth-dispatch-notes.md).

See [the repeat summary](../results/27b-mtp-chain-256k-rerun1/summary.md) for request
counts, cache reuse, draft acceptance, GPU and RAM use, and correctness checks.
The [repeat plan](../results/27b-mtp-chain-256k-rerun1/PLAN.md) defines its controls.
The [validation record](../results/27b-mtp-chain-256k-rerun1/validation.json)
and [cleanup record](../results/27b-mtp-chain-256k-rerun1/cleanup.json) retain the checks.
The [chaining review](../results/27b-mtp-chain-256k/chaining-audit.md) records the
bounded source and tool-parsing audit. The optional path uses the existing
binary and weights. It does not install another component.

<!-- measured-configurations -->
Measured configurations on the RTX 4070 SUPER, 2026-10-17/18. All keep weights
and attention computation on the GPU. K means 1024 tokens. First-token times
below use a filled, repetitive prompt and 64 generated tokens. These single
runs measure capacity and speed. They do not rank answer quality.
The recorded Bonsai runs used `medium` reasoning and an 8 GiB RAM cache.
Current launch defaults use `xhigh` and a 32 GiB RAM cache. These combined
defaults were tested in the separate 27B OpenCode comparison above. The
filled-context rates below retain their earlier measured settings. The other
presets have not been measured with the combined new defaults.
The `27b-256k` filled-context row used MTP depth 1. The preset now launches
depth 2 with chaining on.

| Preset | Decoder | KV | Host cache GiB | First token s | Filled-context tok/s | GPU MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `9b-128k` | mtp 2 | q8_0 | 8 | 64.95 | 136.84 | 6511 |
| `9b-192k` | mtp 2 | q8_0 | 8 | 113.72 | 106.28 | 7751 |
| `9b-256k` | mtp 2 | q8_0 | 8 | 174.38 | 100.70 | 8991 |
| `27b-128k` | dflash 3 | q4_0 | 8 | 194.39 | 99.97 | 11002 |
| `27b-192k` | dflash 3 | q4_0 | 8 | 342.01 | 84.63 | 11335 |
| `27b-256k` | mtp 1 | q4_0 | 8 | 535.55 | 25.43 | 11790 |
| `dory-1loop` | 1 loop | FP8 last-loop | 0 | 173.32 | 14.38 | 11185 |
| `dory-2loop` | 2 loops | FP8 last-loop | 2 | 270.71 | 9.16 | 10499 |
| `dory-2loop-nocache` | 2 loops | FP8 last-loop | 0 | 278.06 | 7.39 | 11175 |
| `dory-4loop` | 4 loops | FP8 last-loop | 0 | 459.23 | 5.36 | 11199 |

GPU figures show total device use. Both Dory two-loop values are snapshots;
the other table values are sampled peaks. Windows manages physical residency.
The engine settings do not prove that Windows performs no memory transfers.

The native application prompts use 4096 input tokens. The code prompt asks for
a small edit to supplied code. It can benefit strongly from prompt lookup.

| Preset | Ordinary generation tok/s | Code-copy/edit tok/s |
| --- | ---: | ---: |
| `9b-256k` | 152.69 | 229.65 |
| `9b-256k-edit` | 153.32 | 430.55 |
| `27b-128k` | 77.76 | 138.99 |
| `27b-192k` | 76.84 | 139.82 |
| `27b-256k` | 74.31 | 91.70 |
| `27b-256k-lookup` | 57.43 | 192.08 |

Actual OpenCode coding tasks used an isolated workspace. Each task required
two file reads and one edit. The resulting source and external tests passed.
Native decode excludes prompt processing and tool time. Workflow throughput
includes those costs. The tasks generated different output lengths. Two small
tasks do not establish general repository-task quality.

| Context and decoder | Output tokens | Task s | Native decode tok/s | Workflow output tok/s |
| --- | ---: | ---: | ---: | ---: |
| 192K dflash | 208 | 4.79 | 105.43 | 43.45 |
| 256K mtp | 349 | 7.07 | 87.84 | 49.37 |

Long-conversation and manual-compaction checks passed for 27B DFlash at 128K,
27B MTP at 256K, and 9B MTP with lookup at 256K. Each private OpenCode test used
about 54K initial input tokens and retained all three checked facts after
compaction. Automatic compaction at the context threshold was not tested.

The 64K conversation-switch tests restored 65532 cached tokens. First token
fell from 82.70 s to 1.65 s for 27B DFlash and from 34.44 s to 1.54 s for 9B MTP
with lookup. Their inference-scope RAM peaks were about 9.65 GiB and 9.74 GiB.
The configured prompt-cache budget is 8 GiB, separate from the rest of the server.

Dory's separate 32K host-cache check restored 32752 tokens from RAM with zero
GPU-cache hits. First token fell from 13.19 s to 0.37 s. The runtime reported
660307968 restored bytes. Its 2 GiB host cache also passed the filled 256K check.
The first actual OpenCode compaction trial used a private 2048-token output
limit. Dory used all 2048 tokens for reasoning and produced no summary. That
trial failed. The normal generated profile allows 16000 output tokens. The
small-budget failure does not establish that normal-budget compaction fails.
The separate 16000-token retry completed its first two turns in 223.57 s and
77.30 s. The follow-up returned all three facts. Compaction had no completed
summary after 772.95 s. The trial was then cancelled after Matti excluded Dory
from the OpenCode selection. The cancellation is not a production token-limit
failure. Post-compaction fact retention remains unverified for Dory.

The final [GPU contention check](../results/gpu-contention-check.json) found only
Dory using the Linux GPU device. Ollama had no loaded models. The busy Windows
GPU engine belonged to the WSL VM. NVIDIA reported 31 percent GPU use and 55 W
during generation. After the private trial stopped, use fell to zero and power
to about 7.5 W. Dory remained loaded at 10503 MiB with zero active requests.
No competing inference process was found. The WSL NVIDIA process list was empty
even while Dory was active; device handles and Windows counters supplied the
process attribution.

Normal-batch 27B DFlash at 256K started after the WSL loader fix, but its 4K
ordinary prompt took 64.8 s to the first token and decoded at 9.37 tok/s. Windows
reported 976 MiB shared GPU memory for the VM, versus 160 MiB in the MTP sample.
The smaller 8/8 batch configuration reached 32.22 tok/s and slowed prompt
processing. Neither result is selected as the primary 256K configuration.

The [full summary](../results/optimization-summary.md) retains native 64K timings,
cache evidence, OpenCode usage, RAM, exact flags, failures, and raw source links.
<!-- /measured-configurations -->

## Models

| Profile | Weights | Default context | Runtime |
| --- | --- | ---: | --- |
| `bonsai2-2b` | Official PQ2_0 | 262144 | Prism CUDA 12.8 |
| `bonsai2-4b` | Official PQ2_0 | 262144 | Prism CUDA 12.8 |
| `bonsai2-9b` | Official PQ2_0 | 262144 | Prism CUDA 12.8 |
| `bonsai2-27b` | Official PTQ1_0, no draft head | 262144 | Prism CUDA 12.8 |
| `dory` | NVIDIA NVFP4 | 262144 | Pinned vLLM Dory branch |
| `bonsai-ptq1` | Existing PTQ1_0 MTP | 98304 | Existing CUDA bundle |
| `bonsai-pq2` | Existing official PQ2_0 profile | 65536 | Existing CUDA bundle |
| `bonsai-abliterated` | Existing PQ2_0 MTP | 32768 | Existing CUDA bundle |
| `bonsai-abliterated-ptq1` | Existing PTQ1_0 | 131072 | Existing CUDA bundle |
| `saluki` | Existing IQ2 mix profile | 32768 | Existing CUDA bundle |

`bonsai` and `bonsai2` select `bonsai2-27b` in the unified command.
The existing `bonsai2-server` command retains its original default.
`localmodel list` shows which weights are installed.
The missing legacy PQ2 and Saluki files are not fetched unless selected.
The new Bonsai profiles serve text. Vision projectors are not installed.

The new Bonsai runtime is Prism release `prism-b10777-e3de715`, commit
`e3de715222b2667c54a267441159a5729f11155e`. It includes the sm89 CUDA target.
The launcher sets an explicit single-GPU tensor split. This avoids a
[WSL draft-loader failure](https://github.com/ggml-org/llama.cpp/issues/29044)
when CUDA reports zero free memory. It does not enable memory fitting.
The runtime archive and each GGUF have a pinned size and SHA256 in `config/`.
The new Bonsai defaults use FP16 K and V cache for 2B and 4B, Q8_0 for 9B,
and Q4_0 for 27B. All four use FlashAttention, one sequence, 256 batch tokens,
and 64 microbatch tokens.
Use `--batch-tokens`, `--ubatch-tokens`, and `--kv-cache` to test another setting.
`--kv-mean-center FILE` selects a model-specific cache calibration file.

## Runtime controls

The new Bonsai profiles allow a 32768 MiB host prompt cache. It stores
inactive GPU state for later restoration. Transformer layers and attention
computation use the GPU. GPU prefix reuse handles ordinary conversation continuation.
Use `--cache-ram MIB` to change the host cache limit. Zero disables snapshots.

Use `localmodel start PROFILE --mem 32` to set a hard 32 GiB RAM limit.
The limit includes the model process, its children, and Linux file cache.
The process group cannot use swap. Bare numbers use GiB. Sizes such as
`32768M` are also accepted. This flag overrides the profile's process cap.
It does not reduce cache budgets. For example, use `--cache-ram 0` to disable
prompt snapshots when testing a smaller RAM cap.
The cache grows as snapshots are stored. A larger limit can retain more
conversation state. It does not extend active context or increase token
generation speed. The original 8 GiB limit was a conservative test budget.

The host cache shares the server's system RAM cap. At the default 32 GiB cache
limit, the 27B profile has a 40 GiB process cap. Other new Bonsai profiles have
a 36 GiB cap. These caps retain the process headroom from the original
8 GiB-cache trials. Increasing `--cache-ram` also increases the process cap.
Smaller cache limits retain minimum process caps of 16 GiB for 27B and 12 GiB
for the other profiles. Swap is disabled for each server.
These are limits, not memory reserved at startup. Apply new limits and
reasoning effort by restarting the model, then relaunching `opencode-local`.

Dory has a separate native host KV cache. Use `--kv-offload-gb 2` to reserve
2 GiB for cached blocks. Its default is zero. This option works with
`--no-offload`: weights and attention computation stay on the GPU. It does not
use CPU weight offload. It does not require LMCache.
The tested `dory-2loop` preset enables this 2 GiB cache. The base `dory` profile
keeps its zero-cache default. `dory-2loop-nocache` retains the earlier comparison.

Both host caches store state for later reuse. They do not increase the active
context that fits in VRAM. Model-server restarts discard these caches. OpenCode
retains its conversation history, but another model must process that history
again. Compaction creates new summary text. A matching prefix can be reused;
the new summary still needs processing.

Prompt lookup drafts repeated text from the prompt without a second model.
MTP uses a draft head in a merged checkpoint. DFlash uses a separate block
drafter. The main model verifies the proposed tokens. Their speed depends on
the prompt and acceptance rate. They are optional and disabled by default.

```sh
localmodel start bonsai2-9b --prompt-lookup --cache-ram 32768
localmodel fetch bonsai2-9b --speculative mtp
localmodel start bonsai2-9b --speculative mtp --prompt-lookup
localmodel fetch bonsai2-9b --speculative dflash
localmodel start bonsai2-9b --speculative dflash
localmodel start bonsai2-4b --speculative mtp --kv-cache q8_0
localmodel start
localmodel start --no-mtp-chain
localmodel start --draft-tokens 4 --mtp-chain
localmodel start dory --no-offload --kv-offload-gb 2
```

The 4B and 9B optional files are official Prism artifacts. The 27B MTP file
is the existing community lean checkpoint. The tensor audit confirms that all 851 official target tensors and tokenizer
metadata remain unchanged. The pinned 27B DFlash2 draft is also a community
artifact. The pinned Prism runtime includes its required changes.
Unsupported model and decoder selections fail before startup.
`--draft-tokens` sets the neural draft depth. `--draft-kv-cache` controls the
draft cache type. Both defaults are 2 tokens and Q8_0. The 27B example uses
two draft tokens and Q4_0, as in the long OpenCode comparison above.
All draft layers stay on the GPU.

KV mean calibration can improve Q4_0 K-cache accuracy. It is not a speed option.
Generate the bias from representative local text while the model server is
stopped. The calibration and server use the same K rotation and cache type.

```sh
localmodel calibrate bonsai2-27b --text path/to/code-and-text.txt
localmodel start bonsai2-27b --kv-mean-center auto
```

Calibration processes at most 64 chunks of 512 tokens by default. Use `--chunks`
to change this limit. The default output is `state/calibration/MODEL-q4_0.gguf`.
Use `--output FILE` to select another path. The bias is specific to the model.
The runtime checks its model dimensions and rotation basis. Mean calibration
requires Q4_0 K cache. Changing models requires a new calibration file.

`--reasoning-effort` selects the Prism chat-template level. The default is
`xhigh`. Use `medium` for shorter reasoning. The 27B paper uses `xhigh`. The
template must support the selected level. More reasoning can increase the
number of generated tokens. OpenCode receives the selected level. Changing
the level does not reproduce the paper's harness, KV precision, or output budget.
The 27B template accepts low, medium, and xhigh. Its model card says low is not
supported by the model and behaves close to xhigh. Use medium or xhigh for 27B.
The smaller templates do not define the same effort instructions. OpenCode
forwards the configured effort.
The pinned Prism server does not report a separate reasoning-token usage count.
A reported zero in OpenCode therefore does not establish that thinking was off.

```sh
localmodel start 27b-128k --reasoning-effort xhigh
```

Sources: [9B optional decoders](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf#speculative-decoding),
[27B reasoning levels](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf#generation-parameters),
[27B community MTP](https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF/blob/f04a3bd22b7b482675663e99efaba6719347b419/README.md),
[27B community DFlash2](https://huggingface.co/naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2),
and [KV calibration](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/KV-CACHE.md).

## Running model and context

```sh
localmodel status
localmodel status --json
localmodel status --port 8896 --json
localmodel opencode-config
```

Status reads the live server. It gets the served model ID from `/v1/models`.
For vLLM, it reads `max_model_len`. For llama.cpp, it reads the active context
from `/props`. It also checks the existing Dory port 8896.
If two servers are ready, select one with `--port`.
The reported context is total configured capacity. It is not unused space in an
OpenCode conversation. Input, template, history, reasoning, and output share it.
The pinned Prism server uses one token as a boundary guard. For manual capacity
checks at 256K, keep input plus output at or below 262143 tokens.

## OpenCode

Start a model in one terminal. From your project, run:

```sh
opencode-local
opencode-local run --title 'Project explanation' 'Explain this project.'
```

The launcher discovers the running model on each start.
It generates one `local/MODEL` provider entry with the live endpoint and context.
The output allowance is 16000 tokens at 256K. It is reduced for small contexts.
The profile preserves backend sampling and reasoning fields.
Request timeouts are one hour. Subagents, sharing, and warming are disabled.
Web search uses the local SearXNG provider described below.
The launcher starts a private standalone OpenCode server.
Configuration, data, cache, state, and temporary files stay in `state/opencode/`.
Ordinary inference commands fail if no model is ready. They do not select the
previous Qwen endpoint. Service commands, `serve`, `reload`, help, and version
do not require a loaded model. They use the same isolated OpenCode state and
preserve its saved model profile.

```sh
opencode-local service status
opencode-local service stop
opencode-local service start
opencode-local --help
```

The Neovim profile loader also calls `localmodel opencode-config` on setup.
It writes the generated source and runtime profiles at that time.
Restart OpenCode after changing the served model. Existing conversations can
retain their selected model in the full UI. Select the current local model in
that conversation. The `run` and `mini` commands select the current local model
before ordinary prompts unless an explicit `--model` or `-m` is supplied.
An immediate `/compact` after resuming `mini` is not verified to switch the
stored model first. The consolidation does not restart the existing
Neovim service at port 49375.
The saved source profile retains the last selected model. The current Neovim
runtime file retains its previous model route. Its web settings now select
local SearXNG. The running daemon loaded this web update without a restart.
`state/opencode-neovim-next.json` contains the generated profile outside the
watched directory. The loader regenerates the profile on the next setup.

## Local web search

SearXNG runs without Docker in `search/`. It uses a private Python 3.12
environment and Granian 2.8.4. The source is pinned to official commit
`1c10abe2d3e4ab7ddcc4e9da1a7d07dc7ca7f8f4`. DuckDuckGo is the only loaded engine.
No Redis or Valkey server is required.

Run these commands in separate terminals:

```sh
searxng-local
localmodel start 27b-256k
opencode-local
```

`searxng-local` listens at `http://127.0.0.1:8888` and stays in the foreground.
Use Ctrl-C to stop it. The server has a 1 GiB RAM cap and no swap. It does not
use the GPU. `config/searxng.yml` contains its settings. Its secret, caches, and
temporary files stay in `state/searxng/`. Access logging is disabled. Search
requests use POST. Search queries are still sent to DuckDuckGo.
Direct browser or API requests can produce SearXNG's one-time
`X-Forwarded-For nor X-Real-IP header is set!` log entry. The direct-client
configuration uses the connection address. This log entry does not block search.

The native OpenCode `websearch` provider is implemented in
`config/searxng-plugin/`. It posts the search query to the local JSON API.
It uses the enabled engines from `config/searxng.yml`.
It rejects HTTP redirects, including external bang redirects. It returns at
most eight results and 2000 snippet characters per result. A failed local
server or any configured engine produces an error, including partial failures.
There is no hosted fallback.
Search text remains untrusted input to the coding tools.

### Configure search

There are three separate settings:

| Setting | File | Current value |
| --- | --- | --- |
| Upstream search engines | `config/searxng.yml` | DuckDuckGo only |
| OpenCode search provider and tool permissions | `config/opencode-web.json` | Local SearXNG; default tool permissions |
| Inference provider and context | Generated from `localmodel` | The current local model |

Keep the loopback bind address, JSON format, POST method, and disabled access
log. The current 10-second engine timeout, 15-second maximum timeout, and
25-second plugin timeout are suitable starting values. Keep engine suspension
after upstream errors. `server.limiter: false` disables the local inbound
limiter. It does not disable DuckDuckGo's bot protection.

Start with one search at a time. This is a conservative operating choice,
not a guaranteed request allowance. There is no query cache or pacing queue in
the plugin. Avoid immediate retries after HTTP 403, HTTP 429, or a CAPTCHA.
More server threads do not increase an upstream request allowance.

To add another SearXNG engine, add it to `keep_only` and enable its entry.
For example, replace those two sections with this configuration and retain
the other existing sections:

```yaml
use_default_settings:
  engines:
    keep_only:
      - duckduckgo
      - brave

engines:
  - name: duckduckgo
    disabled: false
  - name: brave
    disabled: false
    timeout: 10.0
```

Restart `searxng-local` after a YAML change. OpenCode does not need a plugin
change. SearXNG queries both enabled general-search engines and combines their
results. Both services receive the query. This is not an ordered fallback.
The `brave` engine uses the public website and can also be blocked.
It has not been enabled or tested here. See the
[engine merge rules](https://docs.searxng.org/admin/settings/settings.html) and
[Brave engine documentation](https://docs.searxng.org/dev/engines/online/brave.html).
The keyed Brave API engine is named `braveapi`. It also needs `api_key`,
`inactive: false`, and `disabled: false`. Do not put an API key in a tracked
file. The installed YAML loader does not expand arbitrary `${VARIABLE}` values.

To select a different OpenCode search provider, edit
`config/opencode-web.json`. For example:

```json
{
  "websearch": { "provider": "tavily" },
  "permissions": []
}
```

| Provider ID | Credential environment variable |
| --- | --- |
| `searxng` | None; uses the local plugin |
| `exa` | `EXA_API_KEY` |
| `firecrawl` | `FIRECRAWL_API_KEY` |
| `parallel` | `PARALLEL_API_KEY` |
| `tavily` | `TAVILY_API_KEY` |
| `tinyfish` | `TINYFISH_API_KEY`, optional |

Export the chosen credential before starting `opencode-local`. `/connect`
can also store provider credentials in the isolated OpenCode account store.
The selected hosted provider receives search queries. This does not change
the local inference provider. An API key alone does not change the selection.
Avoid `random` when queries must stay with one provider. Random selection can
switch providers after HTTP 429. See
[OpenCode search providers](https://dev.opencode.ai/v2/docs/websearch/).

The launcher reads `config/opencode-web.json` when it generates a profile.
Relaunch `opencode-local` with a ready model after a change. A Neovim setup also
regenerates its profile. Service administration preserves a saved profile;
`service restart` alone does not regenerate it. Do not edit the generated
`state/opencode/config/opencode/opencode.json` as a persistent configuration.
Set `websearch` to `false` in the source file to disable the search tool.

### Configure webfetch permissions

`websearch` finds links and snippets. `webfetch` reads a selected URL directly.
It does not use SearXNG or the selected search provider. Use search to find the
correct page before a fetch. Prefer official documentation and raw text.
The observed fetch errors used the nonexistent `opencode.dev` host. Current
OpenCode documentation is at `https://opencode.ai/v2/docs/`.

The native fetch tool uses a 30-second default timeout and accepts up to 120
seconds per call. It accepts text, HTML, JSON, and XML. It does not run browser
JavaScript or solve a CAPTCHA. Its response limit is 5 MiB. A longer timeout
does not fix a DNS error, denied access, or a binary response.

The empty `permissions` list keeps native default permissions. To ask before
web access and allow a specific documentation site, put these rules in
`config/opencode-web.json`:

```json
{
  "websearch": { "provider": "searxng" },
  "permissions": [
    { "action": "websearch", "resource": "*", "effect": "ask" },
    { "action": "webfetch", "resource": "*", "effect": "ask" },
    { "action": "webfetch", "resource": "https://docs.searxng.org/*", "effect": "allow" }
  ]
}
```

The last matching permission rule wins. Use V2 `permissions`, not the V1
`permission` object. See [web tools](https://dev.opencode.ai/v2/docs/tools/) and
[permissions](https://dev.opencode.ai/v2/docs/permissions/).

The dependencies are installed. To install them again, run:

```sh
~/Code/local-models/bin/setup-searxng
```

The installer checks the pinned source revision. It records installed Python
versions in `search/installed-requirements.txt`. An actual JSON query returned
ten DuckDuckGo results with no engine errors. See
[the SearXNG smoke record](../results/searxng-smoke.json).
The native OpenCode search API also passed without a loaded model. See
[the OpenCode integration record](../results/searxng-opencode-validation.json).
The later native fetch check and persistent configuration checks are in
[the fetch record](../results/webfetch-native-validation.json) and
[the configuration record](../results/web-config-validation.json).
The temporary test servers are stopped. Start the services with the commands above.

## Relocation and compatibility

The existing repositories and all their data moved to `bonsai/` and `dory/`.
Their Git history, local changes, weights, runtimes, and OpenCode data remain.
The old `~/Code/bonsai2-small-gpu` and `~/Code/dory-local` paths are compatibility
symlinks. Keep them while existing processes or virtual-environment scripts use
the old absolute paths.
The existing `dory`, `bonsai2-server`, `bonsai2-models`, `opencode-dory`, and
`opencode-bonsai` links point into the new directories.
The old model-storage links also point into this directory.
The dotfiles-managed `opencode-local` command forwards to `bin/opencode-local`.

The parent manager uses Python 3.12 and uv. Its Python environment has no extra
runtime dependencies. Server jobs use `~/Code/heavy-management/bin/capped`.
GPU inference is serial. Queue concurrency is unchanged.

## Validation

These measurements used the RTX 4070 SUPER with 12282 MiB VRAM on 2026-10-17.
All profiles used context 262144 and zero CPU inference offload. Bonsai used
GPU layers, GPU KV cache, and disabled automatic fitting. Dory used GPU memory
utilization 0.75. GPU use below includes other users of the GPU. Samples are
one-second peaks except the older two-loop Dory snapshot. These single runs
measure capacity and throughput. They do not rank answer quality.

| Profile | Weights | KV cache | 256K first token | 256K tokens/s | About 4K tokens/s | Observed GPU MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `bonsai2-2b` | PQ2_0 | FP16 | 53.65 s | 91.36 | 228.36 | 5111 |
| `bonsai2-4b` | PQ2_0 | FP16 | 114.83 s | 41.33 | 150.85 | 10869 |
| `bonsai2-9b` | PQ2_0 | Q8_0 | 151.40 s | 45.53 | 115.39 | 8049 |
| `bonsai2-27b` | PTQ1_0 | Q4_0 | 507.72 s | 16.35 | 53.20 | 11644 |
| Dory, 1 loop | NVFP4 | FP8, last loop | 173.32 s | 14.38 | 13.17 | 11185 |
| Dory, 2 loops | NVFP4 | FP8, last loop | 278.06 s | 7.39 | 9.40 | 11175 |
| Dory, 4 loops | NVFP4 | FP8, last loop | 459.23 s | 5.36 | 5.13 | 11199 |

The full-context runs generated 64 output tokens. The short Bonsai runs used
the retrieval prompt and generated 48 to 54 tokens. The short Dory runs used
4096 input tokens and generated 64 tokens. Output lengths and tasks differ.
The records and source paths are in `results/model-comparison.json`.

The 2B, 4B, and 9B trials processed 262079 input tokens and generated 64 output
tokens. They reported no truncation. The 27B trial processed 262080 input tokens
and returned all 64 requested output tokens. It reached the one-token boundary
guard and set `truncated=true`. The strict check rejected that flag. Source
inspection identified a boundary stop, with no input-discard operation.
The raw measurement and the separate boundary analysis remain in `results/`.
The [pinned server code](https://github.com/PrismML-Eng/llama.cpp/blob/e3de715222b2667c54a267441159a5729f11155e/tools/server/server-context.cpp#L1771-L1838)
defines this behavior. Future capacity checks reserve one token.

The 27B profile also returned three correct audit facts from 246137 input
tokens. Facts appeared near 10, 53, and 95 percent of the filler. First token
took 461.94 seconds. Decode speed was 16.79 tokens/s. Each smaller profile passed
the same three-fact check at 4090 input tokens. These checks do not establish
general coding quality or long-context quality for the smaller profiles.

The model downloads passed pinned size and SHA256 checks. Strict Pyright,
shell syntax, symlink, and shutdown checks passed. Private OpenCode validation
selected the live Dory and Bonsai endpoints with context 262144 and output 16000.
An actual OpenCode request to restored Dory returned LOCALMODEL_OK and parsed
the reasoning stream. Dory was restored at port 8896 with two loops, 256K context,
FP8 last-loop KV, no CPU weight offload, and the validated 2 GiB host KV cache.
Matti then requested its shutdown. Dory and its GPU worker are now stopped.
Its first post-move request rebuilt FlashInfer CUDA kernels.
The existing Neovim service remained running throughout the trials.
The earlier filled 256K Dory measurements remain in `dory/results/`.
Restart details and outstanding checks are in [HANDOFF.md](../HANDOFF.md).

## Code and weight audit

The audit found no intentional exfiltration path in the reviewed MTP, DFlash,
template, and Dory patch paths. It traced draft candidates through target
verification before chat and tool-call parsing. The metadata client now rejects
HTTP redirects and disables proxies. Its local smoke check rejected five redirect
status codes and sent zero requests to the redirect destination.
Dory starts with `VLLM_NO_USAGE_STATS=1` and `DO_NOT_TRACK=1`. The final live
process has both settings.

Fresh whole-file SHA256 checks match all ten selected Bonsai GGUF files and the
pinned NVIDIA NVFP4 safetensors file. Direct tensor comparisons found all 426,
427, and 851 target tensors unchanged in the 4B, 9B, and 27B merged MTP files.
Each adds 15 draft tensors and removes none. Token arrays, merges, special-token
IDs, and chat templates match the corresponding base model. The DFlash
tokenizers only add their mask-token ID. The community 27B draft contains new
trained weights; its hash establishes artifact identity, not training intent.
All 78 installed Prism files and 10 symlinks match the official release archive.
The three loaded CUDA libraries match verified official NVIDIA package files.

See the [code and data boundary audit](../results/security-code-audit.md),
[tensor and file identity audit](../results/weights-audit.md), and
[community provenance audit](../results/community-provenance-audit.md).
The reports retain exact revisions, hashes, changed files, evidence, and limits.
They do not prove native memory safety, source-to-binary correspondence, or
benign learned behavior. Legacy abliterated profiles are outside this review.
The original vLLM Dory backend's precompiled wheel and transitive dependencies
are not fully pinned. The new GGUF backend has a pinned source build and
recorded binary hash.

The [native Bonsai source review](../experiments/bonsai-native/source-audit.md)
covers the selected window, lookup, MMA, grammar, and KV-tier changes. It found
no added network or tool-execution path. This is a bounded code review.
The runtime checks use one GPU, one slot, and one model per server process.
They do not establish native memory safety under concurrent allocation.

The official Dory Q5 file matches NVIDIA's published SHA256. However, 158 of
217 retained BF16 vectors differ from the published BF16 checkpoint. This
establishes official artifact identity, without establishing unchanged source
weights. The quantized matrices were not compared numerically. See the
[GGUF audit](../results/native-long-context-20261019/dory-gguf-audit.md) and the
[complete retained-vector comparison](../results/native-long-context-20261019/dory-gguf-vector-stats-20261019T174940Z.md).
Of 1,126,400 BF16 elements, 107,855 differ. All compared values are finite.
The maximum absolute difference is 0.001953125. The differences are not uniformly
one BF16 step. The [distribution report](../results/native-long-context-20261019/dory-gguf-vector-stats-20261019T174940Z.distribution.md)
separates vector roles and near-zero relative errors. These figures do not
bound output differences.

The OpenCode profile routes inference locally. Permitted coding tools can still
read files and use the network. The launcher does not provide an OS egress
sandbox. Normal OpenCode tool permissions remain the execution boundary.

## Published quality

These are vendor-reported scores, rechecked on 2026-10-19. Higher is better.
They are not quality measurements of the local runtime and cache settings.
The [refreshed comparison](../results/native-long-context-20261019/published-quality.md)
records immutable source revisions, score tables, and evaluation limits.

The models do not share one base checkpoint. Bonsai 2 27B derives from
Qwen3.8-27B. Prism keeps the base architecture and trains ternary weights.
Dory uses NVIDIA's looped normalized-GPT architecture with sliding-window and
full attention. NVIDIA reports 6 trillion pretraining tokens and 300 billion
context-extension tokens. Its post-training includes teacher-generated data,
supervised fine-tuning, and reinforcement learning. The card does not describe
Dory as a Qwen derivative. The Qwen columns in its score table are baselines.
Sources: [Bonsai family](https://prismml.com/news/bonsai-2-family),
[Bonsai 27B overview](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf#model-overview),
and [Dory design and training](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16#model-architecture).

| Bonsai 2 | HumanEval+ | MBPP+ | BFCL v3 | GPQA Diamond |
| --- | ---: | ---: | ---: | ---: |
| 27B | 95.12 | 83.07 | 74.92 | 85.76 |

Sources: [27B paper, Table 10](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf#page=11),
and the [27B model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

| Model | GPQA Diamond | IFBench | SWE-bench Verified | Terminal-Bench 2.1 | AA-LCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bonsai 2 27B, xhigh | 85.76 | 74.00 | 60.8 | 52.8 | 77 |
| Bonsai 2 27B, medium | 75.56 | 55.33 | — | — | 74 |
| Dory, 1 loop | 69.19 | 74.29 | 50.3 | 12.5 | 63 |
| Dory, 2 loops | 72.41 | 77.94 | 59.5 | 18.1 | 71 |
| Dory, 4 loops | 72.16 | 79.00 | 61.8 | 22.2 | 73 |

Sources: [NVIDIA Dory card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16#benchmarks)
and [Prism 27B paper, Section 4 and Appendix C](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf#page=7).
The medium row comes from [Table 11](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf#page=12).
That table reports HumanEval+ 93.29, MBPP+ 80.69, LiveCodeBench v6 80.47,
BigCodeBench 59.74, BFCL v3 75.12, and tau2-Bench 70.86. It does not give
separate medium SWE-bench or Terminal-Bench scores.
The papers do not report these agent benchmarks for the three smaller Bonsai
models. The providers use different evaluation settings. Do not treat the
cross-provider table as a controlled ranking.

NVIDIA's table uses BF16, including the table repeated on the NVFP4 card.
It does not quantify the score change from NVFP4 weights, FP8 KV, or last-loop
cache. NVIDIA describes last-loop accuracy as close to per-loop accuracy.
Prism evaluates the ternary models with thinking enabled, EvalScope/vLLM, and
large output budgets. The 27B report uses xhigh reasoning. Its Terminal-Bench
setup uses Harbor and Terminus-2, a 262144 context window, and an 80000-token
output limit. SWE-bench uses mini-swe-agent. Our OpenCode output limit is 16000
and the earlier short Bonsai trials used medium reasoning. The current long
comparison uses xhigh, except for its separate medium control. The published scores do not
validate our Q4 KV cache or accuracy on a filled 256K prompt.

PTQ1_0 and PQ2_0 store the same ternary weights. Their storage and kernels
differ. PTQ1_0 is the practical 27B choice for this Ada GPU and the 256K memory
budget. The model card reports an Ada decode advantage for PTQ1_0. PQ2_0 uses
about 1.26 GB more storage and favors prompt-processing speed.

## Selection

Use 27B as the primary OpenCode candidate when capability takes priority.
The current comparison covers 27B only. Dory results are archived. Earlier smaller-model results
remain historical. `localmodel start` selects `27b-256k-optimized`: MTP2, chaining
on, xhigh effort, Q4 KV, a 16K draft window, and a 32 GiB RAM prompt cache.
It also selects the trained head, checkpoints, replay, and 128-token chunked prefill.
The stock `27b-256k` preset remains available explicitly. It passed all six
long OpenCode phases and the filled 256K capacity check.
Stock measured 54.4 tokens/s before compaction and 90.6 afterward. Cold TTFT
was 402 seconds; warm long-history TTFT was about 1.2 seconds.

Use `27b-256k-native` as the measured native choice with a 16K MTP draft window. The target
context remains 256K. This configuration passed the same checks and measured
64.5 tokens/s before compaction and 88.9 afterward. Its MMA override was off.
The separate forced-MMA improvement used MTP1. The completed MTP2/window/MMA
case measured 64.5 tokens/s before compaction and 92.6 afterward. It reached
66.2 tokens/s at filled capacity, compared with 65.4 for the window-only case.
This does not establish an additional speed benefit. The optional preset is
`27b-256k-native-mma`. The window-only run completed before the reported CPU-load interval.
The native lookup preset measured 59.6 tokens/s on the long workflow. Its
199.7 tokens/s capacity result used repetitive text that favors lookup.

`27b-192k` remains available for DFlash. Its older small coding task measured
105.43 native tokens/s with medium effort and an 8 GiB RAM cache. This is a
different context, task, and setting from the new long xhigh comparison.
The medium-effort case passed all workflow, capacity, and schema checks.
It measured 54.7 tokens/s before compaction, 91.3 afterward, and 72.0 seconds
for warm follow-ups, compared with 75.7 seconds for stock xhigh. Output lengths
can differ. The optional preset is `27b-256k-medium`. The default remains xhigh.
The active KV spill case passed, but long-history decode fell to 29.8 tokens/s.
Its measured GPU headroom option is `27b-256k-native-spill`.

Dory tests stopped at Matti's request. Its native Q5 backend passed filled
256K capacity with Q8 KV on the GPU at all three loop counts. Four loops
passed the complete strict OpenCode workflow, at 16.9 tokens/s before
compaction and 34.8 afterward. Its warm workflow took 616.5 seconds. One and
two loops left correct first repairs but stopped on recovered tool errors.
Their later workflow phases remain unmeasured. Their 107-schema checks failed;
the four-loop schema check was canceled. See the
[saved Dory results](../results/native-long-context-20261019/dory-current/summary.md).

The published evaluations use different KV precision, output limits, tools,
and graders. They do not establish local task accuracy. The controlled repairs
test a small part of repository work. No local result here establishes general
performance on Haskell or Lean proof work.

Sources: [Bonsai 2 family](https://prismml.com/news/bonsai-2-family),
[official 27B weights](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
[KV cache guide](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/KV-CACHE.md),
[Prism runtime release](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715).
