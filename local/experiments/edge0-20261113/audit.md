# Edge0 Linux CUDA path: bounded review

Reviewed source: `/home/tritlo/Code/local-models/local/state/sources/edge0-da9cc526`, commit `da9cc526747759227714ddde0728763d95306c04`. Model files are pinned separately to Hugging Face revision `00f91ca2dc8290d74c79360f0d9941704e0f5c69` in the experiment manifest.

The direct `edge0-strata/serve/server.py` route is suitable for an isolated experiment. The reviewed path does not need `edge0-infer`, its frontend, `start.sh`, `setup.py`, or downloaded `model.json` commands. This review covers the CUDA `strata` target, its CMake dependency path, native file/latent loaders, API startup/config parsing and imported control modules. It is not a full repository or dependency audit. It does not prove that every native parser is memory-safe.

## Security findings

- The native `strata` target links the engine, prefill, speculative-decoding and CUDA libraries. Searches of its source/include tree found no outbound socket/connect, shell or exec call. The local file loaders use read-only mappings and size/layout checks. Latent labels and quantization scales are loaded as data. This is static evidence, not a runtime network trace.
- CMake can fetch llama.cpp if `STRATA_GGML_DIR` is omitted. Supply the pinned local dependency. The other configure-time commands seen here are compiler probes, a Python `import numpy` probe and `git rev-parse`. The optional trellis overlay copies and edits files under the build tree. Build only target `strata`; unrelated experimental targets are incomplete.
- The cached `local/state/sources/maya-ggml-3cf03257` is clean at exact commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, with origin `https://github.com/ggml-org/llama.cpp.git`. Copy its tracked contents to an isolated Edge0 dependency directory. Do not modify the shared checkout. ggml has optional OpenMP and KleidiAI fetch paths. Keep both disabled. This x86 Linux build does not need them.
- API configuration is JSON, not executable Python. Engine startup uses an argv list through `lifeline.popen`. However, unrestricted configuration can name an arbitrary executable, child environment, shared-library directories, MCP commands or a `before_load` shell command. Author the small config below locally. Do not merge downloaded model JSON into it.
- `--offline` refuses configured MCP and `before_load` hooks and blocks the API's remote-image path. This is an application policy, not a kernel sandbox. Keep the existing `experiments/deepseek-strata/network_guard.py` outside the API process. Its seccomp filter is inherited by native children. It denies outgoing connections and addressed sends while permitting inbound loopback API replies.
- Optional API modules can contact remote image URLs and MCP endpoints. These features are inactive without corresponding config. An explicit local `exe` also avoids the component-engine lookup path. Set `drafting: "mtp"` to avoid an automatic drafter choice. Do not pass `--open`; set `open_browser: false` and `api_monitor: false`.
- The chat template uses Jinja2 `ImmutableSandboxedEnvironment` (`serve/frontend.py:50`). Tool-call text becomes structured API output. The text-only server does not execute returned coding tools; OMP controls them.
- Linux native children use `PR_SET_PDEATHSIG` through `serve/lifeline.py`. Keep the independent cgroup/ownership checks as well.
- Reuse the existing minimal Python environment: Jinja2 3.1.6, MarkupSafe 3.0.3, regex 2026.9.10 and psutil 7.2.2. Text-only imports need these. The fork's full requirements file adds unused dependencies and pins Pillow 12.3.0. Official Strata 0.2.13 instead pins Pillow 13.0.0 for a documented OpenJPEG security fix. Our minimal environment has no Pillow. Do not install the fork's broad requirements just to serve text.

## Build plan

Use CUDA 13 from `local/state/builds/strata-cache-route-20261109/cuda-site/nvidia/cu13`. `/usr/bin/nvcc` is CUDA 12.0 and is not the selected compiler. Reuse the dependency validation/RPATH procedure in `local/experiments/strata-cache-reuse-20261113/build.py`.

The following is a command plan, not an executed build. Set `EDGE_SOURCE` to the pinned checkout's `edge0-strata`, `EDGE_BUILD` to isolated build storage, and `EDGE_GGML` to the isolated exact-pin dependency copy. Set `CUDA13`, `LOCAL`, and `CUDA_RUNTIME` to their existing reviewed absolute paths.

```sh
"$LOCAL/.venv/bin/cmake" -S "$EDGE_SOURCE" -B "$EDGE_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89-real \
  -DCMAKE_CUDA_COMPILER="$CUDA13/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA13" \
  -DCMAKE_MAKE_PROGRAM="$LOCAL/.venv/bin/ninja" \
  -DSTRATA_GGML_DIR="$EDGE_GGML" \
  -DGGML_OPENMP_FETCH=OFF -DGGML_CPU_KLEIDIAI=OFF -DGGML_CCACHE=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DCMAKE_INSTALL_RPATH="$EDGE_BUILD;$CUDA_RUNTIME" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_SKIP_BUILD_RPATH=OFF
"$LOCAL/.venv/bin/cmake" --build "$EDGE_BUILD" --target strata --parallel 2
```

Run configure/build serially through heavy-management with the existing 16 GiB cap, no swap and `nice -n 19`. The CMake file requests C++20, despite the guide saying C++17. Native version is 0.1.55. It is not official Strata 0.2.13. Latent support is included under `STRATA_NATIVE_EXPERTS` plus CUDA and does not require the test targets.

## Initial serve plan

The source guides disagree about a minimum. `edge0-infer/docs/edge0-145b.md:10` says at least 16 GiB GPU memory and 20 GiB system memory. The newer `docs/models/edge0-145b.md:140` calls these measured test budgets, not general minimum requirements. Twelve GiB is unverified, not proven impossible. SM89 itself is supported. Start with Q4 D/U and a 16K context. Increase context only after a real request passes with MTP active. Keep a 64 GiB model cgroup and the existing Linux 8 GiB headroom guard.

Use a JSON argv array with these separate values. `MODEL` means the verified model directory; it is not a shell expansion inside the JSON:

```text
--pack MODEL
--native MODEL/model-skeleton.gguf
--native-dense-gguf MODEL/model-skeleton.gguf
--ple-gguf MODEL/ple-fp8.gguf
--ple-io direct
--expert-profile MODEL/profile.bin
--expert-cache auto
--prefill auto
--spec 4
--spec-min-p 0.5
--mtp MODEL/mtp
--max-context 16384
--kv int8
--kv-resident 32768
--resident-budget-gib 40
--vram-reserve-mib 1024
--suffix-draft 0
```

`--spec 4` permits three draft tokens plus one verified token. The template uses this setting. Use `--spec 5` only when deliberately testing four drafts. Disabling suffix lookup keeps MTP acceptance interpretable. The engine otherwise defaults suffix lookup to three and can expand the window.

Use these config keys: `exe` = built absolute path; `cwd` = Edge0 Strata source; `tokenizer` = `MODEL/tokenizer`; `model_name` = `edge0-145b`; `arch` = `qwen4exp`; `gpu` = `[0]`; `drafting` = `mtp`; `mtp` = `true`; `log` = an owned case log; `sampling` = temperature 1.0, top_p 0.95 and the paired seed; `fit_max_tokens` = true; `open_browser` = false; `api_monitor` = false. Omit MCP, image, vision, hook, component and learned-profile settings. Allow only these model environment entries:

```json
{
  "STRATA_LATENT_A": "MODEL/experts.bin",
  "STRATA_LATENT_LABELS": "MODEL/labels.bin",
  "STRATA_LATENT_DENSE": "MODEL/latent_dense_q4.bin",
  "STRATA_LATENT_DENSE_FMT": "4",
  "STRATA_EMBED_VRAM": "1",
  "STRATA_LATENT_FUSED": "1",
  "STRATA_LATENT_FUSED_DU": "1",
  "STRATA_LATENT_U_GPU": "1",
  "STRATA_LAT_FUSED_Q3K": "1"
}
```

For Q8 D/U, change only the dense filename and format to `latent_dense_q8.bin` and `8`. Provide only reviewed CUDA library directories. Preserve `/usr/lib/wsl/lib` in the driver-library search order.

Start the API through the existing guard:

```text
capped --mem 64G PYTHON -I -B LOCAL/experiments/deepseek-strata/network_guard.py
  PYTHON -I -B EDGE_SOURCE/serve/server.py --engine strata
  --config OWNED_CONFIG --host 127.0.0.1 --port 8899 --offline
```

The tokenizer layout is the existing Strata layout: `vocab.json`, `merges.txt`, `token_type.json`, `tokenizer.json`, and `chat_template.jinja`. The direct API exposes health, model discovery, OpenAI completions and native metrics. No frontend-specific adapter is required.

## Memory and compatibility

Pinned artifact sizes: experts 37.324 GiB, skeleton 5.580 GiB, MTP experts+dense 0.767 GiB, FP8 PLE 47.684 GiB. File size is not a complete VRAM estimate. Forty GiB of resident-expert budget covers the CPU complement of this expert file while leaving room inside the 64 GiB cap for KV and other allocations. Keep PLE on direct reads; `ple-io ram` would consume almost 48 GiB by itself.

The D/U loader uploads all 48 layers before auto-cache sizing. The exact layout is `48 * 2 * 8 * 1792 * 2560 / 32 * block_bytes`: Q8 is 3.486 GiB; Q4 is 1.846 GiB. Q4 frees 1.641 GiB for other GPU allocations. The skeleton, MTP, KV, buffers and desktop still have to fit. This calculation does not prove 12 GiB feasibility or a throughput result.

For longer contexts retain `--kv-resident 32768` and increase `--max-context`. The model documentation advertises 256K. MTP startup must contain `strata mtp: draft layer loaded,` (`src/core/mtp.cpp:534`), and a request must show positive offered/accepted draft counters. Do not count a successful start after MTP fallback as the intended configuration.

Current official native flags missing here: `--draft-fallback`, `--draft-mtp-min-p`, `--draft-mtp-spec`, `--draft-rule`, `--lookup-max-t`, `--lookup-sizes`, `--ram-keep-gib`. `STRATA_ADAPT_FETCH` is absent from the reviewed native source. Do not copy those controls into the Edge0 args. Wrapper controls such as `--mem`, `--context`, `--expert-ram-gb` and `--draft-tokens` must be translated to the native controls above. The official cache-reuse patch is not assumed present in this fork.

No build, download or inference was performed for this audit.

## Benchmark runner review, 2026-11-13 21:51 CET

Reviewed `local/experiments/edge0-20261113/run.py` (`06d5a3c5547b35b4fde814559d0319750e1812524c87da5f92454664328107b2`), `server.py` (`700ba7ea200913f7b78d0615018ab005819b667b402ad5b66a2fcc0328974aaf`) and the small shared-fixture change (`adb64e0f18e1132d4234a8cfd2b25ecb276e53a68143631f04a6895066e41507`). The implementation follows the audited argument and environment allowlist. It serves the identity `edge0-145b` and counts input with its own tokenizer and template. Its first case configures 32K capacity and targets 16K input.

The controller checks its 4 GiB no-swap cap. It verifies the API's 64 GiB no-swap scope, keeps model/service locks, and stops the owned scope before removing its active record. Linux parent-death SIGTERM links the model controller to the benchmark driver. The readiness loop records an owned search scope before health checks. Existing search remains untouched. The runner compares normal OMP/Qwen/DeepSeek settings before and after the case. It requires MTP startup and positive offered/accepted counters.

The build receipt verifies the native executable and `libstrata_kq.so`. The child environment removes `LD_PRELOAD` and `LD_AUDIT` and specifies the build plus pinned CUDA runtime directories. This host's linker cache resolves `libcuda.so.1` and `libnvidia-ml.so.1` to `/usr/lib/wsl/lib`; the pinned runtime contains no competing driver library.

The implementation agent reported passing CPU checks in `/tmp/edge0-benchmark-cpu.py`. The reviewed fixture checks all 74 artifact receipts and the actual tokenizer. It exercises SIGTERM cleanup and SIGKILL of the driver with a fake API and an orphan child in the real capped scope. Focused strict Pyright reported no errors or warnings. These checks do not establish GPU fit, MTP correctness under inference, speed or task quality. No blocking issue remains for the first guarded case.
