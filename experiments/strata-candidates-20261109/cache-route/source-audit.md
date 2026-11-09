# DeepSeek cache-route candidate

This experiment uses Strata PR [5596](https://github.com/Niko1221/Strata/pull/5596)
at commit `4294e6aaf398c7612b1e943f7947b94a95292350`.
The base is the installed v0.2.12 source,
`146df3c468739147f520ac3816de806f71232571`.
The PR was open and marked draft when checked on 2026-11-09.

The patch changes `ds4_model.cu`, `ds4_store.hpp`, and documentation.
All three patched files match the pinned upstream head byte for byte.
The copied official source is separate from the production source.
The experiment does not change weights, pack files, runtime pins, or defaults.

## Behavior

`STRATA_DS4_CACHE_ROUTE=RAM,VRAM` adds a selection bias to experts already
held in either memory tier. A single value applies to both tiers.
The target still selects its configured number of experts. Six remains six.
The contribution weights use the original router scores for the selected
experts. The selected experts can differ from the unmodified target.

The bias refreshes at each flow-forward call and every sixteen other
forward calls. It is therefore an approximate cache snapshot.
The RAM hint includes a slot whose data is still loading.
Answers depend on cache contents and prior requests.
This is a quality-for-speed trade, not exact inference acceleration.

MTP layers, layers without a bias tensor, and tensor-parallel layers with
more than one member skip the extra bias. Image routing retains its separate
bias. The experiment uses text, one GPU, six experts, and no drafter.
The three changed routing sites cover flow prefill, asynchronous decode,
and synchronous decode. The existing `--score` path can exercise the change.

When the variable is absent or zero, the refresh returns immediately and
the router receives its original bias pointer. No bias buffer is allocated.
This supports a same-binary off/on comparison. It does not prove numerical
identity with the official binary built using a different toolchain.

## Bounded review

The added native code has no network, file, process, shell, or credential API.
It reads one environment value, allocates small vectors and CUDA buffers,
copies bias rows, reads cache metadata, and prints an activation message.
It adds no model-download or tool-execution path.
Inference must retain the existing outgoing-network guard.

The reviewed code keeps model weights read-only. It does not alter the
configured expert count or bypass target logits. Quality still requires
measurement because expert selection changes.

Limits found during review:

- The new CUDA bias buffers are not added to the normal allocation cleanup
  list. Their total is about 60 KiB for forty layers with 384 experts.
  They allocate once per layer, not per token, and process exit releases them.
  An in-process model unload should free them before production use.
- `ExpertStore::holds` follows the existing lock-free `find` pattern.
  It checks the layer range and relies on callers to provide a valid expert
  index. This caller iterates the layer's expert count. The review does not
  prove freedom from races with background cache updates.
- Numeric parsing uses `atof`. Only the explicit finite values recorded by
  the benchmark should be accepted. Do not expose arbitrary inherited values.
- Multi-GPU, vision, multiple concurrent requests, and MTP combinations are
  outside this experiment's validation scope.

The experimental source retains the upstream patch unchanged. These limits
must remain visible when interpreting the measurements.

## Build

The build follows the upstream portable CUDA DeepSeek recipe with only sm89
machine code. It uses the clean pinned ggml checkout at
`3cf03257f219afbe7334045ff7c6a06ac68c627d`.
`dependency-pins.json` records NVIDIA package versions, wheel hashes, installed
file verification, and licenses. CUDA dependencies are isolated under the
experiment's build directory. No system package or production runtime changes.

The build uses two jobs in a 16 GiB systemd scope with zero swap. The receipt
records the resulting executable, library directory, source pins, and command.
The official binary remains the external control. The custom binary with
routing disabled is the required control for routing-enabled measurements.

The build completed with exit code zero and a 3254 MiB peak. GCC 13.3 uses
`-O3 -DNDEBUG`. The portable ggml CPU backend enables AVX2, FMA, and F16C.
It disables native-host tuning, BMI2, AVX512, and OpenMP. Strata's existing
per-file CPU kernels retain their runtime dispatch.

The NVIDIA wheel layout needs local `lib64 -> lib` and unversioned
`libcudart.so`, `libcublas.so`, and `libcublasLt.so` symlinks for configuration.
These links stay inside the isolated build directory.

The benchmark must use the receipt's `lib_dirs`. They put the official
Strata 0.1.58 CUDA13 libraries first. `ldd -r` resolves all imported symbols
against those libraries without engine execution. This removes the different
build-toolchain library versions from the runtime comparison. The executable's
build RUNPATH still refers to its isolated toolchain; an unwrapped direct run
would use those different libraries and would not reproduce the comparison.
