# Strata candidate benchmark

This harness compares the official Strata v0.2.12 engine with two experimental
controls. It does not change production launchers, installed pins, or defaults.
The API and readiness checks remain the official v0.2.12 installation. A custom
engine needs an explicit binary path, binary SHA256, build receipt, and receipt
SHA256. Its receipt must use the exact installed source base revision.

The driver and helpers come from the completed full DeepSeek tuning harness.
Their inputs are copied to `results/strata-candidates-20261109/harness/`.
The original frozen files and completed cases remain unchanged.

All arms use full Unsloth IQ1_S, no DSpark, all six routed experts, a 48 GiB
expert cache, a 64 GiB hard cap, zero swap, 196608 token context capacity,
12 CPU workers, 4096 token prefill chunks, and 16 Engram readers. The GPU
allocation settings remain unchanged. The API binds to loopback. The outgoing
network guard applies to the API and its native child.

All new arms use temperature 1, top-p 0.95, and seed 20261109. The official
sampling config supports the seed. The driver checks these defaults through
`/props`. It retains the public warmup's explicit temperature 0. All engines
use `--verbose` and write native output to files. The seed and verbosity are
changes from the older comparison. Compare these new arms with each other.
The seed reduces sampling variation. It does not guarantee equal output or
equal token counts across different routing decisions or floating-point paths.

## Variants

- The official control leaves both candidate environment variables unset.
- `--borrow 0` disables borrowed prefill room. `--borrow 1` enables it explicitly.
- The custom route-off control uses the same binary as the route-on arms.
- `--cache-route 0.2` gives a 0.2 selection bonus to experts in RAM or VRAM.
- `--cache-route 0.2,0.3` gives 0.2 to RAM and 0.3 to VRAM.
- `--cache-route off` leaves `STRATA_DS4_CACHE_ROUTE` unset.

Cache-aware routing changes expert selection and can change answers. Repair,
recall, and resume checks must pass. This small public fixture cannot establish
general coding quality. Preserve generated-token denominators, failures, and
tool errors. Do not treat a change in generated output length as a speed gain.

The wrapper inserts only `DS4_BORROW` and `STRATA_DS4_CACHE_ROUTE` in the copied
settings result. It does not change the global environment or weaken the
production environment filter. The driver checks the actual native descendant
PID, executable bytes, arguments, environment, and active seccomp filter.
It also verifies the saved launch config and API sampling defaults. Custom
runtime libraries must match their receipt hashes. The driver records loaded
CUDA and kernel libraries before and after the warmup. Custom arms must load
the exact library paths and bytes from the receipt.

Verbose native logs must show borrowed-room allocation or its allocation
fallback when borrowing is enabled. The disabled arm must not show borrowing.
Cache-aware routing logs its exact bonuses on the first forward pass. The driver
checks this after the public warmup and again after the OMP workload. Route-off
must not show the activation marker. A flag alone is not activation evidence.

## Procedure

First screen 16K input with the official control, official `--borrow 0`, custom
route-off, and custom `--cache-route 0.2,0.3`. Run one case at a time. Use a new
case name for each run. Run matched 64K or growing-context checks for a promising
candidate. The full trajectory is 16K, 64K, 128K, and 180K real input. Capacity
alone is not evidence of filled-context performance. Do not run a full product
of all controls.

Do not start a model until the controller has checked resources and started
its owned search service. Do not stop another person's services or workload.
Admission requires 72 GiB available Linux RAM, CPU load at most 8, no GPU compute
process, GPU usage at most 2 GiB, and GPU utilization at most 5 percent.
The running guard stops the owned worker and server below 8 GiB Linux available
RAM, after three failed memory reads, or if the model scope uses swap.
There is no Windows memory monitor.

The driver has a separate 2 GiB cap and 40 minute scope timeout. The model has
its separate 64 GiB scope. OMP has a separate 4 GiB scope. Each case has a
30 minute alarm. A phase has a 1200 second timeout. Keep the genuine OMP
replacement-edit schema, public corpus, trained routing baseline, and workspace
guard unchanged. The previous root-directory read refusal remains in the guard.

Prepare and inspect the code before freezing it. `freeze.py` refuses to replace
existing pins. Pass the custom binary and receipt pins to include them in the
same freeze as the official engine:

```sh
state/venvs/strata-official-0.2.8/bin/python \
  experiments/strata-candidates-20261109/benchmark/freeze.py \
  --native-binary /absolute/path/to/strata-dsv41 \
  --native-sha256 BINARY_SHA256 \
  --native-receipt /absolute/path/to/build-receipt.json \
  --native-receipt-sha256 RECEIPT_SHA256
```

Run from the repository root:

```sh
experiments/strata-candidates-20261109/benchmark/run-case.sh \
  --name official-control-16k --target 16384
experiments/strata-candidates-20261109/benchmark/run-case.sh \
  --name official-borrow0-16k --target 16384 --borrow 0
experiments/strata-candidates-20261109/benchmark/run-case.sh \
  --name custom-route03-16k --target 16384 --cache-route 0.2,0.3 \
  --native-binary /absolute/path/to/strata-dsv41 \
  --native-sha256 BINARY_SHA256 \
  --native-receipt /absolute/path/to/build-receipt.json \
  --native-receipt-sha256 RECEIPT_SHA256
```

Use the same custom arguments with `--cache-route off` for its control.
Use `--targets 16384 65536 131072 184320` for the growing trajectory.
Add `--dry-run` to validate and print a frozen configuration without inference.

The driver restores expert routing counters, lane calibration, generated server
configuration, and absent active state. Slots stay isolated per case. It checks
normal OMP settings without overwriting concurrent changes. Confirm all cleanup
flags and port 8899 after every case. The original trained routing profile is
identical in every arm. This is not an untrained routing test.

Raw evidence includes API timings, native logs, requested and live variant
records, generated token counts, GPU measurements, cgroup `memory.stat`, memory
events, swap, and disk counters. `cache_io.py CASE ...` collects the saved results.
Expert-store hit or miss counters are reported only if the engine exposes them.
CPU lane percentages are not RAM-cache miss rates. Root-device read counters
include other work and do not establish physical Windows SSD traffic.
