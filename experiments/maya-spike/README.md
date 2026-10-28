# Maya 1017 spike

This experiment uses Maya v1.1.28 at commit
`6afa525bbabc3a8f5fd165f89e9bc72155d9bd39`.
GGML uses the commit required by Maya:
`3cf03257f219afbe7334045ff7c6a06ac68c627d`.

The experiment reuses the five verified ISTA IQ1_M GGUF files.
It copies the existing Strata pack and expert profile into each case directory.
The pack references the original weight bytes. It does not convert weights.
No loader patch is required for this pack.

The source is in `state/sources/maya-1.1.28`.
The private build is in `state/builds/maya-1.1.28`.
The commands, configuration, logs, controller, and results are in
`results/maya-spike/20261025`.

Build only the `maya-ds4` target for CUDA architecture 89.
Use the existing private CUDA 13 compiler and libraries from the Dory environment.
Set the CMake `CUDA_cudart_LIBRARY`, `CUDA_cublas_LIBRARY`, and
`CUDA_cublasLt_LIBRARY` values to those CUDA 13 files.
CMake can otherwise select the older system CUDA 12 libraries.
Put `/usr/lib/wsl/lib` before the CUDA library directory at runtime.
This selects the WSL driver library.

Each model process runs under a 48 GiB hard memory limit with swap disabled.
The existing network guard blocks outgoing connections before Maya starts.
The API listens on `127.0.0.1:18899`.
The controller stops the model if Linux available RAM or Windows free RAM
falls below 8 GiB. It also stops the model when the case exits.

The configuration uses four CPU threads, 4096-token prompt chunks,
six routed experts, and `--mmap-experts on`.
DSpark, vision, and expert skipping are disabled.
Maya has no equivalent to Strata's fixed `--ssd-cache` budget.
Its file cache remains subject to the process memory limit.
Startup read-ahead is limited to 16 decimal GB.
The initial automatic read-ahead test crossed the Windows memory floor.
The controller stopped that test before generation.

OMP uses the reviewed runtime with private configuration, conversation,
and result files. Its requests use this experiment's API endpoint.
The normal `localmodel`, `omp-local`, and `omp-remote` configuration is unchanged.

The final comparison uses the stock Strata API for both native engines.
The Maya API assigns a different role to OMP's project instruction message.
The remaining reference history, tools, and repair request have matching tokens.
The first Maya API trial repeatedly requested commands that need approval.
It was stopped before the code repair. Its logs are retained.

The native Maya engine works through the stock Strata API.
This path uses the existing serving dependencies and `--offline` option.
The network guard remains the enforced outgoing-connection restriction.

Both completed OMP cases pass the repair, 100 assertions, three recalled
facts, and session resume. Maya reads the approximately 16K prompt at
419.7 tokens/s and decodes at 5.07 tokens/s. The fresh Strata control reads
it at 254.3 tokens/s and decodes at 5.61 tokens/s. Both configure 256K
capacity under a 48 GiB model cap. The maximum context is not filled.

Maya's workflow takes 340 seconds. Strata's takes 232 seconds.
Maya generates more tokens and recovers from three tool errors.
These single passes do not isolate workflow time from sampling or host load.
The normal launcher continues to use Strata.

Both model servers are stopped. The prior Strata configuration and expert
profile are restored. Clean model cache pages are released after shutdown.

Read `results/maya-spike/20261025/summary.md` for the full comparison.
The JSON files, raw requests, responses, and resource samples are retained
in that directory. The prototype controller previews its settings by default:

```sh
state/venvs/strata-official/bin/python results/maya-spike/20261025/run.py \
  --frontend strata --context 262144 --target 16384 --mem 48 --warm-gb 16
```

To repeat the experiment, add `--run --name NEW_CASE_NAME`.
Start the existing SearXNG service before an OMP case.
The controller stops its model when the case finishes or fails.

The quiet-machine rerun uses the same settings and reverses the run order.
Strata reads the initial 15961-token prompt at 256.1 tokens/s and decodes at
5.66 tokens/s. Maya reads it at 430.9 tokens/s and decodes at 5.23 tokens/s.
Both pass the same repair and resume checks. Both servers are stopped.
The result preserves the earlier prefill and decode ordering.
Read `results/maya-spike/20261025/quiet-summary.md` for the rerun and limits.
