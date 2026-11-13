# Bonsai candidate OMP benchmark

This helper runs the reviewed OMP runtime. It uses a private configuration,
session, workspace, and state directory. It does not use private conversations.
The default command previews the case. Only `--run` starts a model.

The helper also accepts the pinned `groot-19b` and `lfm3-1.7b` models.
For these cases, add `model`, `memory_gib`, and `windows_floor_gib` to the
launch JSON. Set the command's `--mem` value to match `memory_gib`.
The defaults remain `bonsai2-27b`, 48 GiB, and an 8 GiB Windows floor.
The Linux floor is always 8 GiB. The supported Windows floor is 4 to 8 GiB.

Create a launch file for each controlled case. Use the repository Python
executable so the launcher retains the case-specific peak record path.
For example:

```json
{
  "label": "trained-replay-checkpoints",
  "command": [
    "/home/tritlo/Code/local-models/.venv/bin/python",
    "/home/tritlo/Code/local-models/localmodel.py", "start", "27b-256k-native",
    "--runtime", "bonsai-next", "--mtp-head", "procreations",
    "--replay", "--checkpoint-every", "8192", "--mem", "48G"
  ],
  "environment": {"LD_LIBRARY_PATH": "/usr/lib/wsl/lib"}
}
```

Run one case:

```sh
uv run --no-project --python .venv/bin/python python \
  experiments/bonsai-next/benchmark.py \
  --launch-file /tmp/bonsai-case.json \
  --output results/bonsai-next/trained-replay-checkpoints --run
```

Use `--history-tokens 196608` for the selected long workflow. `16384` is also
accepted, but short conversations can remain below OMP's 20000-token
retention budget. OMP then refuses compaction with `Nothing to compact`.
This is a benchmark limitation. It does not establish a model failure.
The default is
65536 actual input tokens, including the OMP system prompt and tool schemas.
The server capacity remains 262144. Each model response has a 4096-token limit.
The default sampling settings use temperature 1 and seed 424242. Use
`--temperature 0` for a separate deterministic parity check. Do not combine
those measurements with the temperature 1 comparison.

The launch inherits `PATH`. `/usr/lib/wsl/lib` supplies WSL driver libraries.
The normal model launcher also sets its private CUDA library paths.

The workflow uses one long tool result from local public runtime source.
It performs a real read, edit and test loop. It then changes a revision marker
inside the old tool result and resumes OMP. The output records the actual
common prefix position. This tests checkpoints inside a message. Another
repair, cancellation and recovery, soft compaction, and a resumed repair follow.
Independent checks validate the three release decisions and code semantics.

Use separate cases for the teacher and trained heads, replay on and off, and
checkpoint intervals 0 and 8192. Keep all other launch settings equal.
For Bonsai, the helper requires MTP depth 2 and chaining on. Groot and LFM3
use their prepared profiles without speculative decoding. Record the source and build
receipt separately before the run. The helper records the full launch plan
and live model settings. It does not build or fetch a model.

The model uses the declared RAM cap with no swap. OMP and its children have a
separate 4 GiB cap. An independent thread checks Windows free RAM and Linux
available RAM every five seconds. It stops this model below either configured
floor, or after three failed memory readings. It never adopts an existing
server. Existing SearXNG remains required and stays outside the benchmark.
Phase deadlines default to 900 seconds. The total deadline is 2400 seconds.

Each output contains the OMP transcript, exact API requests, streaming timings,
native counter differences per phase, resource samples, source hashes,
validation results and server log. `summary.json` records the final state.
The helper stops its own model and OMP scopes after completion or failure.
It retains all files. Use a new output directory for each case.

The history is constructed from public source. It is not a recorded coding
task. Different heads can produce different tool sequences at temperature 1.
Compare completed workflow time, first-token latency and native decode rates
separately. Acceptance counters describe the generated workload. A single
case does not establish a small speed difference.
