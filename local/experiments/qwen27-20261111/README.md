# Dense Qwen 27B comparison

Compare the pinned Dynamic 3.1 UD-Q4_K_M file with standard decoding and
two-token MTP. Both arms use 196608-token context capacity and the same 16K
constructed public-source history. The workload reuses the existing OMP code
repair, recall, and strict resume checks. It is not a general quality benchmark.

`server.py` owns one model controller and its 48 GiB scope. It requires an idle
GPU, CPU load at most eight, and at least 56 GiB available Linux RAM. It stops
below 8 GiB available Linux RAM, on scope swap use, or after repeated failures
to read required resource counters. It does not check Windows memory. Optional
disk counters can be unavailable. Missing counters are not zero.

The controller verifies the cap and zero-swap setting before writing
`ready.json`. It writes `aborted.json` on failure and always stops its owned
server on exit. Check its CPU-only stop test before the first run:

```bash
.venv/bin/python experiments/qwen27-20261111/server.py --self-test
```

Run the controller through `heavy-management/bin/capped` with a 1 GiB cap.
The model gets a separate 48 GiB cap. The controller exits after 40 minutes
unless stopped earlier. Start an owned `searxng-local` only if none is running.

```bash
.venv/bin/python experiments/qwen27-20261111/server.py \
  --arm none --output results/qwen27-20261111/server-none-r2
```

Read the controller PID from that directory's `ready.json`. Run the workload
through `capped` with a 4 GiB cap. Replace `PID` with the recorded number:

```bash
.venv/bin/python experiments/qwen27-20261111/run.py \
  --output results/qwen27-20261111/none-r2 \
  --server-output results/qwen27-20261111/server-none-r2 \
  --controller-pid PID --speculative none --run
```

The workload attaches only to that controller. It uses a private OMP profile,
replacement edits, temperature 1, top-p 0.95, xhigh, and a 4096-token response
limit. Neither arm sets a fixed seed. Resume checks use the existing stricter
request parameters. Normal OMP state remains unchanged.

Stop the controller with SIGTERM after the workload. Check that its scope,
port, and active state are released. Start a fresh controller with `--arm mtp`
and new output directories. Use `--speculative mtp` for the matching workload.
Stop search afterward only if this experiment started it.

Read native request timings and cumulative counter deltas. Report actual input
and generated token counts with throughput and total task time. The two arms
can generate different answers and tool histories. A single pass per arm does
not establish a small speed improvement. Context capacity is not filled input.

Keep failed attempts. `server-none-r1` stopped before inference because the
first guard treated the missing cgroup I/O counter as a required memory read.
The initial `none-r1` workload also stopped before inference because process
I/O counters were unavailable. Both optional counters now permit missing data.
The completed standard run exposed a cleanup race when the worker had already
exited. Cleanup now accepts that case. Its first summary also incorrectly
reported a fixed seed. The pinned runtime ignores `LLAMA_ARG_SEED`, and all
requests omitted a seed. The summary records this metadata correction.
