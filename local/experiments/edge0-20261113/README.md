# Edge0 comparison

This experiment uses Edge0's own Strata fork. It does not replace the official
Strata runtime used by Qwen or DeepSeek. The [bounded audit](audit.md) records
the reviewed build, API, native loaders, and network restrictions.

The source pin is `da9cc526747759227714ddde0728763d95306c04` in
`Edge0-AI/Edge0`. The weight pin is
`00f91ca2dc8290d74c79360f0d9941704e0f5c69` in `Edge0/Edge0-145B-q3k`.
The 74 selected files total 98.13 GiB. This includes the embedding table and
both Q4 and Q8 D/U projection files. The fetcher excludes startup scripts and
website assets. It verifies file hashes and keeps 20 GiB of disk headroom.

Run commands from the repository root. Preview the fetch without `--run`:

```sh
local/.venv/bin/python local/experiments/edge0-20261113/fetch.py
XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
  local/libexec/capped --mem 2G --time 12h \
  local/.venv/bin/python local/experiments/edge0-20261113/fetch.py --run
```

`build.py` reuses the checked CUDA 13 wheels and official CUDA libraries from
the existing installation. It copies the clean pinned ggml tree into isolated
build storage. It builds only `strata`, for SM89, with two compiler jobs.
It records the executable and kernel-library hashes. The build requires a
16 GiB cap, no swap, and priority 19:

```sh
XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
  local/libexec/capped --mem 16G --time 2h nice -n 19 \
  local/.venv/bin/python local/experiments/edge0-20261113/build.py --run
```

The first case uses 32K capacity and a 16K input. All modes select Q3_K experts,
Q4 D/U projections, a 40 GiB expert budget, direct embedding reads, and a
64 GiB model cap with no swap. The token embedding stays in RAM. The U
projection stays on the GPU. All modes set `STRATA_ROUTE_TAIL_SKIP=0`, so the
older fork does not drop experts at rank seven and above. OMP and its tools
use 4 GiB.

| Mode | Native speculation | MTP drafts | GPU reserve | GPU KV cells | Draft vocabulary |
| --- | --- | ---: | ---: | ---: | --- |
| `plain` (default) | `--spec 2`, no MTP layer | 0 | 800 MiB | 32,768 | None |
| `mtp` | `--spec 4` | 3 | 300 MiB | 32,768 | Default, 106,299 tokens |
| `mtp-small` | `--spec 4` | 3 | 550 MiB | 20,480 | Reviewed `data/draft_vocab_en.bin`, 40,525 tokens |

The smaller draft vocabulary is expected to save about 171 MiB of head VRAM.
The native pack requires a window capacity of at least two. Plain mode uses
`--spec 2`, but processes one target token at a time without MTP. All modes
set `--suffix-draft 0 --lookup-chain 0`; plain mode must record zero offered
and accepted drafts. The rejected window-one attempt is preserved under
`failed-plain-native-window1`.
The native source permits a minimum GPU KV window of 20K cells. These are
bounded experiment choices, not new production defaults. The selected mode
is recorded in the API configuration, live settings, result directory and
benchmark report. MTP modes must load the draft layer and record accepted
drafts. Plain mode must not load the layer or offer speculative drafts.

The controller checks GPU inactivity and CPU load before launch. It stops its
own work if Linux available RAM falls below 8 GiB or a memory check fails.

```sh
local/state/venvs/strata-official-0.2.8/bin/python -B \
  local/experiments/edge0-20261113/run.py --mode plain --run
```

Use `--mode mtp` or `--mode mtp-small` to select either draft configuration.
Omit `--run` to print the configuration. `--selftest` checks all three plans
and the memory guard without querying the GPU. After the short case passes, use
`--case matched64 --context 256k` for four repair checkpoints, or
`--case long --context 256k` to fill about 240K input tokens and test compaction.
The driver refuses to overwrite an existing result. Results and cleanup
evidence are in `local/results/edge0-20261113/`.

The initial `q3k-q4-32768-short-20261113` result is preserved. It measured
37.25 tokens/s decode and about 284 tokens/s initial prefill, but made no tool
calls and failed code and recall checks. The log reported 57 MiB free VRAM
and later warned about zero free VRAM. It predates explicit tail-skip disabling;
differences from the new modes cannot be attributed to MTP alone. New result
names include the mode, so they do not overwrite this initial case.

The API uses the existing outgoing-network guard and an explicit local
configuration. It does not execute the downloaded startup script or merge
downloaded launch commands into that configuration. The private OMP history
uses Edge0's actual model name and tokenizer. Existing search and normal
OMP settings remain in place. The controller stops its owned server before
it returns, including when its benchmark parent dies.
