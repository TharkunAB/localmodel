# Strata release watch

The local `strata-watch.timer` checks the official GitHub release every six hours.
It starts work through `codex queue` in the existing maintenance conversation.
It does not download weights or execute release text.
WSL and the Codex daemon must be available. The timer checks after WSL starts
if a scheduled check was missed. An unavailable daemon causes a visible service
failure. The next timer check retries delivery.

The watcher requires 10 GiB free disk space and 72 GiB available Linux RAM.
It defers work when a model is active, a GPU compute process exists, GPU memory
use exceeds 2 GiB, GPU load exceeds 5%, or CPU load exceeds eight runnable tasks
(half the logical CPU count on smaller machines). These are admission checks.
The benchmark must keep its own resource guards active throughout each run.

The watch state is `state/strata-watch.json`. A queued release is retried after
24 hours until `completed_tag` records that release. This also covers an update
whose pins changed before its measurements finished. A completed but rejected
release is not installed again by the watch.

## Update and measurement procedure

1. Read the current repository instructions and check the worktree. Pull before
   edits when safe. Preserve unrelated changes and create a work branch.
2. Verify the release against `https://github.com/Niko1221/Strata/releases`.
   Review relevant source changes. Keep signature and checksum verification.
   Update the local source, engines, packer, and remote launcher pins together
   where the release applies. Preserve the inference network restrictions.
   Do not deploy to a remote server or push Git commits.
3. Recheck GPU processes and memory, CPU load, Linux memory, and free disk space.
   Defer if another model or a heavy job is running. Do not stop user processes.
   Keep existing weights and results. Do not download another quantization or
   delete files to make space. Retain the previous runtime for rollback until
   validation completes. Check the required download and extraction space.
4. Use Coder with zero private expert cache, a 64 GiB hard memory cap, zero swap,
   256K context capacity, and the current verified runtime settings. Respect
   later explicit user changes. Use `~/Code/heavy-management` for capped serial
   jobs. Do not increase queue concurrency. Stop below 8 GiB available Linux
   memory. Do not monitor Windows memory for these benchmark runs.
5. Establish a matched old-runtime baseline at those settings when no suitable
   result exists. Older zero-cache results used a 70 GiB cap; they are not a
   matched 64 GiB baseline. Use the guarded OMP fixture at 16K, 32K, and 64K filled
   input. Keep the fixture, routing profile, sampling, and settings fixed. Do not
   reuse a frozen driver after changing its recorded source hashes.
6. Measure the new runtime. Record prefill and decode throughput, task time,
   tool errors, repair and recall checks, peak RAM and VRAM, disk reads, and
   expert-cache counters where available. Distinguish context capacity from
   filled input. Retain failures. Do not enable a different drafter or change
   defaults merely because the release advertises it.
7. Always stop benchmark servers and helpers, release scopes, and restore
   routing profiles and any temporary state. Verify that cleanup completed.
   Restore the previous runtime if compatibility or quality checks fail.
   Report regressions and measurement limits. Save results and update `--stats`
   and the relevant report. Commit locally; do not push.
8. After a completed comparison or a verified rejection, set `completed_tag`
   in `state/strata-watch.json` to the checked release tag. Preserve the other
   fields. If the run is blocked or deferred, leave it pending. Report the reason
   in the maintenance conversation.

## Inspect or stop

```sh
systemctl --user list-timers strata-watch.timer
journalctl --user -u strata-watch.service -n 30 --no-pager
systemctl --user disable --now strata-watch.timer
```

Stopping the timer prevents future checks. It does not cancel work already
queued into Codex. Cancel that work in its conversation if required.

The installed service calls `experiments/strata-watch.py` with the conversation
UUID and the absolute Codex path. It runs at low CPU priority with a 256 MiB cap.
Only the release check runs in that service. Benchmark jobs run through the
existing conversation and keep their separate 64 GiB cap.
