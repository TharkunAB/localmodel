# Qwen Flash-Next Engram and context results — 2026-11-13

The separate Q8_0 Engram table works with the existing IQ1_M core and MTP. The four 16K–64K conversation runs decoded at 51.23–53.82 tokens/s. Q8_0 long runs reached repair inputs of 184,030 tokens at 192K capacity and 245,433 tokens at 256K capacity. Their weighted repair/resume decode rates were 50.36 and 49.30 tokens/s. OMP also completed a real compaction at 256K and retained the requested facts. No large throughput regression appeared in the matched comparison. These measurements do not establish a general quality or speed advantage for Q8_0.

Both table variants have 320,001,536 rows of 160 values. Q8_0 changes the precision of those values; it does not add learned rows, experts or context capacity. The bundled IQ1_M artifact uses IQ4_NL for this table. The separate Q8_0 artifact adds 50.66 GiB on disk. The original core shards remain installed.

The table is pinned to Unsloth revision `f5d175de2cf7f720d45d6064019a81fcf056e986`. Its size is 54,400,261,312 bytes and its SHA256 is `e3d76a2747c1fe53e3420d1bb7fd9593aafd03035fee4926317db80ad5c5296a`. Download verification established the file hash. The startup check verifies the recorded file identity and its Q8_0 tensor header. The file remains byte-for-byte unchanged. It is a standalone GGUF table, so the launcher passes all three original core shards explicitly through `--native-dense-gguf` and selects the table through `--ple-gguf`. This avoids using the standalone table as the original dense shard set. Native startup logs confirm the selected table path and the loaded MTP layer. The pin and checks are in `local/qwen_engram.py`; the installed receipt is `local/models/qwen38-engram/q8_0/verification.json`.

The user selected Q8_0 as the start/fetch default and 256K as the context default. Use `--context 192k` for the smaller capacity. Use `--engram iq4_nl` for the smaller bundled table; explicit `--engram q8_0` also works. The recorded `default192` and `default256` cases refer to the previous IQ4_NL default.

```sh
localmodel fetch qwen3.8
localmodel start qwen3.8
localmodel start qwen3.8 --context 192k
localmodel start qwen3.8 --engram iq4_nl
```

The measured settings were Strata API v0.2.13 with native engine 0.1.59, a 64 GiB model cap, no swap, a 36 GiB expert RAM budget, four MTP draft tokens, INT8 KV, 65,536 GPU KV cells, direct Engram reads and a 1,024 MiB VRAM reserve. Adaptive fetch and prompt lookup were off. All runs used the same configured seed, public-source OMP fixture and fixed workspace path. The native engine reported about 31.95–31.98 GiB of pinned resident experts. It kept 65,536 KV cells per QSA layer in VRAM at both capacities. Host KV increased from 2.32 GiB at 192K to 3.09 GiB at 256K: about 0.77 GiB more pinned RAM.

The matched runs grew one real OMP conversation through approximately 16K, 32K, 48K and 64K input tokens. Decode rates below divide the total generated tokens by total native decode time across repair and resume requests. They exclude prefill, tools, startup and offline prompt capture.

| Case directory | Generated tokens / native decode seconds | Weighted decode | MTP accepted / offered | Observed model cgroup peak |
| --- | ---: | ---: | ---: | ---: |
| `default192-matched64-r1` | 5,280 / 102.821 | 51.35 t/s | 3,468 / 5,033 (68.91%) | 64.00 GiB |
| `default256-matched64-r1` | 3,767 / 72.786 | 51.75 t/s | 2,568 / 3,437 (74.72%) | 61.06 GiB |
| `q8192-matched64-r1` | 3,584 / 69.961 | 51.23 t/s | 2,242 / 3,098 (72.37%) | 64.00 GiB |
| `q8256-matched64-r1` | 2,207 / 41.009 | 53.82 t/s | 1,562 / 1,921 (81.31%) | 53.01 GiB |
| `q8192-long-r1` | 4,468 / 88.719 | 50.36 t/s | 3,020 / 4,368 (69.14%) | 64.00 GiB |
| `q8256-long-r1` | 3,505 / 71.089 | 49.30 t/s | 2,362 / 3,353 (70.44%) | 58.26 GiB |

The first matched request was identical in all four runs: 15,997 input tokens with the same captured request hash. Cold prefill measured 1,334, 1,104, 1,346 and 1,577 tokens/s in table order. Subsequent checkpoints reused history and processed only about 14K–16K additional tokens. Across all matched repair/resume requests, 92%–93% of input tokens were reused. These are not cold 32K, 48K or 64K prefill measurements. Including small tool-followup prefills, total fresh tokens and prefill time were 61,798/54.841 s, 63,322/64.508 s, 62,399/57.961 s and 63,800/53.029 s.

All 16 matched repair/recall checkpoints passed the independent code checks. All 16 failed the separate strict final-JSON check because the visible response included Markdown fences or prose. The first three configurations passed all four strict resume checks. Q8_0 at 256K returned the correct `{"probe_value":6}` inside fences on all four resumes, so its strict resume score was 0/4. Its recorded `all_quality_checks_passed: false` therefore reflects formatting, not lost recall. Recovered tool errors numbered 1, 2, 4 and 0 in table order; request counts and generated output lengths differed.

The 192K long run reached repair inputs of 15,997, 65,068, 130,850 and 184,030 tokens. All four code, recall and strict resume checks passed. Strict repair JSON remained 0/4 because of fences/prose. At the final checkpoint, the engine reused 131,919 tokens and processed 52,111 new tokens in 35.377 s, or 1,473 tokens/s. Repair/resume decode at that checkpoint was 49.81 tokens/s. The four checkpoints took 281.19 seconds of repair/resume wall time; this is separate from native decode time.

The 256K long run reached repair inputs of 15,997, 65,381, 130,557 and 245,433 tokens. All four code and recall checks passed. All four strict repair and resume format checks failed because of fences/prose; each resume contained the correct probe value. At the final repair checkpoint, the engine reused 131,640 tokens and processed 113,793 new tokens in 78.916 s, or 1,442 tokens/s. Its repair/resume decode rate was 48.91 tokens/s. The four checkpoints took 321.57 seconds of repair/resume wall time.

The final 256K resume exposed a separate latency cost: Strata parked a 246,485-token conversation snapshot of 4,594,699,624 bytes in 16.399 s. The request reused 246,283 tokens and needed only 234 new tokens, but its total reported prompt work was 17.893 s. Decode still ran at 54.6 tokens/s. This snapshot cost is part of long-conversation cache handling. These runs cannot attribute it to Q8_0 table precision alone.

The original 192K compaction probe returned the requested facts, but **did not compact**. Its retained private OMP config still had compaction disabled; its `observed: false` result is preserved. Commit `c08c774` enables compaction in both the template and effective private config. With this correction, the 256K run recorded a persisted `compaction` entry after 246,556 conversation tokens. The post-compaction answer retained all three requested facts and the correct probe value. Its `retention_passed: false` is a strict-format failure because that answer was fenced JSON; this was not a full evaluation of summary accuracy.

The 256K compaction phase took **180.95 seconds**, separately from the repair/resume timings above. It made three summary requests and one final reply request. Client submission order was 62,354, 796, 8,858 and 24,427 input tokens; native execution order was 796, 62,354, 8,858 and 24,427. The 8,858-token request included a native continuation at 9,132 tokens. The final reply used 24,427 input tokens, including 3,589 reused tokens. Keep these extra requests out of the repair/resume decode comparison. Including the probe, the 192K run recorded 3,120 accepted drafts of 4,470 offered; the 256K run recorded 5,523 of 7,962. The table excludes both probes.

Cgroup peaks do not show that Q8_0 saves RAM. At the end of the matched runs, ordinary file-cache charges (`file - shmem`) were 23.56, 20.00, 23.36 and 12.93 GiB. Pinned expert residency remained approximately 32 GiB. Shared file pages can remain charged outside a later run's cgroup, and cache warmth differed across launches. The long runs peaked at 64.00 and 58.26 GiB without swap or OOM. Their lowest sampled Linux available memory was 59.90 and 58.94 GiB. Three matched runs logged approximately 490–497 MiB of free VRAM at one request boundary, without an observed decode collapse.

These are single-trial workflow measurements. Later prompt hashes differ because the conversations retain previous generated output. GPU expert cache placement varied from 1,917 to 2,006 slots in the matched runs. Automatic conversation-cache budgets also differed. Faster task completion can reflect shorter replies or fewer retries. The small decode differences do not establish a causal speedup, and these simple repairs do not establish general coding quality. The long runs demonstrate the selected Q8_0 configuration at their recorded input lengths; there is no corresponding IQ4_NL long run here.

Evidence is under `local/results/qwen-engram-20261113/<case-directory>/`: `measurements.json`, `case-summary.json`, `summary.json`, `engine.log`, captured requests, `*-final-size-identity.json` and memory samples. All six cases recorded successful owned-process cleanup and state/profile restoration. They preserved the preexisting SearXNG service and normal OMP settings. All benchmark-owned processes are stopped.
