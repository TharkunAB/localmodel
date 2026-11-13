# Qwen Engram and context comparison

This driver reuses the measured public-source OMP fixture in `experiments/deepseek-strata/filled_omp.py`. It compares Qwen core quantizations, the IQ4_NL and Q8_0 Engram tables, and context capacities. It does not fetch files or change launcher defaults. Existing commands select IQ1_M and 36 GiB of resident experts.

Use the wrapper from this directory. It puts the controller under a 4 GiB cap. Each model uses 64 GiB with no swap. OMP and its tools use 4 GiB. The validator uses 256 MiB. An owned SearXNG service uses 1 GiB. A healthy existing search service stays running.

Run CPU checks first. Freeze provenance only after production edits and table verification finish:

```sh
./run-case.sh --selftest
./run-case.sh --freeze
./run-case.sh --arm default192 --case matched64 --run
./run-case.sh --arm q8192 --case matched64 --run
./run-case.sh --arm q8256 --case matched64 --run
./run-case.sh --arm default256 --case matched64 --run
```

The four arms use the selected core, seed, workspace path, public source corpus and repair requests. They keep the selected expert RAM budget, four MTP draft tokens, 64K GPU KV, INT8 KV and direct Engram reads. Adaptive fetch and suffix lookup stay off. The driver requires native evidence that MTP loaded and accepted tokens. It checks the selected Engram path in the native startup log. The Q8_0 header and download receipt establish its tensor type.

`matched64` grows one conversation through 16K, 32K, 48K and 64K inputs. Each checkpoint repairs another function, recalls three constraints and resumes the session for an exact answer. The source guard restricts file changes and shell commands. The independent validator checks code behavior.

Run `--case long` on the selected table at both capacities. The last input target is 180K at 192K capacity, or 240K at 256K capacity. The driver then enables normal automatic compaction and probes constraint retention. The probe enables compaction in both its template and its existing private OMP config. The launcher preserves the private config after its first start. The probe records these effective settings and requires a persisted compaction entry as evidence. Successful recall without observed compaction is not a compaction result. The first 192K long run used the earlier probe and remains recorded as `observed: false`.

Repeat close results with `--repeat 2`, preferably in reverse arm order. A repeat changes the sampling seed for every arm in that pair. One pass is a screen, not a general quality ranking. Later checkpoints contain previous generated outputs, so their inputs can differ between arms. Compare saved `*-final-size-identity.json` hashes before making an identical-prompt speed claim. Generated token counts and tool paths also affect task time.

The driver refuses admission unless GPU compute is empty, GPU utilization stays at most 5%, GPU memory use stays at most 2 GiB, one-minute CPU load is at most 8, and Linux has 72 GiB available. It stops owned work if available Linux RAM falls below 8 GiB, model swap is used, an OOM occurs, or three resource checks fail. Windows memory is not part of this guard. The controller deadline is 75 minutes. It leaves time for cleanup instead of using a short outer forced-kill deadline.

Results go to `local/results/qwen-engram-20261113/<arm>-<case>-r<repeat>`. The stable `work` directory moves there only after owned processes stop. A leftover `work` directory requires inspection before another run. Existing results and provenance are never replaced. Native metrics, request bodies, actual token counts, phase timing, objective checks, memory events, GPU samples and shared disk counters remain in each case. Shared disk reads are not model cache misses. The input expert profile and normal OMP settings are checked. The input profile is restored if the owned run changes it. This Qwen configuration has no persistent lane profile or learned-profile output.

Use `--quant q2_k_xl` or `--quant iq3_xxs` to select other installed weights. Use `--expert-ram-gb` to change the resident expert budget. The model cap stays at 64 GiB. For example, screen Q2_K_XL at 256K with Q8_0 Engram and 36 GiB of resident experts. Then test the longer conversation with 48 GiB if the first case is useful:

```sh
./run-case.sh --quant q2_k_xl --arm q8256 --freeze
./run-case.sh --quant q2_k_xl --arm q8256 --case matched64 --run
./run-case.sh --quant q2_k_xl --expert-ram-gb 48 --arm q8256 --freeze
./run-case.sh --quant q2_k_xl --expert-ram-gb 48 --arm q8256 --case long --run
```

A nondefault quantization or RAM budget selects `local/results/qwen-quants-20261113/<quant>-ram<GiB>g-<arm>-<case>-r<repeat>`. Each quantization and budget has its own provenance file. Freeze each combination after all code edits and artifact verification finish. The snapshot records every selected model shard and hashes its selected manifest. The fixture uses the selected source and tokenizer directories. It does not require the IQ1_M weights when another quantization is selected. Live checks require the selected quantization and expert budget.

The additional study uses its own stable workspace path. Its request identities can differ from those of the original Engram study. Compare quantizations at the same RAM budget. A 48 GiB run also changes expert residency. Cgroup memory peaks can include different amounts of charged file cache; a lower peak alone does not prove that a quantization needs less RAM.

All commands are previews unless `--run` or `--freeze` is present. Do not run arms concurrently. Keep other work idle during each timed case. No inference is part of `--selftest`.
