# Strata candidate measurements

This experiment tests four changes against Strata 0.2.12, native 0.1.58.
Production defaults and official runtime files remain unchanged.

1. [DeepSeek prefill borrowing](https://github.com/Niko1221/Strata/issues/5570):
   compare the current allocation with `DS4_BORROW=0`.
2. [Cache-biased routing](https://github.com/Niko1221/Strata/pull/5596):
   compare a separately built engine with the option off and at `0.2,0.3`.
   This option changes expert selection and can change answers.
3. [DSML alias strings](https://github.com/Niko1221/Strata/pull/5598):
   reproduce the parser defect and test the proposed correction.
4. [Linux arena admission](https://github.com/Niko1221/Strata/pull/5531):
   test the host predicates and determine whether our memory configuration
   reaches the changed branch.

## Method

The performance target is full Unsloth DeepSeek UD-IQ1_S. Keep all six experts,
a 48 GiB expert cache, a 64 GiB process cap, zero swap, 192K context capacity,
12 CPU workers, and 4096-token prefill chunks. Keep DSpark disabled.
The model and its children cannot open outgoing connections. Stop below
8 GiB available Linux RAM. Do not monitor Windows RAM during these runs.

Use the existing public OMP repair and resume workload with replacement edits.
Every new arm uses temperature 1 and seed 20261109. Record input and output
token counts, native timing denominators, task checks, resource use, and
cleanup. Start with 16K input. Repeat promising settings at longer filled
context. Do not infer filled-context performance from capacity alone.

The source-built control separates build differences from the routing option.
The native scorer compares 4,995 teacher-forced positions in five public text
segments. The text is tokenized with the installed DeepSeek tokenizer.
The upstream committed token IDs belong to Qwen and must not be reused.
Scoring uses 8K capacity and 16-token windows. It is a short distributional
comparison, not a coding benchmark or a long-context quality guarantee.

## Artifacts

- `cache-route/`: exact patch, source audit, and isolated build instructions.
- `parser-arena/`: baseline and candidate CPU checks and applicability review.
- `benchmark/`: isolated OMP controller and launch variants.
- `quality/score.py`: guarded native distribution scoring.
- Raw local evidence: `results/strata-candidates-20261109/`.

Measurements are complete. See the
[report](../../docs/strata-candidates-20261109.md) for settings, timings,
correctness limits, and recommendations.

Borrowing-off reduces decode from 9.21 to 8.31 tokens/s at 16K input.
Keep borrowing enabled at the current fixed chunk size.
Cache routing improves decode from 9.14 to 12.63 tokens/s at 16K and from
9.51 to 12.15 at 64K. It changes expert selection; 8.71% of scored positions
have a different top token. Keep it experimental. The parser fix passes all
156 targeted fixtures. The arena change does not apply to the current profiles.

All six OMP cases pass the recorded repair, recall, and resume checks.
Both 64K repair replies violate the requested JSON-only format, which the
existing repair checker does not enforce. See the report for this limit.
All three scoring runs and six OMP runs stop their owned processes. Final
verification finds no GPU compute process, open test port, active model state,
or changed frozen runtime file. No production default or weight changes.

`benchmark/results-16k.json`, `benchmark/results-64k.json`, and
`quality/results.json` contain compact measurements. The local lifecycle smoke
probes remain outside version control. Their exact bytes stay on disk because
the frozen local harness records their hashes.
