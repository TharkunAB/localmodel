# Cache-routing distribution check

The official engine and the custom engine with routing disabled produce
byte-identical score files for all 4,995 positions. The SHA256 is
`28ec0af436b785660b32e9f4c3c2a191357a0e772651ff10ed2edddde3062b05`.
This controls for numerical differences in this build on this corpus.

The `0.2,0.3` routing bonuses change the top predicted token at 8.71% of
positions. Aggregate perplexity changes from 5.783 to 5.735. This small
improvement does not establish better coding quality. Segment results differ.

| Segment | Positions | Top-1 agreement | Control perplexity | Routing perplexity |
|---|---:|---:|---:|---:|
| Code | 999 | 92.79% | 3.932 | 3.967 |
| Documentation | 999 | 87.99% | 21.594 | 19.870 |
| English | 999 | 88.49% | 4.205 | 4.286 |
| Other languages | 999 | 91.49% | 6.686 | 6.905 |
| Agent transcript | 999 | 95.70% | 2.710 | 2.659 |
| All | 4,995 | 91.29% | 5.783 | 5.735 |

Top-1 agreement compares the two engines. It is not task accuracy. The
upstream evaluator also reports mean approximate KL 0.04772 and p99 0.5157.
It computes this over the reference's top 64 tokens, renormalizes that set,
and substitutes the candidate's 64th log-probability for missing tokens.
This is not a full-vocabulary KL measurement.

## Method and limits

The input is Strata's public five-segment quality corpus, truncated to 1,000
tokens per segment. The installed DeepSeek tokenizer produced the token IDs.
Each segment has 999 scored positions. The reference is the installed full
Unsloth IQ1_S target, not BF16 or Q8. These results measure the additional
routing change. They do not measure the quantization loss of IQ1_S.

The native scorer uses 8K context capacity and 16-token teacher-forced windows.
All three runs use the same read-only expert profile, 48 GiB expert cache,
64 GiB memory cap, zero swap, and outbound-network guard. They keep all six
routed experts and disable DSpark. The routing activation marker is present
in the candidate log. All runs verify native isolation and stop their owned
processes and scopes. The expert profile remains unchanged.

The runs take 448.5, 432.5, and 322.3 seconds respectively. Do not treat these
elapsed times as OMP throughput. File-cache state and cgroup memory charging
differ between runs. The separate OMP comparison records native timing and
generated-token denominators.

The official run observes 64.00 GiB in its cgroup; the custom control observes
51.78 GiB; routing observes 48.19 GiB. These are sampled charges, including
file cache. They do not establish a reduction in required resident memory.
Available Linux RAM stays above 53.54 GiB throughout all three runs. Swap
remains zero. One routing attempt is deferred at admission because the
preceding scorer's load average is still above 8; it starts no engine.

`results.json` contains the exact summaries and upstream comparison results.
Raw scores, resource samples, logs, corpus provenance, and the deferred
attempt remain under `results/strata-candidates-20261109/quality/` locally.
The exact custom build and library pins are in `../cache-route/`.

The comparison uses the pinned upstream
[`tools/quant_quality.py`](https://github.com/Niko1221/Strata/blob/146df3c468739147f520ac3816de806f71232571/tools/quant_quality.py).
The cache-routing change is
[PR 5596](https://github.com/Niko1221/Strata/pull/5596).
