# Strata 0.2.8 review

Reviewed on 2026-10-30. This review covers the native engine changes that affect
our NVIDIA profiles and the changed output parser. It is not a complete security
audit. No GPU benchmark or remote server restart was part of this review.

## Selected update

Use the official 0.1.54 native engines from the 0.2.8 release. Keep the official
0.2.7.1 Python server and parser at commit
`46c152a1310f77cd2f0ce2c7cd213962df14fe83`.

The 0.2.8 source at commit `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`
has a parser regression. It can convert a quoted example into a tool call.
Keeping the old parser avoids this identified regression. It does not establish
that the old parser has no other defects. This choice also excludes the new
server-side tool recovery and stop-token fixes.

## Parser regression

The new `OutputParser._stray_think_at()` accepts a closing `</think>` at the start
of a line even inside a code fence. The content parser then clears the fence
state. A following quoted call can become an actual tool-call event.

This model output reproduces the problem when the request declares `bash` with
a string `command` parameter:

````text
Plan.
</think>
Example, do not execute:
```xml
</think>
<tool_call>
<function=bash>
<parameter=command>
echo quoted-example
</parameter>
</function>
</tool_call>
```
````

Results from direct parser checks:

- The 0.2.7.1 parser keeps the example as text. It emits no tool call.
- The 0.2.8 parser emits a `bash` call with command `echo quoted-example`.
- Both whole-input and one-character-at-a-time checks reproduce the difference.
- Setting `tool_call_recovery` to `false` does not prevent the conversion.
- An ordinary fenced example without the extra `</think>` remains text.
- A call to an undeclared tool remains text.

DeepSeek is affected too. `DsmlOutputParser` translates DSML into the Qwen tool
format and inherits `OutputParser`. The same fenced example with DSML tags emits
a call in 0.2.8. These checks inspect parser events only. They do not run the
command in the example.

The affected code is in
[frontend.py](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/serve/frontend.py#L1309)
and the inherited parser is in
[dsml.py](https://github.com/Niko1221/Strata/blob/870aedd9e1ebe1c91f94da93b59af17729f6c5c0/serve/dsml.py#L166).
The new code addresses malformed output. This finding is evidence of a defect,
not evidence of malicious intent.

The upstream quoted-value and tool-corpus checks pass: six test methods completed
in 0.47 seconds. They cover nested calls inside string arguments and undeclared
names. They do not cover the counterexample above. The new lenient argument
reader uses JSON parsing and bounded string processing. No new execution or
network operation was found in the reviewed parser changes.

## Expected effect on our profiles

The [upstream release notes](https://github.com/Niko1221/Strata/releases/tag/v0.2.8)
report these measurements:

- Qwen IQ3_S on one RTX A6000: story decode increased from 105.9 to 111.4
  tokens/s; code increased from 139.8 to 145.2. Prefill was unchanged.
- DeepSeek 2.25 on one RTX A6000: decode increased by 4.3%.
- `DS4_FAT_ROWS` increased DeepSeek prefill by 2% to 5% on two RTX A6000 cards.
  It is off by default. It changes floating-point summation order, so output
  need not be identical.

These are upstream measurements on other hardware and workloads. They are not
measurements of this installation.

**Local RTX 4070, Qwen:** expect similar speed or a small increase. The A6000
result is a useful comparison, but it does not establish an Ada speedup. Our
launcher already sets `STRATA_ROUTE_PRIOR=0`. The upstream decision to disable
this routing change on 12 GB cards therefore does not remove a speedup from our
current configuration. Our routing policy already avoids that change, which
upstream associated with repeated text in long conversations.

**Local RTX 4070, DeepSeek:** CPU and SSD expert access can remain the limiting
factor. Do not budget a speed increase from this release alone. A small increase
is possible, but the A6000 result does not measure this offloaded configuration.

**Remote four A40 cards, DeepSeek with DSpark:** a small single-digit percentage
increase is a reasonable planning estimate. The earlier measured average was
55.5 tokens/s. Applying 4.3% arithmetically gives 57.9 tokens/s. That is an
illustration, not a prediction or a new benchmark. Treat roughly the current
55 to 60 tokens/s as the planning range until a controlled comparison is run.

## Changes that do not directly apply

Both DeepSeek launchers set `arch=deepseek41`. The new server fix for a missing
family declaration does not address our current settings. Both also pass an
explicit `--ssd-cache` value. The corrected automatic RAM-store calculation does
not increase that explicit budget.

The Qwen wrapper uses the automatic GPU expert cache. The new reserve calculation
accounts for approximately 245 MiB of allocations made after the cache is sized.
It can leave more reliable headroom by retaining fewer experts. It is not a
guaranteed throughput increase.

The shared-prefix fix applies when an idle `--batch` slot is parked. Our default
single-request profiles do not enable that batching path. The large published
Arc and Radeon improvements target different hardware. NVIDIA builds still use
eight-row windows.

`DS4_FAT_ROWS` stays off. Both launchers clear inherited `DS4_` tuning variables,
so setting it in the shell is not an effective way to enable it here. A separate
reviewed launcher option and an A/B test would be required. No such option was
added in this update.

The new DeepSeek engine also fixes a deadlock during chunked prefill with a layer
split. That is relevant to our multi-GPU deployment, even if decode throughput
does not change.
