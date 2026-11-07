# Strata 0.2.12 runtime review

The local DeepSeek and Qwen profiles, and the remote DeepSeek profiles, pin
[Strata v0.2.12](https://github.com/Niko1221/Strata/releases/tag/v0.2.12).
The source commit is `146df3c468739147f520ac3816de806f71232571`.
Control and both signed CUDA 13 engine packages are version **0.1.58**.
The remote script version is `2026-11-07-strata-0212-1`.
The pin update preserves target weights, quantization, context and cache defaults,
Python dependency pins, and GPU compatibility rules.

Local measurements, API/native smoke checks, and cleanup are complete.
Qwen MTP4 and tuned DFlash2 passed the long workload through approximately
185K actual input tokens. DFlash2 measured faster native decode, but an OMP
history edit caused extra prefill and increased its total model HTTP time.
The normal Qwen default remains MTP4. Explicit DFlash2 uses confidence 0.8.
See the [measurement report](strata-0.2.12-measurements.md) for exact settings,
timing denominators, both Coder attempts, and resource limits. No remote GPU
inference was tested. The earlier
[Qwen and DeepSeek comparison](local-192k-comparison.md) remains historical
evidence for the previous runtime.

## Artifact verification

The source archive has SHA256
`4c8ed4924f040364e015bdfacacb491e226b91e8cac508715320c5042152f323`.
Its receipt records 5,816 source files and has SHA256
`e6f7b38d79fed49875580986b7c988875f423e0fab406ac1c2ac354162131ddb`.
The receipt is a local integrity record from the official commit archive.
It is not an upstream source signature. GitHub does not report a verified
signature for this release tag.

The release feed has sequence 122. Its signature verifies with the existing
trusted key `5d1bc23abbeffd0d`. Verification used the installed 0.2.11 verifier
before accepting the new artifacts. Both CUDA archives match the signed feed
and GitHub asset digests. Their signed component manifests and all packaged
files passed verification: five DeepSeek files and ten Qwen files.
The native binaries were not independently rebuilt from source.

Local installation retains exact feed-byte pins. Remote installation verifies
the signed feed and the pinned component version, platform, archive URL, size,
hash, and manifest hash. Remote feed bytes remain unpinned because upstream
can update the catalog within a release. Signature and component checks remain
mandatory.

The DeepSeek packer and all eight Qwen pack/MTP preparation dependencies are
byte-identical to 0.2.11. Prepared indices and MTP tensors can remain unchanged.
Receipt migration must check the unchanged preparation inputs, existing file
identities, and prepared-file hashes. It must retain the old receipts and record
the runtime change. Keep the old sources, native engines, and indices for rollback.

## Tuned Qwen DFlash2 artifact

The signed feed includes
`qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack` for CUDA, Metal, and HIP.
The file is 339,694,592 bytes and has SHA256
`4df2e5ab2b47c064afbe662e16e4f06569877125c2d0e2123a4ac755266c664c`.
Its declared base is
[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
at revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`.
Strata provides an Apache-2.0 license and modification notice.

The drafter has tuned tensors. It does not replace the target GGUF weights.
Its signed-feed pin and full file hash passed verification. Native startup
confirmed the tuned selector with no MTP layer. The initial confidence-0.35
screen passed code, recall and required-tool checks, but all three strict
resume checks rejected JSON inside Markdown fences. That failure is retained.

Confidence 0.8 passed both the short screen and all four long checkpoints.
The long run used native window 6, the existing manual memory layout, and
196,608-token capacity. It reached 185,415 actual input tokens and measured
58.41 native decode tokens/s, against MTP4's 52.17. Summed model HTTP times
were 227.75 and 203.40 seconds. An OMP stale-read replacement caused one large
prefix reread in the DFlash2 run. No parked-conversation eviction was logged;
the earlier recurrent-checkpoint replacement sequence is not observable.
These sampled trajectories do not establish a matched whole-workflow speedup.

Factory Auto selects the ordinary DFlash2 pack for this IQ1_M quant.
That separate benchmark asset was prepared from the pinned Inco source with
the official producer. Both source files, the intermediate pack and the final
selector-Q4 pack matched their pinned hashes. Factory screens at 128K and
192K capacity passed all checks through about 66K actual input. Their suffix
lookup behavior differs from the manual configuration. Aggregate speculation
acceptance is not a neural drafter accuracy score. Factory 192K also reached
zero reported free MiB for a prefill super-chunk, so the layer-major prefill
path was disabled. Factory settings were not tested at 180K filled input.

## Source and network review

The bounded review covers changed API parsing and request accounting, Control
downloads, native I/O changes, and the existing pack producers. It found no new
outgoing request or shell-execution path in the changed serving/native code.
This is not a proof that all upstream code is safe.

New Control code can download the tuned drafter from the signed feed during
preparation. It checks the expected size and SHA256. The direct API server
launch does not run this downloader. The local and remote launchers keep the
existing outgoing-connection restriction for the server and its children.
The remote guard and installer are byte-identical to their previous versions.
The launchers also remove unreviewed Strata environment settings.
Explicit installation and fetch commands still require network access.

The new `undeclared_tool_calls` option defaults to `text`. This update does not
enable undeclared calls. Parser recovery supports additional misspelled call
openers and unambiguous declared tool aliases. The API returns calls to the
client; it does not execute tools. Crafted quoted text can still resemble a
call. Client tool permissions remain necessary. See the earlier
[parser boundary](strata-0.2.10-coder.md).

The API fixes token and timing accounting across batch admissions and prompt
reads that yield to other requests. Dashboard rates can change because of
corrected accounting. Controlled comparisons must retain raw token counts
and use the same request pattern.

## Adaptive fetch

In the Qwen engine, `STRATA_ADAPT_FETCH=1` allows PCIe expert transfers to
populate the adaptive VRAM tier. Mode `2` also disables that tier's separate copies from RAM.
Both modes are optional and default to off. The transfer uses the existing
memory path. It does not need network access or `io_uring`.

Upstream disables adaptive fetch in resident RAM mode. It also requires an
active adaptive tier, a GPU plan, and a nonzero PCIe share. Some execution modes
are incompatible. A benchmark must confirm activation in the startup log.
Use a matched control with the same RAM and PCIe settings. A flag alone does
not prove that adaptive fetch ran. Local mode-1 and mode-2 startup markers
confirmed activation with mapped experts and MTP4. Modes 0, 1 and 2 measured
44.26, 44.46 and 42.68 native decode tokens/s; new-prefill rates were 667.09,
690.38 and 593.18. All three passed code, recall, required-tool and strict-format
checks through about 66K actual input. These single trajectories do not
establish a repeatable adaptive-fetch gain. The default remains off.

Serving has no separate adaptive admission/hit counter. GPU lookup misses
are not SSD misses. Mapped expert payload counters can count the same bytes
repeatedly. Zero private expert RAM still permits OS page-cache use. All model
scopes retained the 64 GiB cap and zero swap. No model OOM or swap was recorded;
charged file pages and cap-reclaim events remain part of the memory totals.
