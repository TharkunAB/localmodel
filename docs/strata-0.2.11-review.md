# Strata 0.2.11 runtime review

The local DeepSeek and Qwen profiles, and the remote DeepSeek profiles, pin
[Strata v0.2.11](https://github.com/Niko1221/Strata/releases/tag/v0.2.11).
The source commit is `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`.
Control and both signed CUDA 13 engine packages are version **0.1.57**.
The runtime update does not change model weights, quantization, context defaults,
cache defaults, Python dependency pins, or GPU compatibility rules.

Qwen explicitly selects `drafting: mtp`. This keeps three draft tokens
(`--spec 4`) when upstream Auto supports DFlash2. It does not fetch a new drafter.
Coder keeps `--ssd-cache 0`, a 64 GiB cap, and 256K context capacity.
It does not select `--mmap-experts off` or DSpark.
The new server's `engine_args` returned the same native arguments for both
configurations in a hardware-mocked check. This check did not run an engine.

## Artifact verification

The source archive has SHA-256
`85aa74cc4203ecdfcf17e2e325b0806ef388d6b216da202f7bd3a48a8c234cce`.
Its source receipt records 5,780 files. The manifest pins the archive and receipt.
The two CUDA packages match the signed release feed and GitHub asset digests.
Their component manifests, signatures, and every packaged file passed verification.
The trusted signing key is `5d1bc23abbeffd0d`. Verification used the trusted keys
and verifier from the existing 0.2.10 source before the new runtime was accepted.

Local installation retains exact feed-byte pins. Remote installation verifies
the signed feed and the exact pinned component version, platform, archive URL,
size, hash, and manifest hash. The remote feed itself remains unpinned by bytes
because upstream can update it within a release. The signature and exact
component pins remain mandatory.

The DeepSeek packer is byte-identical to 0.2.10. Its imports use only the Python
standard library. Existing Coder indices can be regenerated with the new source
without downloading or modifying GGUF files. Preserve the previous pack receipt
before regeneration.

The Qwen packing and MTP tools, path helper, TLS helper, draft vocabulary, and
expert profile are byte-identical to 0.2.9. The GGUF reader has one comment change;
its Python syntax tree is unchanged. Its external preparation dependencies and
GGUF library pins also remain unchanged. A runtime-only artifact receipt update
must verify these facts, existing file identities, and prepared-file hashes.
It must preserve the previous receipt and record the runtime change.

## Source review and limits

The bounded review covered changed serving, stream ownership, drafting,
component loading, hardware detection, image handling, and telemetry code.
It found no new outgoing request or shell-execution path in these serving changes.
Both launchers retain their inherited outgoing-connection block. Installation
and explicit downloads still require network access.

The server gives concurrent requests separate control state. It closes replica
streams before another request takes ownership. These changes address output
cross-delivery and stalled requests. A stream write now times out after 60 seconds
without progress. Console output uses a bounded queue so a blocked terminal does
not hold the engine. These changes can improve reliability; they do not establish
a throughput gain for a single request.

The image path restricts decoder formats. Image generation writes prompts to an
owner-only temporary file instead of command arguments. Our serving environment
has no Pillow dependency and vision remains disabled. The four minimal serving
dependencies remain Jinja2, MarkupSafe, regex, and psutil at their existing pins.

This review does not prove that all upstream code is safe. Native engines are
signed official binaries; they were not independently rebuilt from source.
The known parser risk remains: crafted quoted content can resemble a tool call.
Client tool permissions remain necessary. See the earlier
[parser review](strata-0.2.10-coder.md) for that boundary.
Remote packaging checks do not establish remote inference or throughput.
