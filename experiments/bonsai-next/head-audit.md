# Trained Bonsai 27B MTP head

Audit date: 2026-10-26. This audit covers the head download and graft. Runtime inference tests are separate.

The file `fetch_head.py` downloads the ProCreations head. It copies the 15 head tensors onto the local official PTQ1 target. It uses the reviewed local tools in `libexec/graft`. Packaging copied these files without changes from `bonsai/graft/tools`. It does not execute downloaded code. It does not download a donor runtime or the full donor model.

## Source pins

| Source | Revision | File | SHA256 |
| --- | --- | --- | --- |
| [Official target](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/b072e1d3b35a0a630cece372c2127528e0994386) | `b072e1d3b35a0a630cece372c2127528e0994386` | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| [Trained donor](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP/tree/efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51) | `efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51` | `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` | Published full-file hash: `3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4` |
| [Graft recipe](https://github.com/professorpalmer/bonsai-ada-surgery/blob/fe77c29cb8b185a382df17013b22d952aac341ed/build/make_mtp_procreations.ps1) | `fe77c29cb8b185a382df17013b22d952aac341ed` | `build/make_mtp_procreations.ps1` | Recipe reviewed; not executed |

The donor card identifies the trained checkpoint as `r3-mtp`. Its matrices use Q8_0. Its normalization tensors use F32 effective multipliers. The donor manifest identifies the unchanged PQ2 target revision as `6ed5e12bf84b7a63069882c91dd9e9218647d17b`. This graft uses our existing PTQ1 target instead.

The remote donor header contains 866 tensors. The 15 `blk.64.*` tensors contain 451,319,808 bytes. The header contains 11,121,984 bytes. Its measured SHA256 is `4b51bd557e88d576da040669ffc3323f7861e116ea4680c2e2e9aca82e76cc88`.

## Ready-made file rejected

The [CaryPalmer artifact](https://huggingface.co/CaryPalmer/Ternary-Bonsai-2-27B-262k-GGUF/tree/70ed43ce8bc6e8a09a00eda5997a53f7f5a0c431) has 6,397,970,272 bytes. Its published SHA256 is `5f212d02ff57cb8eaad260fd7ff57bfaff87ae2bc9183a21dd2f127c27252505`.

Its header keeps all 851 target tensor shapes, types, and relative offsets. Its `tokenizer.chat_template` differs from our official target. It adds author-specific command and evidence instructions under `Work rules`. These instructions do not request network access or secret disclosure. They still change the target prompt and invalidate a claim that all tokenizer metadata is unchanged. We do not use this artifact.

Our current teacher-head lean GGUF and our official PTQ1 file have identical tokenizer metadata, including the chat template and the `xhigh` behavior. The encoded chat-template SHA256 is `57c17aeb437457e667d18dcef233fc61b7a558b21f1d5e7e16dc4038dcfbac48`. The new graft preserves it.

## Download and code review

The original PowerShell recipe clones an unpinned Git branch and resolves the donor through `main`. We do not execute it. The original sparse downloader checks HTTP 206 but does not check the exact `Content-Range` or reject a short successful response. We do not execute that downloader.

Our helper pins the donor revision and header hash. It checks the response status, final HTTPS URL, exact byte range, advertised length, content encoding, premature end, and excess bytes. It bounds each write to the requested range. It rejects ranges outside the pinned file. It downloads the header and the 15 tensor ranges: 462,441,792 bytes in total. Temporary sparse files use about this much disk space. Their logical size does not mean that the full donor was downloaded.

The local graft tools copy tensor spans without dequantization. Their SHA256 values are checked before import:

| Tool | SHA256 |
| --- | --- |
| `gguf_raw.py` | `da8f79c9ced91e562a5dc2cf05476e45b72a6933c721f34bceb356a692c7cf6e` |
| `extract_head.py` | `d37515241b1cfe1c91cc5f1e1786b456379bcdf149e834c77fc216e270ffac10` |
| `merge.py` | `986f5d3e4858f9273e9c6f4beac1a46e5a602769689211affbcbc842d13fa84b` |

The helper imports only these local tools and the Python standard library. The reviewed copy paths do not execute shell commands or make network requests. Only the helper makes HTTPS requests for public model bytes. It does not read prompts, credentials, or conversation logs.

The graft uses `--no-embed-tokens` behavior. It adds no donor embedding table. It preserves every original metadata value except `qwen35.block_count`, which changes from 64 to 65. It adds `qwen35.nextn_predict_layers` and four `graft.*` provenance keys. The target and the draft continue to use the current runtime's embedding handling.

## Completed verification

The fetch, graft, and checks completed in 47.711 seconds under a `heavy-management` 4 GiB memory scope with swap disabled. Before the run, Linux had about 86 GiB available. Windows had about 75 GiB free. The model disk had about 216 GiB free. The scope exited with status 0. Its recorded peak was 4,124 MiB. This scope accounting includes file cache.

The output file is `bonsai/models/bonsai2-27b-procreations/Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf`.

- Size: **6,397,969,888 bytes**.
- SHA256: **`ed55a7aa3202505b00d3314b9a63a15bf178dc347611dae7e8570d12ba647594`**.
- All **851 original target tensor payloads** have identical SHA256 values.
- Every original metadata value, including all tokenizer arrays and the chat template, remains identical except the required block count.
- All **15 head tensor payloads** match the bytes downloaded from the pinned donor ranges.
- Stripping the head in memory reproduces the complete official PTQ1 file hash.
- Seven malformed-response cases and four invalid-range cases were rejected in a temporary smoke test. No permanent tests were added.

The adjacent `.gguf.audit.json` file records every target tensor hash, every head tensor hash, tokenizer metadata hashes, source pins, and the full-donor verification limitation. The helper also pins the final output hash for subsequent fetches. It refuses to replace an existing model file.

## Limits

We did not download the complete donor file. We cannot verify its published full-file hash from sparse ranges. The donor header hash and the 15 head hashes record the bytes retrieved over HTTPS from the immutable revision. We did not independently reproduce Q8_0 quantization from the trained BF16 master weights. The helper records these limits.

The head is **100,311,040 bytes**, or about **95.67 MiB**, larger than the current teacher head. A full 256K context can therefore require more VRAM. Runtime tests must measure this cost and the acceptance change. A successful tensor identity check does not establish inference speed, numerical parity across batch sizes, or coding quality.

Preview:

```sh
.venv/bin/python experiments/bonsai-next/fetch_head.py \
  --trunk bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --output bonsai/models/bonsai2-27b-procreations/Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf
```

Add `--run` to download and graft. Use `localmodel fetch` for the normal capped workflow after catalog integration.
