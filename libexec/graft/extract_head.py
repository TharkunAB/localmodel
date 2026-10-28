#!/usr/bin/env python3
"""Extract the multi-token-prediction (nextn) block(s) from a Qwen 3.8 GGUF.

The MTP head in Qwen 3.8 is a full decoder block stored after the trunk
(blk.64 in the 27B: attention, FFN, norms) plus the nextn-specific tensors
(nextn.eh_proj, nextn.enorm, nextn.hnorm, nextn.shared_head_norm). The head
predicts token t+2 from the trunk's hidden state and the embedding of token t+1,
so it also needs an embedding table. Qwen ties it to token_embd and the GGUF
export drops the copy; by default this tool re-materializes it as
blk.N.nextn.embed_tokens.weight from the donor's token_embd.weight, because the
graft target (Ternary Bonsai 2) stores its own token_embd rotated into a
Hadamard basis that the MTP graph in llama.cpp does not undo.

Tensors are copied as opaque byte spans (see gguf_raw.py), so quant types are
preserved exactly (Q4_K, Q6_K, Q8_0, F32 in the unsloth files).

Usage:
    python3 extract_head.py DONOR.gguf HEAD.gguf [--no-embed-tokens] [--shared-head]

  --no-embed-tokens  do not add blk.N.nextn.embed_tokens.weight
  --shared-head      also add blk.N.nextn.shared_head_head.weight from the donor's
                     output.weight (an fp LM head for the draft, about 1 GB in Q6_K;
                     without it the draft uses the target model's own LM head)
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_raw as gr  # noqa: E402

BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

TOOL_VERSION = "bonsai2-mtp extract_head 1"


def head_block_indices(g: gr.GGUFFile) -> tuple[int, int, list[int]]:
    arch = g.arch
    n_all = g.get(f"{arch}.block_count")
    n_nextn = g.get(f"{arch}.nextn_predict_layers", 0)
    if not n_nextn:
        raise SystemExit(f"{g.path}: no {arch}.nextn_predict_layers key, this file carries no MTP head")
    return n_all, n_nextn, list(range(n_all - n_nextn, n_all))


def head_entries(g: gr.GGUFFile, embed_tokens: bool = True, shared_head: bool = False):
    """(TensorInfo, TensorData) pairs for the head, plus a list of notes."""
    _, _, idx = head_block_indices(g)
    wanted = set(idx)
    entries = []
    notes = []
    names = {t.name for t in g.tensors}
    for t in g.tensors:
        m = BLK_RE.match(t.name)
        if m and int(m.group(1)) in wanted:
            entries.append((gr.TensorInfo(t.name, list(t.dims), t.ttype, 0), g.spans[t.name]))
    if not entries:
        raise SystemExit(f"{g.path}: no blk.{idx}.* tensors found")
    for i in idx:
        emb_name = f"blk.{i}.nextn.embed_tokens.weight"
        if embed_tokens and emb_name not in names:
            src = g.tensor("token_embd.weight")
            if src is None:
                raise SystemExit(f"{g.path}: no token_embd.weight to build {emb_name} from")
            entries.append((gr.TensorInfo(emb_name, list(src.dims), src.ttype, 0), g.spans[src.name]))
            notes.append(f"{emb_name} <- token_embd.weight ({gr.type_name(src.ttype)}, {g.spans[src.name].length} bytes)")
        head_name = f"blk.{i}.nextn.shared_head_head.weight"
        if shared_head and head_name not in names:
            src = g.tensor("output.weight") or g.tensor("token_embd.weight")
            entries.append((gr.TensorInfo(head_name, list(src.dims), src.ttype, 0), g.spans[src.name]))
            notes.append(f"{head_name} <- {src.name} ({gr.type_name(src.ttype)}, {g.spans[src.name].length} bytes)")
    return entries, notes


def head_kvs(g: gr.GGUFFile, idx: list[int]) -> list[gr.KV]:
    arch = g.arch
    n_all, n_nextn, _ = head_block_indices(g)
    kvs = [
        gr.KV("general.architecture", gr.STRING, arch),
        gr.KV("general.type", gr.STRING, "mtp-head"),
        gr.KV("general.name", gr.STRING, f"{g.get('general.name', 'unknown')} nextn head"),
        gr.KV(f"{arch}.block_count", gr.UINT32, n_all),
        gr.KV(f"{arch}.nextn_predict_layers", gr.UINT32, n_nextn),
        gr.KV("graft.donor.name", gr.STRING, os.path.basename(g.path)),
        gr.KV("graft.donor.size", gr.UINT64, g.file_size),
        gr.KV("graft.head_blocks", gr.ARRAY, (gr.UINT32, list(idx))),
        gr.KV("graft.tool", gr.STRING, TOOL_VERSION),
    ]
    for key in ("general.quantized_by", "general.file_type", f"{arch}.embedding_length", f"{arch}.attention.head_count",
                f"{arch}.attention.head_count_kv", f"{arch}.attention.key_length", f"{arch}.attention.value_length",
                f"{arch}.feed_forward_length"):
        kv = g.kv(key)
        if kv is not None:
            kvs.append(kv.clone())
    return kvs


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    if len(args) != 2:
        print(__doc__)
        return 2
    donor_path, out_path = args
    g = gr.read_header(donor_path)
    n_all, n_nextn, idx = head_block_indices(g)
    print(f"donor: {donor_path}\n  arch {g.arch}, {n_all} blocks, {n_nextn} nextn layer(s): blocks {idx}")
    entries, notes = head_entries(g, embed_tokens="--no-embed-tokens" not in flags, shared_head="--shared-head" in flags)
    total = 0
    for info, data in entries:
        print(f"  {info.name} {info.dims} {gr.type_name(info.ttype)} {data.length} bytes")
        total += data.length
    for n in notes:
        print(f"  note: {n}")
    print(f"  {len(entries)} tensors, {total / 2**20:.1f} MiB")
    gr.write_gguf(out_path, head_kvs(g, idx), entries, g.alignment)
    out = gr.read_header(out_path)
    probs = gr.check_spans(out)
    print(f"wrote {out_path}: {out.file_size} bytes, {len(out.tensors)} tensors, span check {'ok' if not probs else probs}")
    print(f"sha256 {gr.sha256_file(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
