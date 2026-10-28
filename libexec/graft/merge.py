#!/usr/bin/env python3
"""Graft an extracted MTP head onto a trunk GGUF, or strip it off again.

merge: every trunk tensor is copied byte for byte and keeps its offset, the head
tensors are appended after them under the next free block index, and the header
gets two edits: <arch>.block_count grows by the number of nextn layers and
<arch>.nextn_predict_layers is added (llama.cpp treats the last
nextn_predict_layers blocks as the MTP head and skips them in the main pass).
A few graft.* keys record provenance; llama.cpp ignores keys it does not know.

strip: the inverse. Removes the head tensors and the added keys and restores
block_count, so strip(merge(trunk, head)) reproduces the trunk byte for byte.

Usage:
    python3 merge.py TRUNK.gguf HEAD.gguf OUT.gguf [--force-arch]
    python3 merge.py --strip MERGED.gguf OUT.gguf

  --force-arch  proceed when trunk and head carry different architecture labels
                (the head's keys are rewritten to the trunk's label; llama.cpp
                will only load this if the two architectures share the MTP graph)
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_raw as gr  # noqa: E402

BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

TOOL_VERSION = "bonsai2-mtp merge 1"


def plan_merge(trunk: gr.GGUFFile, head: gr.GGUFFile, force_arch: bool = False):
    """Returns (kvs, entries, summary dict) without writing anything."""
    arch = trunk.arch
    if head.arch != arch:
        msg = f"architecture mismatch: trunk {arch!r}, head {head.arch!r}"
        if not force_arch:
            raise SystemExit(msg + " (pass --force-arch to graft anyway)")
        print(f"warning: {msg}, rewriting the head's keys to {arch!r}", file=sys.stderr)
    head_arch = head.arch
    n_trunk = trunk.get(f"{arch}.block_count")
    if n_trunk is None:
        raise SystemExit(f"{trunk.path}: missing {arch}.block_count")
    if trunk.kv(f"{arch}.nextn_predict_layers") is not None:
        raise SystemExit(f"{trunk.path}: already has {arch}.nextn_predict_layers, refusing to graft twice")
    n_nextn = head.get(f"{head_arch}.nextn_predict_layers")
    if not n_nextn:
        raise SystemExit(f"{head.path}: missing {head_arch}.nextn_predict_layers")
    old_idx = head.get("graft.head_blocks")
    if old_idx is None:
        n_all = head.get(f"{head_arch}.block_count")
        old_idx = list(range(n_all - n_nextn, n_all))
    if len(old_idx) != n_nextn:
        raise SystemExit(f"{head.path}: {len(old_idx)} head blocks but nextn_predict_layers is {n_nextn}")
    new_idx = [n_trunk + k for k in range(n_nextn)]
    rename = {o: n for o, n in zip(old_idx, new_idx)}

    trunk_names = {t.name for t in trunk.tensors}
    entries = gr.entries_of(trunk)
    for t in head.tensors:
        m = BLK_RE.match(t.name)
        if not m or int(m.group(1)) not in rename:
            raise SystemExit(f"{head.path}: unexpected tensor {t.name}, the head must only hold blk.N.* tensors")
        name = f"blk.{rename[int(m.group(1))]}.{m.group(2)}"
        if name in trunk_names:
            raise SystemExit(f"name collision: {name} exists in the trunk")
        entries.append((gr.TensorInfo(name, list(t.dims), t.ttype, 0), head.spans[t.name]))

    kvs = []
    for kv in trunk.kvs:
        kv = kv.clone()
        if kv.key == f"{arch}.block_count":
            kv.value = n_trunk + n_nextn
            kvs.append(kv)
            kvs.append(gr.KV(f"{arch}.nextn_predict_layers", gr.UINT32, n_nextn))
        else:
            kvs.append(kv)
    kvs.append(gr.KV("graft.donor.name", gr.STRING, head.get("graft.donor.name", os.path.basename(head.path))))
    kvs.append(gr.KV("graft.head_blocks", gr.ARRAY, (gr.UINT32, new_idx)))
    kvs.append(gr.KV("graft.head_tensor_count", gr.UINT32, len(head.tensors)))
    kvs.append(gr.KV("graft.tool", gr.STRING, TOOL_VERSION))
    summary = {"arch": arch, "n_trunk": n_trunk, "n_nextn": n_nextn, "old_idx": old_idx, "new_idx": new_idx,
               "n_tensors": len(entries), "head_bytes": sum(head.spans[t.name].length for t in head.tensors)}
    return kvs, entries, summary


def plan_strip(merged: gr.GGUFFile):
    arch = merged.arch
    n_all = merged.get(f"{arch}.block_count")
    n_nextn = merged.get(f"{arch}.nextn_predict_layers")
    if not n_nextn:
        raise SystemExit(f"{merged.path}: no {arch}.nextn_predict_layers, nothing to strip")
    drop = set(range(n_all - n_nextn, n_all))
    entries = []
    for t in merged.tensors:
        m = BLK_RE.match(t.name)
        if m and int(m.group(1)) in drop:
            continue
        entries.append((t, merged.spans[t.name]))
    kvs = []
    for kv in merged.kvs:
        if kv.key == f"{arch}.nextn_predict_layers" or kv.key.startswith("graft."):
            continue
        kv = kv.clone()
        if kv.key == f"{arch}.block_count":
            kv.value = n_all - n_nextn
        kvs.append(kv)
    return kvs, entries, {"arch": arch, "n_all": n_all, "n_nextn": n_nextn, "dropped": len(merged.tensors) - len(entries)}


def main(argv: list[str]) -> int:
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    if "--strip" in flags and len(args) == 2:
        merged = gr.read_header(args[0])
        kvs, entries, s = plan_strip(merged)
        print(f"strip: {args[0]} -> {args[1]}: dropping {s['dropped']} tensors of the last {s['n_nextn']} block(s), block_count {s['n_all']} -> {s['n_all'] - s['n_nextn']}")
        gr.write_gguf(args[1], kvs, entries, merged.alignment, progress=False)
        print(f"wrote {args[1]}: {os.path.getsize(args[1])} bytes")
        print(f"sha256 {gr.sha256_file(args[1])}")
        return 0
    if len(args) != 3:
        print(__doc__)
        return 2
    trunk = gr.read_header(args[0])
    head = gr.read_header(args[1])
    kvs, entries, s = plan_merge(trunk, head, force_arch="--force-arch" in flags)
    print(f"trunk: {args[0]}: {len(trunk.tensors)} tensors, {s['n_trunk']} blocks, arch {s['arch']}")
    print(f"head:  {args[1]}: {len(head.tensors)} tensors, {s['head_bytes'] / 2**20:.1f} MiB, blocks {s['old_idx']} -> {s['new_idx']}")
    print(f"out:   {args[2]}: {s['n_tensors']} tensors, block_count {s['n_trunk'] + s['n_nextn']}, nextn_predict_layers {s['n_nextn']}")
    gr.write_gguf(args[2], kvs, entries, trunk.alignment, progress=False)
    out = gr.read_header(args[2])
    probs = gr.check_spans(out)
    print(f"wrote {args[2]}: {out.file_size} bytes, {len(out.tensors)} tensors, span check {'ok' if not probs else probs}")
    print(f"sha256 {gr.sha256_file(args[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
