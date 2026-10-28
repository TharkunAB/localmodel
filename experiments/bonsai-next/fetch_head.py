#!/usr/bin/env python3
"""Fetch the pinned trained MTP head and keep the official PTQ1 target unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import BinaryIO, Callable, cast
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DONOR_REPO = "ProCreations/Ternary-Bonsai-2-27B-MTP"
DONOR_REVISION = "efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51"
DONOR_FILENAME = "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"
DONOR_SIZE = 7_657_489_728
DONOR_FULL_SHA256 = "3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4"
HEADER_SIZE = 11_121_984
HEADER_SHA256 = "4b51bd557e88d576da040669ffc3323f7861e116ea4680c2e2e9aca82e76cc88"
HEAD_BYTES = 451_319_808
TRUNK_SIZE = 5_946_648_928
TRUNK_SHA256 = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
OUTPUT_SIZE = 6_397_969_888
OUTPUT_SHA256 = "ed55a7aa3202505b00d3314b9a63a15bf178dc347611dae7e8570d12ba647594"
TOOL_HASHES = {
    "gguf_raw.py": "da8f79c9ced91e562a5dc2cf05476e45b72a6933c721f34bceb356a692c7cf6e",
    "extract_head.py": "d37515241b1cfe1c91cc5f1e1786b456379bcdf149e834c77fc216e270ffac10",
    "merge.py": "986f5d3e4858f9273e9c6f4beac1a46e5a602769689211affbcbc842d13fa84b",
}
URL = f"https://huggingface.co/{DONOR_REPO}/resolve/{DONOR_REVISION}/{DONOR_FILENAME}"

for tool_name, tool_hash in TOOL_HASHES.items():
    with (ROOT / "libexec/graft" / tool_name).open("rb") as tool_handle:
        if hashlib.file_digest(tool_handle, "sha256").hexdigest() != tool_hash:
            raise ValueError(f"Reviewed local graft tool changed: {tool_name}")

from libexec.graft import extract_head, gguf_raw as gr, merge  # noqa: E402

pack_value = cast(Callable[[int, object], bytes], gr._pack_value)  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_range(start: int, length: int, output: BinaryIO) -> str:
    """Accept one exact range. Retry the complete range after an I/O failure."""
    end = start + length - 1
    if start < 0 or length <= 0 or end >= DONOR_SIZE:
        raise ValueError("Range is outside the pinned donor file")
    for attempt in range(4):
        checksum = hashlib.sha256()
        request = urllib.request.Request(
            f"{URL}?localmodel-range={start}-{end}",
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if not response.url.startswith("https://"):
                    raise ValueError("Donor redirect did not use HTTPS")
                if response.status != 206:
                    raise ValueError(f"Expected HTTP 206; received {response.status}")
                expected_range = f"bytes {start}-{end}/{DONOR_SIZE}"
                if response.headers.get("Content-Range") != expected_range:
                    raise ValueError("Donor returned a different Content-Range")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != length:
                    raise ValueError("Donor returned a different Content-Length")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Donor returned an encoded range")
                output.seek(start)
                remaining = length
                while remaining:
                    chunk = response.read(min(1 << 20, remaining))
                    if not chunk:
                        raise OSError("Donor range ended before the requested end")
                    output.write(chunk)
                    checksum.update(chunk)
                    remaining -= len(chunk)
                if response.read(1):
                    raise ValueError("Donor returned bytes beyond the requested end")
            return checksum.hexdigest()
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2)
    raise AssertionError("Unreachable retry state")


def tensor_digest(data: gr.TensorData, length: int | None) -> str:
    if length is None:
        raise ValueError("Unknown tensor size")
    checksum = hashlib.sha256()
    with open(data.path, "rb") as handle:
        handle.seek(data.abs_offset)
        remaining = length
        while remaining:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError("Tensor extends beyond the file")
            checksum.update(chunk)
            remaining -= len(chunk)
    return checksum.hexdigest()


def verify(trunk: gr.GGUFFile, result: gr.GGUFFile, head_hashes: dict[str, str]) -> dict[str, object]:
    """Compare every target tensor and every encoded target metadata value."""
    if gr.check_spans(result):
        raise ValueError(f"Invalid output tensor spans: {gr.check_spans(result)}")
    tensors = {tensor.name: tensor for tensor in result.tensors}
    if len(tensors) != 866 or len(result.tensors) != 866:
        raise ValueError("Output must contain 851 target tensors and 15 head tensors")
    trunk_hashes: dict[str, str] = {}
    for tensor in trunk.tensors:
        other = tensors[tensor.name]
        if (tensor.dims, tensor.ttype) != (other.dims, other.ttype):
            raise ValueError(f"Target tensor shape or type changed: {tensor.name}")
        original_hash = tensor_digest(trunk.spans[tensor.name], tensor.nbytes)
        if original_hash != tensor_digest(result.spans[tensor.name], other.nbytes):
            raise ValueError(f"Target tensor bytes changed: {tensor.name}")
        trunk_hashes[tensor.name] = original_hash
    extras = set(tensors) - set(trunk_hashes)
    if extras != set(head_hashes):
        raise ValueError("Unexpected added tensors")
    for name, expected in head_hashes.items():
        if tensor_digest(result.spans[name], tensors[name].nbytes) != expected:
            raise ValueError(f"Head tensor bytes changed: {name}")
    metadata = {kv.key: kv for kv in result.kvs}
    tokenizer_hashes = {}
    for kv in trunk.kvs:
        other = metadata[kv.key]
        if kv.key == "qwen35.block_count":
            if other.vtype != kv.vtype or other.value != 65:
                raise ValueError("Incorrect block count after graft")
            continue
        encoded = pack_value(kv.vtype, kv.value)
        if kv.vtype != other.vtype or encoded != pack_value(other.vtype, other.value):
            raise ValueError(f"Target metadata changed: {kv.key}")
        if kv.key.startswith("tokenizer."):
            tokenizer_hashes[kv.key] = hashlib.sha256(encoded).hexdigest()
    allowed = {"qwen35.nextn_predict_layers", "graft.donor.name", "graft.head_blocks",
               "graft.head_tensor_count", "graft.tool"}
    if set(metadata) - {kv.key for kv in trunk.kvs} != allowed:
        raise ValueError("Unexpected added metadata")
    kvs, entries, _ = cast(tuple[list[gr.KV], list[tuple[gr.TensorInfo, gr.TensorData]], dict[str, object]],
                           merge.plan_strip(result))
    checksum = hashlib.sha256(gr.serialize_header(kvs, gr.layout_tensors(entries, trunk.alignment), trunk.alignment))
    for _, data in entries:
        for chunk in data.iter_chunks(1 << 20):
            checksum.update(chunk)
    if checksum.hexdigest() != TRUNK_SHA256:
        raise ValueError("Stripping the head does not reproduce the official target file")
    return {"target_tensors_verified": 851, "head_tensors_verified": 15,
            "target_tensor_sha256": trunk_hashes, "tokenizer_metadata_sha256": tokenizer_hashes,
            "stripped_sha256": checksum.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trunk", type=Path, required=True, help="Official unmodified PTQ1_0 file")
    parser.add_argument("--output", type=Path, required=True, help="New merged GGUF path")
    parser.add_argument("--run", action="store_true", help="Download and write files; otherwise show the plan")
    args = parser.parse_args()
    trunk_path = args.trunk.resolve()
    output = args.output.resolve()
    plan = {"trunk": str(trunk_path), "trunk_sha256": TRUNK_SHA256, "output": str(output),
            "donor_repo": DONOR_REPO, "donor_revision": DONOR_REVISION,
            "donor_filename": DONOR_FILENAME, "download_bytes": HEADER_SIZE + HEAD_BYTES,
            "expected_output_bytes": OUTPUT_SIZE, "expected_output_sha256": OUTPUT_SHA256, "run": args.run}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        return 0
    if not trunk_path.is_file() or trunk_path.stat().st_size != TRUNK_SIZE or digest(trunk_path) != TRUNK_SHA256:
        raise ValueError("The trunk does not match the official pinned PTQ1_0 file")
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing model: {output}")
    if shutil.disk_usage(output.parent if output.parent.exists() else output.parent.parent).free < 8 << 30:
        raise OSError("At least 8 GiB of free disk space is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".procreations-head-", dir=output.parent) as temporary:
        staging = Path(temporary)
        sparse = staging / "procreations-pq2-mtp.sparse.gguf"
        with sparse.open("w+b") as handle:
            handle.truncate(DONOR_SIZE)
            if fetch_range(0, HEADER_SIZE, handle) != HEADER_SHA256:
                raise ValueError("The pinned donor header hash differs")
            handle.flush()
            donor = gr.read_header(str(sparse))
            if len(donor.tensors) != 866 or len({t.name for t in donor.tensors}) != 866 or gr.check_spans(donor):
                raise ValueError("Invalid pinned donor tensor table")
            if donor.arch != "qwen35" or donor.get("qwen35.block_count") != 65 or donor.get("qwen35.nextn_predict_layers") != 1:
                raise ValueError("Unexpected donor architecture")
            entries, _ = cast(tuple[list[tuple[gr.TensorInfo, gr.TensorData]], list[str]],
                              extract_head.head_entries(donor, embed_tokens=False))
            if len(entries) != 15 or sum(data.length for _, data in entries) != HEAD_BYTES:
                raise ValueError("Unexpected donor head")
            head_hashes: dict[str, str] = {}
            for info, data in entries:
                nbytes = info.nbytes
                if nbytes is None:
                    raise ValueError("Unknown head tensor size")
                if not re.fullmatch(r"blk\.64\.[a-z_]+(?:\.[a-z_]+)?\.weight", info.name):
                    raise ValueError("Unexpected head tensor name")
                print(f"Fetch {info.name}: {nbytes} bytes", flush=True)
                head_hashes[info.name] = fetch_range(data.abs_offset, nbytes, handle)
            handle.flush()
        head = gr.GGUFFile(str(sparse), donor.version, extract_head.head_kvs(donor, [64]),
                           [info for info, _ in entries], donor.alignment, donor.data_start,
                           donor.file_size, {info.name: data for info, data in entries})
        trunk = gr.read_header(str(trunk_path))
        if len(trunk.tensors) != 851 or trunk.get("qwen35.block_count") != 64 or gr.check_spans(trunk):
            raise ValueError("Unexpected official target layout")
        kvs, merged_entries, _ = cast(tuple[list[gr.KV], list[tuple[gr.TensorInfo, gr.TensorData]], dict[str, object]],
                                    merge.plan_merge(trunk, head))
        partial = staging / "merged.gguf"
        gr.write_gguf(str(partial), kvs, merged_entries, trunk.alignment)
        if partial.stat().st_size != OUTPUT_SIZE:
            raise ValueError("The merged file has an unexpected size")
        result = gr.read_header(str(partial))
        checks = verify(trunk, result, head_hashes)
        report = {**plan, **checks, "output_bytes": partial.stat().st_size,
                  "output_sha256": digest(partial), "head_tensor_sha256": head_hashes,
                  "donor_header_sha256": HEADER_SHA256, "donor_full_sha256_published": DONOR_FULL_SHA256,
                  "donor_full_sha256_verified": False,
                  "provenance_limit": "Only the pinned header and 15 head ranges were downloaded. The full donor digest was not verified.",
                  "graft_tools_sha256": TOOL_HASHES, "elapsed_seconds": round(time.monotonic() - started, 3)}
        if report["output_sha256"] != OUTPUT_SHA256:
            raise ValueError("The merged file does not match the recorded deterministic output hash")
        receipt = output.with_name(output.name + ".audit.json")
        os.link(partial, output)
        partial.unlink()
        receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["output", "output_bytes", "output_sha256", "target_tensors_verified", "head_tensors_verified", "elapsed_seconds"]}, indent=2), flush=True)
    print(f"Audit: {receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
