"""Compare the unquantized bridge tensors with the pinned Tencent checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from libexec.graft.gguf_raw import read_header

OUT = ROOT / "results/bonsai-dspine-20261108"
REVISION = "ea909c0f5fab8865f9e14600334f55f7ce47308f"
URL = f"https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/resolve/{REVISION}/model.safetensors"
DRAFT = ROOT / "models/bonsai2-27b-dspine/Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"


def remote_range(start: int, size: int) -> bytes:
    """Require an exact bounded byte range from the pinned public checkpoint."""
    end = start + size - 1
    request = urllib.request.Request(f"{URL}?range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise RuntimeError("The server did not honor the exact byte range")
        data = response.read(size + 1)
    if len(data) != size:
        raise RuntimeError("The response has the wrong length")
    return data


def main() -> None:
    """Check bridge weights and bias; exclude the quantized codebook."""
    with DRAFT.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != "0c6c984ac18611288d7b8f497bc347c4921e8231b1b6abf28650481eaf35ff55":
        raise RuntimeError("The GGUF differs from its published pin")
    header_length = struct.unpack("<Q", remote_range(0, 8))[0]
    if not 2 <= header_length <= 1024 * 1024:
        raise RuntimeError("The official tensor header exceeds the size bound")
    header = json.loads(remote_range(8, header_length))
    data_start = header_length + 8
    draft = read_header(str(DRAFT))
    mappings = [
        ("dspine_gate.weight", "gate.weight", 0),
        ("dspine_gate.bias", "gate.bias", 0),
        ("dspine_pred_gate.weight", "predecessor_gate.weight", 0),
        ("dspine_prev.weight", "prev.weight", 0),
    ]
    for layer in range(5):
        mappings.append((f"blk.{layer}.dspine_read.weight", "early_read_weights" if layer < 4 else "read_weight", layer if layer < 4 else 0))
        mappings.append((f"blk.{layer}.dspine_write.weight", "feature_write_weights", layer))
    rows: list[dict[str, Any]] = []
    for target_name, source_suffix, index in mappings:
        source_name = "selected_prefix_bridge." + source_suffix
        source = header[source_name]
        target = draft.tensor(target_name)
        if target is None or source["dtype"] != "BF16" or target.ttype not in (0, 30):
            raise RuntimeError(f"Unexpected tensor type for {target_name}")
        size = target.n_elements * 2
        start = source["data_offsets"][0] + index * size
        if start + size > source["data_offsets"][1]:
            raise RuntimeError(f"Source tensor slice is out of bounds: {target_name}")
        original = remote_range(data_start + start, size)
        expected = original if target.ttype == 30 else b"".join(struct.pack("<I", value << 16) for (value,) in struct.iter_unpack("<H", original))
        with DRAFT.open("rb") as stream:
            stream.seek(draft.data_start + target.offset)
            actual = stream.read(len(expected))
        if actual != expected:
            raise RuntimeError(f"Tensor differs from the official source: {target_name}")
        rows.append({"target": target_name, "source": source_name, "slice": index, "source_bytes": size,
                     "target_bytes": len(actual), "source_sha256": hashlib.sha256(original).hexdigest(),
                     "target_sha256": hashlib.sha256(actual).hexdigest(), "equal": True,
                     "conversion": "none" if target.ttype == 30 else "exact BF16 to F32 expansion"})
        print(f"Verified {target_name}", flush=True)
    report = {"official_revision": REVISION, "draft_revision": "62e92414e114702e792ea7a4c37eba57b25bac8b",
              "draft_sha256": digest,
              "verified_tensors": rows, "source_bytes_checked": sum(row["source_bytes"] for row in rows),
              "limitation": "The quantized codebook and other quantized tensors were not compared with the BF16 checkpoint."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bridge-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(rows)} bridge tensors")


if __name__ == "__main__":
    main()
