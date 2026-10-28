#!/usr/bin/env python3
"""Byte-range GGUF reader and writer.

Parses a GGUF v3 header (metadata key/values and tensor infos) and treats every
tensor as an opaque span of bytes: from its offset to the next tensor's offset,
the last one to the end of the file. Nothing here needs the element size of a
quant type, so files that carry types unknown to gguf-py (PrismML PTQ1_0, type
id 143, and PQ2_0, type id 142) can be inspected, copied, merged and split
without decoding a single block.

The header is re-serialized field by field in the original order, so a no-op
rewrite of a well-formed file reproduces it byte for byte (tests check this).

Usage as a tool:
    python3 gguf_raw.py info  <file.gguf>            # header summary
    python3 gguf_raw.py copy  <in.gguf> <out.gguf>   # no-op rewrite
"""
from __future__ import annotations

import hashlib
import mmap
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Iterator, Optional

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# metadata value types (GGUF spec)
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

_SCALAR_FMT = {
    UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h", UINT32: "<I", INT32: "<i",
    FLOAT32: "<f", BOOL: "<?", UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}

VALUE_TYPE_NAMES = {
    UINT8: "UINT8", INT8: "INT8", UINT16: "UINT16", INT16: "INT16", UINT32: "UINT32",
    INT32: "INT32", FLOAT32: "FLOAT32", BOOL: "BOOL", STRING: "STRING", ARRAY: "ARRAY",
    UINT64: "UINT64", INT64: "INT64", FLOAT64: "FLOAT64",
}

# ggml tensor type id -> (block size in elements, block size in bytes).
# Only used for sanity checks and reporting; the copy path never needs it.
# Standard entries match gguf-py, the two Prism entries come from the PrismML
# llama.cpp fork (ggml/src/ggml-common.h: QK_PQ2_0 = QK_PTQ1_0 = 128,
# block_pq2_0 = 2 + 32 bytes, block_ptq1_0 = 2 + 24 + 2 bytes).
GGML_TYPE_SIZES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17),
    40: (64, 36), 41: (128, 18),
    142: (128, 34),  # PQ2_0, PrismML
    143: (128, 28),  # PTQ1_0, PrismML
}

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 142: "PQ2_0", 143: "PTQ1_0",
}


def type_name(ttype: int) -> str:
    return GGML_TYPE_NAMES.get(ttype, f"TYPE_{ttype}")


def align_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


def tensor_nbytes(dims: list[int], ttype: int) -> Optional[int]:
    """Exact byte size for a known type, None for an unknown type id."""
    sizes = GGML_TYPE_SIZES.get(ttype)
    if sizes is None:
        return None
    block, tsize = sizes
    n = 1
    for d in dims:
        n *= d
    if n % block != 0:
        raise ValueError(f"element count {n} not a multiple of block size {block} for type {type_name(ttype)}")
    return n // block * tsize


@dataclass
class KV:
    key: str
    vtype: int
    value: Any  # scalar, str, or for ARRAY a tuple (elem_type, list)

    def clone(self) -> "KV":
        v = self.value
        if self.vtype == ARRAY:
            v = (v[0], list(v[1]))
        return KV(self.key, self.vtype, v)


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    ttype: int
    offset: int  # relative to the start of the data region

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self) -> Optional[int]:
        return tensor_nbytes(self.dims, self.ttype)


@dataclass
class TensorData:
    """Where a tensor's bytes live: a file path, an absolute byte offset and a length.

    length is the span up to the next tensor (or end of file), so it can include
    up to alignment - 1 bytes of padding.
    """
    path: str
    abs_offset: int
    length: int

    def read(self) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(self.abs_offset)
            return f.read(self.length)

    def iter_chunks(self, chunk: int = 64 << 20) -> Iterator[bytes]:
        with open(self.path, "rb") as f:
            f.seek(self.abs_offset)
            left = self.length
            while left > 0:
                b = f.read(min(chunk, left))
                if not b:
                    raise IOError(f"short read in {self.path} at {self.abs_offset + self.length - left}")
                left -= len(b)
                yield b


@dataclass
class GGUFFile:
    path: str
    version: int
    kvs: list[KV]
    tensors: list[TensorInfo]
    alignment: int
    data_start: int  # absolute file offset of the data region
    file_size: int
    spans: dict[str, TensorData] = field(default_factory=dict)

    # convenience
    def kv(self, key: str) -> Optional[KV]:
        for kv in self.kvs:
            if kv.key == key:
                return kv
        return None

    def get(self, key: str, default: Any = None) -> Any:
        kv = self.kv(key)
        if kv is None:
            return default
        return kv.value[1] if kv.vtype == ARRAY else kv.value

    @property
    def arch(self) -> str:
        return self.get("general.architecture")

    def tensor(self, name: str) -> Optional[TensorInfo]:
        for t in self.tensors:
            if t.name == name:
                return t
        return None

    def span(self, name: str) -> TensorData:
        return self.spans[name]


# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------

class _Cursor:
    def __init__(self, buf, pos: int = 0):
        self.buf = buf
        self.pos = pos

    def take(self, n: int) -> bytes:
        b = self.buf[self.pos:self.pos + n]
        if len(b) != n:
            raise ValueError(f"truncated GGUF header at {self.pos}")
        self.pos += n
        return bytes(b)

    def scalar(self, vtype: int):
        fmt = _SCALAR_FMT[vtype]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.scalar(UINT64)
        return self.take(n).decode("utf-8")

    def value(self, vtype: int):
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            etype = self.scalar(UINT32)
            n = self.scalar(UINT64)
            return (etype, [self.value(etype) for _ in range(n)])
        if vtype in _SCALAR_FMT:
            return self.scalar(vtype)
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_header(path: str) -> GGUFFile:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            c = _Cursor(mm)
            if c.take(4) != GGUF_MAGIC:
                raise ValueError(f"{path}: not a GGUF file")
            version = c.scalar(UINT32)
            if version != GGUF_VERSION:
                raise ValueError(f"{path}: unsupported GGUF version {version}")
            n_tensors = c.scalar(UINT64)
            n_kv = c.scalar(UINT64)
            kvs = []
            for _ in range(n_kv):
                key = c.string()
                vtype = c.scalar(UINT32)
                kvs.append(KV(key, vtype, c.value(vtype)))
            tensors = []
            for _ in range(n_tensors):
                name = c.string()
                n_dims = c.scalar(UINT32)
                dims = [c.scalar(UINT64) for _ in range(n_dims)]
                ttype = c.scalar(UINT32)
                offset = c.scalar(UINT64)
                tensors.append(TensorInfo(name, dims, ttype, offset))
            header_end = c.pos
    alignment = DEFAULT_ALIGNMENT
    for kv in kvs:
        if kv.key == "general.alignment":
            alignment = int(kv.value)
    data_start = align_up(header_end, alignment)
    g = GGUFFile(path, version, kvs, tensors, alignment, data_start, size)
    g.spans = _build_spans(g)
    return g


def _build_spans(g: GGUFFile) -> dict[str, TensorData]:
    order = sorted(g.tensors, key=lambda t: t.offset)
    spans = {}
    for i, t in enumerate(order):
        start = g.data_start + t.offset
        end = g.data_start + order[i + 1].offset if i + 1 < len(order) else g.file_size
        if end < start:
            raise ValueError(f"{g.path}: tensor {t.name} has offset past the end of the file")
        spans[t.name] = TensorData(g.path, start, end - start)
    return spans


def check_spans(g: GGUFFile) -> list[str]:
    """Sanity checks that need no decoding: offsets aligned, spans cover known sizes."""
    problems = []
    for t in g.tensors:
        if t.offset % g.alignment:
            problems.append(f"{t.name}: offset {t.offset} not aligned to {g.alignment}")
        nb = t.nbytes
        sp = g.spans[t.name]
        if nb is not None:
            if sp.length < nb:
                problems.append(f"{t.name}: span {sp.length} shorter than expected {nb} bytes")
            elif sp.length - nb >= g.alignment:
                problems.append(f"{t.name}: span {sp.length} exceeds expected {nb} by a full alignment")
    return problems


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------

def _pack_scalar(vtype: int, v) -> bytes:
    return struct.pack(_SCALAR_FMT[vtype], v)


def _pack_string(s) -> bytes:
    b = s.encode("utf-8") if isinstance(s, str) else bytes(s)
    return struct.pack("<Q", len(b)) + b


def _pack_value(vtype: int, v) -> bytes:
    if vtype == STRING:
        return _pack_string(v)
    if vtype == ARRAY:
        etype, items = v
        out = [struct.pack("<I", etype), struct.pack("<Q", len(items))]
        out.extend(_pack_value(etype, x) for x in items)
        return b"".join(out)
    if vtype in _SCALAR_FMT:
        return _pack_scalar(vtype, v)
    raise ValueError(f"unknown GGUF value type {vtype}")


def serialize_header(kvs: list[KV], tensors: list[TensorInfo], alignment: int, pad: bool = True) -> bytes:
    """Header bytes; with pad=True zero-padded to the data region start."""
    out = [GGUF_MAGIC, struct.pack("<I", GGUF_VERSION), struct.pack("<Q", len(tensors)), struct.pack("<Q", len(kvs))]
    for kv in kvs:
        out.append(_pack_string(kv.key))
        out.append(struct.pack("<I", kv.vtype))
        out.append(_pack_value(kv.vtype, kv.value))
    for t in tensors:
        out.append(_pack_string(t.name))
        out.append(struct.pack("<I", len(t.dims)))
        out.extend(struct.pack("<Q", d) for d in t.dims)
        out.append(struct.pack("<I", t.ttype))
        out.append(struct.pack("<Q", t.offset))
    hdr = b"".join(out)
    if pad:
        hdr += b"\0" * (align_up(len(hdr), alignment) - len(hdr))
    return hdr


def layout_tensors(entries: list[tuple[TensorInfo, TensorData]], alignment: int) -> list[TensorInfo]:
    """Assign data-relative offsets in order, each span aligned. Returns new TensorInfos."""
    out = []
    pos = 0
    for info, data in entries:
        pos = align_up(pos, alignment)
        out.append(TensorInfo(info.name, list(info.dims), info.ttype, pos))
        pos += data.length
    return out


def write_gguf(out_path: str, kvs: list[KV], entries: list[tuple[TensorInfo, TensorData]],
               alignment: int = DEFAULT_ALIGNMENT, progress: bool = False) -> None:
    """Write a GGUF: header from kvs and entries, then every span copied verbatim.

    entries: (TensorInfo, TensorData) in file order. Offsets in TensorInfo are
    recomputed here, so callers pass any value.
    """
    infos = layout_tensors(entries, alignment)
    hdr = serialize_header(kvs, infos, alignment)
    total = sum(d.length for _, d in entries)
    done = 0
    with open(out_path, "wb") as out:
        out.write(hdr)
        pos = 0  # data-relative
        for info, (_, data) in zip(infos, entries):
            if info.offset > pos:
                out.write(b"\0" * (info.offset - pos))
                pos = info.offset
            for chunk in data.iter_chunks():
                out.write(chunk)
                done += len(chunk)
            pos += data.length
            if progress:
                print(f"  {done / total * 100:5.1f}%  {info.name}", file=sys.stderr)


def entries_of(g: GGUFFile, names: Optional[list[str]] = None) -> list[tuple[TensorInfo, TensorData]]:
    """(info, span) pairs in header order, optionally only for the given names."""
    keep = None if names is None else set(names)
    return [(t, g.spans[t.name]) for t in g.tensors if keep is None or t.name in keep]


def sha256_file(path: str, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def describe(g: GGUFFile, tensors: bool = False) -> str:
    lines = [f"{g.path}", f"  size {g.file_size} bytes, version {g.version}, alignment {g.alignment}, data starts at {g.data_start}",
             f"  {len(g.kvs)} kv pairs, {len(g.tensors)} tensors, arch {g.arch}"]
    for kv in g.kvs:
        if kv.vtype == ARRAY:
            etype, items = kv.value
            shown = repr(items[:8])
            if len(items) > 8:
                shown = shown[:-1] + ", ...]"
            lines.append(f"  {kv.key}: ARRAY[{VALUE_TYPE_NAMES.get(etype, etype)} x {len(items)}] {shown}")
        else:
            v = repr(kv.value)
            if len(v) > 100:
                v = v[:100] + "..."
            lines.append(f"  {kv.key}: {VALUE_TYPE_NAMES.get(kv.vtype, kv.vtype)} {v}")
    hist: dict[str, int] = {}
    for t in g.tensors:
        hist[type_name(t.ttype)] = hist.get(type_name(t.ttype), 0) + 1
    lines.append("  tensor types: " + ", ".join(f"{k} x {v}" for k, v in sorted(hist.items())))
    if tensors:
        for t in g.tensors:
            lines.append(f"  {t.name} {t.dims} {type_name(t.ttype)} offset {t.offset} span {g.spans[t.name].length}")
    probs = check_spans(g)
    lines.append(f"  span check: {'ok' if not probs else '; '.join(probs)}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "info":
        g = read_header(argv[1])
        print(describe(g, tensors="--tensors" in argv))
        return 0
    if len(argv) == 3 and argv[0] == "copy":
        g = read_header(argv[1])
        write_gguf(argv[2], g.kvs, entries_of(g), g.alignment, progress=True)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
