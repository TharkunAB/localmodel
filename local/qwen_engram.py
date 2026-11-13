"""Manage the optional pinned Q8_0 table without changing the Qwen core files."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parent
REVISION = "f5d175de2cf7f720d45d6064019a81fcf056e986"
NAME = "Qwen3.8-Flash-Next-engram-Q8_0.gguf_file"
PIN: dict[str, Any] = {
    "size": 54400261312,
    "sha256": "e3d76a2747c1fe53e3420d1bb7fd9593aafd03035fee4926317db80ad5c5296a",
    "url": f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REVISION}/engram/{NAME}",
}


def directory() -> Path:
    """Require an isolated directory with only managed regular files."""
    folder = ROOT / "models/qwen38-engram/q8_0"
    for path in (folder.parent.parent, folder.parent, folder):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise RuntimeError(f"The Qwen Engram directory is not an ordinary directory: {path}")
    if folder.exists():
        allowed = {NAME, NAME + ".part", "verification.json"}
        for path in folder.iterdir():
            if path.name not in allowed or path.is_symlink() or not path.is_file():
                raise RuntimeError(f"The Qwen Engram directory contains an unknown file or link: {path}")
    return folder


def table_path() -> Path:
    """Return the optional table path without requiring a download."""
    return directory() / NAME


def helpers() -> ModuleType:
    """Use the existing reviewed download, digest, and identity functions."""
    spec = importlib.util.spec_from_file_location("qwen_artifacts", ROOT / "experiments/qwen-strata/runtime.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("The Qwen artifact helper is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_table(path: Path) -> None:
    """Read only the bounded GGUF header and require the published Q8_0 tensor."""
    with path.open("rb") as stream:
        header = stream.read(4096)
    position = 0

    def number(fmt: str) -> int:
        nonlocal position
        value = int(struct.unpack_from("<" + fmt, header, position)[0])
        position += struct.calcsize("<" + fmt)
        return value

    def string() -> bytes:
        nonlocal position
        length = number("Q")
        if length > len(header) - position:
            raise ValueError("GGUF string exceeds the header budget")
        value = header[position:position + length]
        position += length
        return value

    try:
        if number("I") != 0x46554747 or number("I") != 3 or number("Q") != 1:
            raise ValueError("Expected a GGUF v3 file with one tensor")
        count = number("Q")
        if count > 32:
            raise ValueError("Too many GGUF metadata fields")
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        for _ in range(count):
            key, kind = string(), number("I")
            if key == b"general.alignment":
                if kind != 4 or number("I") != 32:
                    raise ValueError("Unexpected GGUF alignment")
            elif kind == 8:
                string()
            elif kind in sizes:
                position += sizes[kind]
            else:
                raise ValueError("Unsupported GGUF metadata type")
        if (string() != b"per_layer_token_embd.weight" or number("I") != 2
                or [number("Q"), number("Q")] != [160, 320001536]
                or number("I") != 8 or number("Q") != 0):
            raise ValueError("Expected the Q8_0 PLE table with shape [160, 320001536]")
        if (position + 31) // 32 * 32 + 320001536 * 170 != path.stat().st_size:
            raise ValueError("The table byte count differs from its header")
    except (ValueError, struct.error) as error:
        raise RuntimeError(f"Invalid Qwen Engram table: {error}") from error


def verify() -> Path:
    """Check the receipt identity and tensor metadata before startup."""
    path = table_path()
    receipt = path.parent / "verification.json"
    if not path.is_file() or not receipt.is_file():
        raise RuntimeError("Q8_0 Engram is absent. Run localmodel fetch qwen3.8 --engram q8_0.")
    loaded: object = json.loads(receipt.read_text())
    if not isinstance(loaded, dict):
        raise RuntimeError("The Qwen Engram verification receipt must be an object.")
    record = cast(dict[str, Any], loaded)
    identity = helpers().identity(path)
    if (record.get("sha256") != PIN["sha256"] or record.get("revision") != REVISION
            or record.get("file") != identity or identity["size"] != PIN["size"]):
        raise RuntimeError("Q8_0 Engram changed. Reverify with localmodel fetch qwen3.8 --engram q8_0.")
    check_table(path)
    return path


def fetch() -> None:
    """Download and verify the table; the caller holds the Qwen model lock."""
    path = table_path()
    receipt = path.parent / "verification.json"
    if receipt.exists():
        try:
            verify()
            print("Qwen Q8_0 Engram is already verified.")
            return
        except (RuntimeError, ValueError):
            pass
    partial = path.with_name(NAME + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    if have > PIN["size"]:
        raise RuntimeError("The Q8_0 Engram partial exceeds its pinned size.")
    remaining = 0 if path.exists() else PIN["size"] - have
    if shutil.disk_usage(ROOT).free < remaining + 10 * 2**30:
        raise RuntimeError(f"Q8_0 Engram needs {remaining / 2**30:.2f} GiB free plus 10 GiB headroom.")
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime = helpers()
    if not path.exists() and have == PIN["size"]:
        if runtime.digest(partial) != PIN["sha256"]:
            raise RuntimeError("The completed Q8_0 Engram partial differs from its pin.")
        partial.rename(path)
    else:
        runtime.download(PIN, path)
    check_table(path)
    receipt.write_text(json.dumps({"revision": REVISION, "sha256": PIN["sha256"],
                                   "file": runtime.identity(path)}, indent=2) + "\n")
    verify()
    print(f"Verified Qwen Q8_0 Engram: {path}")


def remove(dry_run: bool = False) -> None:
    """Delete only the optional table directory under the Qwen model lock."""
    folder = directory()
    if not folder.exists():
        print("Qwen Q8_0 Engram is already absent.")
        return
    size = sum(path.stat().st_size for path in folder.iterdir())
    print(f"{'Would remove' if dry_run else 'Removing'} {folder} ({size / 2**30:.2f} GiB).", flush=True)
    if dry_run:
        return
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            if any(arg.startswith(os.fsencode(str(folder) + "/"))
                   for arg in (process / "cmdline").read_bytes().split(b"\0")):
                raise RuntimeError(f"Qwen Q8_0 Engram is in use by PID {process.name}.")
        except OSError:
            continue
    shutil.rmtree(folder)
    print("Deleted only the optional Q8_0 table. Restore with localmodel fetch qwen3.8 --engram q8_0.")


if __name__ == "__main__":
    fetch()
