"""Manage the pinned DeepSeek UD-IQ1_S files and the reviewed Strata server."""

from __future__ import annotations

import argparse
from collections.abc import Generator
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, cast

from project_paths import CAPPED, read_manifest

ROOT = Path(__file__).resolve().parent
NAME = "deepseek-v4.1-1017-iq1-s"
HERE = ROOT / "experiments/deepseek-strata"
PYTHON = Path(read_manifest(HERE / "manifest.json")["runtime"]["venv"]) / "bin/python"
SOURCE = Path(read_manifest(HERE / "manifest.json")["source_directory"])
PACK_SOURCE = Path(read_manifest(HERE / "manifest.json").get("pack_source_directory", str(SOURCE)))
PACK_ASSETS = frozenset({"ds4.txt", "tensors.txt", "engram.bin",
                         "tokenizer/chat_template.jinja", "tokenizer/token_type.json",
                         "tokenizer/merges.txt", "tokenizer/vocab.json", "tokenizer/tokenizer.json"})


def manifest() -> dict[str, Any]:
    """Read the pinned model and runtime definitions."""
    return read_manifest(HERE / "manifest.json")


def directory() -> Path:
    """Use the managed native Linux directory. Reject directory links."""
    parent = ROOT / "models"
    path = Path(manifest()["model_directory"])
    if path.parent != parent or parent.is_symlink() or path.is_symlink():
        raise RuntimeError("Use a regular DeepSeek directory directly inside models.")
    return path


@contextmanager
def model_lock(shared: bool = False) -> Generator[None]:
    """Exclude weight changes while a managed server uses the model."""
    state = ROOT / "state"
    state.mkdir(exist_ok=True)
    with (state / "models-deepseek-strata.lock").open("a") as lock:
        try:
            fcntl.flock(lock, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("DeepSeek is serving, fetching, or removing files. Stop that operation first.") from None
        yield


def installed() -> bool:
    """Require all five pinned files at their expected sizes."""
    folder = directory()
    return all(not (folder / item["name"]).is_symlink() and (folder / item["name"]).is_file()
               and (folder / item["name"]).stat().st_size == item["size"] for item in manifest()["files"])


def verification_receipt() -> dict[str, Any]:
    """Record file identities after a complete successful checksum check."""
    info = manifest()
    files: list[dict[str, Any]] = []
    for item in info["files"]:
        path = directory() / item["name"]
        stat = path.stat()
        files.append({"name": item["name"], "sha256": item["sha256"], "size": stat.st_size,
                      "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
                      "inode": stat.st_ino, "device": stat.st_dev})
    return {"revision": info["model_revision"], "files": files}


def verified() -> bool:
    """Reuse checksum evidence only while all file identities match."""
    try:
        return installed() and json.loads((directory() / "verification.json").read_text()) == verification_receipt()
    except (OSError, ValueError):
        return False


def environment() -> dict[str, str]:
    """Use the matching CUDA runtime and the local user service manager."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("STRATA_", "MAYA_", "DS4_"))}
    env.update({"XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
                "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{os.getuid()}/bus",
                "HEAVY_PEAKS": str(ROOT / "state/deepseek-lifecycle-peaks")})
    return env


def pack_files() -> dict[str, Path]:
    """Reject linked, unknown, or nonregular pack entries."""
    pack = directory() / "strata-pack"
    if pack.is_symlink() or (pack.exists() and not pack.is_dir()):
        raise RuntimeError("The DeepSeek pack must be a regular directory.")
    files: dict[str, Path] = {}
    for path in pack.rglob("*"):
        name = str(path.relative_to(pack))
        if path.is_symlink() or (path.is_dir() and name != "tokenizer"):
            raise RuntimeError("The DeepSeek pack contains an unexpected directory or link.")
        if path.is_dir():
            continue
        if not path.is_file() or name not in PACK_ASSETS | {"localmodel-pack.json", "expert-profile.bin"}:
            raise RuntimeError("The DeepSeek pack contains an unknown or nonregular file.")
        if name == "expert-profile.bin" and path.stat().st_size != 40 * 384 * 4:
            raise RuntimeError("The Strata expert profile has an unexpected size.")
        files[name] = path
    return files


def check_pack() -> None:
    """Require every generated pack asset and verify its recorded bytes."""
    try:
        paths = pack_files()
        record = cast(dict[str, Any], json.loads(paths["localmodel-pack.json"].read_text()))
        if record["model_revision"] != manifest()["model_revision"] or record["source_revision"] != manifest().get("pack_source_revision", manifest()["source_revision"]):
            raise ValueError("Pack pins changed.")
        if set(record["files"]) != set(PACK_ASSETS) or set(paths) - {"expert-profile.bin"} != set(PACK_ASSETS) | {"localmodel-pack.json"}:
            raise ValueError("The pack asset list is incomplete.")
        for name, expected in record["files"].items():
            with paths[name].open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("Pack bytes changed.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("DeepSeek pack is missing or changed. Remove strata-pack/localmodel-pack.json and run localmodel fetch deepseek.") from error


def fetch() -> None:
    """Fetch verified weights, check the runtime, and create its small index."""
    with model_lock():
        receipt = directory() / "verification.json"
        if receipt.is_symlink() or (receipt.exists() and not receipt.is_file()):
            raise RuntimeError("The DeepSeek verification receipt must be a regular file.")
        pack_files()
        env = environment()
        subprocess.run([str(CAPPED), "--mem", "2G", "--time", "15m", sys.executable,
                        str(HERE / "runtime.py"), "--run"], env=env, check=True)
        if not verified():
            subprocess.run([str(CAPPED), "--mem", "4G", "--time", "12h", str(PYTHON),
                            str(HERE / "fetch.py"), "--directory", str(directory()), "--run"], env=env, check=True)
            (directory() / "verification.json").write_text(json.dumps(verification_receipt(), indent=2) + "\n")
        else:
            print("DeepSeek UD-IQ1_S is already installed. Checksum receipt matches all five files.", flush=True)
        pack = directory() / "strata-pack"
        if not (pack / "localmodel-pack.json").is_file():
            subprocess.run([str(CAPPED), "--mem", "4G", "--time", "10m", str(PYTHON),
                            str(PACK_SOURCE / "tools/dsv41/pack_gguf.py"), "--model", str(directory()),
                            "--out", str(pack), "--shards", *[item["name"] for item in manifest()["files"]]],
                           env=env, check=True)
            record = {"model_revision": manifest()["model_revision"], "source_revision": manifest().get("pack_source_revision", manifest()["source_revision"]),
                      "files": {str(path.relative_to(pack)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in pack.rglob("*") if path.is_file() and str(path.relative_to(pack)) in PACK_ASSETS}}
            (pack / "localmodel-pack.json").write_text(json.dumps(record, indent=2) + "\n")
        check_pack()


def check_ready() -> None:
    """Check build, model identities, and generated pack bytes before serving."""
    if not verified():
        raise RuntimeError("DeepSeek files lack current checksum evidence. Run localmodel fetch deepseek.")
    subprocess.run([sys.executable, str(HERE / "runtime.py"), "--check"], env=environment(), check=True,
                   stdout=subprocess.DEVNULL)
    check_pack()


def remove(dry_run: bool = False) -> None:
    """Delete only this model's data. Refuse linked or unexpected entries."""
    folder = directory()
    if not folder.exists():
        print("DeepSeek weights are already absent.")
        return
    names = {item["name"] + suffix for item in manifest()["files"] for suffix in ("", ".part", ".moving")}
    names |= {"strata-pack", "verification.json"}
    if any(path.name not in names or path.is_symlink() or
           (not path.is_dir() if path.name == "strata-pack" else not path.is_file()) for path in folder.iterdir()):
        raise RuntimeError("The DeepSeek directory contains unknown files or links. No files were deleted.")
    pack_files()
    size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file() and not path.is_symlink())
    print(f"{'Would remove' if dry_run else 'Removing'} {folder} ({size / 2**30:.2f} GiB).", flush=True)
    if dry_run:
        return
    with model_lock():
        for process in Path("/proc").iterdir():
            if not process.name.isdecimal():
                continue
            try:
                if process.stat().st_uid != os.getuid():
                    continue
                arguments = (process / "cmdline").read_bytes().split(b"\0")
                if any(argument.startswith(os.fsencode(str(folder) + "/")) for argument in arguments):
                    raise RuntimeError(f"DeepSeek is in use by PID {process.name}. Stop it before removal.")
            except OSError:
                continue
        shutil.rmtree(folder)
    print(f"Deleted DeepSeek model data. Restore it with localmodel fetch deepseek. Freed {size / 2**30:.2f} GiB.")


def settings(args: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
    """Generate the reviewed server settings from explicit launcher controls."""
    if args.no_offload:
        raise RuntimeError("DeepSeek UD-IQ1_S needs RAM and SSD experts on this GPU. --no-offload cannot fit its weights.")
    if args.runtime != "stock" or args.speculative != "none" or args.prompt_lookup:
        raise RuntimeError("This Strata profile uses its pinned runtime with speculation off.")
    if args.cache_ram is not None or args.kv_cache or args.kv_mean_center or args.cpu_offload_gb or args.kv_offload_gb:
        raise RuntimeError("Use --expert-ram-gb for Strata. llama.cpp prompt/KV cache and Dory offload controls do not apply.")
    if any(getattr(args, key) is not None for key in ("cpu_moe_layers", "moe_cache_mib", "batch_tokens", "draft_window", "kv_vram_cells", "mma_decode_min_kv")):
        raise RuntimeError("Flint MoE, llama.cpp batching, and Bonsai draft/KV controls do not apply to Strata.")
    if args.gpu_expert_cache_gb is not None:
        raise RuntimeError("Official Strata uses --gpu-expert-fraction instead of --gpu-expert-cache-gb.")
    ram = args.expert_ram_gb if args.expert_ram_gb is not None else 36
    threads = args.cpu_threads if args.cpu_threads is not None else 4
    reserve = args.vram_reserve_mib if args.vram_reserve_mib is not None else 700
    engram = args.engram_cache_mib if args.engram_cache_mib is not None else 256
    readers = args.engram_threads if args.engram_threads is not None else 16
    if not 0 <= ram <= 128 or not 0 <= threads <= 64 or not 0 <= reserve <= 12288 or not 0 <= engram <= 4096 or not 1 <= readers <= 64:
        raise RuntimeError("Use expert RAM 0-128 GiB, CPU threads 0-64, VRAM reserve 0-12288 MiB, embedding cache 0-4096 MiB, and embedding threads 1-64.")
    fraction = args.gpu_expert_fraction
    if fraction is not None and not 0 <= fraction <= 1:
        raise RuntimeError("Use 0 to 1 for --gpu-expert-fraction.")
    chunk = args.prefill_chunk if args.prefill_chunk is not None else 4096
    if chunk is not None and chunk < 1:
        raise RuntimeError("Use a positive --prefill-chunk token count.")
    env = {"DS4_ENGRAM_CACHE_MB": str(engram), "DS4_ENGRAM_THREADS": str(readers),
           "STRATA_DS4_DISK_SKIP_W": "0", "STRATA_DS4_SLOT_KEEP": "0"}
    budget = args.thinking_budget if args.thinking_budget is not None else 32768
    if not 0 <= budget <= 1048576:
        raise RuntimeError("Use a thinking budget from 0 to 1048576 tokens.")
    context = args.context or item["context"]
    if not 1024 <= context <= 1048576:
        raise RuntimeError("Use 1024 to 1048576 context tokens for Strata.")
    native_args = ["--pack", str(directory() / "strata-pack"), "--max-context", str(context),
                   "--ssd-cache", str(ram * 2**30 / 10**9), "--cpu-threads", str(threads),
                   "--reserve-mib", str(reserve), "--dspark", "none", "--experts", "6", "--vision", "off"]
    if fraction is not None:
        native_args += ["--vram-experts", str(fraction)]
    if chunk is not None:
        native_args += ["--prefill-chunk", str(chunk)]
    runtime = manifest()["runtime"]
    binary = Path(runtime["directory"]) / "engine-dsv41" / f"{runtime['version']}-linux-x86_64-cuda-cuda13/strata-dsv41"
    return {"exe": str(binary), "arch": "deepseek41", "args": native_args,
            "cwd": str(SOURCE), "tokenizer": str(directory() / "strata-pack/tokenizer"),
            "model_name": item["alias"], "gpu": [0], "log": str(ROOT / "state/deepseek-engine.log"),
            "env": env, "lib_dirs": [str(binary.parent)],
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "reasoning_effort": None if (args.reasoning_effort or "xhigh") == "xhigh" else args.reasoning_effort,
            "reasoning_budget_tokens": budget, "fit_max_tokens": True, "api_monitor": False,
            "allowed_hosts": [], "trusted_origins": [], "cors_origins": [], "engine_ready_s": 3600}


def command(args: argparse.Namespace, item: dict[str, Any]) -> list[str]:
    """Build the local API command. The general --mem flag can override its cap."""
    settings(args, item)
    return [str(CAPPED), "--mem", str(manifest()["memory_limit"]), str(PYTHON), "-I", "-B", str(HERE / "network_guard.py"),
            str(PYTHON), "-I", "-B", str(SOURCE / "serve/server.py"),
            "--engine", "strata", "--config", str(ROOT / "state/deepseek-server.json"),
            "--host", "127.0.0.1", "--port", str(args.port), "--offline"]
