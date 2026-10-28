"""Manage the pinned Qwen3.8 IQ2_XS files and the Strata server."""

from __future__ import annotations

import argparse
from collections.abc import Generator
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, cast

from project_paths import CAPPED, read_manifest

ROOT = Path(__file__).resolve().parent
NAME = "qwen3.8"
HERE = ROOT / "experiments/qwen-strata"
PYTHON = ROOT / "state/venvs/strata-official/bin/python"


def manifest() -> dict[str, Any]:
    """Read the pinned model and runtime definitions."""
    return read_manifest(HERE / "manifest.json")


def directory() -> Path:
    """Require the dedicated model directory without symbolic links."""
    folder = ROOT / "models/qwen38-ista-iq2-xs"
    if Path(manifest()["model_directory"]) != folder:
        raise RuntimeError("The Qwen model directory does not match its managed path.")
    if folder.parent.is_symlink() or folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise RuntimeError("The Qwen model directory must be a regular directory.")
    return folder


@contextmanager
def model_lock(shared: bool = False) -> Generator[None]:
    """Exclude model changes while a managed server uses the files."""
    state = ROOT / "state"
    state.mkdir(exist_ok=True)
    path = state / "models-qwen-strata.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        try:
            fcntl.flock(lock, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Qwen is serving, fetching, or removing files. Stop that operation first.") from None
        yield


def installed() -> bool:
    """Check raw download sizes and the completed artifact receipt."""
    folder = directory()
    receipt = folder / "artifacts.json"
    return receipt.is_file() and not receipt.is_symlink() and all(
        not (folder / entry["name"]).is_symlink()
        and (folder / entry["name"]).is_file()
        and (folder / entry["name"]).stat().st_size == entry["size"]
        for entry in manifest()["files"]
    )


def environment() -> dict[str, str]:
    """Clear inherited engine tuning and keep the reviewed routing policy."""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("STRATA_", "MAYA_", "DS4_"))}
    env.update({"XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
                "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{os.getuid()}/bus",
                "HEAVY_PEAKS": str(ROOT / "state/qwen-lifecycle-peaks"),
                "STRATA_ROUTE_PRIOR": "0"})
    return env


def fetch() -> None:
    """Download and verify the model, draft layer, and native runtime."""
    with model_lock():
        directory()
        if not PYTHON.exists():
            subprocess.run([str(CAPPED), "--mem", "2G", "--time", "15m", sys.executable,
                            "-B", str(ROOT / "experiments/deepseek-strata/runtime.py"), "--run"],
                           env=environment(), check=True)
        subprocess.run([str(CAPPED), "--mem", "4G", "--time", "12h", str(PYTHON),
                        "-B", str(HERE / "runtime.py"), "--run"], env=environment(), check=True)


def check_ready() -> None:
    """Validate pinned artifacts before a server can start."""
    directory()
    subprocess.run([str(PYTHON), "-B", str(HERE / "runtime.py"), "--check"],
                   env=environment(), check=True, stdout=subprocess.DEVNULL)


def remove(dry_run: bool = False) -> None:
    """Remove only known model files after a locked inventory check."""
    with model_lock():
        folder = directory()
        if not folder.exists():
            print("Qwen weights are already absent.")
            return
        result = subprocess.run([str(PYTHON), "-B", str(HERE / "runtime.py"), "--inventory"],
                                env=environment(), check=True, capture_output=True, text=True)
        inventory: object = json.loads(result.stdout)
        if not isinstance(inventory, list):
            raise RuntimeError("The Qwen artifact inventory must be a list of paths.")
        entries = cast(list[object], inventory)
        if any(not isinstance(name, str) or not name
                or Path(name).is_absolute() or ".." in Path(name).parts for name in entries):
            raise RuntimeError("The Qwen artifact inventory contains an invalid path.")
        names = cast(list[str], entries)
        allowed = set(names)
        parents = {str(parent) for name in names for parent in Path(name).parents if str(parent) != "."}
        size = 0
        for path in folder.rglob("*"):
            name = str(path.relative_to(folder))
            if path.is_symlink() or (path.is_dir() and name not in parents) or (
                    not path.is_dir() and (not path.is_file() or name not in allowed)):
                raise RuntimeError(f"Qwen contains an unknown file, directory, or link: {name}. No files were deleted.")
            if path.is_file():
                size += path.stat().st_size
        print(f"{'Would remove' if dry_run else 'Removing'} {folder} ({size / 2**30:.2f} GiB).", flush=True)
        if dry_run:
            return
        for process in Path("/proc").iterdir():
            if not process.name.isdecimal():
                continue
            try:
                if process.stat().st_uid != os.getuid():
                    continue
                arguments = (process / "cmdline").read_bytes().split(b"\0")
                if any(argument.startswith(os.fsencode(str(folder) + "/")) for argument in arguments):
                    raise RuntimeError(f"Qwen is in use by PID {process.name}. Stop it before removal.")
            except OSError:
                continue
        shutil.rmtree(folder)
        print(f"Deleted Qwen model data. Restore it with localmodel fetch qwen3.8. Freed {size / 2**30:.2f} GiB.")


def settings(args: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
    """Build Qwen engine settings from the supported launcher controls."""
    if args.no_offload:
        raise RuntimeError("Qwen IQ2_XS needs system RAM for experts on this GPU. --no-offload cannot fit its weights.")
    if args.runtime != "stock" or args.speculative != "mtp" or args.prompt_lookup:
        raise RuntimeError("This Qwen profile requires its pinned Strata runtime with MTP. Use a verify window of 2 or more tokens.")
    unsupported = ("cpu_threads", "gpu_expert_cache_gb", "gpu_expert_fraction", "engram_cache_mib",
                   "engram_threads", "cpu_moe_layers", "moe_cache_mib", "batch_tokens", "draft_window",
                   "kv_vram_cells", "mma_decode_min_kv", "cache_ram", "checkpoint_every", "slot_save_path")
    if any(getattr(args, key, None) is not None for key in unsupported) or any(
            getattr(args, key, None) for key in ("kv_cache", "kv_mean_center", "cpu_offload_gb", "kv_offload_gb")):
        raise RuntimeError("Use --expert-ram-gb, --vram-reserve-mib, and --prefill-chunk for Qwen. DeepSeek and llama.cpp tuning flags do not apply.")
    if args.replay or args.gdn_mma is not None:
        raise RuntimeError("Replay and GDN MMA controls require a Bonsai runtime.")
    ram = args.expert_ram_gb if args.expert_ram_gb is not None else 32
    reserve = args.vram_reserve_mib if args.vram_reserve_mib is not None else 700
    context = args.context or item["context"]
    drafts = args.draft_tokens if args.draft_tokens is not None else 3
    chunk = args.prefill_chunk
    budget = args.thinking_budget if args.thinking_budget is not None else 32768
    if not 0 < ram <= 128 or not 0 <= reserve <= 12288:
        raise RuntimeError("Use expert RAM above 0 and at most 128 GiB, and VRAM reserve 0-12288 MiB.")
    if not 1024 <= context <= 262144:
        raise RuntimeError("Use 1024 to 262144 context tokens for this Qwen profile.")
    if not 1 <= drafts <= 7:
        raise RuntimeError("Use 1 to 7 draft tokens for Qwen MTP.")
    if chunk is not None and chunk < 1:
        raise RuntimeError("Use a positive --prefill-chunk token count.")
    if not 0 <= budget <= 1048576:
        raise RuntimeError("Use a thinking budget from 0 to 1048576 tokens.")
    info = manifest()
    source = Path(info["source_directory"])
    binary = Path(info["runtime"]["binary"])
    native_args = ["--pack", info["pack_directory"], "--native", info["first_shard"],
                   "--ple-gguf", info["ple_shard"],
                   "--expert-profile", str(source / "data/expert-profile.bin"), "--expert-cache", "auto",
                   "--prefill", "auto" if chunk is None else str(chunk),
                   "--spec", str(drafts + 1), "--spec-min-p", "0.5", "--suffix-draft", "0",
                   "--mtp", info["mtp_directory"],
                   "--max-context", str(context), "--kv", "int8",
                   "--resident-budget-gib", str(ram), "--vram-reserve-mib", str(reserve)]
    return {"exe": str(binary), "arch": "qwen4exp", "args": native_args,
            "cwd": str(source), "tokenizer": str(Path(info["pack_directory"]) / "tokenizer"),
            "model_name": item["alias"], "gpu": [0], "log": str(ROOT / "state/qwen-engine.log"),
            "env": {"STRATA_ROUTE_PRIOR": "0"}, "lib_dirs": [str(binary.parent)],
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "reasoning_effort": None if (args.reasoning_effort or "xhigh") == "xhigh" else args.reasoning_effort,
            "reasoning_budget_tokens": budget, "fit_max_tokens": True, "api_monitor": False,
            "allowed_hosts": [], "trusted_origins": [], "cors_origins": [], "engine_ready_s": 3600}


def command(args: argparse.Namespace, item: dict[str, Any]) -> list[str]:
    """Build the local API command with the existing outgoing connection guard."""
    settings(args, item)
    source = Path(manifest()["source_directory"])
    return [str(CAPPED), "--mem", str(manifest()["memory_limit"]), str(PYTHON), "-I", "-B",
            str(ROOT / "experiments/deepseek-strata/network_guard.py"),
            str(PYTHON), "-I", "-B", str(source / "serve/server.py"),
            "--engine", "strata", "--config", str(ROOT / "state/qwen-server.json"),
            "--host", "127.0.0.1", "--port", str(args.port), "--offline"]
