"""Manage the pinned Qwen3.8 Unsloth UD-IQ1_M files and the Strata server."""

from __future__ import annotations

import argparse
from collections.abc import Generator
from contextlib import contextmanager
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, cast

from project_paths import CAPPED, read_manifest
import qwen_cache_reuse
import qwen_engram

ROOT = Path(__file__).resolve().parent
NAME = "qwen3.8"
HERE = ROOT / "experiments/qwen-strata"
PYTHON = Path(read_manifest(HERE / "manifest.json")["runtime"]["venv"]) / "bin/python"


def manifest() -> dict[str, Any]:
    """Read the pinned model and runtime definitions."""
    return read_manifest(HERE / "manifest.json")


def directory() -> Path:
    """Require the dedicated model directory without symbolic links."""
    folder = ROOT / "models/qwen38-unsloth-iq1-m"
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


def fetch(speculative: str = "mtp", engram: str = "default") -> None:
    """Download and verify the model, draft layer, and native runtime."""
    if speculative not in ("mtp", "dflash2"):
        raise RuntimeError("Qwen supports only mtp or dflash2 artifacts.")
    if engram not in ("default", "q8_0"):
        raise RuntimeError("Qwen Engram must be default or q8_0.")
    with model_lock():
        directory()
        if not PYTHON.exists():
            subprocess.run([str(CAPPED), "--mem", "2G", "--time", "15m", sys.executable,
                            "-B", str(ROOT / "experiments/deepseek-strata/runtime.py"), "--run"],
                           env=environment(), check=True)
        subprocess.run([str(CAPPED), "--mem", "4G", "--time", "12h", str(PYTHON),
                        "-B", str(HERE / "runtime.py"), "--run", "--speculative", speculative], env=environment(), check=True)
        if engram == "q8_0":
            subprocess.run([str(CAPPED), "--mem", "2G", "--time", "12h", str(PYTHON),
                            "-B", str(ROOT / "qwen_engram.py")], env=environment(), check=True)


def check_ready(speculative: str = "mtp", cache_reuse_fix: bool = False, engram: str = "default") -> None:
    """Validate pinned artifacts before a server can start."""
    if speculative not in ("mtp", "dflash2"):
        raise RuntimeError("Qwen supports only mtp or dflash2 artifacts.")
    directory()
    subprocess.run([str(PYTHON), "-B", str(HERE / "runtime.py"), "--check", "--speculative", speculative],
                   env=environment(), check=True, stdout=subprocess.DEVNULL)
    if cache_reuse_fix:
        qwen_cache_reuse.verify()
    if engram == "q8_0":
        qwen_engram.verify()
    elif engram != "default":
        raise RuntimeError("Qwen Engram must be default or q8_0.")


def remove(dry_run: bool = False, speculative: str = "mtp", engram: str = "default") -> None:
    """Remove only known model files after a locked inventory check."""
    if speculative not in ("mtp", "dflash2"):
        raise RuntimeError("Qwen supports only mtp or dflash2 artifacts.")
    with model_lock():
        if engram == "q8_0":
            qwen_engram.remove(dry_run)
            return
        if engram != "default":
            raise RuntimeError("Qwen Engram must be default or q8_0.")
        model = directory()
        folder = model / "dflash2" if speculative == "dflash2" else model
        if folder.is_symlink():
            raise RuntimeError("The optional draft directory must not be a symbolic link.")
        if not folder.exists():
            print("Qwen weights are already absent.")
            return
        result = subprocess.run([str(PYTHON), "-B", str(HERE / "runtime.py"), "--inventory", "--speculative", speculative],
                                env=environment(), check=True, capture_output=True, text=True)
        inventory: object = json.loads(result.stdout)
        if not isinstance(inventory, list):
            raise RuntimeError("The Qwen artifact inventory must be a list of paths.")
        entries = cast(list[object], inventory)
        if any(not isinstance(name, str) or not name
                or Path(name).is_absolute() or ".." in Path(name).parts for name in entries):
            raise RuntimeError("The Qwen artifact inventory contains an invalid path.")
        names = cast(list[str], entries)
        if speculative == "dflash2":
            names = [str(Path(name).relative_to("dflash2")) for name in names]
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
        suffix = " --speculative dflash2" if speculative == "dflash2" else ""
        print(f"Deleted {folder}. Restore it with localmodel fetch qwen3.8{suffix}. Freed {size / 2**30:.2f} GiB.")


def runtime_status(offset: int) -> dict[str, Any]:
    """Read positive activation evidence from this server start only."""
    result: dict[str, Any] = {"effective_speculative": None, "adaptive_fetch_active": None,
                              "effective_engram_type": None}
    try:
        with (ROOT / "state/qwen-engine.log").open("rb") as stream:
            stream.seek(offset)
            text = stream.read(1024 * 1024).decode("utf-8", errors="replace")
    except OSError:
        return result
    if "DFlash2 drafter on (" in text:
        result["effective_speculative"] = "dflash2"
    elif "runs without the draft layer" in text or "runs without draft layer" in text:
        result["effective_speculative"] = "none"
    elif "strata mtp: draft layer loaded," in text:
        result["effective_speculative"] = "mtp"
    active = re.search(r"strata serve: STRATA_ADAPT_FETCH=([12]):", text)
    if active:
        result["adaptive_fetch_active"] = int(active.group(1))
    elif "STRATA_ADAPT_FETCH is off (" in text:
        result["adaptive_fetch_active"] = 0
    table = re.search(r"PLE on, table 320001536 rows of (\S+)", text)
    if table:
        result["effective_engram_type"] = table.group(1)
    return result


def settings(args: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
    """Build Qwen engine settings from the supported launcher controls."""
    if args.no_offload:
        raise RuntimeError("Qwen UD-IQ1_M needs system RAM for experts on this GPU. --no-offload cannot fit its weights.")
    if args.runtime != "stock" or args.speculative not in ("mtp", "dflash2"):
        raise RuntimeError("This Qwen profile requires pinned Strata with --speculative mtp or dflash2.")
    dflash = args.speculative == "dflash2"
    if dflash and args.prompt_lookup:
        raise RuntimeError("Use --no-prompt-lookup for the explicit Qwen DFlash2 experiment.")
    if args.lookup_type != "ngram-simple" or args.lookup_min_tokens is not None:
        raise RuntimeError("Qwen uses native suffix lookup. --lookup-type ngram-mod and --lookup-min-tokens are llama.cpp controls.")
    unsupported = ("gpu_expert_cache_gb", "gpu_expert_fraction", "engram_cache_mib",
                   "engram_threads", "cpu_moe_layers", "moe_cache_mib", "batch_tokens", "draft_window",
                   "mma_decode_min_kv", "cache_ram", "checkpoint_every", "slot_save_path")
    if any(getattr(args, key, None) is not None for key in unsupported) or any(
            getattr(args, key, None) for key in ("kv_cache", "kv_mean_center", "cpu_offload_gb", "kv_offload_gb")):
        raise RuntimeError("These DeepSeek or llama.cpp controls do not apply to Qwen. Use the Qwen controls described by localmodel start --help.")
    if args.replay or args.gdn_mma is not None:
        raise RuntimeError("Replay and GDN MMA controls require a Bonsai runtime.")
    ram = args.expert_ram_gb if args.expert_ram_gb is not None else 36
    adaptive = getattr(args, "adaptive_fetch", None)
    if adaptive is None:
        adaptive = 1 if ram == 0 else 0
    confidence = getattr(args, "draft_conf", None)
    confidence = 0.8 if confidence is None else confidence
    if adaptive not in (0, 1, 2) or (adaptive and ram != 0):
        raise RuntimeError("Qwen --adaptive-fetch 1 or 2 requires --expert-ram-gb 0 (file-backed experts).")
    if not 0 <= confidence <= 1:
        raise RuntimeError("Use a DFlash2 confidence threshold from 0 to 1.")
    reserve = args.vram_reserve_mib if args.vram_reserve_mib is not None else 1024
    context = args.context or item["context"]
    drafts = args.draft_tokens if args.draft_tokens is not None else 5 if dflash else item.get("draft_tokens", 4)
    kv_cells = args.kv_vram_cells
    if kv_cells is None:
        kv_cells = min(item.get("kv_vram_cells", context), context)
    ple_io = getattr(args, "ple_io", None)
    if ple_io is None:
        ple_io = item.get("ple_io", "direct")
    chunk = args.prefill_chunk
    budget = args.thinking_budget if args.thinking_budget is not None else 32768
    if not 0 <= ram <= 128 or not 0 <= reserve <= 12288:
        raise RuntimeError("Use expert RAM from 0 to 128 GiB, and VRAM reserve 0-12288 MiB.")
    if not 1024 <= context <= 262144:
        raise RuntimeError("Use 1024 to 262144 context tokens for this Qwen profile.")
    if not 1 <= drafts <= 7:
        raise RuntimeError("Use 1 to 7 draft tokens for Qwen; the verify window adds one accepted token.")
    if args.prompt_lookup and not 2 <= args.lookup_tokens <= 16:
        raise RuntimeError("Use 2 to 16 tokens for the Qwen lookup verification window.")
    if args.prompt_lookup and args.lookup_tokens < drafts + 1:
        raise RuntimeError("The Qwen lookup window must hold the MTP drafts plus one accepted token.")
    if args.cpu_threads is not None and not 0 <= args.cpu_threads <= 64:
        raise RuntimeError("Use 0 to 64 Qwen CPU workers; 0 selects the automatic count.")
    if not 1024 <= kv_cells <= context:
        raise RuntimeError("Use 1024 to the target context length for Qwen --kv-vram-cells.")
    if ple_io not in ("direct", "mmap", "ram"):
        raise RuntimeError("Use direct, mmap, or ram for Qwen --ple-io.")
    if chunk is not None and chunk < 1:
        raise RuntimeError("Use a positive --prefill-chunk token count.")
    if not 0 <= budget <= 1048576:
        raise RuntimeError("Use a thinking budget from 0 to 1048576 tokens.")
    info = manifest()
    engram = getattr(args, "engram", "default")
    if engram not in ("default", "q8_0"):
        raise RuntimeError("Qwen Engram must be default or q8_0.")
    ple_table = str(qwen_engram.table_path()) if engram == "q8_0" else info["ple_shard"]
    source = Path(info["source_directory"])
    cache_reuse_fix = getattr(args, "cache_reuse_fix", False)
    binary = qwen_cache_reuse.binary_path() if cache_reuse_fix else Path(info["runtime"]["binary"])
    native_args = ["--pack", info["pack_directory"], "--native", info["first_shard"],
                   "--ple-gguf", ple_table,
                   "--expert-profile", str(source / "data/expert-profile.bin"), "--expert-cache", "auto",
                   "--prefill", "auto" if chunk is None else str(chunk),
                   "--spec", str(drafts + 1), "--suffix-draft", "3" if args.prompt_lookup else "0",
                   "--max-context", str(context), "--kv", "int8", "--vram-reserve-mib", str(reserve)]
    native_args += ["--mmap-experts"] if ram == 0 else ["--resident-budget-gib", str(ram)]
    if dflash:
        native_args += ["--draft", "dflash2", "--draft-model",
                        str(directory() / "dflash2" / info["dflash2"]["name"]),
                        "--draft-fallback", "none", "--draft-conf", str(confidence),
                        "--mtp-draft-vocab", str(Path(info["mtp_directory"]) / "draft_vocab.bin")]
    else:
        native_args += ["--mtp", info["mtp_directory"], "--spec-min-p", "0.5"]
    if args.prompt_lookup:
        native_args += ["--lookup-max-t", str(args.lookup_tokens)]
    if args.cpu_threads is not None:
        native_args += ["--pool-workers", str(args.cpu_threads)]
    native_args += ["--kv-resident", str(kv_cells), "--ple-io", ple_io]
    # The public alias is outside the upstream automatic drafter policy.
    # Keep explicit native DFlash2 arguments; do not let that policy select another pack.
    return {"exe": str(binary), "arch": "qwen4exp", **({} if dflash else {"drafting": "mtp"}), "args": native_args,
            "expert_memory": "cache" if ram == 0 else "resident", "expert_ram_gb": ram,
            "engram": engram, "engram_type": "Q8_0" if engram == "q8_0" else "IQ4_NL", "engram_path": ple_table,
            "adaptive_fetch": adaptive, "vram_reserve_mib": reserve,
            "user_args": ["--kv-resident", str(kv_cells), "--vram-reserve-mib", str(reserve)],
            "cwd": str(source), "tokenizer": str(Path(info["pack_directory"]) / "tokenizer"),
            "model_name": item["alias"], "gpu": [0], "log": str(ROOT / "state/qwen-engine.log"),
            "env": {"STRATA_ROUTE_PRIOR": "0", "STRATA_HC_Q8": "1", "STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_ADAPT_FETCH": str(adaptive)},
            "lib_dirs": qwen_cache_reuse.lib_dirs() if cache_reuse_fix else [str(binary.parent)],
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
