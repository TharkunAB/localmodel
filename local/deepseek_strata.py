"""Manage the pinned DeepSeek variants and the reviewed Strata server."""

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
CODER_NAME = "deepseek-v4.1-1017-coder-iq2-xxs"
NAMES = (NAME, CODER_NAME)
HERE = ROOT / "experiments/deepseek-strata"
PYTHON = Path(read_manifest(HERE / "manifest.json")["runtime"]["venv"]) / "bin/python"
SOURCE = Path(read_manifest(HERE / "manifest.json")["source_directory"])
PACK_SOURCE = Path(read_manifest(HERE / "manifest.json").get("pack_source_directory", str(SOURCE)))
PACK_ASSETS = frozenset({"ds4.txt", "tensors.txt", "engram.bin",
                         "tokenizer/chat_template.jinja", "tokenizer/token_type.json",
                         "tokenizer/merges.txt", "tokenizer/vocab.json", "tokenizer/tokenizer.json"})


def manifest_path(model: str = NAME) -> Path:
    """Select the pinned weight definition."""
    if model not in NAMES:
        raise ValueError(f"Unknown DeepSeek profile: {model}.")
    return ROOT / "experiments" / ("deepseek-coder" if model == CODER_NAME else "deepseek-strata") / "manifest.json"


def manifest(model: str = NAME) -> dict[str, Any]:
    """Use one runtime definition with the selected model's weight pins."""
    base = read_manifest(HERE / "manifest.json")
    path = manifest_path(model)
    return {**base, **read_manifest(path)} if model == CODER_NAME else base


def directory(model: str = NAME) -> Path:
    """Use the managed native Linux directory. Reject directory links."""
    parent = ROOT / "models"
    path = Path(manifest(model)["model_directory"])
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


def drafter_directory(model: str = NAME) -> Path:
    """Reject linked, unknown, or nonregular draft files."""
    folder = directory(model) / "drafter"
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise RuntimeError("The DeepSeek drafter must be a regular directory.")
    names = {manifest(model)["dspark"]["name"] + suffix for suffix in ("", ".part", ".moving")}
    names.add("verification.json")
    if folder.exists() and any(path.name not in names or path.is_symlink() or not path.is_file()
                               for path in folder.iterdir()):
        raise RuntimeError("The DeepSeek drafter contains unknown files or links.")
    return folder


def installed(draft: bool = False, model: str = NAME) -> bool:
    """Require the selected pinned files at their expected sizes."""
    folder = drafter_directory(model) if draft else directory(model)
    files = [manifest(model)["dspark"]] if draft else manifest(model)["files"]
    return all(not (folder / item["name"]).is_symlink() and (folder / item["name"]).is_file()
               and (folder / item["name"]).stat().st_size == item["size"] for item in files)


def file_record(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    """Record the identity of one checksum-verified file."""
    stat = path.stat()
    return {"name": item["name"], "sha256": item["sha256"], "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino, "device": stat.st_dev}


def shared_files(model: str) -> list[tuple[str, dict[str, Any]]]:
    """Find files whose full-file pins are identical in the other profile."""
    peer = CODER_NAME if model == NAME else NAME
    other = {item["name"]: item for item in manifest(peer)["files"]}
    return [(peer, item) for item in manifest(model)["files"]
            if item["name"] in other and all(item[key] == other[item["name"]][key] for key in ("size", "sha256"))]


def file_receipt(model: str, item: dict[str, Any]) -> dict[str, Any] | None:
    """Read checksum evidence for one file, even when other shards are absent."""
    folder = directory(model)
    path, receipt = folder / item["name"], folder / "verification.json"
    try:
        if path.is_symlink() or receipt.is_symlink() or not path.is_file() or path.stat().st_size != item["size"]:
            return None
        record = cast(dict[str, Any], json.loads(receipt.read_text()))
        if record["revision"] == manifest(model)["model_revision"] and file_record(path, item) in record["files"]:
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def refresh_link_receipt(model: str, item: dict[str, Any], record: dict[str, Any]) -> None:
    """Refresh only the known file's identity after a hard-link count change."""
    folder = directory(model)
    record["files"] = [file_record(folder / item["name"], item) if row["name"] == item["name"] else row
                       for row in record["files"]]
    (folder / "verification.json").write_text(json.dumps(record, indent=2) + "\n")


def reuse_shared_files(model: str) -> None:
    """Link verified identical files while the shared lifecycle lock is held."""
    folder = directory(model)
    folder.mkdir(parents=True, exist_ok=True)
    for peer, item in shared_files(model):
        target = folder / item["name"]
        if target.exists() or target.is_symlink():
            continue
        record = file_receipt(peer, item)
        if record is not None:
            os.link(directory(peer) / item["name"], target)
            refresh_link_receipt(peer, item, record)
            print(f"Reused verified {item['name']} from {peer}.", flush=True)


def verification_receipt(draft: bool = False, model: str = NAME) -> dict[str, Any]:
    """Record file identities after a complete successful checksum check."""
    info = manifest(model)
    folder = drafter_directory(model) if draft else directory(model)
    files: list[dict[str, Any]] = []
    for item in [info["dspark"]] if draft else info["files"]:
        path = folder / item["name"]
        files.append(file_record(path, item))
    return {"revision": info["dspark"]["revision"] if draft else info["model_revision"], "files": files}


def verified(draft: bool = False, model: str = NAME) -> bool:
    """Reuse checksum evidence only while all file identities match."""
    try:
        folder = drafter_directory(model) if draft else directory(model)
        return installed(draft, model) and json.loads((folder / "verification.json").read_text()) == verification_receipt(draft, model)
    except (OSError, ValueError):
        return False


def environment() -> dict[str, str]:
    """Use the matching CUDA runtime and the local user service manager."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("STRATA_", "MAYA_", "DS4_"))}
    env.update({"XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
                "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{os.getuid()}/bus",
                "HEAVY_PEAKS": str(ROOT / "state/deepseek-lifecycle-peaks")})
    return env


def pack_directory(speculative: str = "none", model: str = NAME) -> Path:
    """Keep the baseline and DSpark packs separate."""
    if model == CODER_NAME and speculative != "none":
        raise RuntimeError("DeepSeek Coder currently supports --speculative none only.")
    if speculative not in ("none", "dspark"):
        raise RuntimeError("DeepSeek supports --speculative none or dspark.")
    return directory(model) / ("strata-pack-dspark" if speculative == "dspark" else "strata-pack")


def pack_files(speculative: str = "none", model: str = NAME) -> dict[str, Path]:
    """Reject linked, unknown, or nonregular pack entries."""
    pack = pack_directory(speculative, model)
    info = manifest(model)
    layers, experts = int(info.get("layer_count", 40)), int(info.get("expert_count", 384))
    profile_sizes = (layers * experts * 4, (layers + 3) * experts * 4) if model == NAME else (layers * experts * 4,)
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
        if name == "expert-profile.bin" and path.stat().st_size not in profile_sizes:
            raise RuntimeError("The Strata expert profile has an unexpected size.")
        files[name] = path
    return files


def check_pack(speculative: str = "none", model: str = NAME) -> None:
    """Require every generated pack asset and verify its recorded bytes."""
    try:
        paths = pack_files(speculative, model)
        record = cast(dict[str, Any], json.loads(paths["localmodel-pack.json"].read_text()))
        if record["model_revision"] != manifest(model)["model_revision"] or record["source_revision"] != manifest(model).get("pack_source_revision", manifest(model)["source_revision"]):
            raise ValueError("Pack pins changed.")
        if record.get("draft") != (manifest(model)["dspark"] if speculative == "dspark" else None):
            raise ValueError("Draft pins changed.")
        if set(record["files"]) != set(PACK_ASSETS) or set(paths) - {"expert-profile.bin"} != set(PACK_ASSETS) | {"localmodel-pack.json"}:
            raise ValueError("The pack asset list is incomplete.")
        model_dirs = [line.removeprefix("model_dir ") for line in paths["ds4.txt"].read_text().splitlines()
                      if line.startswith("model_dir ")]
        if model_dirs != [str(directory(model))]:
            raise ValueError(f"The pack model_dir must be {directory(model)}; rebuild the pack after moving weights.")
        for name, expected in record["files"].items():
            with paths[name].open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("Pack bytes changed.")
        expected_files = [item["name"] for item in manifest(model)["files"]]
        if speculative == "dspark":
            expected_files.append("drafter/" + manifest(model)["dspark"]["name"])
        actual_files = [line.split(maxsplit=2)[2] for line in paths["tensors.txt"].read_text().splitlines()
                        if line.startswith("file ")]
        if actual_files != expected_files:
            raise ValueError("The pack references unexpected weight files.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        suffix = " --speculative dspark" if speculative == "dspark" else ""
        raise RuntimeError(f"DeepSeek pack is missing or changed: {error} Remove {pack_directory(speculative, model) / 'localmodel-pack.json'} and run localmodel fetch {model}{suffix}.") from error


def fetch(speculative: str = "none", model: str = NAME) -> None:
    """Fetch verified weights, check the runtime, and create its small index."""
    pack = pack_directory(speculative, model)
    with model_lock():
        receipt = directory(model) / "verification.json"
        if receipt.is_symlink() or (receipt.exists() and not receipt.is_file()):
            raise RuntimeError("The DeepSeek verification receipt must be a regular file.")
        pack_files(speculative, model)
        env = environment()
        subprocess.run([str(CAPPED), "--mem", "2G", "--time", "15m", sys.executable,
                        str(HERE / "runtime.py"), "--run"], env=env, check=True)
        if not verified(model=model):
            reuse_shared_files(model)
            subprocess.run([str(CAPPED), "--mem", "4G", "--time", "12h", str(PYTHON),
                            str(HERE / "fetch.py"), "--manifest", str(manifest_path(model)), "--directory", str(directory(model)), "--run"], env=env, check=True)
            (directory(model) / "verification.json").write_text(json.dumps(verification_receipt(model=model), indent=2) + "\n")
        else:
            print(f"{model} is installed. Checksum receipt matches its pinned files.", flush=True)
        if speculative == "dspark" and not verified(draft=True, model=model):
            draft = drafter_directory(model)
            subprocess.run([str(CAPPED), "--mem", "4G", "--time", "2h", str(PYTHON),
                            str(HERE / "fetch.py"), "--manifest", str(manifest_path(model)), "--directory", str(draft), "--draft", "--run"], env=env, check=True)
            (draft / "verification.json").write_text(json.dumps(verification_receipt(draft=True, model=model), indent=2) + "\n")
        if not (pack / "localmodel-pack.json").is_file():
            if speculative == "none" and (any(directory(model).parent.glob("dspark-*.gguf")) or
                    (directory(model).parent / "dspark/dspark.json").exists()):
                raise RuntimeError("The stock packer found an external draft beside the model. Move it before rebuilding the baseline pack.")
            draft_args = (["--dspark-gguf", str(drafter_directory(model) / manifest(model)["dspark"]["name"])]
                          if speculative == "dspark" else [])
            subprocess.run([str(CAPPED), "--mem", "4G", "--time", "10m", str(PYTHON),
                            str(PACK_SOURCE / "tools/dsv41/pack_gguf.py"), "--model", str(directory(model)),
                            "--out", str(pack), "--shards", *[item["name"] for item in manifest(model)["files"]], *draft_args],
                           env=env, check=True)
            record = {"model_revision": manifest(model)["model_revision"], "source_revision": manifest(model).get("pack_source_revision", manifest(model)["source_revision"]),
                      "draft": manifest(model)["dspark"] if speculative == "dspark" else None,
                      "files": {str(path.relative_to(pack)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in pack.rglob("*") if path.is_file() and str(path.relative_to(pack)) in PACK_ASSETS}}
            (pack / "localmodel-pack.json").write_text(json.dumps(record, indent=2) + "\n")
        check_pack(speculative, model)


def check_ready(speculative: str = "none", model: str = NAME) -> None:
    """Check build, model identities, and generated pack bytes before serving."""
    if not verified(model=model):
        raise RuntimeError(f"DeepSeek files lack current checksum evidence. Run localmodel fetch {model}.")
    if speculative == "dspark" and not verified(draft=True, model=model):
        raise RuntimeError(f"The DSpark draft lacks current checksum evidence. Run localmodel fetch {model} --speculative dspark.")
    subprocess.run([sys.executable, str(HERE / "runtime.py"), "--check"], env=environment(), check=True,
                   stdout=subprocess.DEVNULL)
    check_pack(speculative, model)


def remove(dry_run: bool = False, speculative: str = "none", model: str = NAME) -> None:
    """Delete only this model's data. Refuse linked or unexpected entries."""
    folder = directory(model)
    pack_directory(speculative, model)
    if not folder.exists():
        print("DeepSeek weights are already absent.")
        return
    draft = drafter_directory(model)
    if model == NAME:
        pack_files("dspark", model)
    targets = [draft, pack_directory("dspark", model)] if speculative == "dspark" else [folder]
    if speculative == "none":
        names = {item["name"] + suffix for item in manifest(model)["files"] for suffix in ("", ".part", ".moving")}
        folders = {"strata-pack", "strata-pack-dspark", "drafter"} if model == NAME else {"strata-pack"}
        names |= folders | {"verification.json"}
        if any(path.name not in names or path.is_symlink() or
               (not path.is_dir() if path.name in folders else not path.is_file()) for path in folder.iterdir()):
            raise RuntimeError("The DeepSeek directory contains unknown files or links. No files were deleted.")
        pack_files(model=model)
    inodes: dict[tuple[int, int], tuple[int, int, int]] = {}
    for target in targets:
        for path in target.rglob("*"):
            if path.is_file():
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino)
                count = inodes.get(key, (0, 0, 0))[0]
                inodes[key] = (count + 1, stat.st_nlink, stat.st_blocks * 512)
    size = sum(blocks for count, links, blocks in inodes.values() if count == links)
    print(f"{'Would remove' if dry_run else 'Removing'} {', '.join(map(str, targets))}; "
          f"{size / 2**30:.2f} GiB of unshared file data.", flush=True)
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
        receipts: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        if speculative == "none":
            for peer, item in shared_files(model):
                source, remaining = folder / item["name"], directory(peer) / item["name"]
                if source.is_file() and remaining.is_file() and source.samefile(remaining):
                    record = file_receipt(peer, item)
                    if record is not None:
                        receipts.append((peer, item, record))
        free_before = shutil.disk_usage(folder.parent).free
        for target in targets:
            if target.exists():
                shutil.rmtree(target)
        for peer, item, record in receipts:
            refresh_link_receipt(peer, item, record)
        free_change = shutil.disk_usage(folder.parent).free - free_before
    suffix = " --speculative dspark" if speculative == "dspark" else ""
    print(f"Deleted selected DeepSeek data. Restore it with localmodel fetch {model}{suffix}. "
          f"Observed free-space change: {free_change / 2**30:.2f} GiB.")


def settings(args: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
    """Generate the reviewed server settings from explicit launcher controls."""
    model = str(item["alias"])
    if args.no_offload:
        raise RuntimeError("DeepSeek needs RAM or SSD experts on this GPU. --no-offload cannot fit its weights.")
    if args.runtime != "stock" or args.speculative not in ("none", "dspark") or args.prompt_lookup:
        raise RuntimeError("This Strata profile uses its pinned runtime with --speculative none or dspark.")
    pack = pack_directory(args.speculative, model)
    confidence = getattr(args, "dspark_conf", None)
    if confidence is not None and args.speculative != "dspark":
        raise RuntimeError("--dspark-conf requires --speculative dspark.")
    if args.speculative == "dspark" and (not 1 <= args.draft_tokens <= 64 or
            (confidence is not None and not 0 <= confidence <= 1)):
        raise RuntimeError("Use 1 to 64 DSpark draft tokens and confidence from 0 to 1.")
    if args.cache_ram is not None or args.kv_cache or args.kv_mean_center or args.cpu_offload_gb or args.kv_offload_gb:
        raise RuntimeError("Use --expert-ram-gb for Strata. llama.cpp prompt/KV cache and Dory offload controls do not apply.")
    if any(getattr(args, key) is not None for key in ("cpu_moe_layers", "moe_cache_mib", "batch_tokens", "draft_window", "kv_vram_cells", "mma_decode_min_kv")):
        raise RuntimeError("Flint MoE, llama.cpp batching, and Bonsai draft/KV controls do not apply to Strata.")
    if args.gpu_expert_cache_gb is not None:
        raise RuntimeError("Official Strata uses --gpu-expert-fraction instead of --gpu-expert-cache-gb.")
    memory = getattr(args, "expert_memory", None) or manifest(model).get("expert_memory", "cache")
    if memory not in ("cache", "resident"):
        raise RuntimeError("Use cache or resident for --expert-memory.")
    if memory == "resident" and args.expert_ram_gb not in (None, 0):
        raise RuntimeError("--expert-memory resident requires --expert-ram-gb 0 or no explicit cache budget.")
    ram = (args.expert_ram_gb if args.expert_ram_gb is not None else manifest(model)["expert_ram_budget_gib"]) if memory == "cache" else 0
    threads = args.cpu_threads if args.cpu_threads is not None else 12
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
    slots = args.strata_slots if getattr(args, "strata_slots", None) is not None else True
    env = {"DS4_ENGRAM_CACHE_MB": str(engram), "DS4_ENGRAM_THREADS": str(readers),
           "STRATA_DS4_DISK_SKIP_W": "0", "STRATA_DS4_SLOT_KEEP": str(int(slots))}
    budget = args.thinking_budget if args.thinking_budget is not None else 32768
    if not 0 <= budget <= 1048576:
        raise RuntimeError("Use a thinking budget from 0 to 1048576 tokens.")
    context = args.context or item["context"]
    if not 1024 <= context <= 1048576:
        raise RuntimeError("Use 1024 to 1048576 context tokens for Strata.")
    info = manifest(model)
    if slots:
        # Keep incompatible weights, packs, runtimes, and context limits in separate directories.
        identity = {"weights": [file["sha256"] for file in info["files"]],
                    "draft": info["dspark"]["sha256"] if args.speculative == "dspark" else None,
                    "source": info["source_revision"],
                    "packer": info.get("pack_source_revision", info["source_revision"]),
                    "native": info["runtime"]["binary_sha256"],
                    "context": context}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        env.update(STRATA_DS4_SLOT_CAP="1", STRATA_DS4_SLOT_GB="2",
                   STRATA_DS4_SLOT_DIR=str(ROOT / "state/deepseek-slots" / key))
    native_args = ["--pack", str(pack), "--max-context", str(context),
                   "--ssd-cache", str(ram * 2**30 / 10**9) if memory == "cache" else "0", "--cpu-threads", str(threads),
                   "--reserve-mib", str(reserve), "--dspark", "on" if args.speculative == "dspark" else "none",
                   "--experts", "6", "--vision", "off"]
    if memory == "resident":
        native_args += ["--mmap-experts", "off"]
    if args.speculative == "dspark":
        native_args += ["--dspark-block", str(args.draft_tokens), "--dspark-conf", str(0.65 if confidence is None else confidence)]
    if fraction is not None:
        native_args += ["--vram-experts", str(fraction)]
    if chunk is not None:
        native_args += ["--prefill-chunk", str(chunk)]
    runtime = info["runtime"]
    binary = Path(runtime["directory"]) / "engine-dsv41" / f"{runtime['version']}-linux-x86_64-cuda-cuda13/strata-dsv41"
    return {"exe": str(binary), "arch": "deepseek41", "args": native_args,
            "expert_memory": memory, "expert_ram_gb": ram,
            "cwd": str(SOURCE), "tokenizer": str(pack / "tokenizer"),
            "model_name": item["alias"], "gpu": [0], "log": str(ROOT / "state/deepseek-engine.log"),
            "env": env, "lib_dirs": [str(binary.parent)],
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "reasoning_effort": None if (args.reasoning_effort or "xhigh") == "xhigh" else args.reasoning_effort,
            "reasoning_budget_tokens": budget, "fit_max_tokens": True, "api_monitor": False,
            "allowed_hosts": [], "trusted_origins": [], "cors_origins": [], "engine_ready_s": 3600}


def command(args: argparse.Namespace, item: dict[str, Any]) -> list[str]:
    """Build the local API command. The general --mem flag can override its cap."""
    model = str(item["alias"])
    settings(args, item)
    return [str(CAPPED), "--mem", str(manifest(model)["memory_limit"]), str(PYTHON), "-I", "-B", str(HERE / "network_guard.py"),
            str(PYTHON), "-I", "-B", str(SOURCE / "serve/server.py"),
            "--engine", "strata", "--config", str(ROOT / "state/deepseek-server.json"),
            "--host", "127.0.0.1", "--port", str(args.port), "--offline"]
