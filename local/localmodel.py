"""Fetch, remove, start, and inspect local models, including Groot and LFM3."""

from __future__ import annotations

import argparse
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from typing import Any, cast
import urllib.request

import presets
import deepseek_strata
import qwen_strata
import strata_monitor

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
CAPPED = ROOT / "libexec/capped"
ALIASES = {"inkling-2-mini": "inkling", "bonsai": "bonsai2-27b", "bonsai2": "bonsai2-27b", "bonsai-ptq2": "bonsai-pq2",
           "groot": "groot-19b", "lfm": "lfm3-1.7b", "lfm3": "lfm3-1.7b",
           "deepseek": deepseek_strata.NAME, "deepseek-full": deepseek_strata.NAME,
           "deepseek-coder": deepseek_strata.CODER_NAME, "deepseek-iq1-s": deepseek_strata.NAME, "strata": deepseek_strata.NAME}
DEFAULT_CACHE_RAM_MIB = 32768
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_START_MODEL = "inkling"
INKLING_PREPARE = ROOT / "experiments/inkling-mini/prepare.sh"
INKLING_UNSLOTH_PREPARE = INKLING_PREPARE.with_name("prepare-dspark.py")
INKLING_QUANTS = ("iq1_s", "iq2_xxs")
GGUF_RUNTIMES = ("flint", "llamacpp", "inkling")


def native_runtime_path(name: str) -> Path:
    """Return an isolated source-built runtime directory."""
    if name not in ("bonsai-native", "bonsai-next", "dory-native"):
        raise ValueError(f"Unknown native runtime: {name}.")
    return STATE / "builds" / name / "bin"


def verify_native_build(name: str) -> None:
    """Require a build record that matches the candidate manifest and binary."""
    binary = native_runtime_path(name) / "llama-server"
    record = read_json(binary.parent.parent / "localmodel-build.json")
    manifest = ROOT / "experiments" / name / "manifest.json"
    if record.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise RuntimeError(f"{name} was built from another manifest. Build the reviewed candidate again.")
    with binary.open("rb") as stream:
        if record.get("binary_sha256") != hashlib.file_digest(stream, "sha256").hexdigest():
            raise RuntimeError(f"{name} binary changed after its recorded build.")


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text())


def catalog() -> dict[str, dict[str, Any]]:
    """Read model download definitions and launch defaults."""
    entries = cast(dict[str, dict[str, Any]], read_json(ROOT / "config/models.json"))
    return {"dory": {"context": 262144, "alias": "dory", "runtime": "dory"}, **entries}


def tokens(value: str, maximum: int = 262144) -> int:
    """Read a token count with an optional binary K or M suffix."""
    multiplier = {"k": 1024, "m": 1024**2}.get(value[-1:].lower(), 1)
    try:
        count = int(value[:-1] if multiplier > 1 else value) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a token count such as 65536 or 256k.") from error
    if not 1 <= count <= maximum:
        raise argparse.ArgumentTypeError(f"Use 1 to {maximum} tokens.")
    return count


def memory_size(value: str) -> str:
    """Read a positive systemd memory size. Bare numbers use GiB."""
    if not re.fullmatch(r"[1-9][0-9]*[KMGT]?", value, re.IGNORECASE):
        raise argparse.ArgumentTypeError("Use a positive size such as 32, 32G, or 32768M. Bare numbers use GiB.")
    return value.upper() + ("G" if value[-1].isdigit() else "")


class NoMetadataRedirect(urllib.request.HTTPRedirectHandler):
    """Keep metadata requests at the selected loopback endpoint."""

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int,
                         msg: str, headers: Any, newurl: str) -> None:
        return None


def request(port: int, route: str) -> Any:
    """Read loopback metadata without a proxy or redirect."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoMetadataRedirect())
    with opener.open(f"http://127.0.0.1:{port}{route}", timeout=2) as response:
        body = response.read()
    return json.loads(body) if body else None


def endpoint(port: int) -> dict[str, Any]:
    """Read the live served model and its configured context capacity."""
    request(port, "/health")
    models = request(port, "/v1/models")["data"]
    if len(models) != 1:
        raise RuntimeError(f"Expected one loaded model at port {port}.")
    info = models[0]
    model = str(info["id"])
    context = info.get("max_model_len")
    backend = "dory" if context is not None and "dory" in str(info.get("root", model)).lower() else "llamacpp"
    if model in (*deepseek_strata.NAMES, qwen_strata.NAME):
        backend = "strata"
    if context is None:
        props = request(port, "/props")
        context = props["default_generation_settings"]["n_ctx"]
    if not isinstance(context, int) or context < 1:
        raise RuntimeError(f"The model at port {port} did not report a context capacity.")
    settings = next((item for item in catalog().values() if item["alias"] == model or item.get("iq2_xxs", {}).get("alias") == model), dict[str, Any]())
    return {
        "running": True, "model": model, "context": context,
        **({"quant": "iq1_s" if model == settings["alias"] else "iq2_xxs"} if settings.get("runtime") == "inkling" else {}),
        "base_url": f"http://127.0.0.1:{port}/v1", "port": port,
        "backend": backend, "output": min(32768 if "flint" in model.lower() else 16000, max(1, context // 2)),
        "temperature": settings.get("temperature", 0.6 if "saluki" in model else 1.0),
        "top_p": settings.get("top_p", 0.95),
        "reasoning_field": "reasoning" if backend == "dory" else "reasoning_content",
    }


def current(port: int | None = None) -> dict[str, Any]:
    """Find the running model. Use live metadata instead of saved capacity."""
    ports = [port] if port is not None else [8899, 8896]
    managed: dict[str, Any] = {}
    if (STATE / "active.json").exists():
        managed = cast(dict[str, Any], read_json(STATE / "active.json"))
        candidate = managed.get("port")
        if port is None and isinstance(candidate, int) and candidate not in ports:
            ports.insert(0, candidate)
    found: list[dict[str, Any]] = []
    for candidate in ports:
        try:
            found.append(endpoint(candidate))
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
    if not found:
        raise RuntimeError("No local model is ready. Run localmodel start first.")
    if len(found) > 1:
        raise RuntimeError("More than one model server is ready. Select one with --port PORT.")
    result = found[0]
    if managed.get("port") == result["port"] and managed.get("model") == result["model"]:
        result["preset"] = managed.get("preset")
        result["profile"] = managed.get("profile")
        result["settings_profile"] = managed.get("settings_profile", "default")
        result["pid"] = managed.get("controller_pid")
        if managed.get("profile") == "inkling":
            result["kv_device"] = managed.get("kv_device")
            result["fit_target_mib"] = managed.get("fit_target_mib")
        for key in ("cpu_offload_gb", "cpu_moe_layers", "gpu_layers", "moe_cache_mib", "batch_tokens", "ubatch_tokens", "command", "runtime", "rounds", "memory_limit",
                    "expert_memory", "expert_ram_gb", "cpu_threads", "gpu_expert_cache_gb", "gpu_expert_fraction", "prefill_chunk", "vram_reserve_mib", "engram_cache_mib", "engram_threads", "thinking_budget", "strata_slots", "strata_slot_directory", "ple_io", "expert_tail_skip", "adaptive_fetch", "draft_conf"):
            result[key] = managed.get(key)
        if result["backend"] == "dory":
            result["rounds"] = managed.get("rounds")
            result["kv_offload_gb"] = managed.get("kv_offload_gb", 0)
        else:
            for key in ("speculative", "draft_tokens", "dspark_conf", "draft_kv_cache", "mtp_chain", "mtp_head", "checkpoint_every", "replay", "gdn_mma", "slot_save_path", "prompt_lookup", "lookup_type", "lookup_tokens", "lookup_min_tokens", "draft_window", "kv_vram_cells", "mma_decode_min_kv", "cache_ram_mib", "kv_cache", "kv_mean_center", "reasoning_effort"):
                result[key] = managed.get(key)
        if managed.get("runtime") == "strata-qwen" and isinstance(managed.get("engine_log_offset"), int):
            result.update(qwen_strata.runtime_status(managed["engine_log_offset"]))
    return result


@contextmanager
def inkling_weight_lock(name: str) -> Generator[None, None, None]:
    """Keep the selected weights installed while Inkling serves requests."""
    directory = ROOT / "bonsai/state"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"models-{name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"{name} weights are being fetched or removed.") from None
        yield


def installed(name: str, item: dict[str, Any]) -> bool:
    """Check whether the selected model files are present."""
    if item.get("runtime") == "strata-qwen":
        return qwen_strata.installed()
    if item.get("runtime") == "strata":
        return deepseek_strata.installed(model=name)
    if name == "dory":
        snapshots = ROOT / "dory/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4/snapshots"
        return any(snapshots.glob("*/*.safetensors"))
    directory = ROOT / ("models" if name == "inkling" else "bonsai/models")
    path = directory / item["directory"] / item["filename"]
    return path.is_file() and path.stat().st_size == item["size"]


def runtime_path(runtime: str = "stock") -> Path:
    """Return the pinned runtime directory for the selected model family."""
    if runtime == "inkling":
        return STATE / "runtimes/inkling-mini-6196d98-scales/bin"
    if runtime == "inkling-unsloth":
        return STATE / "runtimes/inkling-unsloth-b12588-mix-c0c3de8"
    info = read_json(ROOT / "config" / (f"{runtime}-runtime.json" if runtime in ("flint", "llamacpp") else "runtime.json"))
    directory = STATE / "runtimes" / runtime if runtime in ("flint", "llamacpp") else ROOT / "bonsai/runtime"
    return directory / info["release"] / ("llama-" + info["release"])


def inkling_runtime(args: argparse.Namespace) -> str:
    """Use the pinned Unsloth build for profiles that request it, DSpark or automatic placement."""
    return "inkling-unsloth" if getattr(args, "unsloth_runtime", False) or args.speculative == "dspark" or args.moe_cache_mib == "auto" else "inkling"


def fetch_runtime(runtime: str = "stock") -> None:
    """Download and verify the pinned CUDA runtime."""
    if runtime == "inkling":
        subprocess.run(["bash", str(INKLING_PREPARE)], check=True)
        return
    if runtime == "inkling-unsloth":
        subprocess.run([sys.executable, str(INKLING_UNSLOTH_PREPARE)], check=True)
        return
    target = runtime_path(runtime)
    if (target / "llama-server").is_file():
        return
    info = read_json(ROOT / "config" / (f"{runtime}-runtime.json" if runtime in ("flint", "llamacpp") else "runtime.json"))
    downloads = STATE / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / info["filename"]
    valid = archive.is_file() and archive.stat().st_size == info["size"]
    if valid:
        with archive.open("rb") as stream:
            valid = hashlib.file_digest(stream, "sha256").hexdigest() == info["sha256"]
    if not valid:
        print(f"Fetching runtime {info['release']}...", flush=True)
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", str(archive), info["url"]], check=True)
        with archive.open("rb") as stream:
            if archive.stat().st_size != info["size"] or hashlib.file_digest(stream, "sha256").hexdigest() != info["sha256"]:
                raise RuntimeError("The runtime archive failed its size or SHA256 check.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(target.parent, filter="data")
    if not (target / "llama-server").is_file():
        raise RuntimeError(f"The runtime archive did not contain {target / 'llama-server'}.")
    print(f"Installed runtime {info['release']}. SHA256 verified.", flush=True)


def fetch(name: str, item: dict[str, Any], speculative: str = "none", mtp_head: str = "teacher",
          *, quant: str | None = None, unsloth_runtime: bool = False) -> None:
    """Fetch a pinned model and its required runtime."""
    if speculative == "dflash2" and item["runtime"] != "strata-qwen":
        raise RuntimeError("DFlash2 artifacts require Strata Qwen.")
    if item["runtime"] == "strata-qwen":
        if speculative not in ("none", "mtp", "dflash2"):
            raise RuntimeError("Qwen supports only none, mtp, or dflash2 artifacts.")
        qwen_strata.fetch("dflash2" if speculative == "dflash2" else "mtp")
        return
    if item["runtime"] == "strata":
        deepseek_strata.fetch(speculative, model=name)
        return
    if name == "dory":
        with (STATE / "models-dory.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another fetch or remove is running for dory.") from None
            if installed(name, item):
                print("Dory NVFP4 is already installed.")
                return
            subprocess.run([str(ROOT / "dory/setup.sh")], check=True)
        return
    if speculative != "none" and speculative not in item:
        raise RuntimeError(f"{name} has no pinned {speculative} artifact.")
    variant_name = "mtp_procreations" if speculative == "mtp" and mtp_head == "procreations" else speculative
    variants = [None, speculative] if speculative in ("dflash", "dspark") else [None if speculative == "none" else variant_name]
    if name == "inkling" and quant == "iq2_xxs":
        variants[0] = "iq2_xxs"
    for variant in variants:
        if variant == "mtp_procreations":
            output = item["mtp_procreations"]
            path = ROOT / "bonsai/models" / output["directory"] / output["filename"]
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT / "bonsai/models"):
                raise RuntimeError("Use a regular trained-head file inside the managed model directory.")
            if not path.exists():
                fetch(name, item)
            with (ROOT / "bonsai/state" / f"models-{name}.lock").open("w") as trunk_lock, \
                    (ROOT / "bonsai/state" / f"models-{name}-{variant}.lock").open("w") as output_lock:
                try:
                    fcntl.flock(trunk_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("Another fetch or remove is using the trained head or its target.") from None
                if not path.exists():
                    subprocess.run([str(CAPPED), "--mem", "4G", "--time", "30m", str(ROOT / ".venv/bin/python"),
                                    str(ROOT / "experiments/bonsai-next/fetch_head.py"),
                                    "--trunk", str(ROOT / "bonsai/models" / item["directory"] / item["filename"]),
                                    "--output", str(path), "--run"], check=True)
                if not installed(name, output):
                    raise RuntimeError("The generated trained-head artifact has the wrong size.")
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != output["sha256"]:
                        raise RuntimeError("The generated trained-head artifact failed its SHA256 check.")
                print("Trained MTP weights are installed. SHA256 verified.", flush=True)
            continue
        command = [str(ROOT / ".venv/bin/python"), str(ROOT / "libexec/bonsai2-models"), "pull", name]
        if variant:
            command += ["--variant", variant]
        subprocess.run(command, check=True)
    if item["runtime"] == "prism" or item["runtime"] in GGUF_RUNTIMES:
        with (STATE / "runtime.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            fetch_runtime("inkling-unsloth" if name == "inkling" and (speculative == "dspark" or unsloth_runtime)
                          else "stock" if item["runtime"] == "prism" else item["runtime"])


def remove(name: str, speculative: str = "none", dry_run: bool = False, mtp_head: str = "teacher",
           *, quant: str | None = None) -> None:
    """Delete selected weights and download data. Keep configuration and runtimes."""
    if speculative == "dflash2" and name != qwen_strata.NAME:
        raise RuntimeError("DFlash2 artifacts require Strata Qwen.")
    if name == qwen_strata.NAME:
        if speculative not in ("none", "mtp", "dflash2"):
            raise RuntimeError("Qwen supports only none, mtp, or dflash2 artifacts.")
        qwen_strata.remove(dry_run, "dflash2" if speculative == "dflash2" else "mtp")
        return
    if name in deepseek_strata.NAMES:
        deepseek_strata.remove(dry_run, speculative, model=name)
        return
    if name == "dory":
        hub = ROOT / "dory/.cache/huggingface/hub"
        repository = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4"
        path = hub / ("models--" + repository.replace("/", "--"))
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT / "dory/.cache"):
            raise RuntimeError(f"Use a model cache directory inside {hub}: {path}")
        with (STATE / "models-dory.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another fetch or remove is running for dory.") from None
            for process in Path("/proc").iterdir():
                if not process.name.isdecimal():
                    continue
                try:
                    if process.stat().st_uid != os.getuid():
                        continue
                    arguments = (process / "cmdline").read_bytes().split(b"\0")
                    if any(os.fsdecode(argument) == repository or
                           (argument.startswith(b"/") and Path(os.fsdecode(argument)).is_relative_to(path))
                           for argument in arguments):
                        raise RuntimeError(f"Dory is in use by PID {process.name}. Stop it before removing the weights.")
                except OSError:
                    continue
            size = sum(entry.stat().st_size for entry in path.rglob("*")
                       if entry.is_file() and not entry.is_symlink())
            if dry_run:
                print(f"Would remove {path} ({size / 2**30:.2f} GiB).")
            else:
                if path.exists():
                    shutil.rmtree(path)
                print(f"Deleted dory: removed {size / 2**30:.2f} GiB. Restore with localmodel fetch dory.")
        return
    variant_name = "mtp_procreations" if speculative == "mtp" and mtp_head == "procreations" else speculative
    variants = [None, "dflash"] if speculative == "dflash" else [None if speculative == "none" else variant_name]
    if name == "inkling" and quant == "iq2_xxs" and speculative == "none":
        variants = ["iq2_xxs"]
    for variant in variants:
        command = [str(ROOT / ".venv/bin/python"), str(ROOT / "libexec/bonsai2-models"), "delete", name]
        if variant:
            command += ["--variant", variant]
        if dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)


def launch_command(args: argparse.Namespace, item: dict[str, Any]) -> list[str]:
    """Build a command with GPU weights and GPU KV cache."""
    if args.gpu_layers is not None and (item["runtime"] != "llamacpp" or "cpu_moe_layers" in item):
        raise RuntimeError("--gpu-layers requires a dense llama.cpp profile. Use --cpu-moe-layers for MoE profiles.")
    if item["runtime"] == "strata-qwen":
        return qwen_strata.command(args, item)
    if item["runtime"] == "strata":
        return deepseek_strata.command(args, item)
    context = args.context or item["context"]
    cpu_moe_layers = getattr(args, "cpu_moe_layers", None)
    if args.runtime != "stock" and item["runtime"] != "prism":
        raise RuntimeError("--runtime selects an alternate Bonsai Prism runtime.")
    if item["runtime"] not in ("prism", "dory-native", *GGUF_RUNTIMES) and (args.speculative != "none" or args.prompt_lookup or args.cache_ram is not None or args.kv_mean_center or args.kv_cache or args.reasoning_effort):
        raise RuntimeError("These runtime controls require a new Bonsai profile.")
    if any(value is not None for value in (args.draft_window, args.kv_vram_cells, args.mma_decode_min_kv)) and args.runtime not in ("bonsai-native", "bonsai-next"):
        raise RuntimeError("Draft windows, active KV spill, and MMA controls require a native Bonsai runtime.")
    if any(value is not None for value in (args.checkpoint_every, args.replay, args.gdn_mma)) and args.runtime != "bonsai-next":
        raise RuntimeError("Checkpoints, replay rollback, and chunked GDN controls require --runtime bonsai-next.")
    if args.slot_save_path is not None and item["runtime"] != "prism":
        raise RuntimeError("--slot-save-path requires a Bonsai Prism model.")
    if args.draft_window is not None and args.speculative != "mtp":
        raise RuntimeError("--draft-window requires --speculative mtp.")
    if args.lookup_type != "ngram-simple" and not args.prompt_lookup:
        raise RuntimeError("--lookup-type requires --prompt-lookup.")
    if (cpu_moe_layers is not None or args.moe_cache_mib is not None) and "cpu_moe_layers" not in item:
        raise RuntimeError("CPU expert layers and GPU expert caching require a Flint or Groot MoE profile.")
    if cpu_moe_layers is not None and cpu_moe_layers > item.get("layers", 49):
        raise RuntimeError(f"Use at most {item.get('layers', 49)} CPU expert layers for this model.")
    if args.lookup_min_tokens is not None and (not args.prompt_lookup or args.lookup_type != "ngram-mod"):
        raise RuntimeError("--lookup-min-tokens requires --prompt-lookup --lookup-type ngram-mod.")
    if item["runtime"] in GGUF_RUNTIMES:
        allowed_draft = "dspark" if item["runtime"] == "inkling" else "mtp"
        if args.speculative != "none" and (args.speculative != allowed_draft or allowed_draft not in item):
            raise RuntimeError("This profile has no configured artifact for the selected draft decoder.")
        if args.kv_mean_center or (args.reasoning_effort and "reasoning_effort" not in item):
            raise RuntimeError("This profile has no configured KV calibration or reasoning-effort control.")
        if args.cpu_offload_gb or args.kv_offload_gb:
            raise RuntimeError("Use --cpu-moe-layers for MoE profiles. Attention and KV stay on the GPU.")
        cpu_moe = 0 if args.no_offload else cpu_moe_layers if cpu_moe_layers is not None else item.get("cpu_moe_layers", 0)
        cache_ram = args.cache_ram if args.cache_ram is not None else item.get("cache_ram", DEFAULT_CACHE_RAM_MIB)
        weights = item["mtp"] if args.speculative == "mtp" else item
        directory = ROOT / ("models" if item["runtime"] == "inkling" else "bonsai/models")
        model = directory / weights["directory"] / weights["filename"]
        if item["runtime"] == "inkling":
            host_kv = getattr(args, "host_kv", False)
            if host_kv and args.no_offload:
                raise RuntimeError("The Inkling 1M placement keeps KV in RAM and cannot use --no-offload.")
            placement = ".*=CUDA0" if args.no_offload else getattr(args, "tensor_placement", ".*=CUDA0")
            auto_fit = args.moe_cache_mib == "auto"
            if not auto_fit and args.moe_cache_mib and not (cpu_moe or "_exps" in placement):
                raise RuntimeError("The Inkling MoE cache requires routed experts in RAM. Select a host-expert profile or use --cpu-moe-layers.")
            fit = (["--gpu-layers", "auto", "--fit", "on", "--fit-ctx", str(context), "--fit-target", str(args.fit_target_mib)]
                   if auto_fit else ["--gpu-layers", "999", "--fit", "off", "--n-cpu-moe", str(cpu_moe), "--override-tensor", placement])
            draft = ["--spec-type", "none"]
            if args.speculative == "dspark":
                artifact = item["dspark"]
                draft = ["--spec-type", "draft-dspark", "--model-draft", str(directory / artifact["directory"] / artifact["filename"]),
                         "--spec-draft-n-max", str(args.draft_tokens), "--n-gpu-layers-draft", "99", "--spec-draft-p-min", "0",
                         "--cache-type-k-draft", args.draft_kv_cache, "--cache-type-v-draft", args.draft_kv_cache]
            return [str(CAPPED), "--mem", f"{item['memory_mib']}M", sys.executable,
                    str(ROOT / "experiments/deepseek-strata/network_guard.py"),
                    str(runtime_path(inkling_runtime(args)) / "llama-server"), "--model", str(model), "--alias", item["alias"],
                    "--host", "127.0.0.1", "--port", str(args.port), *fit,
                    "--load-mode", "none" if auto_fit else getattr(args, "load_mode", "auto"),
                    "--parallel", "1", "--ctx-size", str(context),
                    "--cache-type-k", args.kv_cache or item["kv_cache"], "--cache-type-v", args.kv_cache or item["kv_cache"],
                    "--flash-attn", "auto", "--batch-size", str(args.batch_tokens or item["batch_tokens"]),
                    "--ubatch-size", str(args.ubatch_tokens), "--threads", "4", "--jinja", "--offline", "--no-webui",
                    "--cache-ram", str(cache_ram), "--moe-cache-mib", str(args.moe_cache_mib or 0),
                    *(["--no-kv-offload"] if host_kv else []), *draft]
        if args.moe_cache_mib and cpu_moe < 3:
            raise RuntimeError("The MoE cache needs routed experts in RAM. Use --cpu-moe-layers 3 or more and omit --no-offload.")
        gpu_layers = 99 if args.no_offload else args.gpu_layers if args.gpu_layers is not None else item.get("gpu_layers", 99)
        command = [str(runtime_path(item["runtime"]) / "llama-server"), "--model", str(model),
                   "--alias", item["alias"], "--ctx-size", str(context),
                   "--gpu-layers", str(gpu_layers), "--n-cpu-moe", str(cpu_moe), "--fit", "on" if gpu_layers == "auto" else "off",
                   "--flash-attn", "on", "--parallel", "1",
                   "--cache-type-k", args.kv_cache or item["kv_cache"],
                   "--cache-type-v", args.kv_cache or item["kv_cache"],
                   "--host", "127.0.0.1", "--port", str(args.port), "--metrics", "--jinja",
                   "--offline", "--no-context-shift", "--load-mode", "none",
                   "--batch-size", str(args.batch_tokens or 2048), "--ubatch-size", str(args.ubatch_tokens),
                   "--cache-ram", str(cache_ram), "--reasoning", "on",
                   "--temp", str(item.get("temperature", 1.0)), "--top-p", str(item.get("top_p", 0.95)),
                   "--top-k", str(item.get("top_k", 20)), "--min-p", "0.0",
                   "--repeat-penalty", str(item.get("repeat_penalty", 1.0))]
        if args.moe_cache_mib is not None:
            command += ["--moe-cache-mib", str(args.moe_cache_mib)]
        if "reasoning_effort" in item:
            command += ["--reasoning-effort", args.reasoning_effort or item["reasoning_effort"]]
        speculation: list[str] = []
        if args.speculative == "mtp":
            speculation.append("draft-mtp")
            command += ["--spec-draft-n-max", str(args.draft_tokens)]
        if args.prompt_lookup:
            speculation.append(args.lookup_type)
        if speculation or args.prompt_lookup is not None:
            command += ["--spec-type", ",".join(speculation) or "none"]
        if args.prompt_lookup and args.lookup_type == "ngram-mod":
            command += ["--spec-ngram-mod-n-max", str(args.lookup_tokens),
                        "--spec-ngram-mod-n-min", str(args.lookup_min_tokens)]
        memory_mib = item.get("memory_mib", 32768 + cache_ram)
        if "memory_mib" in item:
            memory_mib += max(0, cache_ram - item["cache_ram"])
        return [str(CAPPED), "--mem", f"{memory_mib}M", *command]
    if item["runtime"] == "dory-native":
        if args.speculative != "none" or args.prompt_lookup or args.kv_mean_center or args.reasoning_effort:
            raise RuntimeError("Dory GGUF does not support Bonsai speculation, KV calibration, or reasoning-effort controls.")
        if args.cpu_offload_gb or args.kv_offload_gb:
            raise RuntimeError("Dory GGUF keeps weights and active KV on the GPU. Host block caching is a vLLM control.")
        model = ROOT / "bonsai/models" / item["directory"] / item["filename"]
        command = [str(native_runtime_path("dory-native") / "llama-server"), "--model", str(model),
                   "--alias", item["alias"], "--ctx-size", str(context),
                   "--gpu-layers", "999", "--fit", "off", "--flash-attn", "on", "--parallel", "1",
                   "--cache-type-k", args.kv_cache or "q8_0", "--cache-type-v", args.kv_cache or "q8_0",
                   "--host", "127.0.0.1", "--port", str(args.port), "--metrics", "--jinja",
                   "--override-kv", f"dory.num_loops=int:{args.rounds},dory.recurrent_kv_cache_mode=str:last_loop",
                   "--override-tensor", r"^blk\.[0-9]+\.(residual_alpha|attn_qk_scale|attn_gate_scale|ffn_gate_scale|ffn_up_scale)\.weight$=CUDA0,^output_scale\.weight$=CUDA0,^token_embd\.weight$=CUDA0",
                   "--offline", "--no-context-shift",
                   "--load-mode", "none", "--threads", "1",
                   "--batch-size", str(args.batch_tokens or 2048), "--ubatch-size", str(args.ubatch_tokens),
                   "--cache-ram", str(args.cache_ram if args.cache_ram is not None else DEFAULT_CACHE_RAM_MIB),
                   "--temp", "1.0", "--top-p", "0.95", "--top-k", "0", "--min-p", "0.0",
                   "--repeat-penalty", "1.0", "--reasoning", "on",
                   "--chat-template-kwargs", '{"force_nonempty_content":true}']
        return [str(CAPPED), "--mem", f"{16384 + (args.cache_ram if args.cache_ram is not None else DEFAULT_CACHE_RAM_MIB)}M", *command]
    if item["runtime"] == "dory":
        command = [str(ROOT / "dory/bin/dory"), "serve", "--context", str(context),
                   "--port", str(args.port), "--rounds", str(args.rounds)]
        if args.cpu_offload_gb:
            command += ["--cpu-offload-gb", str(args.cpu_offload_gb)]
        else:
            command += ["--no-offload"]
        if args.kv_offload_gb:
            command += ["--kv-offload-gb", str(args.kv_offload_gb)]
        if args.batch_tokens:
            command += ["--batch-tokens", str(args.batch_tokens)]
        return command
    if args.cpu_offload_gb:
        raise RuntimeError("CPU offloading is supported only by the Dory backend.")
    if args.kv_offload_gb:
        raise RuntimeError("Host KV caching is supported only by the Dory backend.")
    if item["runtime"] == "legacy":
        command = [str(ROOT / "bonsai/local/bonsai2-server"), "--model", args.model,
                   "--context", str(context), "--port", str(args.port)]
    else:
        if args.speculative != "none" and args.speculative not in item:
            raise RuntimeError(f"{args.model} has no pinned {args.speculative} artifact.")
        weights = item["mtp_procreations" if args.mtp_head == "procreations" else "mtp"] if args.speculative == "mtp" else item
        model = ROOT / "bonsai/models" / weights["directory"] / weights["filename"]
        selected_runtime = native_runtime_path(args.runtime) if args.runtime != "stock" else runtime_path()
        command = [str(selected_runtime / "llama-server"), "--model", str(model),
                   "--alias", item["alias"], "--ctx-size", str(context),
                   "--gpu-layers", "99", "--tensor-split", "1", "--fit", "off", "--flash-attn", "on", "--parallel", "1",
                   "--cache-type-k", args.kv_cache or item["kv_cache"],
                   "--cache-type-v", args.kv_cache or item["kv_cache"],
                   "--host", "127.0.0.1", "--port", str(args.port), "--metrics",
                   "--jinja", "--reasoning-effort", args.reasoning_effort or DEFAULT_REASONING_EFFORT, "--temp", "1.0",
                   "--top-p", "0.95", "--top-k", "20"]
        command += ["--cache-ram", str(args.cache_ram if args.cache_ram is not None else DEFAULT_CACHE_RAM_MIB)]
        speculation: list[str] = []
        if args.prompt_lookup:
            speculation.append(args.lookup_type)
            if args.lookup_type == "ngram-mod":
                command += ["--spec-ngram-mod-n-max", str(args.lookup_tokens)]
                if args.lookup_min_tokens is not None:
                    command += ["--spec-ngram-mod-n-min", str(args.lookup_min_tokens)]
        if args.speculative != "none":
            speculation.append("draft-" + args.speculative)
            command += ["--spec-draft-n-max", str(args.draft_tokens), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0"]
            command += ["--cache-type-k-draft", args.draft_kv_cache, "--cache-type-v-draft", args.draft_kv_cache]
        if args.speculative == "dflash":
            draft = item["dflash"]
            command += ["--model-draft", str(ROOT / "bonsai/models" / draft["directory"] / draft["filename"]),
                        "--gpu-layers-draft", "99"]
        if speculation:
            command += ["--spec-type", ",".join(speculation)]
        if args.draft_window is not None:
            command += ["--spec-draft-window", str(args.draft_window)]
        if args.kv_vram_cells is not None:
            command += ["--kv-vram-cells", str(args.kv_vram_cells)]
        if args.checkpoint_every is not None:
            command += ["--checkpoint-every-nt", str(args.checkpoint_every)]
        if args.slot_save_path is not None:
            if not args.dry_run and not args.slot_save_path.is_dir():
                raise RuntimeError("--slot-save-path must name an existing directory.")
            command += ["--slot-save-path", str(args.slot_save_path.resolve()), "--slots"]
    command += ["--batch-size", str(args.batch_tokens or 256), "--ubatch-size", str(args.ubatch_tokens)]
    if args.kv_mean_center:
        if (args.kv_cache or item["kv_cache"]) != "q4_0":
            raise RuntimeError("KV mean calibration requires --kv-cache q4_0.")
        bias = STATE / "calibration" / f"{args.model}-{args.kv_cache or item['kv_cache']}.gguf" if str(args.kv_mean_center) == "auto" else args.kv_mean_center
        if not args.dry_run and not bias.is_file():
            raise RuntimeError(f"KV calibration is missing: {bias}. Run localmodel calibrate {args.model} --text FILE.")
        command += ["--kv-mean-center", str(bias)]
    memory_limit_mib = 16384 if args.model == "bonsai2-27b" else 12288
    if item["runtime"] == "prism":
        cache_ram_mib = args.cache_ram if args.cache_ram is not None else DEFAULT_CACHE_RAM_MIB
        # Increase the process cap by the added snapshot cache budget.
        memory_limit_mib += max(0, cache_ram_mib - 8192)
        if args.kv_vram_cells is not None:
            memory_limit_mib += 8192
    return [str(CAPPED), "--mem", f"{memory_limit_mib}M", *command]


def check_free(port: int = 8899) -> None:
    """Refuse GPU work while a managed or known model server is running."""
    ports = {8896, 8899, port}
    if (STATE / "active.json").exists():
        ports.add(read_json(STATE / "active.json").get("port", port))
    for candidate in ports:
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", candidate)) == 0:
                raise RuntimeError(f"Port {candidate} is already in use. Stop that model server before starting another GPU job.")


def prism_environment(runtime: str = "stock") -> dict[str, str]:
    """Use private CUDA libraries and matching KV rotation for model jobs."""
    environment = os.environ.copy()
    if runtime == "inkling":
        libraries = [str(runtime_path(runtime))]
    elif runtime in ("stock", "flint", "llamacpp"):
        libraries = [str(runtime_path(runtime)), str(ROOT / "bonsai/runtime/bonsai2/lib")]
    else:
        libraries = [str(native_runtime_path(runtime)), str(native_runtime_path(runtime).parent / "lib")]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    if runtime not in ("stock", *GGUF_RUNTIMES):
        cuda_libraries = ROOT / "dory/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
        environment["LD_LIBRARY_PATH"] += ":" + str(cuda_libraries)
    environment.pop("LLAMA_ATTN_ROT_DISABLE", None)
    return environment


def calibrate(args: argparse.Namespace, item: dict[str, Any]) -> None:
    """Generate a model-specific KV bias from local calibration text."""
    if item["runtime"] != "prism":
        raise RuntimeError("KV calibration requires a new Bonsai profile.")
    kv = args.kv_cache or item["kv_cache"]
    model = ROOT / "bonsai/models" / item["directory"] / item["filename"]
    output = args.output or STATE / "calibration" / f"{args.model}-{kv}.gguf"
    command = [str(CAPPED), "--mem", "12G", str(runtime_path() / "llama-kv-mean-center"),
               "--model", str(model), "--file", str(args.text), "--output", str(output),
               "--gpu-layers", "99", "--ctx-size", "512", "--batch-size", "512", "--ubatch-size", "64",
               "--cache-type-k", kv, "--cache-type-v", kv, "--flash-attn", "on", "--chunks", str(args.chunks)]
    if args.dry_run:
        print(shlex.join(command))
        return
    if not args.text.is_file() or not installed(args.model, item):
        raise RuntimeError("Install the model and select an existing calibration text file.")
    STATE.mkdir(exist_ok=True)
    with (STATE / "server.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another localmodel GPU job is running. Stop it before calibration.") from None
        check_free()
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, env=prism_environment(), check=True)
    print(f"Saved {output}. Start with --kv-mean-center {output}.")


def wait_with_monitor(process: subprocess.Popen[bytes], port: int, console: Path) -> int:
    """Show telemetry after startup and retain server ownership after detaching."""
    print(f"Starting Strata. Startup output: {console}", flush=True)
    monitor: subprocess.Popen[bytes] | None = None
    try:
        while process.poll() is None:
            try:
                request(port, "/health")
                break
            except (OSError, ValueError):
                time.sleep(0.5)
        if process.poll() is not None:
            return process.returncode
        monitor = subprocess.Popen(strata_monitor.command(port))
        while process.poll() is None:
            if monitor.poll() is not None:
                if monitor.returncode == 130:
                    raise KeyboardInterrupt
                if monitor.returncode != 0:
                    raise RuntimeError(f"The Strata monitor exited with code {monitor.returncode}.")
                print(f"Monitor detached. Strata is still running. Ctrl-C stops it. Output: {console}", flush=True)
                return process.wait()
            time.sleep(0.2)
        return process.returncode
    finally:
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        if process.poll() is not None:
            print(f"Strata exited with code {process.returncode}. Output: {console}", flush=True)


def stop_child(process: subprocess.Popen[bytes]) -> None:
    """Stop the owned server process group before releasing its metadata."""
    for number, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError(f"The server process group {process.pid} did not stop.")


def serve(args: argparse.Namespace, item: dict[str, Any]) -> int:
    """Start one model and retain its metadata until the process stops."""
    strata = qwen_strata if item["runtime"] == "strata-qwen" else deepseek_strata
    command = launch_command(args, item)
    if args.mem is not None:
        if command[:2] == [str(CAPPED), "--mem"]:
            command = command[3:]
        command = [str(CAPPED), "--mem", args.mem, *command]
    environment = (prism_environment(args.runtime) if item["runtime"] == "prism" else
                   prism_environment(item["runtime"]) if item["runtime"] in GGUF_RUNTIMES else
                   prism_environment("dory-native") if item["runtime"] == "dory-native" else
                   strata.environment() if item["runtime"] in ("strata", "strata-qwen") else os.environ.copy())
    if item["runtime"] == "inkling":
        environment.update(HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4")
    if item["runtime"] in GGUF_RUNTIMES:
        if args.prompt_lookup is not None:
            environment.pop("LLAMA_ARG_SPEC_TYPE", None)
        if args.moe_cache_mib is not None:
            environment.pop("LLAMA_ARG_MOE_CACHE_MIB", None)
    if args.mma_decode_min_kv is not None:
        environment["GGML_CUDA_FA_MMA_DECODE_MIN_KV"] = str(args.mma_decode_min_kv)
    if args.runtime == "bonsai-next":
        environment["LLAMA_RS_REPLAY"] = str(int(bool(args.replay)))
        if args.gdn_mma is not None:
            if args.gdn_mma:
                environment.pop("GGML_CUDA_GDN_MMA_DISABLE", None)
            else:
                environment["GGML_CUDA_GDN_MMA_DISABLE"] = "1"
    mtp = item["runtime"] == "prism" and args.speculative == "mtp"
    if mtp:
        chain = str(int(args.mtp_chain)) if args.mtp_chain is not None else environment.get("LLAMA_MTP_CHAIN", "1")
        if chain not in ("0", "1"):
            raise RuntimeError("LLAMA_MTP_CHAIN must be 0 or 1. Use --mtp-chain or --no-mtp-chain to override it.")
        environment["LLAMA_MTP_CHAIN"] = chain
    if args.dry_run:
        if item["runtime"] in ("strata", "strata-qwen"):
            print(json.dumps({"command": command, "settings_profile": args.settings_profile,
                              **({"monitor_command": strata_monitor.command(args.port)} if args.tui else {}),
                              "configuration": strata.settings(args, item)}, indent=2))
            return 0
        variables = ([f"LLAMA_MTP_CHAIN={environment['LLAMA_MTP_CHAIN']}"] if mtp else [])
        if args.mma_decode_min_kv is not None:
            variables.append(f"GGML_CUDA_FA_MMA_DECODE_MIN_KV={args.mma_decode_min_kv}")
        if args.runtime == "bonsai-next":
            variables.append(f"LLAMA_RS_REPLAY={environment['LLAMA_RS_REPLAY']}")
            if args.gdn_mma is False:
                variables.append(f"GGML_CUDA_GDN_MMA_DISABLE={environment['GGML_CUDA_GDN_MMA_DISABLE']}")
        unset: list[str] = []
        if args.runtime == "bonsai-next" and args.gdn_mma:
            unset += ["-u", "GGML_CUDA_GDN_MMA_DISABLE"]
        if item["runtime"] in GGUF_RUNTIMES:
            if args.prompt_lookup is not None:
                unset += ["-u", "LLAMA_ARG_SPEC_TYPE"]
            if args.moe_cache_mib is not None:
                unset += ["-u", "LLAMA_ARG_MOE_CACHE_MIB"]
        prefix = ["env", *unset, *variables] if unset or variables else []
        print(shlex.join(prefix + command))
        return 0
    if args.tui and not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise RuntimeError("--tui requires an interactive terminal. Use --dry-run to preview the command.")
    fetch_hint = f"localmodel fetch {args.model}"
    if item["runtime"] == "inkling":
        fetch_hint += f" {args.settings_profile} --quant {args.quant or 'iq1_s'} --speculative {args.speculative}"
        if args.moe_cache_mib == "auto":
            fetch_hint += " --moe-cache-mib auto"
    weights = item["mtp_procreations" if args.mtp_head == "procreations" else "mtp"] if args.speculative == "mtp" and item["runtime"] != "strata-qwen" else item
    if not installed(args.model, weights):
        if item["runtime"] == "inkling":
            raise RuntimeError(f"Model is missing. Run {fetch_hint} first.")
        suffix = f" --speculative {args.speculative}" if args.speculative != "none" else ""
        if args.mtp_head == "procreations":
            suffix += " --mtp-head procreations"
        raise RuntimeError(f"Model is missing. Run localmodel fetch {args.model}{suffix} first.")
    if args.speculative == "dflash" and not installed(args.model, item["dflash"]):
        raise RuntimeError(f"Optional weights are missing. Run localmodel fetch {args.model} --speculative {args.speculative}.")
    if item["runtime"] in ("prism", "dory-native", *GGUF_RUNTIMES):
        runtime = (inkling_runtime(args) if item["runtime"] == "inkling"
                   else args.runtime if item["runtime"] == "prism" else item["runtime"])
        binary = (runtime_path(runtime) if runtime in ("stock", "inkling-unsloth", *GGUF_RUNTIMES) else native_runtime_path(runtime)) / "llama-server"
        if not binary.is_file():
            hint = fetch_hint if runtime in ("stock", "inkling-unsloth", *GGUF_RUNTIMES) else f"python experiments/build-native.py {runtime} --run"
            raise RuntimeError(f"Runtime is missing. Run {hint} first.")
        if runtime == "inkling-unsloth":
            environment["LD_LIBRARY_PATH"] = subprocess.check_output(
                [sys.executable, str(INKLING_UNSLOTH_PREPARE), "--library-path"], text=True).strip()
        elif runtime == "inkling":
            subprocess.run(["bash", str(INKLING_PREPARE), "--verify"], check=True)
        elif runtime not in ("stock", *GGUF_RUNTIMES):
            verify_native_build(runtime)
    STATE.mkdir(exist_ok=True)
    storage_lock = strata.model_lock(shared=True) if item["runtime"] in ("strata", "strata-qwen") else nullcontext()
    draft_lock = nullcontext()
    if item["runtime"] == "inkling":
        iq2 = args.quant == "iq2_xxs"
        storage_lock = inkling_weight_lock("inkling-iq2_xxs" if iq2 else "inkling")
        # The IQ1 target and the DSpark drafter share one download lock.
        if iq2 and args.speculative == "dspark":
            draft_lock = inkling_weight_lock("inkling")
    with (STATE / "server.lock").open("w") as lock, storage_lock, draft_lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another localmodel server is running. Stop it before switching models.") from None
        check_free(args.port)
        if item["runtime"] == "inkling":
            if not installed(args.model, item):
                raise RuntimeError(f"Inkling weights are missing. Run {fetch_hint}.")
            if args.speculative == "dspark":
                artifact = item["dspark"]
                path = ROOT / "models" / artifact["directory"] / artifact["filename"]
                if path.is_symlink() or not installed(args.model, artifact):
                    raise RuntimeError(f"Inkling DSpark weights are missing or invalid. Run {fetch_hint}.")
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != artifact["sha256"]:
                        raise RuntimeError(f"Inkling DSpark weights failed SHA256 verification. Run {fetch_hint}.")
        strata_config: dict[str, Any] = {}
        if item["runtime"] in ("strata", "strata-qwen"):
            if item["runtime"] == "strata":
                deepseek_strata.check_ready(args.speculative, model=args.model)
            else:
                qwen_strata.check_ready(args.speculative, args.cache_reuse_fix)
            if args.tui:
                strata_monitor.ensure()
            config_name = "qwen-server.json" if item["runtime"] == "strata-qwen" else "deepseek-server.json"
            strata_config = strata.settings(args, item)
            if item["runtime"] == "strata" and strata_config["env"]["STRATA_DS4_SLOT_KEEP"] == "1":
                slot_directory = Path(strata_config["env"]["STRATA_DS4_SLOT_DIR"])
                for directory in (slot_directory.parent, slot_directory):
                    directory.mkdir(mode=0o700, exist_ok=True)
                    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        os.fchmod(descriptor, 0o700)
                    finally:
                        os.close(descriptor)
            (STATE / config_name).write_text(json.dumps(strata_config, indent=2) + "\n")
        qwen_log = STATE / "qwen-engine.log"
        log_offset = qwen_log.stat().st_size if item["runtime"] == "strata-qwen" and qwen_log.exists() else 0
        console = STATE / ("qwen-console.log" if item["runtime"] == "strata-qwen" else "deepseek-console.log")
        with console.open("a") if args.tui else nullcontext() as output:
            process = subprocess.Popen(command, env=environment, start_new_session=True,
                                       stdout=output, stderr=subprocess.STDOUT if args.tui else None)
        active = STATE / "active.json"
        active.write_text(json.dumps({"pid": process.pid, "controller_pid": os.getpid(), "profile": args.model,
            "command": command,
            "memory_limit": command[2] if command[:2] == [str(CAPPED), "--mem"] else None,
            "gpu_layers": (("auto" if args.moe_cache_mib == "auto" else 999) if item["runtime"] == "inkling" else
                           (99 if args.no_offload else args.gpu_layers if args.gpu_layers is not None else item.get("gpu_layers", 99))
                           if item["runtime"] == "llamacpp" else None),
            "preset": args.preset,
            "settings_profile": args.settings_profile,
            "runtime": inkling_runtime(args) if item["runtime"] == "inkling" else args.runtime if item["runtime"] == "prism" else item["runtime"],
            "model": item["alias"], "port": args.port,
            "context": args.context or item["context"],
            "cpu_offload_gb": args.cpu_offload_gb,
            "cpu_moe_layers": (0 if args.no_offload else args.cpu_moe_layers if args.cpu_moe_layers is not None else item["cpu_moe_layers"]) if "cpu_moe_layers" in item and args.moe_cache_mib != "auto" else None,
            "moe_cache_mib": args.moe_cache_mib if "cpu_moe_layers" in item else None,
            "batch_tokens": None if item["runtime"] in ("strata", "strata-qwen") else args.batch_tokens or item.get("batch_tokens", 512 if item["runtime"] == "dory" else 2048 if item["runtime"] in ("dory-native", *GGUF_RUNTIMES) else 256),
            "ubatch_tokens": args.ubatch_tokens if item["runtime"] in ("prism", "dory-native", *GGUF_RUNTIMES) else None,
            "rounds": args.rounds if item["runtime"] in ("dory", "dory-native") else None,
            "kv_offload_gb": args.kv_offload_gb if item["runtime"] == "dory" else None,
            "speculative": args.speculative, "draft_tokens": 0 if item["runtime"] == "inkling" and args.speculative == "none" else args.draft_tokens,
            "draft_conf": (args.draft_conf if args.draft_conf is not None else 0.8) if args.speculative == "dflash2" else None,
            "adaptive_fetch": strata_config.get("adaptive_fetch"),
            "tui": args.tui,
            "cache_reuse_fix": args.cache_reuse_fix if item["runtime"] == "strata-qwen" else None,
            "engine_log_offset": log_offset if item["runtime"] == "strata-qwen" else None,
            "dspark_conf": (0.0 if item["runtime"] == "inkling" else args.dspark_conf if args.dspark_conf is not None else 0.65) if args.speculative == "dspark" else None,
            "mtp_chain": environment.get("LLAMA_MTP_CHAIN") == "1" if mtp else None,
            "mtp_head": args.mtp_head if mtp else None,
            "checkpoint_every": args.checkpoint_every,
            "replay": environment.get("LLAMA_RS_REPLAY") == "1" if args.runtime == "bonsai-next" else None,
            "gdn_mma": "GGML_CUDA_GDN_MMA_DISABLE" not in environment if args.runtime == "bonsai-next" else None,
            "slot_save_path": str(args.slot_save_path.resolve()) if args.slot_save_path else None,
            "draft_kv_cache": None if item["runtime"] == "inkling" and args.speculative == "none" else args.draft_kv_cache, "prompt_lookup": args.prompt_lookup,
            "lookup_type": ("suffix" if item["runtime"] == "strata-qwen" else args.lookup_type) if args.prompt_lookup else None,
            "lookup_tokens": args.lookup_tokens if args.prompt_lookup and (item["runtime"] == "strata-qwen" or args.lookup_type == "ngram-mod") else None,
            "lookup_min_tokens": args.lookup_min_tokens if args.prompt_lookup and args.lookup_type == "ngram-mod" else None,
            "draft_window": args.draft_window, "kv_vram_cells": args.kv_vram_cells,
            "mma_decode_min_kv": args.mma_decode_min_kv,
            "cache_ram_mib": args.cache_ram if args.cache_ram is not None else item.get("cache_ram", DEFAULT_CACHE_RAM_MIB if item["runtime"] in ("prism", "dory-native", "flint") else None),
            "kv_cache": args.kv_cache or item.get("kv_cache"),
            **({"kv_device": "cpu" if getattr(args, "host_kv", False) else "cuda",
                "fit_target_mib": args.fit_target_mib, "quant": args.quant or "iq1_s"} if item["runtime"] == "inkling" else {}),
            "kv_mean_center": str(args.kv_mean_center) if args.kv_mean_center else None,
            "reasoning_effort": (args.reasoning_effort or item.get("reasoning_effort", DEFAULT_REASONING_EFFORT)) if item["runtime"] in ("prism", "strata", "strata-qwen") or "reasoning_effort" in item else None,
            "expert_memory": strata_config.get("expert_memory"),
            "expert_ram_gb": strata_config.get("expert_ram_gb"),
            "ple_io": args.ple_io if item["runtime"] == "strata-qwen" else None,
            "expert_tail_skip": int(strata_config["env"]["STRATA_ROUTE_TAIL_SKIP"]) if item["runtime"] == "strata-qwen" else None,
            "cpu_threads": args.cpu_threads if args.cpu_threads is not None else 12 if item["runtime"] == "strata" else 4 if item["runtime"] == "inkling" else None,
            "strata_slots": strata_config.get("env", {}).get("STRATA_DS4_SLOT_KEEP") == "1" if item["runtime"] == "strata" else None,
            "strata_slot_directory": strata_config.get("env", {}).get("STRATA_DS4_SLOT_DIR"),
            "gpu_expert_cache_gb": args.gpu_expert_cache_gb,
            "gpu_expert_fraction": args.gpu_expert_fraction,
            "prefill_chunk": (args.prefill_chunk if args.prefill_chunk is not None else 4096) if item["runtime"] == "strata" else args.prefill_chunk if item["runtime"] == "strata-qwen" else None,
            "vram_reserve_mib": strata_config.get("vram_reserve_mib") if item["runtime"] == "strata-qwen" else (args.vram_reserve_mib if args.vram_reserve_mib is not None else 700) if item["runtime"] == "strata" else None,
            "engram_cache_mib": (args.engram_cache_mib if args.engram_cache_mib is not None else 256) if item["runtime"] == "strata" else None,
            "engram_threads": (args.engram_threads if args.engram_threads is not None else 16) if item["runtime"] == "strata" else None,
            "thinking_budget": (args.thinking_budget if args.thinking_budget is not None else 32768) if item["runtime"] in ("strata", "strata-qwen") else None,
            "started": time.time()}, indent=2) + "\n")
        def interrupt(_number: int, _frame: Any) -> None:
            """Stop the child process when the launcher receives SIGTERM."""
            raise KeyboardInterrupt

        previous_handler = signal.signal(signal.SIGTERM, interrupt)
        try:
            return wait_with_monitor(process, args.port, console) if args.tui else process.wait()
        except KeyboardInterrupt:
            if process.poll() is not None:
                return process.returncode
            return 130
        finally:
            stop_child(process)
            signal.signal(signal.SIGTERM, previous_handler)
            if process.poll() is not None:
                active.unlink(missing_ok=True)


def stop() -> None:
    """Stop only the verified model controller started by this manager."""
    path = STATE / "active.json"
    if not path.exists():
        raise RuntimeError("No managed model is running. Stop legacy servers with Ctrl-C.")
    info = read_json(path)
    pid = info.get("controller_pid")
    if not isinstance(pid, int):
        raise RuntimeError("The current server has no controller record. Stop it with Ctrl-C.")
    process = Path("/proc") / str(pid)
    expected = os.fsencode(ROOT / "localmodel.py")
    if not process.exists() or process.stat().st_uid != os.getuid() or expected not in (process / "cmdline").read_bytes().split(b"\0"):
        raise RuntimeError("The saved controller is not running. No process was stopped.")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 45
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if path.exists():
        raise RuntimeError("The model is still stopping. Inspect its server log before starting another.")
    print(f"Stopped {info['profile']}.")


def stats_text() -> str:
    """Show recorded measurements without starting a model."""
    return ("2026-11-21 Inkling 2 Mini IQ1_S on the 4070 SUPER, one slot, Q4 KV, no SSD offload.\n"
            "128k-fixed (default): 120,049 uncached coding tokens at 193.8 prefill and 32.5 decode tok/s.\n"
            "Cached follow-up: 32.2 tok/s. All 26 complete 128-token blocks reached 30 tok/s. Minimum free VRAM: 2,355 MiB.\n"
            "128K --auto, 4,088-token retrieval (three runs): median 363.7 prefill and 27.0 decode tok/s.\n"
            "Earlier fixed placements:\n"
            "128K: 124,992 uncached input, 154.5 prefill and 26.1 decode tok/s, 1,207 MiB free VRAM.\n"
            "256K: 259,994 uncached input, 89.8 prefill and 16.2 decode tok/s, 840 MiB free VRAM.\n"
            "Both filled prompts retrieved all three values. At 128K with the same 4,088-token input,\n"
            "the 256 MiB expert cache changed prefill/decode from 316.8/33.6 to 310.9/42.0 tok/s;\n"
            "wall time fell from 18.99 s to 17.34 s. Both passed. The explicit 128k profile uses zero cache.\n"
            "Earlier 128K placement: 768 MiB expert cache, input embedding and first two / last six expert banks in RAM.\n"
            "On 124,992 uncached input: 151.0 prefill, 30.7 decode tok/s, 887 MiB free, retrieval 3/3.\n"
            "Two-turn coding (24,625 cold input): 40.2 / 44.8 decode tok/s, 159.4 s total, 10/10 checks.\n"
            "128k-tuned, no draft, one two-turn coding run: 120,049 uncached input at 173.4 prefill and 33.5 decode tok/s.\n"
            "Follow-up reused 121,130 tokens, decoded at 34.1 tok/s. Total 798.6 s, 9/10 checks, 1,175 MiB free.\n"
            "128K DSpark comparison, 124,992 cold input, same 512 MiB placement, no draft / depth 2 / no draft.\n"
            "Cold decode was 32.7 / 16.8 / 30.2 tok/s. Cached follow-ups were 30.6–33.4 without a draft and\n"
            "22.8–23.6 with depth 2. All nine retrieval answers passed. Background CPU load varied.\n"
            "1M allocation, short prompt only: 4,088 uncached input and 188 output; 19.39 prefill, 9.15 decode tok/s.\n"
            "Wall time 231.28 s, retrieval 3/3, minimum free VRAM 2,016 MiB; host KV 9,272.25 MiB, zero swap.\n"
            "The 768 MiB expert cache hit on 6.27% of lookups. Full-window throughput and recall are unmeasured.\n\n"
            "2026-11-11 dense Qwen3.8-27B Dynamic 3.1 UD-Q4_K_M, llama.cpp b11946.\n"
            "192K capacity, Q4 KV, auto GPU fitting, 48G cap, xhigh; same 15997-token initial OMP input.\n"
            "Standard / MTP2: decode 2.99 / 5.37 tok/s; initial prefill 383.90 / 310.31 tok/s.\n"
            "Repair plus strict resume: 340.07 / 210.04 s, with 842 / 698 generated tokens. Both pass all checks.\n"
            "MTP accepts 431/528 draft tokens (81.63%). Single sampled runs; shorter output also reduces task time.\n"
            "Peak scope RAM 24.19 / 14.73 GiB, whole-GPU VRAM 11741 / 11517 MiB; no swap or OOM.\n"
            "RAM charges include file cache; their difference does not establish an MTP memory saving.\n"
            "Use localmodel start qwen3.8-27b mtp. Flash-Next remains the faster local choice in earlier measurements.\n"
            "All benchmark services stopped. See docs/qwen27-20261111.md.\n\n"
            "2026-11-09 official Strata 0.2.13 / native 0.1.59: existing defaults retained.\n"
            "Full Unsloth IQ1_S, 192K capacity, cache48, 64G cap, CPU12, prefill4096, reserve700, six experts, no DSpark.\n"
            "Matched 16K/32K/64K decode, 0.2.12 -> 0.2.13: 8.97 -> 9.17, 8.72 -> 9.12, 8.56 -> 8.81 tok/s.\n"
            "Initial prefill: 261.11 -> 256.33, 347.54 -> 346.42, 417.30 -> 417.96 tok/s; actual input 15961/32331/65306.\n"
            "All six repair/recall/strict-resume checks pass. Supplemental repair JSON-only compliance: old 3/3, new 2/3.\n"
            "Output lengths and tool paths differ; the 16K/32K tasks take longer. These single passes do not establish a general speed gain.\n"
            "Peak RAM 64 GiB and whole-GPU VRAM 11.79 GiB; no OOM or swap. Qwen MTP4, API tools and OMP discovery pass.\n"
            "All benchmark services stopped. See docs/strata-0.2.13-review.md for resources, formatting limits and cache behavior.\n\n"
            "2026-11-07 full DeepSeek tuning, Strata 0.2.12 / native 0.1.58: defaults retained.\n"
            "192K capacity, 64G cap, cache48, CPU12, prefill4096, reserve700, six experts, no DSpark.\n"
            "Nine 16K OMP screens: controls 9.19/9.16 decode tok/s; cache40 8.49, cache52 9.37, CPU8 8.15, reserve1536 8.48.\n"
            "DSpark depths 1/3: 8.50/8.74 decode tok/s. Chunk2048: 9.61 decode, but first prefill 189.91 versus 267.44/270.11 tok/s.\n"
            "All repair/recall/strict-resume checks pass; no OOM or swap. Actual input 16571-18287; these screens do not fill 192K.\n"
            "Outputs and tool paths differ; a single 2% cache52 gain is not established. All owned services stopped.\n"
            "See docs/deepseek-full-0.2.12-tuning.md for settings, resources and limitations.\n\n"
            "2026-11-07 full Unsloth DeepSeek on Strata 0.2.12 / native 0.1.58. deepseek now selects full UD-IQ1_S.\n"
            "192K capacity, 64G cap, cache48, CPU12, prefill4096, six experts, no DSpark.\n"
            "Growing 16K/64K/128K/180K OMP: 8.99 decode / 373.09 new-prefill tok/s, versus 9.55 / 355.32 on 0.2.11.\n"
            "All four checkpoints pass through 184758 actual input; 3118 assertions. Peak RAM 64 GiB, whole-GPU 11.624 GiB; no OOM or swap.\n"
            "2626 output tokens versus 3887 before; shorter OMP wall time (825.69 versus 966.70 s) is not a fixed-work speedup.\n"
            "One recovered read('.') refusal came from the unchanged benchmark guard. See docs/deepseek-full-0.2.12.md.\n\n"
            "2026-11-07 Strata 0.2.12 / native 0.1.58 comparison; 64G cap and zero swap.\n"
            "Qwen Unsloth IQ1_M, 192K capacity: normal default MTP4; optional tuned DFlash2 uses 5 drafts and confidence 0.8.\n"
            "Qwen default: 36 GiB resident expert budget, adaptive fetch off. lowram profile: mapped experts, zero dedicated expert RAM, adaptive fetch 1. Both use a 64G cap.\n"
            "Resident experts, growing 16K/32K/64K OMP: MTP4 53.63 decode / 1319.75 new-prefill tok/s; DFlash2 58.57 / 1452.02.\n"
            "Resident experts, matched 16K/64K/128K/180K: MTP4 52.17 decode / 1545.45 prefill; DFlash2 58.41 / 1701.80.\n"
            "Maximum actual input 185243 / 185415; both pass all code, recall, tool and strict JSON checks.\n"
            "Long-run model HTTP time 203.40 / 227.75 s; output 4448 / 4744 tokens. Faster decode did not establish faster whole tasks.\n"
            "OMP replaced an older read result; DFlash's run reread 62193 unchanged tokens from an earlier recurrent checkpoint. No parked-chat eviction.\n"
            "Factory Auto (ordinary DFlash2): 61.63 decode at 128K capacity, 56.82 at 192K; actual input only through 66K.\n"
            "Factory settings left little GPU headroom; the 192K prefill path reported zero free MiB. They are not the launcher defaults.\n"
            "Adaptive fetch 0/1/2 with mapped experts: 44.26 / 44.46 / 42.68 decode tok/s; no useful gain. Resident mode disables it.\n"
            "Coder 0.2.11 -> 0.2.12: 12.55 -> 11.74 decode; 356.54 -> 412.22 new-prefill tok/s.\n"
            "First new Coder attempt timed out; same-budget repeat passed all 16K/32K/64K checks (maximum input 66002).\n"
            "Single sampled runs with different outputs. All owned servers stopped; no model OOM or swap.\n"
            "See docs/strata-0.2.12-measurements.md and docs/strata-0.2.12-review.md.\n\n"
            "Earlier measurements retain their original runtime labels:\n\n"
            "qwen3.8: Unsloth UD-IQ1_M v3.1 + MTP4, Strata 0.2.11 / native 0.1.57.\n"
            "Default: 192K capacity, 64K GPU KV window, 64G cap, 36 GiB expert budget, direct PLE reads, 1024 MiB VRAM reserve. Expert tail skipping and suffix lookup are off.\n"
            "2026-11-07 growing 16K/64K/128K/180K OMP: 4782 tokens / 89.8841 decode seconds = 53.20 tok/s; maximum input 185531.\n"
            "Decode 54.84/55.05/52.30/50.99; initial/new-suffix prefill 1354/1677/1619/1517 tok/s. Later requests reuse prefixes.\n"
            "MTP accepted 3257/4449 drafts (73.21%). RAM peak 64 GiB; whole-GPU peak 11.38 GiB; zero model swap/OOM.\n"
            "All four code/recall/strict-resume checkpoints pass. Minimum reported free VRAM 498 MiB, with one warning below 512 MiB.\n"
            "Actual resident experts 31.90 GiB; zero logical expert file reads. See docs/local-192k-comparison.md for metrics and limits.\n"
            "Earlier 32 GiB / 768 MiB reserve run, 2026-11-06: 3290 tokens / 59.5461 decode seconds = 55.25 tok/s.\n"
            "Decode by checkpoint: 54.65/57.62/52.66/56.27; first-request new-suffix prefill: 1548/1920/1883/1783 tok/s. Later requests reuse prefixes.\n"
            "Last input 185151 tokens. All four code/recall/strict-resume checkpoints pass, 3118 assertions, zero tool errors. Draft acceptance 2270/2992 (75.87%).\n"
            "Peak 64 GiB, no OOM/model swap; 31.89 GiB expert RAM, 2.93 GiB GPU expert cache, 11.72 GiB whole-device VRAM peak.\n"
            "Zero logical expert file blobs during requests; PLE stays file-backed. Shared guest disk counters include other processes and do not measure host SSD activity.\n"
            "Short full-expert MTP2/3/4: 53.50/53.91/55.63 tok/s. mmap and lookup did not improve; 36 GiB expert budget added no residency.\n"
            "The 256K full-GPU-KV control disabled MTP for lack of VRAM and averaged 35.54 tok/s through 128K.\n"
            "Across 17 checkpoints, code/recall pass 17/17 and strict resume 12/17; five earlier short answers used fenced correct JSON. No automatic-compaction test.\n"
            "See docs/qwen-unsloth-192k.md and results/qwen-unsloth-20261106/final-results.json.\n"
            "Historical ISTA IQ2_XS, 2026-11-05: 53.36 default / 70.05 with 64K GPU KV through 128K. Different VM storage, desktop VRAM use and expert-skipping policy prevent a matched quant comparison.\n"
            "Historical release comparison: 0.2.9 -> 0.2.11 decode 51.65 -> 51.38; prefill improved 2.25-2.82x. See docs/qwen-0.2.11-tuning.md.\n"
            "Full 0.2.9, 2026-10-31, 16K OMP: 431.28 prefill / 47.95 decode tok/s; peak 64 GiB, no swap/OOM, file-cache reclaim at the cap.\n"
            "Repair, 100 assertions, recall and strict JSON resume pass. One edit-format error recovered. No low-VRAM warning.\n"
            "Earlier native-only update with API 0.2.7.1: 768 MiB reserve 443.07 prefill / 49.08 decode; strict resume used fenced JSON (fail).\n"
            "That earlier API with 700 MiB reserve: 549.00 prefill / 47.19 decode; all quality checks pass, but only 468 MiB VRAM free.\n"
            "Single passes; see docs/strata-0.2.9-review.md and results/strata-029-20261031/.\n"
            "Previous native 0.1.52, 256K/64G, 16K OMP: 520.1 prompt / 35.49 decode tok/s; all benchmark checks pass.\n"
            "Peak 64 GiB; no swap/OOM. Linux memory monitoring only. Source: results/strata-0271-linux-guard-20261028/summary.md.\n"
            "Previous 128K capacity / 48G cap: 16K OMP 1396 prompt / 61.68 decode tok/s.\n"
            "Previous 48G code/recall pass; resume used fenced JSON (strict format fail). 40G/16 GiB: 609/46.34 tok/s, all checks pass.\n"
            "Source: results/strata-027-20261027/summary.md.\n\n"
            "deepseek-coder (optional): ISTA GSQ-RCO Coder IQ2_XXS, Strata 0.2.11 / native 0.1.57.\n"
            "Default: 192K capacity, 64G cap, file-backed experts with zero private cache, 12 workers, 4096-token chunks, retained slots, no draft.\n"
            "2026-11-07 growing 16K/64K/128K/180K OMP: 3034 tokens / 254.6245 decode seconds = 11.92 tok/s; maximum input 184811.\n"
            "Decode 13.02/11.05/11.11/11.62; new-suffix prefill 412/487/488/483 tok/s. Later requests reuse prefixes.\n"
            "All four code/recall/resume checkpoints pass, 3118 assertions. Peak about 64 GiB; zero model swap/OOM.\n"
            "Coder is optional; deepseek now selects full Unsloth. See docs/local-192k-comparison.md for tool-policy refusals, coding diagnostics, memory costs, and the full-Unsloth comparison.\n"
            "2026-11-05 matched default comparison, growing 16K/32K/64K OMP: 0.2.10 -> 0.2.11 weighted decode 12.499 -> 12.460 tok/s (-0.32%).\n"
            "Decode by checkpoint: 13.10/12.60/11.31 -> 12.78/12.63/11.84; first-request prefill 409.9/470.3/602.5 -> 413.2/482.6/605.8 tok/s.\n"
            "Later prefill measures the new suffix after prefix reuse. Generated 2239/2244 tokens; repair+resume 372.03/371.97 s.\n"
            "All repair, recall and strict resume checks pass. Recovered guard refusals 3/2; settings and profiles restored.\n"
            "Both peak at 64 GiB RAM and 11731 MiB VRAM; no model swap/OOM. Minimum Linux available RAM 91.90/92.11 GiB.\n"
            "Shared root reads 69.10/56.28 GiB are not model-only reads. Expert-cache miss counters unavailable.\n"
            "Single pair, unchanged weights and routing; no filled-256K or automatic-compaction test. See docs/strata-0.2.11-coder.md.\n"
            "2026-11-03 resident comparison, growing 16K/32K/64K OMP: weighted decode 13.80 tok/s at 64G versus 13.25 on 0.2.9 (+4.2%).\n"
            "New prefill 482.7/618.3/756.5 tok/s; old control 498.5/639.6/768.3. GPU expert count rose from 242 to 353.\n"
            "New 64G: 6900/7327 CPU experts in RAM, 427 file-backed. New 70G: all 7327 in RAM, 14.04 tok/s (+1.7%).\n"
            "Use --expert-memory resident --mem 70G for measured full expert residency. Engram stays on SSD. Prefill was unchanged from new 64G.\n"
            "Old64/new64/new70: generated 2080/2902/2372 tokens; repair+resume 331.85/391.89/342.01 s.\n"
            "All final checks passed; recovered tool errors 1/1/2. Peaks reached their caps; no OOM or swap.\n"
            "Same frozen routing profile, fresh slots, single sampled trajectories. Output lengths differ; no end-to-end speed claim.\n"
            "No filled-256K or automatic-compaction test. See docs/strata-0.2.10-coder.md.\n"
            "File-backed sharing test, 70G and CPU weight 10: --expert-memory cache --expert-ram-gb 0.\n"
            "Quiet 16K/32K/64K: 12.80 decode tok/s; prefill 397.5/515.7/599.3. About 8.8% lower decode than resident70G.\n"
            "Controlled 32 GiB competitor: file cache 64.12 -> 42.36 GiB, 16K decode 6.40 tok/s; recovery 11.49 tok/s and 54.74 GiB cache.\n"
            "All five checkpoints pass; no OOM/swap; minimum Linux headroom 59.48 GiB. CPU weight was experiment-only. Default cap remains 64G.\n"
            "32 GiB demand cache, same 70G cap/CPU weight: quiet 9.08, pressure 9.14, recovery 8.77 decode tok/s.\n"
            "Guarded retry: all five checkpoints pass; peak 53.30 GiB, no OOM/swap, minimum Linux headroom 31.27 GiB.\n"
            "An initial OMP attempt failed; its unintended approval-config edit was restored. Future fixture runs enforce workspace boundaries.\n"
            "Single passes with different output lengths and prefix reuse. See docs/coder-elastic-memory.md.\n"
            "Historical native 0.1.55 model comparison, same 64G/48 GiB demand cache: Coder 11.24 vs full Unsloth 8.16 tok/s.\n"
            "See docs/deepseek-coder-comparison.md for those earlier measurements and quality limits.\n"
            "\n"
            "deepseek / deepseek-full: Unsloth Dynamic v3.1 UD-IQ1_S. The following results used Strata 0.2.11 / native 0.1.57, 192K capacity.\n"
            "Default: 12 CPU workers, 4096-token chunks, retained conversation slots, 48 GiB expert cache, 64G cap, no draft.\n"
            "2026-11-07 growing 16K/64K/128K/180K OMP: 3887 tokens / 406.9776 decode seconds = 9.55 tok/s; maximum input 184898.\n"
            "Decode 9.81/9.42/9.26/9.25; new-suffix prefill 276/432/461/447 tok/s. All four code/recall/resume checkpoints pass.\n"
            "Peak 64 GiB, zero model swap/OOM. Later prefill reuses prefixes. Single trajectories; output lengths differ.\n"
            "16K screens: cache0=7.39, cache48=9.25, DSpark3=9.02, DSpark5=9.07, prefill8192=8.21, resident=7.62 decode tok/s.\n"
            "DSpark and resident copies did not improve this selection. See docs/local-192k-comparison.md for settings, correctness and measurement limits.\n"
            "Historical measurements below use native 0.1.55 and 256K capacity.\n"
            "0.2.9 tuning, 16K OMP: four workers 5.20 decode; twelve workers 8.51-8.80 tok/s.\n"
            "Growing 16K/32K/64K repeat: 8.07/8.28/7.82 decode tok/s; all checks pass, no tool errors. Peak 63.97 GiB, no swap/OOM/cap events.\n"
            "DSpark2 with twelve workers: 8.38 tok/s, 84.8% acceptance; remains optional.\n"
            "Retained-prefix replay: resume prompt time 47.15 -> 26.19 s. Restart reused 66,165 tokens, reading one new token in 0.54 s.\n"
            "One early growing-candidate checkpoint missed required test execution after approval rejections; code/recall/resume passed.\n"
            "See docs/deepseek-029-tuning.md for full workflow results, retries, and measurement limits.\n"
            "Earlier four-worker 0.2.9, 2026-10-31, 64K OMP: 5.21 decode / 419.71 prefill tok/s; peak 59.16 GiB, no swap/OOM or cap events.\n"
            "Repair, 100 assertions, recall and strict resume pass; no tool errors.\n"
            "Generated 527 tokens over 5 requests; repair+resume 302.29 s. No material throughput change from 0.1.54.\n"
            "Previous native 0.1.54, 64K OMP, both with a 70G cap: 48 GiB cache 5.22 decode / 423.04 prefill tok/s; 64 GiB cache 5.48 / 425.46.\n"
            "Peak RAM 61.72/70.00 GiB. Both pass repair, 100 assertions, recall and resume; no swap/OOM.\n"
            "The larger cache gained 5.1% decode and 0.6% prefill. The default uses 48 GiB to retain RAM headroom.\n"
            "16K OMP cache comparison: 48 GiB/64G cap 5.24 decode / 278.80 prefill tok/s; 64 GiB/70G cap 5.49 / 292.14.\n"
            "Peak RAM 48.45/58.81 GiB; both pass repair, 100 assertions, recall and resume. No swap/OOM or memory-limit events.\n"
            "Single passes, no OS page-cache flush; generated 694/515 tokens. No filled-256K or compaction test.\n"
            "Earlier DSpark comparison used a 36 GiB expert cache and a 64G cap:\n"
            "16K OMP: none 4.98 decode / 275.41 prefill tok/s; DSpark5 4.60 / 276.10. Both pass code, 100 assertions, recall and resume.\n"
            "Generated 562/1347 tokens; repair+resume 211.19/422.34 s. Different output lengths affect elapsed time.\n"
            "Peak cgroup RAM 38.28/39.50 GiB; zero swap/OOM. DSpark acceptance 79.08%; remains optional.\n"
            "Single passes with background CPU activity; no filled-256K or compaction test. See experiments/deepseek-strata/README.md.\n"
            "Historical GSQ-RCO IQ1_M results below do not measure this quantization.\n"
            "Strata 0.2.6, 16K OMP, six experts, 4096-token chunks: 48G/36 GiB cache gave 257.7 prompt and 5.4 decode tok/s;\n"
            "64G/48 GiB cache gave 262.0 prompt and 5.6 decode tok/s. Peaks: 40.3/59.0 GiB. Single passes; see experiments/deepseek-strata/README.md.\n"
            "Strata 0.2.7 / engine 0.1.52: 16K OMP 284 prompt / 5.31 decode tok/s, 48G cap / 36 GiB cache; all checks pass.\n"
            "Strata 0.2.7.1 / engine 0.1.52: 16K OMP 264.7 prompt / 5.15 decode tok/s, 64G cap / 36 GiB cache; all checks pass.\n"
            "Peak 47.58 GiB. No measured speed gain. See results/strata-0271-20261028/summary.md.\n"
            "Historical Abyss 16K OMP task: 7.2 prompt tok/s, 6.1 decode tok/s, 43.5 GiB peak RAM.\n"
            "Those timings do not measure the official Strata backend. See experiments/deepseek-strata/README.md.\n\n"
            "Groot/LFM OMP, 2026-10-28: 256K capacity, actual 16K and 192K inputs, llama.cpp b11946, no MTP.\n"
            "lfm (Q8_0): Q8 KV, GPU weights, 16 GiB cap, 4 GiB prompt snapshots.\n"
            "  16K: 15186 prompt / 160.7 decode tok/s; 165.7 after compaction. Cold TTFT 1.09 s.\n"
            "  192K: 6959 prompt / 95.7 decode tok/s; 160.1 after compaction. Cold TTFT 28.35 s.\n"
            "  Code tests pass; exact recall and tool instructions fail. 192K: 10 tool errors (8 approval, 2 edit format).\n"
            "groot (Q4_K_M): Q8 KV, 24 CPU expert layers, 32 GiB cap, 8 GiB prompt snapshots.\n"
            "  16K: 867 prompt / 34.0 decode tok/s. Cold TTFT 18.91 s. Completed checks pass; too short to compact.\n"
            "  192K: 759 prompt / 26.2 decode tok/s; 31.4 after compaction. Cold TTFT 259.28 s. All checks pass.\n"
            "  192K measured workflow: LFM 267 s / 22658 generated tokens; Groot 613 s / 6064 tokens.\n"
            "Largest sampled RAM / whole-card VRAM: LFM 3.03 / 5.83 GiB; Groot 18.00 / 10.42 GiB.\n"
            "RAM includes charged file cache; warm weights affect peaks. Single runs, constructed history; no full 256K test.\n"
            "Source: results/groot-lfm-20261028/summary.md.\n\n"
            "Legacy vLLM Dory host KV offload was tested at 2 GiB.\n\n"
            + presets.stats_text())


def main() -> int:
    """Parse model management and discovery commands."""
    help_tail = (f"Start defaults to {DEFAULT_START_MODEL} (Inkling 2 Mini IQ1_S, 128K, Q4 KV, no draft). Fixed placement with a 640 MiB expert cache. Use --auto for Unsloth sizing.\n"
                 "Models: inkling (IQ1_S or IQ2_XXS), bonsai2-27b, bonsai2-9b, qwen3.8 (Flash-Next), qwen3.8-27b (dense), deepseek, deepseek-coder, deepseek-full, flint, groot, lfm, dory, dory-gguf.\n"
                 "Use localmodel list for installed models and settings profiles; localmodel presets lists existing presets.\n"
                 "Select a settings profile with localmodel start MODEL PROFILE, for example qwen3.8 lowram.\n"
                 "Use localmodel start --help for flags and localmodel start MODEL --dry-run to preview a launch.\n"
                 "Use localmodel --stats for recorded measurements and their limitations.")
    parser = argparse.ArgumentParser(prog="localmodel", description=__doc__, epilog=help_tail,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", action="store_true", help="Show recorded performance measurements and exit.")
    commands = parser.add_subparsers(dest="action")
    commands.add_parser("list", help="List models, settings profiles, and installed weights.")
    commands.add_parser("stop", help="Stop the model started by localmodel.")
    preset_parser = commands.add_parser("presets", help="Show named settings and measured performance.")
    preset_parser.add_argument("--json", action="store_true")
    fetch_parser = commands.add_parser("fetch", aliases=["pull"], help="Fetch pinned weights and the required runtime.")
    fetch_parser.add_argument("model")
    fetch_parser.add_argument("--speculative", choices=("none", "mtp", "dflash", "dflash2", "dspark"), default="none",
                              help="Fetch a supported optional decoder (default: plain weights; Qwen includes MTP). Inkling dspark fetches the target, Q8 drafter and Unsloth runtime. Qwen dflash2 also fetches its optional signed draft pack.")
    fetch_parser.add_argument("--moe-cache-mib", choices=("auto",), help="Inkling: prepare the Unsloth automatic placer. The drafter is optional.")
    fetch_parser.add_argument("--mtp-head", choices=("teacher", "procreations"), default="teacher",
                              help="MTP head to fetch (default: selected preset, otherwise teacher). Requires --speculative mtp.")
    removal = commands.add_parser("remove", help="Delete selected weights and download data. Keep configuration and runtimes.",
                                  description="Remove the artifacts selected by fetch. Refuse weights that a running process uses.")
    removal.add_argument("model", help="Model name, alias, or preset. Presets select their configured decoder.")
    removal.add_argument("--speculative", choices=("none", "mtp", "dflash", "dflash2", "dspark"), default="none",
                         help="mtp removes MTP weights (the complete profile for Qwen); dflash removes plain and draft weights; dflash2 removes the Qwen draft pack; dspark removes only the Inkling drafter or the DeepSeek draft and its pack. For Inkling, none removes only the target.")
    removal.add_argument("--mtp-head", choices=("teacher", "procreations"), default="teacher",
                         help="Select the same MTP artifact as fetch (default: selected preset, otherwise teacher).")
    removal.add_argument("--dry-run", action="store_true", help="Show the selected paths and storage use without deleting files.")
    calibration = commands.add_parser("calibrate", help="Build a local model-specific KV mean file. Stop the server first.")
    calibration.add_argument("model")
    calibration.add_argument("--text", type=Path, required=True, help="Representative local calibration text.")
    calibration.add_argument("--output", type=Path)
    calibration.add_argument("--kv-cache", choices=("q4_0",), default="q4_0")
    calibration.add_argument("--chunks", type=int, default=64, help="512-token calibration chunks (default: 64).")
    calibration.add_argument("--dry-run", action="store_true")
    for action in ("status", "opencode-config"):
        sub = commands.add_parser(action, help="Read live model metadata." if action == "status" else "Print an OpenCode profile for the running model.")
        sub.add_argument("--port", type=int)
        if action == "status":
            sub.add_argument("--json", action="store_true")
    start = commands.add_parser("start", aliases=["serve"], help="Start a model or preset. Stop it with Ctrl-C.",
                                epilog=help_tail, formatter_class=argparse.RawDescriptionHelpFormatter)
    start.add_argument("model", nargs="?", default=DEFAULT_START_MODEL,
                       help=f"Model or preset to start (default: {DEFAULT_START_MODEL}).")
    start.add_argument("settings_profile", nargs="?", default="default", metavar="PROFILE",
                       help="Inkling defaults to 128k-fixed. Its 128k-dspark, 192k-dspark and 256k-dspark profiles select the optional drafter. Flash-Next also supports lowram; dense Qwen supports mtp. Use localmodel list to see available profiles.")
    for selection in (fetch_parser, removal):
        selection.add_argument("settings_profile", nargs="?", default="default", metavar="PROFILE",
                               help="Select a named Inkling placement. See localmodel list.")
    for selection in (fetch_parser, removal, start):
        selection.add_argument("--quant", choices=INKLING_QUANTS,
                               help="Inkling target weights: iq1_s (default) or iq2_xxs. Both share the DSpark drafter.")
    start.add_argument("--context", "--ctx", "--ctxt", type=lambda value: tokens(value, maximum=1048576),
                       help="Context length in tokens. K and M are binary units. Only Inkling supports --ctx 1m locally.")
    start.add_argument("--port", type=int, default=8899)
    start.add_argument("--tui", action="store_true",
                       help="Show Strata telemetry in this terminal. q detaches; Ctrl-C stops the server.")
    start.add_argument("--cache-reuse-fix", action=argparse.BooleanOptionalAction, default=False,
                       help="Use the separately verified Qwen CUDA cache-reuse backport (default: off).")
    start.add_argument("--mem", type=memory_size,
                       help="Hard RAM cap for the model process and its children, including Linux file cache. Bare numbers use GiB, for example --mem 32. Swap is disabled for this process tree. Omit to keep the profile cap. This does not reduce --cache-ram automatically.")
    start.add_argument("--expert-memory", choices=("cache", "resident"),
                       help="DeepSeek CPU expert storage (default: profile). cache uses --expert-ram-gb. resident copies variable-size experts within the native RAM allowance, with an 8 GiB margin. Remaining experts and Engram can still use SSD.")
    start.add_argument("--expert-ram-gb", type=float,
                       help="Strata expert demand-cache budget in GiB, inside --mem (Coder: 0; deepseek-full: 48; Qwen: 36, or 0 with lowram). Zero leaves CPU expert caching to Linux. With --expert-memory resident, omit this flag or use 0. Separate from llama.cpp --cache-ram.")
    start.add_argument("--cpu-threads", type=int, help="Strata CPU expert workers, 0-64 (DeepSeek default: 12; Qwen: automatic). 0 selects the automatic count. Qwen also uses the host thread. I/O threads are separate.")
    start.add_argument("--strata-slots", action=argparse.BooleanOptionalAction, default=None,
                       help="Retain DeepSeek Strata conversation checkpoints (default: on). Keeps one saved conversation in RAM and a 2 GiB disk-cache budget per weights/runtime/context configuration.")
    start.add_argument("--gpu-expert-cache-gb", type=float, help="Legacy control. Official Strata requires --gpu-expert-fraction instead.")
    start.add_argument("--gpu-expert-fraction", type=float, help="DeepSeek Strata GPU expert-cache fraction, from 0 to 1, of free VRAM after dense weights and the reserve (default: stock automatic).")
    start.add_argument("--prefill-chunk", type=int, help="Strata prefill chunk tokens (DeepSeek: 4096, minimum effective 256; Qwen: automatic unless set).")
    start.add_argument("--vram-reserve-mib", type=int, help="Strata VRAM reserve for CUDA and desktop use, in MiB (DeepSeek: 700; Qwen: 1024).")
    start.add_argument("--engram-cache-mib", type=int, help="DeepSeek Strata hashed-embedding row-cache budget in MiB (default: 256; 0 disables).")
    start.add_argument("--engram-threads", type=int, help="DeepSeek Strata hashed-embedding SSD read threads (default: 16).")
    start.add_argument("--ple-io", choices=("direct", "mmap", "ram"),
                       help="Qwen embedding-table reads: direct (default), mmap (reclaimable page cache), or ram (full RAM copy; needs more memory).")
    start.add_argument("--thinking-budget", type=int, help="Strata chat-template thinking budget in tokens (default: 32768). Client requests can turn thinking off.")
    start.add_argument("--runtime", choices=("stock", "bonsai-native", "bonsai-next"), default="stock",
                       help="Bonsai runtime (default: selected preset; stock for base model names). bonsai-next adds reviewed checkpoint and replay patches in a separate build.")
    start.add_argument("--rounds", "--loops", type=int, choices=(1, 2, 3, 4), default=2,
                       help="Dory recurrent rounds per token (default: 2).")
    offload = start.add_mutually_exclusive_group()
    offload.add_argument("--no-offload", action="store_true", help="Keep transformer layers and attention on the GPU. For Flint and Groot this also sets --cpu-moe-layers 0; weights plus context may exceed VRAM. RAM snapshots and active KV spill use separate controls.")
    offload.add_argument("--gpu-layers", type=int,
                         help="Dense llama.cpp layers on the GPU; the rest use RAM. Dense Qwen defaults to automatic fitting. An explicit count disables fitting; --no-offload requests all layers.")
    offload.add_argument("--cpu-offload-gb", type=float, default=0, help="Dory CPU weight offload in GiB.")
    offload.add_argument("--cpu-moe-layers", type=int,
                         help="Place routed experts in the first N layers in RAM; keep attention on the GPU. Flint: default 33, range 0-49. Groot: default 24, range 0-40.")
    cache_size = start.add_mutually_exclusive_group()
    cache_size.add_argument("--auto", dest="moe_cache_mib", action="store_const", const="auto",
                            help="Inkling: fit expert placement and cache size to free VRAM at startup. Keep the requested GPU KV context.")
    cache_size.add_argument("--moe-cache-mib", type=lambda value: "auto" if value == "auto" else int(value),
                       help="GPU cache for host experts, in MiB (0 disables). Inkling accepts auto to let Unsloth place experts for the requested context. 256k-dspark uses auto; choose a manual placement profile for a numerical budget.")
    start.add_argument("--fit-target-mib", type=int, help="Inkling: enable the automatic placer with this free-VRAM target in MiB (default with auto: 2816).")
    start.add_argument("--kv-offload-gb", type=float, default=0,
                       help="Legacy vLLM Dory host KV cache in GiB. Attention stays on the GPU (default: 0).")
    start.add_argument("--batch-tokens", type=int)
    start.add_argument("--ubatch-tokens", type=int, help="Physical batch size (default: 64; Dory GGUF and Flint: 1024; Groot and LFM3: 256).")
    start.add_argument("--kv-cache", choices=("q4_0", "q8_0", "f16"), help="llama.cpp KV cache type (default: model profile).")
    start.add_argument("--kv-mean-center", type=Path, help="Prism model-specific KV mean file.")
    start.add_argument("--reasoning-effort", choices=("default", "low", "medium", "xhigh"),
                       help=f"Prism or dense Qwen chat-template effort (default: {DEFAULT_REASONING_EFFORT}). The template must support the level.")
    start.add_argument("--cache-ram", type=int,
                       help=f"llama.cpp prompt snapshot cache in MiB (default: {DEFAULT_CACHE_RAM_MIB}; Groot: 8192; LFM3: 4096; 0 disables). This differs from active KV spill.")
    start.add_argument("--prompt-lookup", action=argparse.BooleanOptionalAction, default=None,
                       help="Enable prompt lookup for Bonsai, Flint, Groot, LFM3, or Qwen. Qwen uses a three-token suffix match alongside MTP. Explicit Flint flags override LLAMA_ARG_SPEC_TYPE; omission preserves it.")
    start.add_argument("--lookup-type", choices=("ngram-simple", "ngram-mod"), default="ngram-simple",
                       help="llama.cpp prompt lookup algorithm (default: ngram-simple). Requires --prompt-lookup. Qwen uses its native suffix algorithm.")
    start.add_argument("--lookup-tokens", type=int, default=32,
                       help="ngram-mod lookup limit (default: 32), or Qwen verification window including the accepted token (2-16; default: 12). Neural depth still uses --draft-tokens.")
    start.add_argument("--lookup-min-tokens", type=int,
                       help="Minimum ngram-mod draft length (0 to --lookup-tokens). Requires prompt lookup with ngram-mod. Flint default: min(8, maximum); Bonsai keeps its runtime default unless set.")
    start.add_argument("--draft-window", type=tokens,
                       help="Native Bonsai MTP context window. Omit to use the full target context. Requires a native Bonsai runtime.")
    start.add_argument("--kv-vram-cells", type=int,
                       help="Active KV positions kept in VRAM; remaining KV uses RAM. Native Bonsai or Qwen Strata. Qwen default: 65536, capped by context. Pass the context capacity to keep full KV in VRAM.")
    start.add_argument("--mma-decode-min-kv", type=int,
                       help="Added native Bonsai wide-GQA MMA dispatch threshold (0 disables this override). Other MMA kernels remain available. Requires --runtime bonsai-native.")
    start.add_argument("--speculative", choices=("none", "mtp", "dflash", "dflash2", "dspark"), default="none",
                       help="Optional draft decoder (default: none; Qwen: mtp). DFlash2 is an explicit Qwen experiment. DSpark supports Inkling and deepseek-full.")
    start.add_argument("--draft-tokens", type=int, default=2, help="Maximum neural draft tokens per step (default: 2; Qwen MTP: 4, DFlash2: 5; DeepSeek DSpark: 5). Qwen verifies N+1 tokens, including the accepted token.")
    start.add_argument("--adaptive-fetch", type=int, choices=(0, 1, 2),
                       help="Qwen experimental PCIe expert admission: 0 off, 1 supplement tier copies, 2 replace tier copies. Default: 1 with mapped experts, 0 with resident experts. Modes 1/2 require --expert-ram-gb 0; inspect native activation logs.")
    start.add_argument("--draft-conf", type=float, help="Qwen DFlash2 confidence threshold, 0 to 1 (default: 0.8).")
    start.add_argument("--dspark-conf", type=float, help="DSpark confidence threshold, 0 to 1 (default: 0.65). Requires --speculative dspark.")
    start.add_argument("--mtp-chain", action=argparse.BooleanOptionalAction, default=None,
                       help="Prism MTP chained drafting (default: on). Preset settings override LLAMA_MTP_CHAIN=0/1. Explicit flags take priority.")
    start.add_argument("--mtp-head", choices=("teacher", "procreations"), default="teacher",
                       help="MTP head (default: selected preset, otherwise teacher). procreations uses a trained head with unchanged target tensors.")
    start.add_argument("--checkpoint-every", type=int,
                       help="Create a checkpoint inside a long message every N prompt tokens (0 disables). Requires bonsai-next. Existing minimum spacing is 8192 tokens.")
    start.add_argument("--replay", action=argparse.BooleanOptionalAction, default=None,
                       help="Use CUDA recurrent replay rollback to save VRAM (default: off). Requires bonsai-next, MTP and one GPU slot.")
    start.add_argument("--gdn-mma", action=argparse.BooleanOptionalAction, default=None,
                       help="Enable chunked GDN prefill (native default: on). Requires bonsai-next; ubatch128 or larger can engage the kernel with MTP2.")
    start.add_argument("--slot-save-path", type=Path,
                       help="Existing directory for manual Bonsai slot save/restore through the local API. Files contain prompt state; this does not enable automatic persistence.")
    start.add_argument("--draft-kv-cache", choices=("q4_0", "q8_0", "f16"), default="q8_0",
                       help="Draft decoder KV cache type. Inkling DSpark uses the selected placement's draft KV. Other llama.cpp profiles default to q8_0.")
    start.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stats:
        if args.action is not None:
            parser.error("Use --stats without a command.")
        print(stats_text())
        return 0
    if args.action is None:
        parser.error("Choose a command or use --stats.")
    if args.action == "stop":
        stop()
        return 0
    if args.action == "presets":
        print(json.dumps(presets.load(), indent=2) if args.json else presets.stats_text())
        return 0
    if getattr(args, "port", None) is not None and not 1 <= args.port <= 65535:
        parser.error("Use a port from 1 to 65535.")
    if args.action in ("status", "opencode-config"):
        info = current(args.port)
        if args.action == "opencode-config":
            from opencode_local import profile
            print(json.dumps(profile(info), indent=2))
        elif args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"{info['model']}: {info['context']} context tokens at {info['base_url']}")
        return 0
    models = catalog()
    if args.action == "list":
        print(f"{'MODEL':<28} {'WEIGHTS':<10} {'CONTEXT':>8}  {'RUNTIME':<8}  {'PROFILES':<18}  OPTIONAL DECODERS / QUANTS")
        for name, item in models.items():
            state = "installed" if installed(name, item) else "missing"
            optional = ", ".join(f"{variant}:{'ready' if installed(name, item[variant]) else 'missing'}"
                                 for variant in ("mtp", "mtp_procreations", "dflash") if variant in item) or "-"
            profiles = ", ".join(("default", *item.get("profiles", {})))
            if item["runtime"] == "inkling":
                profiles = ", ".join(p + (" (default)" if p == item["default_profile"] else "") for p in item["profiles"])
                dspark = "ready" if installed(name, item["dspark"]) else "missing"
                iq2 = "installed" if installed(name, item["iq2_xxs"]) else "missing"
                optional = f"dspark:{dspark} | iq1_s:{state} (default), iq2_xxs:{iq2}"
            print(f"{name:<28} {state:<10} {item['context']:>8}  {item['runtime']:<8}  {profiles:<18}  {optional}")
        return 0
    explicit = {argument.split("=", 1)[0][2:].replace("-", "_")
                for argument in sys.argv[2:] if argument.startswith("--")}
    if "auto" in explicit:
        explicit.remove("auto")
        explicit.add("moe_cache_mib")
    for alias, destination in (("ctx", "context"), ("ctxt", "context"), ("loops", "rounds"), ("no_prompt_lookup", "prompt_lookup"),
                               ("no_mtp_chain", "mtp_chain"), ("no_replay", "replay"), ("no_gdn_mma", "gdn_mma")):
        if alias in explicit:
            explicit.add(destination)
    args.preset = args.model if args.model in presets.load() else None
    if args.preset:
        settings = presets.get(args.preset)["settings"]
        for key, value in settings.items():
            if key == "model" or (args.action in ("start", "serve", "fetch", "pull", "remove") and key not in explicit):
                setattr(args, key, Path(value) if key == "kv_mean_center" else value)
    args.model = ALIASES.get(args.model, args.model)
    if args.model not in models:
        parser.error("Unknown model. Use localmodel list.")
    if args.model != "inkling" and (getattr(args, "moe_cache_mib", None) == "auto" or getattr(args, "fit_target_mib", None) is not None):
        parser.error("Automatic expert placement requires Inkling.")
    if args.model != "inkling" and (getattr(args, "context", None) or 0) > 262144:
        parser.error("Use 1 to 262144 tokens for this model.")
    if getattr(args, "quant", None) is not None and args.model != "inkling":
        parser.error("--quant requires Inkling.")
    if args.model == "inkling" and getattr(args, "quant", None) == "iq2_xxs":
        models["inkling"] = {**models["inkling"], **models["inkling"]["iq2_xxs"]}
    if args.model != "inkling" and args.action in ("fetch", "pull", "remove") and args.settings_profile != "default":
        parser.error("Named fetch and remove profiles require Inkling.")
    if getattr(args, "tui", False) and models[args.model]["runtime"] not in ("strata", "strata-qwen"):
        parser.error("--tui requires a Strata model.")
    if getattr(args, "cache_reuse_fix", False) and args.model != qwen_strata.NAME:
        parser.error("--cache-reuse-fix requires Strata Qwen.")
    if args.model == "inkling" and args.action in ("start", "serve"):
        item = models[args.model]
        inkling_profiles = cast(dict[str, dict[str, Any]], item["profiles"])
        context = args.context or item["context"]
        if args.settings_profile == "default" and (context != item["context"] or args.speculative == "dspark"):
            args.settings_profile = min(
                (name for name, settings in inkling_profiles.items()
                 if context <= settings["context"] and settings.get("speculative", "none") == args.speculative
                 and (args.speculative != "none" or not settings.get("unsloth_runtime"))),
                key=lambda name: (inkling_profiles[name]["context"], name),
                default=None,
            )
            if args.settings_profile is None:
                parser.error("No Inkling placement supports this context and decoder. See localmodel list.")
        elif args.settings_profile == "default":
            args.settings_profile = item["default_profile"]
        elif args.settings_profile in inkling_profiles and args.context and args.context > inkling_profiles[args.settings_profile]["context"]:
            parser.error("The requested context exceeds this Inkling placement. Omit PROFILE to select by --ctx.")
        settings = inkling_profiles.get(args.settings_profile, {})
        decoder = args.speculative if "speculative" in explicit else settings.get("speculative", "none")
        if decoder not in ("none", "dspark"):
            parser.error("Inkling supports --speculative none or dspark.")
        if decoder != "dspark" and explicit & {"draft_tokens", "draft_kv_cache"}:
            parser.error("Inkling draft options require --speculative dspark.")
        if decoder == "dspark":
            if settings.get("host_kv") or context > 262144:
                parser.error("Inkling DSpark supports GPU KV through 256K.")
            if not 1 <= args.draft_tokens <= 7:
                parser.error("Inkling DSpark supports 1 to 7 draft tokens.")
            if "draft_kv_cache" not in explicit:
                args.draft_kv_cache = settings.get("draft_kv_cache", "f16")
        if args.fit_target_mib is not None:
            if args.fit_target_mib < 1 or args.moe_cache_mib not in (None, "auto"):
                parser.error("--fit-target-mib must be positive and requires automatic expert placement.")
            args.moe_cache_mib = "auto"
            explicit.add("moe_cache_mib")
        if settings.get("moe_cache_mib") == "auto" and isinstance(args.moe_cache_mib, int):
            parser.error("This Inkling profile uses automatic expert placement. Select an explicit manual placement profile to set a numerical --moe-cache-mib.")
        if args.no_offload and args.moe_cache_mib is None and settings.get("moe_cache_mib") != "auto":
            args.moe_cache_mib = 0
            explicit.add("moe_cache_mib")
        if args.moe_cache_mib is None:
            args.moe_cache_mib = settings.get("moe_cache_mib", 0)
        if args.moe_cache_mib == "auto":
            if settings.get("host_kv") or explicit & {"cpu_moe_layers", "no_offload"}:
                parser.error("Automatic expert placement requires GPU KV and chooses the expert layers itself.")
            if args.fit_target_mib is None:
                args.fit_target_mib = settings.get("fit_target_mib", 2816)
        supported = {"context", "ctx", "ctxt", "port", "mem", "dry_run", "batch_tokens", "ubatch_tokens",
                     "cpu_moe_layers", "moe_cache_mib", "kv_cache", "no_offload", "cache_ram", "fit_target_mib", "quant",
                     "speculative", "draft_tokens", "draft_kv_cache"}
        if unsupported := explicit - supported:
            parser.error("Unsupported Inkling option: " + ", ".join("--" + key.replace("_", "-") for key in sorted(unsupported)))
    elif args.model == "inkling" and args.action in ("fetch", "pull") and args.settings_profile == "default":
        args.settings_profile = models["inkling"]["default_profile"]
    if getattr(args, "settings_profile", "default") != "default":
        profiles = models[args.model].get("profiles", {})
        if args.settings_profile not in profiles:
            available = ", ".join(("default", *profiles))
            parser.error(f"Unknown profile {args.settings_profile!r} for {args.model}. Available: {available}.")
        profile_settings: dict[str, Any] = dict(profiles[args.settings_profile])
        if args.model == "inkling":
            if args.action in ("fetch", "pull", "remove"):
                profile_settings = {"speculative": profile_settings.get("speculative", "none"),
                            "unsloth_runtime": bool(profile_settings.get("unsloth_runtime") or profile_settings.get("moe_cache_mib") == "auto")}
            if "speculative" in explicit and args.speculative != profile_settings.get("speculative", "none"):
                profile_settings.pop("draft_tokens", None)
        for key, value in profile_settings.items():
            if key not in explicit:
                setattr(args, key, Path(value) if key in ("kv_mean_center", "slot_save_path") else value)
        explicit.update(profile_settings)
    if getattr(args, "speculative", "none") == "dflash2" and args.model != qwen_strata.NAME:
        parser.error("--speculative dflash2 requires Strata Qwen.")
    if getattr(args, "draft_conf", None) is not None and (args.model != qwen_strata.NAME or args.speculative != "dflash2"):
        parser.error("--draft-conf requires Qwen with --speculative dflash2.")
    if getattr(args, "adaptive_fetch", None) is not None and args.model != qwen_strata.NAME:
        parser.error("--adaptive-fetch requires Strata Qwen.")
    if getattr(args, "speculative", "none") == "dspark" and args.model not in (deepseek_strata.NAME, "inkling"):
        parser.error("DSpark requires Inkling or deepseek-full.")
    if getattr(args, "dspark_conf", None) is not None and (args.model != deepseek_strata.NAME or args.speculative != "dspark"):
        parser.error("--dspark-conf requires deepseek-full with --speculative dspark.")
    if args.model == deepseek_strata.NAME and args.action in ("start", "serve") and args.speculative == "dspark":
        if "draft_tokens" not in explicit:
            args.draft_tokens = 5
    if args.model == qwen_strata.NAME and args.action in ("start", "serve"):
        if "speculative" not in explicit:
            args.speculative = "mtp"
        if "draft_tokens" not in explicit:
            args.draft_tokens = 5 if args.speculative == "dflash2" else models[args.model].get("draft_tokens", 4)
        if "lookup_tokens" not in explicit:
            args.lookup_tokens = 12
        if args.kv_vram_cells is None:
            context = args.context or models[args.model]["context"]
            args.kv_vram_cells = min(models[args.model].get("kv_vram_cells", context), context)
        if args.ple_io is None:
            args.ple_io = models[args.model].get("ple_io", "direct")
    if getattr(args, "mtp_head", "teacher") == "procreations" and (args.speculative != "mtp" or "mtp_procreations" not in models[args.model]):
        parser.error("The ProCreations head requires a supported model and --speculative mtp.")
    if args.action == "calibrate":
        if args.chunks < 1:
            parser.error("Calibration chunks must be positive.")
        calibrate(args, models[args.model])
        return 0
    if args.action in ("fetch", "pull", "remove"):
        STATE.mkdir(exist_ok=True)
        if args.speculative != "none" and args.speculative not in models[args.model] and not (
                (args.model == qwen_strata.NAME and args.speculative in ("mtp", "dflash2")) or
                (args.model == deepseek_strata.NAME and args.speculative == "dspark")):
            parser.error("This profile has no pinned artifact for the selected decoder.")
        if args.action == "remove":
            remove(args.model, args.speculative, args.dry_run, args.mtp_head, quant=args.quant)
        else:
            fetch(args.model, models[args.model], args.speculative, args.mtp_head, quant=args.quant,
                  unsloth_runtime=args.model == "inkling" and inkling_runtime(args) == "inkling-unsloth")
        return 0
    if not 0 <= args.cpu_offload_gb < float("inf"):
        parser.error("CPU offload must be finite and nonnegative.")
    if not 0 <= args.kv_offload_gb < float("inf"):
        parser.error("Host KV cache size must be finite and nonnegative.")
    if args.cpu_moe_layers is not None and not 0 <= args.cpu_moe_layers <= 49:
        parser.error("Use 0 to 49 CPU expert layers (Groot supports at most 40).")
    if args.gpu_layers is not None and not 0 <= args.gpu_layers <= 999:
        parser.error("Use 0 to 999 GPU layers.")
    if isinstance(args.moe_cache_mib, int) and args.moe_cache_mib < 0:
        parser.error("The MoE cache size must be nonnegative.")
    if models[args.model]["runtime"] != "flint" and args.prompt_lookup is None:
        args.prompt_lookup = False
    if models[args.model]["runtime"] in ("flint", "llamacpp") and args.prompt_lookup and args.lookup_type == "ngram-mod" and args.lookup_min_tokens is None:
        args.lookup_min_tokens = min(8, args.lookup_tokens)
    if models[args.model]["runtime"] in ("strata", "strata-qwen") and args.ubatch_tokens is not None:
        parser.error("--ubatch-tokens is a llama.cpp control. Use --prefill-chunk for Strata.")
    if args.ubatch_tokens is None:
        args.ubatch_tokens = models[args.model].get("ubatch_tokens", 256 if models[args.model]["runtime"] == "llamacpp" else 1024 if models[args.model]["runtime"] in ("dory-native", "flint") else 64)
    if (args.batch_tokens is not None and args.batch_tokens < 1) or args.ubatch_tokens < 1:
        parser.error("Batch sizes must be positive.")
    if args.cache_ram is not None and args.cache_ram < 0:
        parser.error("RAM cache size must be nonnegative.")
    if args.draft_tokens < 1:
        parser.error("Draft token count must be positive.")
    if args.lookup_tokens < 1:
        parser.error("Lookup token count must be positive.")
    if args.lookup_min_tokens is not None and not 0 <= args.lookup_min_tokens <= min(args.lookup_tokens, 1024):
        parser.error("Use 0 to --lookup-tokens (at most 1024) for --lookup-min-tokens.")
    if models[args.model]["runtime"] in ("flint", "llamacpp") and args.lookup_tokens > 1024:
        parser.error("The ngram-mod draft limit must not exceed 1024 tokens.")
    if args.draft_window is not None and args.draft_window > (args.context or models[args.model]["context"]):
        parser.error("The draft window must not exceed the target context.")
    if args.kv_vram_cells is not None and not 1 <= args.kv_vram_cells <= (args.context or models[args.model]["context"]):
        parser.error("Use 1 to the target context length for --kv-vram-cells.")
    if args.mma_decode_min_kv is not None and args.mma_decode_min_kv < 0:
        parser.error("The MMA threshold must be nonnegative.")
    if args.checkpoint_every is not None and args.checkpoint_every < 0:
        parser.error("Checkpoint spacing must be nonnegative.")
    if args.replay and args.speculative != "mtp":
        parser.error("--replay requires --speculative mtp.")
    if args.mtp_chain is not None and (models[args.model]["runtime"] != "prism" or args.speculative != "mtp"):
        parser.error("--mtp-chain and --no-mtp-chain require a Prism MTP decoder.")
    if args.expert_memory is not None and models[args.model]["runtime"] != "strata":
        parser.error("--expert-memory requires Strata DeepSeek.")
    if args.strata_slots is not None and models[args.model]["runtime"] != "strata":
        parser.error("--strata-slots and --no-strata-slots require Strata DeepSeek.")
    if args.ple_io is not None and models[args.model]["runtime"] != "strata-qwen":
        parser.error("--ple-io requires Strata Qwen.")
    if models[args.model]["runtime"] not in ("strata", "strata-qwen") and any(getattr(args, key) is not None for key in
            ("expert_ram_gb", "cpu_threads", "gpu_expert_cache_gb", "gpu_expert_fraction", "prefill_chunk", "vram_reserve_mib", "engram_cache_mib", "engram_threads", "thinking_budget")):
        parser.error("These expert-cache, CPU-lane, embedding, and thinking-budget flags require the Strata DeepSeek model.")
    return serve(args, models[args.model])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
    except KeyboardInterrupt:
        sys.exit(130)
