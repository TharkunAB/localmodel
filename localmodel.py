"""Fetch, remove, start, and inspect local models, including Groot and LFM3."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
CAPPED = ROOT / "libexec/capped"
ALIASES = {"bonsai": "bonsai2-27b", "bonsai2": "bonsai2-27b", "bonsai-ptq2": "bonsai-pq2",
           "groot": "groot-19b", "lfm": "lfm3-1.7b", "lfm3": "lfm3-1.7b",
           "deepseek": deepseek_strata.NAME, "deepseek-iq1-s": deepseek_strata.NAME, "strata": deepseek_strata.NAME}
DEFAULT_CACHE_RAM_MIB = 32768
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_START_MODEL = "27b-256k-optimized"


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


def tokens(value: str) -> int:
    """Read a context count. A k suffix means 1024 tokens."""
    try:
        count = int(value[:-1]) * 1024 if value.lower().endswith("k") else int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a token count such as 65536 or 256k.") from error
    if not 1 <= count <= 262144:
        raise argparse.ArgumentTypeError("Use 1 to 262144 tokens.")
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
    if model in (deepseek_strata.NAME, qwen_strata.NAME):
        backend = "strata"
    if context is None:
        props = request(port, "/props")
        context = props["default_generation_settings"]["n_ctx"]
    if not isinstance(context, int) or context < 1:
        raise RuntimeError(f"The model at port {port} did not report a context capacity.")
    settings = next((item for item in catalog().values() if item["alias"] == model), dict[str, Any]())
    return {
        "running": True, "model": model, "context": context,
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
        result["pid"] = managed.get("controller_pid")
        for key in ("cpu_offload_gb", "cpu_moe_layers", "moe_cache_mib", "batch_tokens", "ubatch_tokens", "command", "runtime", "rounds", "memory_limit",
                    "expert_ram_gb", "cpu_threads", "gpu_expert_cache_gb", "gpu_expert_fraction", "prefill_chunk", "vram_reserve_mib", "engram_cache_mib", "engram_threads", "thinking_budget"):
            result[key] = managed.get(key)
        if result["backend"] == "dory":
            result["rounds"] = managed.get("rounds")
            result["kv_offload_gb"] = managed.get("kv_offload_gb", 0)
        else:
            for key in ("speculative", "draft_tokens", "draft_kv_cache", "mtp_chain", "mtp_head", "checkpoint_every", "replay", "gdn_mma", "slot_save_path", "prompt_lookup", "lookup_type", "lookup_tokens", "lookup_min_tokens", "draft_window", "kv_vram_cells", "mma_decode_min_kv", "cache_ram_mib", "kv_cache", "kv_mean_center", "reasoning_effort"):
                result[key] = managed.get(key)
    return result


def installed(name: str, item: dict[str, Any]) -> bool:
    """Check whether the selected model files are present."""
    if item.get("runtime") == "strata-qwen":
        return qwen_strata.installed()
    if item.get("runtime") == "strata":
        return deepseek_strata.installed()
    if name == "dory":
        snapshots = ROOT / "dory/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4/snapshots"
        return any(snapshots.glob("*/*.safetensors"))
    path = ROOT / "bonsai/models" / item["directory"] / item["filename"]
    return path.is_file() and path.stat().st_size == item["size"]


def runtime_path(runtime: str = "stock") -> Path:
    """Return the pinned runtime directory for the selected model family."""
    info = read_json(ROOT / "config" / (f"{runtime}-runtime.json" if runtime in ("flint", "llamacpp") else "runtime.json"))
    directory = STATE / "runtimes" / runtime if runtime in ("flint", "llamacpp") else ROOT / "bonsai/runtime"
    return directory / info["release"] / ("llama-" + info["release"])


def fetch_runtime(runtime: str = "stock") -> None:
    """Download and verify the pinned CUDA runtime."""
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


def fetch(name: str, item: dict[str, Any], speculative: str = "none", mtp_head: str = "teacher") -> None:
    """Fetch a pinned model and its required runtime."""
    if item["runtime"] == "strata-qwen":
        qwen_strata.fetch()
        return
    if item["runtime"] == "strata":
        deepseek_strata.fetch()
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
    variants = [None, "dflash"] if speculative == "dflash" else [None if speculative == "none" else variant_name]
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
    if item["runtime"] in ("prism", "flint", "llamacpp"):
        with (STATE / "runtime.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            fetch_runtime(item["runtime"] if item["runtime"] in ("flint", "llamacpp") else "stock")


def remove(name: str, speculative: str = "none", dry_run: bool = False, mtp_head: str = "teacher") -> None:
    """Delete selected weights and download data. Keep configuration and runtimes."""
    if name == qwen_strata.NAME:
        qwen_strata.remove(dry_run)
        return
    if name == deepseek_strata.NAME:
        deepseek_strata.remove(dry_run)
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
    for variant in variants:
        command = [str(ROOT / ".venv/bin/python"), str(ROOT / "libexec/bonsai2-models"), "delete", name]
        if variant:
            command += ["--variant", variant]
        if dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)


def launch_command(args: argparse.Namespace, item: dict[str, Any]) -> list[str]:
    """Build a command with GPU weights and GPU KV cache."""
    if item["runtime"] == "strata-qwen":
        return qwen_strata.command(args, item)
    if item["runtime"] == "strata":
        return deepseek_strata.command(args, item)
    context = args.context or item["context"]
    cpu_moe_layers = getattr(args, "cpu_moe_layers", None)
    if args.runtime != "stock" and item["runtime"] != "prism":
        raise RuntimeError("--runtime selects an alternate Bonsai Prism runtime.")
    if item["runtime"] not in ("prism", "dory-native", "flint", "llamacpp") and (args.speculative != "none" or args.prompt_lookup or args.cache_ram is not None or args.kv_mean_center or args.kv_cache or args.reasoning_effort):
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
    if item["runtime"] in ("flint", "llamacpp"):
        if args.speculative != "none" or args.kv_mean_center or args.reasoning_effort:
            raise RuntimeError("This profile has no configured draft decoder, KV calibration, or reasoning-effort control.")
        if args.cpu_offload_gb or args.kv_offload_gb:
            raise RuntimeError("Use --cpu-moe-layers for MoE profiles. Attention and KV stay on the GPU.")
        cpu_moe = 0 if args.no_offload else cpu_moe_layers if cpu_moe_layers is not None else item.get("cpu_moe_layers", 0)
        if args.moe_cache_mib and cpu_moe < 3:
            raise RuntimeError("The MoE cache needs routed experts in RAM. Use --cpu-moe-layers 3 or more and omit --no-offload.")
        cache_ram = args.cache_ram if args.cache_ram is not None else item.get("cache_ram", DEFAULT_CACHE_RAM_MIB)
        model = ROOT / "bonsai/models" / item["directory"] / item["filename"]
        command = [str(runtime_path(item["runtime"]) / "llama-server"), "--model", str(model),
                   "--alias", item["alias"], "--ctx-size", str(context),
                   "--gpu-layers", "99", "--n-cpu-moe", str(cpu_moe), "--fit", "off",
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
        if args.prompt_lookup is not None:
            command += ["--spec-type", args.lookup_type if args.prompt_lookup else "none"]
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
    libraries = ([str(runtime_path(runtime)), str(ROOT / "bonsai/runtime/bonsai2/lib")] if runtime in ("stock", "flint", "llamacpp")
                 else [str(native_runtime_path(runtime)), str(native_runtime_path(runtime).parent / "lib")])
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    if runtime not in ("stock", "flint", "llamacpp"):
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


def serve(args: argparse.Namespace, item: dict[str, Any]) -> int:
    """Start one model and retain its metadata until the process stops."""
    strata = qwen_strata if item["runtime"] == "strata-qwen" else deepseek_strata
    command = launch_command(args, item)
    if args.mem is not None:
        if command[:2] == [str(CAPPED), "--mem"]:
            command = command[3:]
        command = [str(CAPPED), "--mem", args.mem, *command]
    environment = (prism_environment(args.runtime) if item["runtime"] == "prism" else
                   prism_environment(item["runtime"]) if item["runtime"] in ("flint", "llamacpp") else
                   prism_environment("dory-native") if item["runtime"] == "dory-native" else
                   strata.environment() if item["runtime"] in ("strata", "strata-qwen") else os.environ.copy())
    if item["runtime"] in ("flint", "llamacpp"):
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
            print(json.dumps({"command": command, "configuration": strata.settings(args, item)}, indent=2))
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
        if item["runtime"] in ("flint", "llamacpp"):
            if args.prompt_lookup is not None:
                unset += ["-u", "LLAMA_ARG_SPEC_TYPE"]
            if args.moe_cache_mib is not None:
                unset += ["-u", "LLAMA_ARG_MOE_CACHE_MIB"]
        prefix = ["env", *unset, *variables] if unset or variables else []
        print(shlex.join(prefix + command))
        return 0
    weights = item["mtp_procreations" if args.mtp_head == "procreations" else "mtp"] if args.speculative == "mtp" and item["runtime"] != "strata-qwen" else item
    if not installed(args.model, weights):
        suffix = f" --speculative {args.speculative}" if args.speculative != "none" else ""
        if args.mtp_head == "procreations":
            suffix += " --mtp-head procreations"
        raise RuntimeError(f"Model is missing. Run localmodel fetch {args.model}{suffix} first.")
    if args.speculative == "dflash" and not installed(args.model, item["dflash"]):
        raise RuntimeError(f"Optional weights are missing. Run localmodel fetch {args.model} --speculative {args.speculative}.")
    if item["runtime"] in ("prism", "dory-native", "flint", "llamacpp"):
        runtime = item["runtime"] if item["runtime"] in ("dory-native", "flint", "llamacpp") else args.runtime
        binary = (runtime_path(runtime) if runtime in ("stock", "flint", "llamacpp") else native_runtime_path(runtime)) / "llama-server"
        if not binary.is_file():
            hint = f"localmodel fetch {args.model}" if runtime in ("stock", "flint", "llamacpp") else f"python experiments/build-native.py {runtime} --run"
            raise RuntimeError(f"Runtime is missing. Run {hint} first.")
        if runtime not in ("stock", "flint", "llamacpp"):
            verify_native_build(runtime)
    STATE.mkdir(exist_ok=True)
    with (STATE / "server.lock").open("w") as lock, (strata.model_lock(shared=True) if item["runtime"] in ("strata", "strata-qwen") else nullcontext()):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another localmodel server is running. Stop it before switching models.") from None
        check_free(args.port)
        if item["runtime"] in ("strata", "strata-qwen"):
            strata.check_ready()
            config_name = "qwen-server.json" if item["runtime"] == "strata-qwen" else "deepseek-server.json"
            (STATE / config_name).write_text(json.dumps(strata.settings(args, item), indent=2) + "\n")
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        active = STATE / "active.json"
        active.write_text(json.dumps({"pid": process.pid, "controller_pid": os.getpid(), "profile": args.model,
            "command": command,
            "memory_limit": command[2] if command[:2] == [str(CAPPED), "--mem"] else None,
            "preset": args.preset,
            "runtime": args.runtime if item["runtime"] == "prism" else item["runtime"],
            "model": item["alias"], "port": args.port,
            "context": args.context or item["context"],
            "cpu_offload_gb": args.cpu_offload_gb,
            "cpu_moe_layers": (0 if args.no_offload else args.cpu_moe_layers if args.cpu_moe_layers is not None else item["cpu_moe_layers"]) if "cpu_moe_layers" in item else None,
            "moe_cache_mib": args.moe_cache_mib if "cpu_moe_layers" in item else None,
            "batch_tokens": None if item["runtime"] in ("strata", "strata-qwen") else args.batch_tokens or (512 if item["runtime"] == "dory" else 2048 if item["runtime"] in ("dory-native", "flint", "llamacpp") else 256),
            "ubatch_tokens": args.ubatch_tokens if item["runtime"] in ("prism", "dory-native", "flint", "llamacpp") else None,
            "rounds": args.rounds if item["runtime"] in ("dory", "dory-native") else None,
            "kv_offload_gb": args.kv_offload_gb if item["runtime"] == "dory" else None,
            "speculative": args.speculative, "draft_tokens": args.draft_tokens,
            "mtp_chain": environment.get("LLAMA_MTP_CHAIN") == "1" if mtp else None,
            "mtp_head": args.mtp_head if mtp else None,
            "checkpoint_every": args.checkpoint_every,
            "replay": environment.get("LLAMA_RS_REPLAY") == "1" if args.runtime == "bonsai-next" else None,
            "gdn_mma": "GGML_CUDA_GDN_MMA_DISABLE" not in environment if args.runtime == "bonsai-next" else None,
            "slot_save_path": str(args.slot_save_path.resolve()) if args.slot_save_path else None,
            "draft_kv_cache": args.draft_kv_cache, "prompt_lookup": args.prompt_lookup,
            "lookup_type": args.lookup_type if args.prompt_lookup else None,
            "lookup_tokens": args.lookup_tokens if args.prompt_lookup and args.lookup_type == "ngram-mod" else None,
            "lookup_min_tokens": args.lookup_min_tokens if args.prompt_lookup and args.lookup_type == "ngram-mod" else None,
            "draft_window": args.draft_window, "kv_vram_cells": args.kv_vram_cells,
            "mma_decode_min_kv": args.mma_decode_min_kv,
            "cache_ram_mib": args.cache_ram if args.cache_ram is not None else item.get("cache_ram", DEFAULT_CACHE_RAM_MIB if item["runtime"] in ("prism", "dory-native", "flint") else None),
            "kv_cache": args.kv_cache or item.get("kv_cache"),
            "kv_mean_center": str(args.kv_mean_center) if args.kv_mean_center else None,
            "reasoning_effort": (args.reasoning_effort or DEFAULT_REASONING_EFFORT) if item["runtime"] in ("prism", "strata", "strata-qwen") else None,
            "expert_ram_gb": args.expert_ram_gb if args.expert_ram_gb is not None else 36 if item["runtime"] == "strata" else 32 if item["runtime"] == "strata-qwen" else None,
            "cpu_threads": args.cpu_threads if args.cpu_threads is not None else 4 if item["runtime"] == "strata" else None,
            "gpu_expert_cache_gb": args.gpu_expert_cache_gb,
            "gpu_expert_fraction": args.gpu_expert_fraction,
            "prefill_chunk": (args.prefill_chunk if args.prefill_chunk is not None else 4096) if item["runtime"] == "strata" else args.prefill_chunk if item["runtime"] == "strata-qwen" else None,
            "vram_reserve_mib": (args.vram_reserve_mib if args.vram_reserve_mib is not None else 700) if item["runtime"] in ("strata", "strata-qwen") else None,
            "engram_cache_mib": (args.engram_cache_mib if args.engram_cache_mib is not None else 256) if item["runtime"] == "strata" else None,
            "engram_threads": (args.engram_threads if args.engram_threads is not None else 16) if item["runtime"] == "strata" else None,
            "thinking_budget": (args.thinking_budget if args.thinking_budget is not None else 32768) if item["runtime"] in ("strata", "strata-qwen") else None,
            "started": time.time()}, indent=2) + "\n")
        def interrupt(_number: int, _frame: Any) -> None:
            """Stop the child process when the launcher receives SIGTERM."""
            raise KeyboardInterrupt

        previous_handler = signal.signal(signal.SIGTERM, interrupt)
        try:
            return process.wait()
        except KeyboardInterrupt:
            if process.poll() is not None:
                return process.returncode
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            return 130
        finally:
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
    return ("qwen3.8: IQ2_XS + MTP3, 256K capacity, 64G cap/32 GiB resident budget.\n"
            "256K/64G, 16K OMP: 520.1 prompt / 35.49 decode tok/s; all benchmark checks pass.\n"
            "Peak 64 GiB; no swap/OOM. Linux memory monitoring only. Source: results/strata-0271-linux-guard-20261028/summary.md.\n"
            "Previous 128K capacity / 48G cap: 16K OMP 1396 prompt / 61.68 decode tok/s.\n"
            "Previous 48G code/recall pass; resume used fenced JSON (strict format fail). 40G/16 GiB: 609/46.34 tok/s, all checks pass.\n"
            "Source: results/strata-027-20261027/summary.md.\n\n"
            "deepseek: Unsloth Dynamic v3.1 UD-IQ1_S, 256K capacity. No local measurements yet.\n"
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
    help_tail = (f"Start defaults to {DEFAULT_START_MODEL}. Explicit flags override preset settings.\n"
                 "Models: bonsai2-27b, bonsai2-9b, qwen3.8, deepseek, flint, groot, lfm, dory, dory-gguf.\n"
                 "Use localmodel list for installed profiles and localmodel presets for named settings.\n"
                 "Use localmodel start --help for flags and localmodel start MODEL --dry-run to preview a launch.\n"
                 "Use localmodel --stats for recorded measurements and their limitations.")
    parser = argparse.ArgumentParser(prog="localmodel", description=__doc__, epilog=help_tail,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", action="store_true", help="Show recorded performance measurements and exit.")
    commands = parser.add_subparsers(dest="action")
    commands.add_parser("list", help="List model profiles and installed weights.")
    commands.add_parser("stop", help="Stop the model started by localmodel.")
    preset_parser = commands.add_parser("presets", help="Show named settings and measured performance.")
    preset_parser.add_argument("--json", action="store_true")
    fetch_parser = commands.add_parser("fetch", aliases=["pull"], help="Fetch pinned weights and the required runtime.")
    fetch_parser.add_argument("model")
    fetch_parser.add_argument("--speculative", choices=("none", "mtp", "dflash"), default="none",
                              help="Fetch a supported optional decoder (default: plain weights).")
    fetch_parser.add_argument("--mtp-head", choices=("teacher", "procreations"), default="teacher",
                              help="MTP head to fetch (default: selected preset, otherwise teacher). Requires --speculative mtp.")
    removal = commands.add_parser("remove", help="Delete selected weights and download data. Keep configuration and runtimes.",
                                  description="Remove the artifacts selected by fetch. Refuse weights that a running process uses.")
    removal.add_argument("model", help="Model name, alias, or preset. Presets select their configured decoder.")
    removal.add_argument("--speculative", choices=("none", "mtp", "dflash"), default="none",
                         help="Match fetch: none removes plain weights; mtp removes MTP weights; dflash removes plain and draft weights. Shared files affect all profiles that use them.")
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
    start.add_argument("--context", "--ctx", type=tokens)
    start.add_argument("--port", type=int, default=8899)
    start.add_argument("--mem", type=memory_size,
                       help="Hard RAM cap for the model process and its children, including Linux file cache. Bare numbers use GiB, for example --mem 32. Swap is disabled for this process tree. Omit to keep the profile cap. This does not reduce --cache-ram automatically.")
    start.add_argument("--expert-ram-gb", type=float, help="Strata expert RAM budget in GiB, inside --mem (DeepSeek: 36 with a 64 GiB cap; Qwen: 32 with a 64 GiB cap). Separate from llama.cpp --cache-ram.")
    start.add_argument("--cpu-threads", type=int, help="DeepSeek Strata CPU expert compute threads (default: 4; 0 uses the stock automatic count). I/O threads are separate.")
    start.add_argument("--gpu-expert-cache-gb", type=float, help="Legacy control. Official Strata requires --gpu-expert-fraction instead.")
    start.add_argument("--gpu-expert-fraction", type=float, help="DeepSeek Strata GPU expert-cache fraction, from 0 to 1, of free VRAM after dense weights and the reserve (default: stock automatic).")
    start.add_argument("--prefill-chunk", type=int, help="Strata prefill chunk tokens (DeepSeek: 4096, minimum effective 256; Qwen: automatic unless set).")
    start.add_argument("--vram-reserve-mib", type=int, help="Strata VRAM reserve for CUDA and desktop use, in MiB (default: 700).")
    start.add_argument("--engram-cache-mib", type=int, help="DeepSeek Strata hashed-embedding row-cache budget in MiB (default: 256; 0 disables).")
    start.add_argument("--engram-threads", type=int, help="DeepSeek Strata hashed-embedding SSD read threads (default: 16).")
    start.add_argument("--thinking-budget", type=int, help="Strata chat-template thinking budget in tokens (default: 32768). Client requests can turn thinking off.")
    start.add_argument("--runtime", choices=("stock", "bonsai-native", "bonsai-next"), default="stock",
                       help="Bonsai runtime (default: selected preset; stock for base model names). bonsai-next adds reviewed checkpoint and replay patches in a separate build.")
    start.add_argument("--rounds", "--loops", type=int, choices=(1, 2, 3, 4), default=2,
                       help="Dory recurrent rounds per token (default: 2).")
    offload = start.add_mutually_exclusive_group()
    offload.add_argument("--no-offload", action="store_true", help="Keep transformer layers and attention on the GPU. For Flint and Groot this also sets --cpu-moe-layers 0; weights plus context may exceed VRAM. RAM snapshots and active KV spill use separate controls.")
    offload.add_argument("--cpu-offload-gb", type=float, default=0, help="Dory CPU weight offload in GiB.")
    offload.add_argument("--cpu-moe-layers", type=int,
                         help="Place routed experts in the first N layers in RAM; keep attention on the GPU. Flint: default 33, range 0-49. Groot: default 24, range 0-40.")
    start.add_argument("--moe-cache-mib", type=int,
                       help="Flint/Groot GPU cache for host expert weights, in MiB. Requires CPU expert layers. 0 disables; omitted inherits LLAMA_ARG_MOE_CACHE_MIB (native default: 0). --fit off does not make room automatically.")
    start.add_argument("--kv-offload-gb", type=float, default=0,
                       help="Legacy vLLM Dory host KV cache in GiB. Attention stays on the GPU (default: 0).")
    start.add_argument("--batch-tokens", type=int)
    start.add_argument("--ubatch-tokens", type=int, help="Physical batch size (default: 64; Dory GGUF and Flint: 1024; Groot and LFM3: 256).")
    start.add_argument("--kv-cache", choices=("q4_0", "q8_0", "f16"), help="llama.cpp KV cache type (default: model profile).")
    start.add_argument("--kv-mean-center", type=Path, help="Prism model-specific KV mean file.")
    start.add_argument("--reasoning-effort", choices=("default", "low", "medium", "xhigh"),
                       help=f"Prism chat-template effort (default: {DEFAULT_REASONING_EFFORT}). The template must support the level.")
    start.add_argument("--cache-ram", type=int,
                       help=f"llama.cpp prompt snapshot cache in MiB (default: {DEFAULT_CACHE_RAM_MIB}; Groot: 8192; LFM3: 4096; 0 disables). This differs from active KV spill.")
    start.add_argument("--prompt-lookup", action=argparse.BooleanOptionalAction, default=None,
                       help="Bonsai, Flint, Groot, or LFM3 prompt lookup speculation. No draft weights required. Explicit Flint flags override LLAMA_ARG_SPEC_TYPE; omission preserves it.")
    start.add_argument("--lookup-type", choices=("ngram-simple", "ngram-mod"), default="ngram-simple",
                       help="Prompt lookup algorithm (default: ngram-simple). Requires --prompt-lookup.")
    start.add_argument("--lookup-tokens", type=int, default=32,
                       help="Independent ngram-mod lookup limit (default: 32). Neural depth still uses --draft-tokens.")
    start.add_argument("--lookup-min-tokens", type=int,
                       help="Minimum ngram-mod draft length (0 to --lookup-tokens). Requires prompt lookup with ngram-mod. Flint default: min(8, maximum); Bonsai keeps its runtime default unless set.")
    start.add_argument("--draft-window", type=tokens,
                       help="Native Bonsai MTP context window. Omit to use the full target context. Requires a native Bonsai runtime.")
    start.add_argument("--kv-vram-cells", type=int,
                       help="Native Bonsai active KV positions kept in VRAM. Remaining KV uses RAM. Requires --runtime bonsai-native.")
    start.add_argument("--mma-decode-min-kv", type=int,
                       help="Added native Bonsai wide-GQA MMA dispatch threshold (0 disables this override). Other MMA kernels remain available. Requires --runtime bonsai-native.")
    start.add_argument("--speculative", choices=("none", "mtp", "dflash"), default="none",
                       help="Prism optional draft decoder (default: none).")
    start.add_argument("--draft-tokens", type=int, default=2, help="Maximum neural draft tokens per step (default: 2; Qwen: 3).")
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
                       help="Draft decoder KV cache type (default: q8_0).")
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
        print(f"{'MODEL':<28} {'WEIGHTS':<10} {'CONTEXT':>8}  {'RUNTIME':<8}  OPTIONAL DECODERS")
        for name, item in models.items():
            state = "installed" if installed(name, item) else "missing"
            optional = ", ".join(f"{variant}:{'ready' if installed(name, item[variant]) else 'missing'}"
                                 for variant in ("mtp", "mtp_procreations", "dflash") if variant in item) or "-"
            print(f"{name:<28} {state:<10} {item['context']:>8}  {item['runtime']:<8}  {optional}")
        return 0
    args.preset = args.model if args.model in presets.load() else None
    if args.preset:
        settings = presets.get(args.preset)["settings"]
        explicit = {argument.split("=", 1)[0][2:].replace("-", "_")
                    for argument in sys.argv[2:] if argument.startswith("--")}
        for alias, destination in (("ctx", "context"), ("loops", "rounds"), ("no_prompt_lookup", "prompt_lookup"),
                                   ("no_mtp_chain", "mtp_chain"), ("no_replay", "replay"), ("no_gdn_mma", "gdn_mma")):
            if alias in explicit:
                explicit.add(destination)
        for key, value in settings.items():
            if key == "model" or (args.action in ("start", "serve", "fetch", "pull", "remove") and key not in explicit):
                setattr(args, key, Path(value) if key == "kv_mean_center" else value)
    args.model = ALIASES.get(args.model, args.model)
    if args.model not in models:
        parser.error("Unknown model. Use localmodel list.")
    if args.model == qwen_strata.NAME and args.action in ("start", "serve"):
        explicit_flags = {arg.split("=", 1)[0] for arg in sys.argv[2:]}
        if "--speculative" not in explicit_flags:
            args.speculative = "mtp"
        if "--draft-tokens" not in explicit_flags:
            args.draft_tokens = 3
    if getattr(args, "mtp_head", "teacher") == "procreations" and (args.speculative != "mtp" or "mtp_procreations" not in models[args.model]):
        parser.error("The ProCreations head requires a supported model and --speculative mtp.")
    if args.action == "calibrate":
        if args.chunks < 1:
            parser.error("Calibration chunks must be positive.")
        calibrate(args, models[args.model])
        return 0
    if args.action in ("fetch", "pull", "remove"):
        STATE.mkdir(exist_ok=True)
        if args.speculative != "none" and args.speculative not in models[args.model] and not (args.model == qwen_strata.NAME and args.speculative == "mtp"):
            parser.error("This profile has no pinned artifact for the selected decoder.")
        if args.action == "remove":
            remove(args.model, args.speculative, args.dry_run, args.mtp_head)
        else:
            fetch(args.model, models[args.model], args.speculative, args.mtp_head)
        return 0
    if not 0 <= args.cpu_offload_gb < float("inf"):
        parser.error("CPU offload must be finite and nonnegative.")
    if not 0 <= args.kv_offload_gb < float("inf"):
        parser.error("Host KV cache size must be finite and nonnegative.")
    if args.cpu_moe_layers is not None and not 0 <= args.cpu_moe_layers <= 49:
        parser.error("Use 0 to 49 CPU expert layers (Groot supports at most 40).")
    if args.moe_cache_mib is not None and args.moe_cache_mib < 0:
        parser.error("The MoE cache size must be nonnegative.")
    if models[args.model]["runtime"] != "flint" and args.prompt_lookup is None:
        args.prompt_lookup = False
    if models[args.model]["runtime"] in ("flint", "llamacpp") and args.prompt_lookup and args.lookup_type == "ngram-mod" and args.lookup_min_tokens is None:
        args.lookup_min_tokens = min(8, args.lookup_tokens)
    if models[args.model]["runtime"] in ("strata", "strata-qwen") and args.ubatch_tokens is not None:
        parser.error("--ubatch-tokens is a llama.cpp control. Use --prefill-chunk for Strata.")
    if args.ubatch_tokens is None:
        args.ubatch_tokens = 256 if models[args.model]["runtime"] == "llamacpp" else 1024 if models[args.model]["runtime"] in ("dory-native", "flint") else 64
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
