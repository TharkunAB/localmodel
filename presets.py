"""Read named launch settings and their measured performance."""
# pyright: strict

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

DATA = Path(__file__).resolve().parent / "config/presets.json"


def load() -> dict[str, dict[str, Any]]:
    """Read presets. Reject unsupported settings and incomplete measurements."""
    entries = cast(dict[str, dict[str, Any]], json.loads(DATA.read_text()))
    try:
        for name, preset in entries.items():
            settings: dict[str, Any] = preset["settings"]
            measured: dict[str, Any] = preset["measurement"]
            model = settings["model"]
            dory = model in ("dory", "dory-gguf")
            flint = model == "flint"
            required = {"model", "context", "cpu_offload_gb"}
            required |= {"cpu_moe_layers", "kv_cache", "cache_ram"} if flint else {"rounds"} if dory else {
                "speculative", "prompt_lookup", "kv_cache", "draft_kv_cache", "draft_tokens", "cache_ram"}
            if model == "dory-gguf":
                required |= {"kv_cache", "cache_ram"}
            optional = ({"batch_tokens", "ubatch_tokens"} if flint else {"kv_offload_gb", "batch_tokens"} if model == "dory" else
                        {"batch_tokens", "ubatch_tokens"} if dory else
                        {"kv_mean_center", "batch_tokens", "ubatch_tokens", "reasoning_effort", "runtime", "mtp_chain",
                         "draft_window", "kv_vram_cells", "mma_decode_min_kv", "lookup_type", "lookup_tokens",
                         "mtp_head", "checkpoint_every", "replay", "gdn_mma"})
            valid = model in ("bonsai2-9b", "bonsai2-27b", "dory", "dory-gguf", "flint") and required <= set(settings) <= required | optional
            if "kv_mean_center" in settings:
                valid &= not dory and isinstance(settings["kv_mean_center"], str) and bool(settings["kv_mean_center"])
            if "reasoning_effort" in settings:
                valid &= settings["reasoning_effort"] in ("default", "low", "medium", "xhigh")
            valid &= type(settings["context"]) is int and 1 <= settings["context"] <= 262144
            valid &= type(settings["cpu_offload_gb"]) in (int, float) and settings["cpu_offload_gb"] == 0 and isinstance(preset["description"], str)
            if flint:
                valid &= type(settings["cpu_moe_layers"]) is int and 0 <= settings["cpu_moe_layers"] <= 49
            elif dory:
                valid &= type(settings["rounds"]) is int and settings["rounds"] in ((1, 2, 4) if model == "dory" else (1, 2, 3, 4))
                host_gib = settings.get("kv_offload_gb", 0)
                valid &= isinstance(host_gib, (int, float)) and not isinstance(host_gib, bool) and 0 <= host_gib < float("inf")
            else:
                valid &= settings["speculative"] in ("none", "mtp", "dflash")
                valid &= type(settings["prompt_lookup"]) is bool
                valid &= all(settings[key] in ("q4_0", "q8_0", "f16") for key in ("kv_cache", "draft_kv_cache"))
                valid &= type(settings["draft_tokens"]) is int and settings["draft_tokens"] > 0
                valid &= settings.get("runtime", "stock") in ("stock", "bonsai-native", "bonsai-next")
                if "mtp_head" in settings:
                    valid &= settings["mtp_head"] in ("teacher", "procreations") and settings["speculative"] == "mtp"
                    valid &= settings["mtp_head"] == "teacher" or model == "bonsai2-27b"
                if "checkpoint_every" in settings:
                    valid &= settings.get("runtime") == "bonsai-next" and type(settings["checkpoint_every"]) is int and settings["checkpoint_every"] >= 0
                for key in ("replay", "gdn_mma"):
                    if key in settings:
                        valid &= settings.get("runtime") == "bonsai-next" and type(settings[key]) is bool
                if settings.get("replay"):
                    valid &= settings["speculative"] == "mtp"
                if "mtp_chain" in settings:
                    valid &= type(settings["mtp_chain"]) is bool and settings["speculative"] == "mtp"
                for key in ("draft_window", "kv_vram_cells", "mma_decode_min_kv"):
                    if key in settings:
                        valid &= settings.get("runtime") in ("bonsai-native", "bonsai-next") and type(settings[key]) is int
                        valid &= settings[key] >= (0 if key == "mma_decode_min_kv" else 1)
                        if key != "mma_decode_min_kv":
                            valid &= settings[key] <= settings["context"]
                if "draft_window" in settings:
                    valid &= settings["speculative"] == "mtp"
                if "lookup_type" in settings:
                    valid &= settings["lookup_type"] in ("ngram-simple", "ngram-mod")
                    valid &= settings["lookup_type"] == "ngram-simple" or settings["prompt_lookup"]
                if "lookup_tokens" in settings:
                    valid &= type(settings["lookup_tokens"]) is int and settings["lookup_tokens"] > 0
                    valid &= settings["prompt_lookup"] and settings.get("lookup_type") == "ngram-mod"
            if model != "dory":
                valid &= settings["kv_cache"] in ("q4_0", "q8_0", "f16")
                valid &= type(settings["cache_ram"]) is int and settings["cache_ram"] >= 0
            for key in ("batch_tokens", "ubatch_tokens"):
                if key in settings:
                    valid &= type(settings[key]) is int and settings[key] > 0
            valid &= measured["status"] in ("ok", "pending", "failed", "unmeasured")
            values: list[Any] = [measured.get(key) for key in ("ttft_seconds", "decode_tps", "gpu_mib")]
            if measured["status"] == "ok":
                valid &= all(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 for value in values)
                valid &= isinstance(measured.get("source"), str) and bool(measured["source"])
            else:
                valid &= all(value is None for value in values)
            for run in measured.get("workloads", []):
                valid &= type(run["prompt_tokens"]) is int and 0 < run["prompt_tokens"] <= settings["context"]
                valid &= all(isinstance(run[key], (int, float)) and run[key] > 0 for key in ("everyday_tps", "code_tps"))
                valid &= isinstance(run["source"], str) and bool(run["source"])
            if "opencode" in measured:
                coding = measured["opencode"]
                valid &= all(isinstance(coding[key], (int, float)) and coding[key] > 0 for key in ("native_tps", "workflow_tps", "wall_seconds"))
                valid &= type(coding["output_tokens"]) is int and coding["output_tokens"] > 0
                valid &= isinstance(coding["source"], str) and bool(coding["source"])
            if "opencode_long" in measured:
                coding = measured["opencode_long"]
                valid &= all(type(coding[key]) in (int, float) and 0 < coding[key] < float("inf")
                             for key in ("pre_tps", "post_tps", "cold_ttft_seconds", "warm_seconds"))
                valid &= type(coding["passed_phases"]) is int and 0 <= coding["passed_phases"] <= 6
                valid &= isinstance(coding["source"], str) and bool(coding["source"].strip())
            if not valid:
                raise ValueError(f"Invalid preset: {name}.")
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid preset data in {DATA}: {error}") from error
    return entries


def get(name: str) -> dict[str, Any]:
    """Return one validated preset."""
    entries = load()
    if name not in entries:
        raise ValueError(f"Unknown preset {name!r}. Choose: {', '.join(entries)}.")
    return entries[name]


def stats_text() -> str:
    """Show launch settings and recorded benchmark results."""
    lines = ["RTX 4070 SUPER 12 GiB. K = 1024 tokens. Each measurement records its date and settings in config/presets.json.",
             "Attention computation uses the GPU. Profile notes identify CPU expert weights. Capacity measurements appear where available.",
             "NAME                     MODEL          CTX   DECODER              KV        RAM MiB  TTFT s / tok/s / GPU MiB"]
    for name, preset in load().items():
        settings: dict[str, Any] = preset["settings"]
        measured: dict[str, Any] = preset["measurement"]
        dory = settings["model"] in ("dory", "dory-gguf")
        decoder = f"{settings['rounds']} rounds" if dory else "none" if settings["model"] == "flint" else f"{settings['speculative']}:{settings['draft_tokens']}" if settings["speculative"] != "none" else "none"
        if settings.get("prompt_lookup"):
            decoder += "+lookup"
        kv = "fp8" if settings["model"] == "dory" else settings["kv_cache"]
        if settings.get("kv_mean_center"):
            kv += "+cal"
        timing = f"{measured['ttft_seconds']:.2f} / {measured['decode_tps']:.2f} / {measured['gpu_mib']:.0f}" if measured["status"] == "ok" else measured["status"]
        host_mib = settings.get('kv_offload_gb', 0) * 1024 if settings["model"] == "dory" else settings['cache_ram']
        lines.append(f"{name:<24} {settings['model']:<14} {settings['context'] // 1024:>3}K  {decoder:<20} {kv:<9} {host_mib:>7.0f}  {timing}")
        if measured.get("note"):
            lines.append(f"  {measured['note']}")
        if settings["model"] == "flint":
            lines.append(f"  Official llama.cpp b11803. CPU MoE cutoff: {settings['cpu_moe_layers']}. Attention and active KV stay on the GPU.")
        if settings.get("runtime") in ("bonsai-native", "bonsai-next"):
            native = {"runtime": settings["runtime"], "draft_window": settings.get("draft_window", "full"),
                      "prompt_lookup": settings["prompt_lookup"], "lookup_type": settings.get("lookup_type", "ngram-simple"),
                      "lookup_tokens": settings.get("lookup_tokens", 32),
                      "kv_vram_cells": settings.get("kv_vram_cells", "all"),
                      "mma_decode_min_kv": settings.get("mma_decode_min_kv", "runtime default"),
                      "mtp_chain": settings.get("mtp_chain", "environment/default"),
                      "mtp_head": settings.get("mtp_head", "teacher"),
                      "checkpoint_every": settings.get("checkpoint_every", 0),
                      "replay": settings.get("replay", False),
                      "gdn_mma": settings.get("gdn_mma", "runtime default"),
                      "batch_tokens": settings.get("batch_tokens", 256),
                      "ubatch_tokens": settings.get("ubatch_tokens", 64),
                      "reasoning_effort": settings.get("reasoning_effort", "xhigh")}
            lines.append("  " + json.dumps(native, sort_keys=True))
    lines.append("GPU figures are total device use. Pending presets have no completed matching measurement.")
    lines.append("Historical capacity runs use repetitive prompts that favor speculation. See README.md for application and cache trials.")
    lines.append("RAM is a host cache limit: llama.cpp prompt snapshots for Bonsai, Flint, and Dory GGUF, or host KV blocks for original Dory.")
    lines.append("Flint additionally stores CPU expert weights in RAM. Its preset has no completed local benchmark.")
    lines.append("Dory GGUF uses a separate native runtime and Q8 KV by default. Its cache_ram stores prompt snapshots.")
    lines.append("Bonsai runtime selects stock or the separately built bonsai-native runtime. Native controls require bonsai-native.")
    lines.append("draft_window limits MTP draft history. Omit it for full history. The target context stays unchanged.")
    lines.append("kv_vram_cells keeps that many active KV positions in VRAM. Remaining active KV uses RAM. This differs from snapshot retention.")
    lines.append("mma_decode_min_kv selects the added native MMA dispatch threshold; 0 disables that override. Other MMA kernels remain available.")
    lines.append("ngram-mod lookup_tokens is independent of neural draft_tokens.")
    lines.append("Prism effort defaults to xhigh. Use --reasoning-effort medium for shorter reasoning.")
    lines.append("Historical 2026-10-17/18 Bonsai capacity timings used medium effort and an 8192 MiB host cache.")
    lines.append("New measurements use their recorded settings. The separate 2026-10-19 27B repeat used xhigh and a 32768 MiB cache.")
    lines.append("27b-256k capacity and long OpenCode rows now match MTP2 with chaining on. Older short MTP1 records remain in its JSON metadata.")
    lines += ["", "Native application prompts. Code copies and edits supplied text; ordinary generation uses different text.",
              "PRESET                   INPUT TOKENS  ORDINARY tok/s  CODE tok/s"]
    for name, preset in load().items():
        for run in preset["measurement"].get("workloads", []):
            lines.append(f"{name:<24} {run['prompt_tokens']:>12}  {run['everyday_tps']:>14.2f}  {run['code_tps']:>10.2f}")
    lines += ["", "Actual OpenCode tasks: two file reads, one edit, and passing external tests.",
              "PRESET                   NATIVE tok/s  WORKFLOW tok/s  TASK s  OUTPUT TOKENS"]
    for name, preset in load().items():
        coding = preset["measurement"].get("opencode")
        if coding:
            lines.append(f"{name:<24} {coding['native_tps']:>12.2f}  {coding['workflow_tps']:>14.2f}  {coding['wall_seconds']:>6.2f}  {coding['output_tokens']:>13}")
    lines.append("Workflow rates include prompt and tool time. Output lengths differ. These small tasks do not rank general coding quality.")
    long_records = [(name, preset["measurement"]["opencode_long"]) for name, preset in load().items()
                    if "opencode_long" in preset["measurement"]]
    if long_records:
        lines += ["", "Long OpenCode with compaction. Native rates exclude prefill and tool time.",
                  "PRESET                   PRE tok/s  POST tok/s  COLD TTFT s  WARM s  PASSED"]
        for name, coding in long_records:
            lines.append(f"{name:<24} {coding['pre_tps']:>9.2f}  {coding['post_tps']:>10.2f}  {coding['cold_ttft_seconds']:>11.2f}  {coding['warm_seconds']:>6.2f}  {coding['passed_phases']}/6")
            lines.append(f"  Source: {coding['source']}")
            if coding.get("note"):
                lines.append(f"  {coding['note']}")
        lines.append("Warm time covers the five follow-ups after the first coding turn, including compaction. Check passed phases before choosing a preset.")
    lines += ["", "Earlier 27B depth/chaining repeat, 2026-10-19 at 14:48 CEST: 256K window, about 213000 input tokens, xhigh, 32 GiB cache.",
              "Stock MTP depth 2 with chaining in that repeat: 53.8 tok/s before compaction; 87.9 tok/s after compaction at about 18000-20000 input tokens.",
              "Warm follow-ups including compaction: 72.1 s. Cold TTFT: 404.5 s; warm long-history TTFT: about 1.3 s.",
              "Depth 2 without chaining: 58.4/92.4 tok/s. Summary lengths differ, so workflow time does not isolate chain speed.",
              "All six cases passed. Depth-1 control rates vary without chain execution. See README.md for limits.",
              "Use: localmodel start. This selects 27b-256k-optimized, two draft tokens, chaining on, and a 16K draft window.",
              "Use --draft-tokens N to change depth and --no-mtp-chain to disable chaining. Depth 1 cannot chain.",
              "Full depth 1/2/4 and chaining repeat: ~/Code/local-models/results/27b-mtp-chain-256k-rerun1/summary.md."]
    lines.append("Full cache, OpenCode, RAM, and audit results: ~/Code/local-models/results/optimization-summary.md and README.md.")
    return "\n".join(lines)
