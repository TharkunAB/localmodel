"""Summarize saved native-runtime experiments without contacting model services."""
# pyright: strict

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import statistics
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUN = HERE / "runs/20261019-quiet-baseline"
DEFAULT_OUTPUT = ROOT / "results/native-long-context-20261019"
spec = importlib.util.spec_from_file_location("native_summary_helpers", ROOT / "results/summarize-mtp-chain.py")
assert spec is not None and spec.loader is not None
helpers: Any = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
Record = dict[str, Any]
PHASES = ["clamp_fix", "index_boundary_fix", "pre_compaction_recall", "manual_compaction",
          "post_compaction_fix", "post_compaction_recall"]


def utc_time(value: Any) -> datetime | None:
    """Read an ISO timestamp only when it includes a time zone."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except ValueError:
        return None


def load_overlap(trial: Record, events: list[Record], now: datetime) -> Record:
    """Compare saved trial intervals with reported host load intervals."""
    if not trial or trial.get("status") in ("pending", "not planned"):
        return {"status": "not measured", "overlapped": False, "events": []}
    start = utc_time(trial.get("started_utc"))
    elapsed = trial.get("elapsed_seconds")
    running = trial.get("status") == "running"
    end = now if running else None
    if not running and start is not None and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and 0 <= elapsed < float("inf"):
        end = start + timedelta(seconds=elapsed)
    if start is None or end is None or end < start:
        return {"status": "unknown trial interval", "overlapped": None, "events": []}
    overlaps: list[Record] = []
    for event in events:
        event_start = utc_time(event.get("start_utc"))
        event_end = now if event.get("end_utc") is None else utc_time(event.get("end_utc"))
        if event_start is None or event_end is None:
            continue
        overlap_start, overlap_end = max(start, event_start), min(end, event_end)
        if overlap_start < overlap_end:
            overlaps.append({**event, "overlap_start_utc": overlap_start.isoformat(),
                             "overlap_end_utc": overlap_end.isoformat(),
                             "overlap_seconds": (overlap_end - overlap_start).total_seconds()})
    return {"status": "reported host load overlap" if overlaps else "no reported overlap", "overlapped": bool(overlaps),
            "trial_start_utc": start.isoformat(), "trial_end_utc": end.isoformat(),
            "end_basis": "current UTC for running case" if running else "started_utc plus elapsed_seconds", "events": overlaps}


def usage_range(requests: list[Record], field: str = "prompt_tokens") -> Record:
    """Keep actual provider token counts separate from configured context."""
    values: list[int] = []
    for request in requests:
        usage: Record = request.get("usage") or {}
        value = usage.get(field)
        if isinstance(value, (int, float)):
            values.append(int(value))
    return {"minimum": min(values) if values else None, "maximum": max(values) if values else None,
            "sum": sum(values) if values else None, "requests_with_usage": len(values)}


def phase(step: Record, requests: list[Record], model: str, running: bool) -> Record:
    """Check saved fixture, tool, routing, recall and compaction evidence."""
    name = str(step["name"])
    selected = [request for request in requests if request.get("phase") == name]
    calls: list[Record] = step.get("tool_calls", [])
    tools_ok = all(call.get("name") in {"read", "edit"} and call.get("state", {}).get("status") == "completed" for call in calls)
    pending = step.get("pending_permissions", {}).get("data", [])
    errors = [request.get("errors") for request in selected if request.get("errors")]
    messages: list[Record] = step.get("assistant_messages", [])
    route_ok = all(message.get("model", {}).get("id") == model and message.get("model", {}).get("providerID") == "local"
                   and not message.get("error") for message in messages)
    checks: Record = {"tools_ok": tools_ok, "tool_count": len(calls), "pending_permissions": pending,
                      "request_errors": errors, "routing_ok": route_ok, "correctness_error": step.get("correctness_error")}
    passed = tools_ok and not pending and not errors and route_ok and not step.get("correctness_error")
    if name.endswith("fix"):
        verified: Record = step.get("verification", {})
        checks["fixture_pass"] = verified.get("test_returncode") == 0 if verified else None
        checks["completed_read_and_edit"] = {"read", "edit"}.issubset({call.get("name") for call in calls})
        passed = passed and checks["fixture_pass"] is True and checks["completed_read_and_edit"]
    if "recall" in name:
        facts: Record = step.get("facts_present") or {}
        checks["facts_present"] = facts
        checks["recall_pass"] = all(facts.values()) if facts else None
        passed = passed and checks["recall_pass"] is True
    compactions: list[Record] = step.get("compactions", [])
    if name == "manual_compaction":
        compact: Record = compactions[0] if compactions else {}
        checks["compaction_completed"] = compact.get("status") == "completed" and bool(compact.get("summary"))
        checks["compaction_routing_ok"] = compact.get("model", {}).get("providerID") == "local" and compact.get("model", {}).get("id") == model
        checks["summary_facts_present"] = step.get("summary_facts_present")
        passed = passed and checks["compaction_completed"] and checks["compaction_routing_ok"]
    if step.get("status") == "failed":
        status = "failed"
    elif passed and "wall_seconds" in step:
        status = "passed"
    elif running and not step.get("status"):
        status = "running"
    else:
        status = "failed" if "wall_seconds" in step else "partial"
    return {"name": name, "status": status, "wall_seconds": step.get("wall_seconds"),
            "native": helpers.weighted([step]), "checks": checks, "actual_input_tokens": usage_range(selected),
            "actual_output_tokens": usage_range(selected, "completion_tokens"),
            "tool_status_counts": step.get("tool_status_counts", {}),
            "requests": [{key: request.get(key) for key in ("request_id", "phase", "ttft_seconds", "wall_seconds", "usage", "timings", "finish_reason", "errors")}
                         for request in selected],
            "compaction_summary_characters": len(str(compactions[0].get("summary", ""))) if compactions else None}


def hook_result(directory: Path, trial: Record, name: str, running: bool) -> Record:
    """Read a saved hook. Keep user cancellation separate from failure."""
    hook: Record = trial.get("hooks", {}).get(name, {})
    result: Record = helpers.read(directory / f"{name}.json") or hook.get("result") or {}
    requested = name in trial.get("commands", {}).get("hooks", {})
    status = result.get("status")
    if not result:
        status = "pending" if requested else "not run"
    elif running and name not in trial.get("hooks", {}):
        status = "running" if status != "ok" else "ok; finishing"
    elif hook.get("returncode") not in (None, 0):
        status = "failed"
    cancellation: Record = helpers.read(directory / "cancellation.json")
    if cancellation.get("status") == "canceled" and cancellation.get("hook") == name:
        status = "canceled"
    return {"status": status or "partial", "returncode": hook.get("returncode"), "result": result}


def summarize(directory: Path, config: Record, planned: bool, events: list[Record] | None = None,
              now: datetime | None = None) -> Record:
    """Read one case from small saved logs and measurement records."""
    trial: Record = helpers.read(directory / "trial.json")
    cancellation: Record = helpers.read(directory / "cancellation.json")
    running = trial.get("status") == "running"
    requests: list[Record] = helpers.rows(directory / "proxy/requests.jsonl")
    requests.sort(key=lambda request: str(request.get("started_at", "")))
    workflow_hook = hook_result(directory, trial, "opencode", running)
    workflow: Record = workflow_hook["result"]
    steps: list[Record] = workflow.get("steps", [])
    settings: Record = trial.get("live_settings", {})
    model = str(settings.get("model") or config.get("model") or workflow.get("localmodel", {}).get("model", "unknown"))
    phases = [phase(step, requests, model, running) for step in steps]
    if trial:
        status = trial.get("status", "partial")
    else:
        status = "partial" if workflow or requests else "pending" if planned else "not planned"
    if not running and cancellation.get("status") == "canceled":
        status = "canceled"
    if trial.get("cleanup_error"):
        status = "failed"
    before = [step for step in steps if step.get("name") in PHASES[:3]]
    after = [step for step in steps if step.get("name") in PHASES[4:]]
    warm = [step for step in steps if step.get("name") in PHASES[1:] and "wall_seconds" in step]
    empty: Record = {}
    first = next((request for request in requests if request.get("phase") == "clamp_fix"), empty)
    compact = next((step for step in phases if step["name"] == "manual_compaction"), empty)
    long_requests = [request for request in requests if request.get("phase") in PHASES[:3]]
    post_requests = [request for request in requests if request.get("phase") in PHASES[4:]]
    warm_first: list[float] = []
    for name in PHASES[1:3]:
        request = next((entry for entry in requests if entry.get("phase") == name), empty)
        if isinstance(request.get("ttft_seconds"), (int, float)):
            warm_first.append(float(request["ttft_seconds"]))
    capacity_hook = hook_result(directory, trial, "capacity", running)
    capacity: Record = capacity_hook["result"]
    grammar_hook = hook_result(directory, trial, "grammar107", running)
    grammar: Record = grammar_hook["result"]
    expected_call: Record = helpers.read(HERE / "tools-107.json").get("expected", {})
    returned_calls: list[Record] = grammar.get("returned_calls", [])
    grammar_requests = [request for request in requests if request.get("phase") == "grammar107"]
    grammar_request: Record = grammar_requests[0] if grammar_requests else {}
    grammar_usage: Record = grammar.get("usage") or grammar_request.get("usage") or {}
    grammar_timings: Record = grammar.get("timings") or grammar_request.get("timings") or {}
    windows_raw: Record = helpers.read(directory / "windows-memory.json") or trial.get("windows_memory") or {}
    windows: Record | None = None
    if windows_raw:
        vm_summaries: list[Record] = windows_raw.get("summary", [])
        windows = {"status": windows_raw.get("status"), "error": windows_raw.get("error"),
                   "phase": "after all timed hooks; sampling requires confirmed idle", "linux_pid_attribution_available": False,
                   "vm_processes": [{"process_name": vm.get("process_name"), "windows_pid": vm.get("windows_pid"),
                                     "valid_counter_samples": vm.get("valid_counter_samples"),
                                     "invalid_counter_samples": vm.get("invalid_counter_samples"),
                                     "timestamps": len(vm.get("sample_totals", [])),
                                     "peak_dedicated_sum_mib": vm.get("peak_dedicated_sum_mib"),
                                     "peak_shared_sum_mib": vm.get("peak_shared_sum_mib")}
                                    for vm in vm_summaries]}
    samples: list[Record] = helpers.rows(directory / "resources.jsonl")
    resource_stats: Record = {"sample_count": len(samples)}
    for key in ("gpu_temperature_c", "gpu_watts", "gpu_sm_clock_mhz", "gpu_memory_clock_mhz", "linux_available_memory_kib"):
        observations = [float(sample[key]) for sample in samples if type(sample.get(key)) in (int, float)]
        resource_stats[key] = {"minimum": min(observations) if observations else None,
                               "median": statistics.median(observations) if observations else None,
                               "maximum": max(observations) if observations else None}
    loads = [float(sample["load_average"][0]) for sample in samples
             if isinstance(sample.get("load_average"), list) and sample["load_average"]
             and type(sample["load_average"][0]) in (int, float)]
    resource_stats["linux_load_1m_peak"] = max(loads) if loads else None
    peak: Record = trial.get("resources", {})
    gpu = [float(sample["gpu_mib"]) for sample in samples if isinstance(sample.get("gpu_mib"), (int, float))]
    scope = [int(sample["memory.peak"]) for sample in samples if str(sample.get("memory.peak", "")).isdigit()]
    rss = [int(sample["rss_kib"]) for sample in samples if isinstance(sample.get("rss_kib"), (int, float))]
    for key, values in (("gpu_mib", gpu), ("scope_memory_bytes", scope), ("rss_kib", rss)):
        if isinstance(peak.get(key), (int, float)):
            values.append(peak[key])
    return {"name": config["name"], "directory": str(directory), "status": status,
            "model": model, "stage": config.get("stage"), "case": config, "live_settings": settings,
            "error": trial.get("error") or workflow.get("error"), "cleanup_error": trial.get("cleanup_error"),
            "cancellation": cancellation or None,
            "workload_version": workflow.get("workload_version"), "seeded_prompt_estimate": workflow.get("seeded_prompt_estimate"),
            "configured_context_tokens": settings.get("context"), "actual_long_input_tokens": usage_range(long_requests),
            "actual_post_compaction_input_tokens": usage_range(post_requests), "cold_ttft_seconds": first.get("ttft_seconds"),
            "warm_long_ttft_median_seconds": statistics.median(warm_first) if warm_first else None,
            "pre_compaction_native": helpers.weighted(before), "post_compaction_native": helpers.weighted(after),
            "warm_workflow_seconds": sum(float(step["wall_seconds"]) for step in warm) if warm else None,
            "warm_workflow_completed_phases": len(warm), "warm_workflow_planned_phases": 5,
            "workflow_seconds": sum(float(step["wall_seconds"]) for step in steps if "wall_seconds" in step) if steps else None,
            "compaction_seconds": compact.get("wall_seconds"),
            "compaction_output_tokens": compact.get("actual_output_tokens", {}).get("sum"),
            "compaction_native": compact.get("native"), "compaction_checks": compact.get("checks"),
            "opencode_status": workflow_hook["status"], "passed_phases": sum(item["status"] == "passed" for item in phases),
            "planned_phases": 6, "correctness_errors": workflow.get("correctness_errors", []), "phases": phases,
            "capacity": {"status": capacity_hook["status"], "error": capacity.get("error"),
                         "filled_context_pass": capacity.get("status") == "ok" and capacity.get("filled_context") is True if capacity else None,
                         "actual_input_tokens": capacity.get("prompt_tokens"), "actual_output_tokens": capacity.get("output_tokens"),
                         "expected_input_tokens": capacity.get("expected_prompt_tokens"), "context_guard_tokens": capacity.get("context_guard_tokens"),
                         "decode_tokens_per_second": capacity.get("decode_tokens_per_second"),
                         "ttft_seconds": capacity.get("time_to_first_token_seconds"), "wall_seconds": capacity.get("elapsed_seconds")},
            "grammar107": {"status": grammar_hook["status"], "valid": grammar.get("status") == "ok" if grammar and grammar_hook["status"] != "canceled" else None,
                           "error": grammar.get("error"), "schema_count": grammar.get("schema_count"), "source_tokens": grammar.get("source_tokens"),
                           "actual_input_tokens": grammar_usage.get("prompt_tokens"), "actual_output_tokens": grammar_usage.get("completion_tokens"),
                           "native_decode_tokens_per_second": grammar_timings.get("predicted_per_second"),
                           "ttft_seconds": grammar_request.get("ttft_seconds"), "wall_seconds": grammar.get("wall_seconds"),
                           "executes_tools": grammar.get("executes_tools"), "returned_call_count": len(returned_calls) if grammar else None,
                           "expected_tool_name": expected_call.get("name"),
                           "returned_tool_names": [call.get("name") for call in returned_calls],
                           "arguments_match": returned_calls[0].get("arguments") == expected_call.get("arguments") if len(returned_calls) == 1 else None},
            "peak_gpu_mib": max(gpu) if gpu else None, "peak_inference_scope_mib": max(scope) / 1024 ** 2 if scope else None,
            "peak_inference_rss_mib": max(rss) / 1024 if rss else None,
            "resource_samples": resource_stats,
            "opencode_peak_scope_mib": workflow.get("memory", {}).get("scope_memory_peak_bytes", 0) / 1024 ** 2 if workflow else None,
            "limits": trial.get("limits"), "chain_evidence": helpers.chain_evidence(directory / "server.log"),
            "windows_memory": windows,
            "load_overlap": load_overlap(trial, events or [], now or datetime.now(timezone.utc)),
            "completed_recorded_requests": len(requests)}


def fmt(value: Any, digits: int = 1) -> str:
    """Keep absent measurements distinct from zero."""
    return "—" if value is None else f"{float(value):.{digits}f}"


def span(value: Record) -> str:
    """Format actual input ranges without implying a filled window."""
    low, high = value["minimum"], value["maximum"]
    return "—" if low is None else f"{low:,}–{high:,}"


def markdown(result: Record) -> str:
    """Write local measurements and correctness evidence for review."""
    lines = ["# Native runtime benchmark results", "", f"Updated: {result['generated_utc']}. Status: {result['status']}.", "",
             "These measurements use constructed native OpenCode history from public pinned source. They do not use private user conversations. "
             "Configured context and actual input are separate fields. Cold TTFT includes the first prompt prefill. "
             "Warm workflow time sums the five phases after the first coding turn, including compaction. Coverage must reach 5/5.", "",
             "Native generation speed divides generated tokens by native decode time. It excludes prefill and tools. "
             "Output lengths, tool calls and compaction histories can differ. Check correctness before choosing a configuration. "
             "The controlled repairs and three-fact checks do not establish general coding quality or published scores. "
             "Published external rates require separate labels and matching hardware and workloads. They are not local measurements in this report.", ""]
    events: list[Record] = result.get("load_events", {}).get("events", [])
    if events:
        lines += ["Reported host load qualifies overlapping trial intervals. These workflow wall times are not quiet-machine results. "
                  "GPU decode rates can still be useful, but host load can affect prefill, tools, compaction, and total workflow time. "
                  "The report retains every raw measurement. It does not discard or rerun a case. "
                  "No reported overlap does not prove that the machine was idle.", ""]
        for event in events:
            lines.append(f"- Reported interval: {event.get('start_utc')} to {event.get('end_utc') or 'unknown end (still open)'}. "
                         f"Source: {event.get('source')}. {event.get('note', '')}")
        affected = [row["label"] for row in result["cases"] if row.get("load_overlap", {}).get("overlapped")]
        lines += ["", "Affected measured cases: " + (", ".join(affected) if affected else "none so far") + ". "
                  "A [host load] marker identifies these cases below. Pending cases are not included.", ""]
    lines += ["| Case | Status | Actual long input | Cold TTFT s | Long native tok/s | Post native tok/s | Warm workflow s | Compaction s / output tokens | Repair/recall phases | GPU / scope MiB |",
              "|---|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for row in result["cases"]:
        marker = " [host load]" if row.get("load_overlap", {}).get("overlapped") else ""
        lines.append(f"| {row['label']}{marker} | {row['status']} | {span(row['actual_long_input_tokens'])} | {fmt(row['cold_ttft_seconds'])} | "
                     f"{fmt(row['pre_compaction_native']['native_tokens_per_second'])} | {fmt(row['post_compaction_native']['native_tokens_per_second'])} | "
                     f"{fmt(row['warm_workflow_seconds'])} ({row['warm_workflow_completed_phases']}/5) | {fmt(row['compaction_seconds'])} / "
                     f"{fmt(row['compaction_output_tokens'], 0)} | {row['passed_phases']}/6 | {fmt(row['peak_gpu_mib'], 0)} / {fmt(row['peak_inference_scope_mib'], 0)} |")
    lines += ["", "GPU memory is the sampled GPU total. Scope RAM covers only the owned inference scope. "
              "Pending and unselected cases have no measurements. Running rows contain partial observations. "
              "A failed attempt can still have timing values; these values do not establish a successful configuration.", "",
              "| Case | Capacity status | Filled input / output | Capacity tok/s | 107-schema status | Grammar input / output | Grammar TTFT s | Grammar native tok/s |",
              "|---|---|---:|---:|---|---:|---:|---:|"]
    for row in result["cases"]:
        capacity, grammar = row["capacity"], row["grammar107"]
        lines.append(f"| {row['label']} | {capacity['status']} | {fmt(capacity['actual_input_tokens'], 0)} / {fmt(capacity['actual_output_tokens'], 0)} | "
                     f"{fmt(capacity['decode_tokens_per_second'])} | {grammar['status']} | {fmt(grammar['actual_input_tokens'], 0)} / "
                     f"{fmt(grammar['actual_output_tokens'], 0)} | {fmt(grammar['ttft_seconds'])} | {fmt(grammar['native_decode_tokens_per_second'])} |")
    lines += ["", "Capacity uses exact token IDs and forced output length. It checks capacity and speed, not answer quality. "
              "The grammar fixture validates returned tool names and structured JSON. It never executes those calls. "
              "The public source target and 107 schemas both contribute to its actual input count.", ""]
    lines += ["## Resource observations", "",
              "Samples span each case, including startup and diagnostic intervals. Clocks are sample medians, not decode-only medians. "
              "GPU totals and Linux load include other host work. Linux load does not measure Windows CPU activity. "
              "One-second samples can miss short peaks.", "",
              "| Case | GPU peak MiB | Scope RAM peak MiB | RSS peak MiB | GPU peak C | GPU peak W | SM median MHz | Memory median MHz | Linux load 1m peak | Available RAM min GiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["cases"]:
        resources = row["resource_samples"]
        available = resources["linux_available_memory_kib"]["minimum"]
        lines.append(f"| {row['label']} | {fmt(row['peak_gpu_mib'], 0)} | {fmt(row['peak_inference_scope_mib'], 0)} | "
                     f"{fmt(row['peak_inference_rss_mib'], 0)} | {fmt(resources['gpu_temperature_c']['maximum'])} | "
                     f"{fmt(resources['gpu_watts']['maximum'])} | {fmt(resources['gpu_sm_clock_mhz']['median'], 0)} | "
                     f"{fmt(resources['gpu_memory_clock_mhz']['median'], 0)} | {fmt(resources['linux_load_1m_peak'])} | "
                     f"{fmt(available / 1024 ** 2 if available is not None else None)} |")
    lines += [""]
    for row in result["cases"]:
        lines += [f"## {row['label']}", "", f"Model: {row['model']}. OpenCode: {row['opencode_status']}. "
                  f"Configured context: {row['configured_context_tokens']}. Actual post-compaction input: {span(row['actual_post_compaction_input_tokens'])}. "
                  f"Warm long TTFT median: {fmt(row['warm_long_ttft_median_seconds'])} s. "
                  f"All phase wall time: {fmt(row['workflow_seconds'])} s.", ""]
        overlap: Record = row.get("load_overlap", {})
        if overlap.get("overlapped"):
            lines += [f"Reported host load overlaps this trial ({overlap['trial_start_utc']} to {overlap['trial_end_utc']}). "
                      "Workflow wall time has this qualification.", ""]
        if row["error"] or row["cleanup_error"]:
            lines += [f"Error: {row['error']}. Cleanup error: {row['cleanup_error']}.", ""]
        if row.get("cancellation"):
            lines += [f"Canceled hook: {row['cancellation']['hook']}. {row['cancellation']['reason']} "
                      "Completed hook results remain valid. The canceled hook has no test outcome.", ""]
        if row["correctness_errors"]:
            lines += ["Correctness errors: `" + json.dumps(row["correctness_errors"], sort_keys=True) + "`.", ""]
        grammar = row["grammar107"]
        if grammar["status"] == "failed":
            lines += [f"107-schema check failed. Expected tool: `{grammar['expected_tool_name']}`. "
                      f"Returned tools: `{json.dumps(grammar['returned_tool_names'])}`. "
                      f"Returned JSON arguments match: `{grammar['arguments_match']}`. "
                      f"Diagnostic error: {grammar['error']}.", ""]
        if row["live_settings"]:
            selected = {key: row["live_settings"].get(key) for key in ("runtime", "rounds", "draft_tokens", "draft_window", "mtp_chain",
                        "prompt_lookup", "lookup_type", "lookup_tokens", "mma_decode_min_kv", "kv_vram_cells", "reasoning_effort", "kv_cache",
                        "draft_kv_cache", "cache_ram_mib", "batch_tokens", "ubatch_tokens", "cpu_offload_gb")}
            lines += ["Actual settings: `" + json.dumps(selected, sort_keys=True) + "`.", ""]
        windows = row["windows_memory"]
        if windows is not None:
            lines += [f"Optional Windows memory observation: {windows['status']}. "
                      "The runner schedules it after all timed hooks and samples only if the loaded model reports idle. "
                      "These are Windows VM-process counters. They do not identify a Linux model PID, "
                      "prove CPU offload or GPU paging, or measure peak memory during generation. "
                      "Partial or invalid counters do not change the inference result. "
                      f"Diagnostic error: {windows['error']}.", "",
                      "| Windows VM process | Sample timestamps | Valid / invalid counters | Dedicated peak MiB | Shared peak MiB |",
                      "|---|---:|---:|---:|---:|"]
            for vm in windows["vm_processes"]:
                lines.append(f"| {vm['process_name']} ({vm['windows_pid']}) | {vm['timestamps']} | "
                             f"{vm['valid_counter_samples']} / {vm['invalid_counter_samples']} | "
                             f"{fmt(vm['peak_dedicated_sum_mib'], 0)} | {fmt(vm['peak_shared_sum_mib'], 0)} |")
            lines += ["", "Peaks compare sums within each sample timestamp. Samples are not added over time.", ""]
        for item in row["phases"]:
            lines += [f"{item['name']}: {item['status']}; {fmt(item['wall_seconds'])} s; "
                      f"actual input {span(item['actual_input_tokens'])}; output {fmt(item['actual_output_tokens']['sum'], 0)} tokens; "
                      f"native {fmt(item['native']['native_tokens_per_second'])} tok/s. Checks: `" + json.dumps(item["checks"], sort_keys=True) + "`.", ""]
        grammar = row["grammar107"]
        if grammar["status"] not in ("not run", "pending"):
            lines += [f"Grammar replay: valid={grammar['valid']}; {grammar['schema_count']} schemas; "
                      f"{grammar['source_tokens']} source tokens; {fmt(grammar['wall_seconds'])} s; "
                      f"{grammar['returned_call_count']} returned calls. Error: {grammar['error']}.", ""]
        evidence = row["chain_evidence"]
        if evidence["reported_draft_calls"] or evidence["reported_chain_decodes"]:
            lines += [f"Observed native timing windows: {evidence['reported_chained_calls']}/{evidence['reported_draft_calls']} chained draft calls; "
                      f"{evidence['reported_graph_reuses']} graph reuses; {evidence['reported_graph_rebuilds']} graph rebuilds. "
                      "Printed windows can omit a final partial window.", ""]
    return "\n".join(lines)


def main() -> int:
    """Aggregate one or more saved run roots with no inference or weight I/O."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-directory", type=Path, action="append", help="Repeat for separate stage run directories.")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", help="Comma-separated model profile names. Default: cases that are not archived. Explicit names include archived cases.")
    args = parser.parse_args()
    roots = [path.resolve() for path in (args.root_directory or [DEFAULT_RUN])]
    now = datetime.now(timezone.utc)
    event_path = HERE / "load-events.json"
    event_data: Record = helpers.read(event_path)
    events: list[Record] = event_data.get("events", [])
    matrix: Record = helpers.read(HERE / "matrix.json")
    configs: list[Record] = matrix["cases"]
    if args.models:
        selected_models = set(str(args.models).split(","))
        if not selected_models <= {str(config["model"]) for config in configs}:
            parser.error("Select known model profile names from the matrix.")
        configs = [config for config in configs if config["model"] in selected_models]
    else:
        configs = [config for config in configs if not config.get("archived")]
    found: dict[str, list[Path]] = {}
    planned: set[str] = set()
    for root in roots:
        plan: Record = helpers.read(root / "plan.json")
        planned.update(str(item["name"]) for item in plan.get("case_plans", []))
        for trial_path in root.rglob("trial.json"):
            trial: Record = helpers.read(trial_path)
            name = str(trial.get("name") or trial_path.parent.name)
            directories = found.setdefault(name, [])
            if trial_path.parent not in directories:
                directories.append(trial_path.parent)
    result_cases: list[Record] = []
    for config in configs:
        name = str(config["name"])
        if config["stage"] == "optional" and name not in found and name not in planned:
            continue
        directories = found.get(name) or [roots[0] / name]
        for directory in directories:
            row = summarize(directory, config, name in planned, events, now)
            row["label"] = name if len(directories) == 1 else f"{name} ({directory.parent.name})"
            result_cases.append(row)
    statuses = {row["status"] for row in result_cases}
    status = ("running" if "running" in statuses else "complete" if statuses == {"ok"} else "pending"
              if statuses <= {"pending", "not planned"} else "complete with failures or cancellations" if statuses <= {"ok", "failed", "canceled"} else "partial")
    result: Record = {"generated_utc": now.isoformat(timespec="seconds"), "status": status,
                      "input_roots": [str(path) for path in roots],
                      "models": sorted({str(config["model"]) for config in configs}),
                      "load_events": {"file": str(event_path), "events": events},
                      "case_coverage": {"passed": sum(row["status"] == "ok" for row in result_cases),
                                        "failed": sum(row["status"] == "failed" for row in result_cases),
                                        "canceled": sum(row["status"] == "canceled" for row in result_cases),
                                        "unselected": sum(row["status"] == "not planned" for row in result_cases),
                                        "reported": len(result_cases)}, "cases": result_cases}
    destination: Path = args.output_directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / "summary.md").write_text(markdown(result))
    print(json.dumps({"status": status, "cases": len(result_cases), "output": str(destination / "summary.md")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
