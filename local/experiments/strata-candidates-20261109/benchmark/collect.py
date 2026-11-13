"""Collect one OMP checkpoint's timings, quality checks and resource evidence.

Use cache_io.py for standalone collection with explicit case paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def read_json(path: Path) -> Record:
    """Return an empty record for a missing or incomplete JSON file."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def rows(path: Path, warnings: list[str]) -> list[Record]:
    """Read saved events and report incomplete lines."""
    if not path.exists():
        return []
    result: list[Record] = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.append(f'{path.name}:{index}: incomplete JSON line excluded')
    return result


def total(values: list[Any]) -> int | float | None:
    """Sum recorded values without treating missing evidence as zero."""
    known = [value for value in values if isinstance(value, (int, float))]
    return sum(known) if known else None


def phase(path: Path, warnings: list[str]) -> Record:
    """Retain the final timing and usage event for each model request ID."""
    events = rows(path, warnings)
    requests: dict[int, Record] = {}
    for event in events:
        if event.get('kind') == 'request' and event.get('url') == 'http://127.0.0.1:8899/v1/chat/completions':
            requests.setdefault(event['id'], {'id': event['id'], 'timings': {}, 'usage': {}, 'started': False})
    for event in events:
        request_id = event.get('id')
        if not isinstance(request_id, int):
            continue
        request = requests.get(request_id)
        if request is None:
            continue
        if event.get('kind') == 'generation_started':
            request['started'] = True
        if event.get('kind') == 'timings':
            for key in ('timings', 'usage'):
                if event.get(key):
                    request[key] = event[key]
        if event.get('kind') in ('response_error', 'budget_exceeded'):
            request['error'] = event.get('error', event['kind'])
    recorded: list[Record] = []
    for request in requests.values():
        timing, usage = request['timings'], request['usage']
        cached = timing.get('cache_n', usage.get('prompt_tokens_details', {}).get('cached_tokens'))
        input_tokens = usage.get('prompt_tokens')
        if input_tokens is None and timing.get('prompt_n') is not None:
            input_tokens = timing['prompt_n'] + (cached or 0)
        recorded.append({
            'request_id': request['id'], 'started': request['started'], 'error': request.get('error'),
            'input_tokens': input_tokens, 'context_reused_tokens': cached,
            'prefill_tokens': timing.get('prompt_n'), 'prefill_ms': timing.get('prompt_ms'),
            'generated_tokens': usage.get('completion_tokens', timing.get('predicted_n')),
            'native_predicted_tokens': timing.get('predicted_n'), 'native_decode_ms': timing.get('predicted_ms'),
            'draft_offered_tokens': timing.get('draft_n'), 'draft_accepted_tokens': timing.get('draft_n_accepted'),
        })
    prefix = path.name.removesuffix('.network.jsonl')
    summary = read_json(path.with_name(prefix + '.summary.json'))
    tools = summary.get('tools')
    if tools is None:
        tools = [{'tool': event.get('toolName'), 'is_error': event.get('isError')}
                 for event in rows(path.with_name(prefix + '.stdout.jsonl'), warnings)
                 if event.get('type') == 'tool_execution_end']
    return {
        'artifact': path.name, 'phase': prefix.split('-', 1)[1], 'exit_code': summary.get('exit_code'),
        'wall_seconds': summary.get('wall_seconds'), 'offline_input_tokens': summary.get('actual_input_tokens_offline'),
        'requests': recorded, 'tools': tools, 'errors': summary.get('errors', []),
    }


def collect_case(path: Path, target: int = 65536) -> Record:
    """Collect quality, native timings, resource samples, and cleanup evidence."""
    driver = read_json(path / 'case-summary.json')
    benchmark = read_json(path / 'summary.json')
    finished = read_json(path / 'finished.json')
    warnings: list[str] = []
    phases = [phase(file, warnings) for kind in ('repair', 'resume')
              for file in sorted(path.glob(f'*-{target}-{kind}.network.jsonl'))]
    requests: list[Record] = [request for item in phases for request in item['requests']]
    timed = [request for request in requests if request['native_predicted_tokens'] is not None
             and request['native_decode_ms'] is not None and request['native_decode_ms'] > 0]
    predicted = total([request['native_predicted_tokens'] for request in timed])
    decode_ms = total([request['native_decode_ms'] for request in timed])
    offered = total([request['draft_offered_tokens'] for request in requests])
    accepted = total([request['draft_accepted_tokens'] for request in requests])
    initial: Record = requests[0] if requests else {}
    initial_ms = initial.get('prefill_ms')
    initial_n = initial.get('prefill_tokens')
    checks: Record = next((check for check in benchmark.get('checks', []) if check.get('target') == target), dict[str, Any]())
    validation: Record = checks.get('validation', {})
    validation_output: Record
    try:
        validation_output = json.loads(validation.get('validation_stdout', ''))
    except json.JSONDecodeError:
        validation_output = {}
    quality: Record = {key: validation.get(key) for key in
               ('passed', 'code_passed', 'recall_passed', 'tests_unchanged', 'read_executed', 'edit_executed', 'bash_executed')}
    quality.update(assertions=validation_output.get('assertions'), strict_resume_passed=checks.get('resume', {}).get('passed'),
                   all_quality_checks_passed=benchmark.get('all_quality_checks_passed'))
    resources = rows(path / 'resources.jsonl', warnings)
    samples = rows(path / 'memory.jsonl', warnings)
    memory_records = [sample.get('model_memory', {}) for sample in samples] + [driver.get('memory', {})]
    peaks = [int(record['memory.peak']) for record in memory_records if 'memory.peak' in record]
    peaks += [sample['model_peak_bytes'] for sample in resources if 'model_peak_bytes' in sample]
    peaks += [sample['model_memory_bytes'] for sample in resources if 'model_memory_bytes' in sample]
    gpu = [sample['gpu_used_mib'] for sample in resources if 'gpu_used_mib' in sample]
    available = [sample['linux_available_bytes'] for sample in samples if 'linux_available_bytes' in sample]
    loads = [sample['loadavg_1m'] for sample in samples if 'loadavg_1m' in sample]
    events: dict[str, int] = {}
    event_texts = [record.get('memory.events', '') for record in memory_records]
    event_texts += [sample.get('memory_events', '') for sample in resources]
    for text in event_texts:
        for line in text.splitlines():
            key, value = line.split()
            events[key] = max(events.get(key, 0), int(value))
    tools = [tool for item in phases for tool in item['tools']]
    tool_errors: dict[str, int] = {}
    for tool in tools:
        if tool.get('is_error'):
            key = str(tool.get('tool'))
            tool_errors[key] = tool_errors.get(key, 0) + 1
    inputs = [request['input_tokens'] for request in requests]
    phase_times = [item['wall_seconds'] for item in phases]
    complete = (driver.get('status') == benchmark.get('status') == finished.get('status') == 'completed')
    return {
        'case': path.name, 'status': driver.get('status', 'missing'), 'benchmark_status': benchmark.get('status'),
        'execution_complete': complete, 'error': driver.get('error', benchmark.get('error')),
        'model': benchmark.get('model', driver.get('live', {}).get('model')),
        'engine_version': benchmark.get('engine_version'), 'speculative': driver.get('speculative'),
        'draft_tokens': driver.get('draft_tokens'), 'context_capacity_tokens': benchmark.get('server_context_capacity'),
        'memory_cap_gib': driver.get('memory_cap_gib'), 'expert_cache_gib': benchmark.get('expert_cache_gib'),
        'quality': quality, 'planned_input': read_json(path / f'{target}-planned.json'),
        'model_request_attempts': len(requests), 'model_requests_started': sum(request['started'] for request in requests),
        'requests_with_generated_token_counts': sum(request['generated_tokens'] is not None for request in requests),
        'generated_tokens': total([request['generated_tokens'] for request in requests]),
        'native_decode_timed_requests': len(timed), 'native_decode_tokens': predicted, 'native_decode_ms': decode_ms,
        'weighted_native_decode_tokens_per_second': predicted * 1000 / decode_ms if predicted is not None and decode_ms else None,
        'initial_input_tokens': initial.get('input_tokens'), 'initial_context_reused_tokens': initial.get('context_reused_tokens'),
        'initial_prefill_tokens': initial_n, 'initial_prefill_ms': initial_ms,
        'initial_prefill_tokens_per_second': initial_n * 1000 / initial_ms if initial_n is not None and initial_ms else None,
        'actual_input_tokens': inputs, 'context_reused_tokens': [request['context_reused_tokens'] for request in requests],
        'repair_resume_wall_seconds': sum(phase_times) if len(phase_times) == 2 and all(t is not None for t in phase_times) else None,
        'recorded_phase_wall_seconds': total(phase_times), 'tool_errors': sum(tool_errors.values()),
        'tool_errors_by_name': tool_errors, 'tool_calls': len(tools),
        'draft_offered_tokens': offered, 'draft_accepted_tokens': accepted,
        'draft_acceptance_percent': accepted * 100 / offered if accepted is not None and offered else None,
        'peak_observed_cgroup_bytes': max(peaks) if peaks else None,
        'peak_observed_gpu_mib': max(gpu) if gpu else None,
        'minimum_linux_available_bytes': min(available) if available else None,
        'loadavg_1m_range': [min(loads), max(loads)] if loads else None,
        'memory_events': events, 'memory_limits': driver.get('limits'),
        'server_stopped': driver.get('server_stopped'), 'worker_cleanup': driver.get('worker_cleanup'),
        'cleanup_errors': {key: value for key, value in driver.items() if 'cleanup_error' in key},
        'aborted': read_json(path / 'aborted.json'), 'warnings': warnings, 'phases': phases,
    }

