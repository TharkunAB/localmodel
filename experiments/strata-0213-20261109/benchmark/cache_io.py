"""Collect native timing and available storage evidence without inferring cache misses."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from benchmark_paths import HERE
import re
from typing import Any

from collect import collect_case, read_json, rows

Record = dict[str, Any]


def stamp(value: str | int | float) -> float:
    """Read epoch seconds or recorder ISO timestamps."""
    return float(value) if isinstance(value, (int, float)) else datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def io_window(samples: list[Record], start: float | None, end: float | None, tokens: int | None) -> Record:
    """Bound shared root reads with inner and enclosing observed counter samples."""
    result: Record = {'start': start, 'end': end, 'token_denominator': tokens,
                     'scope': 'Shared Linux root block device; includes other processes and Engram'}
    if start is None or end is None or end < start:
        return {**result, 'available': False, 'reason': 'Missing or unordered request boundaries.'}
    inner = [s for s in samples if start <= s['at'] <= end]
    before = [s for s in samples if s['at'] <= start]
    after = [s for s in samples if s['at'] >= end]
    def delta(first: Record, last: Record) -> int | None:
        if first['disk']['device'] != last['disk']['device']:
            return None
        difference = int(last['disk']['read_bytes']) - int(first['disk']['read_bytes'])
        return difference if difference >= 0 else None
    low = delta(inner[0], inner[-1]) if len(inner) >= 2 else None
    high = delta(before[-1], after[0]) if before and after else None
    result.update(available=low is not None or high is not None,
                  read_bytes_inner=low, read_bytes_enclosing=high,
                  inner_sample_seconds=inner[-1]['at'] - inner[0]['at'] if len(inner) >= 2 else None,
                  enclosing_extra_seconds=after[0]['at'] - before[-1]['at'] - (end-start) if before and after else None,
                  shared_read_mb_per_token_inner=low / 1e6 / tokens if low is not None and tokens else None,
                  shared_read_mb_per_token_enclosing=high / 1e6 / tokens if high is not None and tokens else None)
    return result


def request_windows(path: Path, samples: list[Record], warnings: list[str]) -> list[Record]:
    """Keep each request's actual prefill and decode intervals separate."""
    requests: dict[int, Record] = {}
    for event in rows(path, warnings):
        ident = event.get('id')
        if not isinstance(ident, int):
            continue
        kind = event.get('kind')
        if kind == 'request' and event.get('url') == 'http://127.0.0.1:8899/v1/chat/completions':
            requests[ident] = {'id': ident, 'start': stamp(event['at']), 'artifact': path.name}
        if ident not in requests:
            continue
        request = requests[ident]
        if kind == 'first_token':
            request['first_token'] = stamp(event['at'])
        if kind == 'timings':
            request.update(end=stamp(event['at']), timings=event.get('timings', {}))
    result: list[Record] = []
    for request in requests.values():
        timing = request.get('timings', {})
        result.append({**request,
            'prefill_shared_io': io_window(samples, request['start'], request.get('first_token'), timing.get('prompt_n')),
            'decode_shared_io': io_window(samples, request.get('first_token'), request.get('end'), timing.get('predicted_n'))})
    return result


def native_facts(path: Path) -> Record:
    """Read placements and cumulative CPU-lane shares from native stderr."""
    if not path.exists():
        return {'available': False}
    lines = path.read_text(errors='replace').splitlines()
    return {'available': True,
        'resident_copies': [dict(copied_experts=int(m.group(1)),cpu_lane_experts=int(m.group(2)),payload_gb=float(m.group(3)),partial=('the others read in place' in line)) for line in lines if (m := re.search(r"(\d+) of the CPU lane's (\d+) experts copied into RAM \(([0-9.]+) GB",line))],
        'placement': [line for line in lines if any(tag in line for tag in ('CUDA0 layers', 'routed experts on the CPU lane', 'RAM store:', 'experts VRAM leaves over', 'copied into RAM'))],
        'cumulative_cpu_lane_percent': [float(match.group(1)) for line in lines
            if (match := re.search(r'CPU lane ([0-9.]+)% of the experts', line))],
        'cpu_lane_note': 'Native serve summaries are cumulative across prefill and decode. They are not RAM-cache miss rates.',
        'exact_store_summaries': [line for line in lines if line.startswith('store:')],
        'cache_counter_lines': [line for line in lines if any(word in line.lower() for word in ('cache hit', 'cache miss', 'cache-route', 'cache-aware routing', 'store hit', 'store miss'))],
        'native_disk_summaries': [line for line in lines if line.startswith('decode reads from the disk:')]}



def sampled_resources(path:Path,warnings:list[str])->Record:
    """Summarize measured GPU fields and model-scope swap without extrapolation."""
    observed=rows(path/'metrics-samples.jsonl',warnings)
    memory=rows(path/'memory.jsonl',warnings)
    gpu:Record={}
    for phase in ('all','reading','generating','idle'):
        selected=[sample for sample in observed if phase=='all' or sample.get('live',{}).get('state')==phase]
        values:Record={}
        for field in ('gpu_util','gpu_mem_used','gpu_power','gpu_temp'):
            known=[float(sample.get('hardware',{}).get(field)) for sample in selected if isinstance(sample.get('hardware',{}).get(field),(int,float))]
            values[field]={'samples':len(known),'mean':sum(known)/len(known) if known else None,'max':max(known) if known else None}
        gpu[phase]=values
    swap=[int(sample['model_memory']['memory.swap.current']) for sample in memory if 'memory.swap.current' in sample.get('model_memory',{})]
    available=[sample['linux_available_bytes'] for sample in memory if isinstance(sample.get('linux_available_bytes'),(int,float))]
    return {'gpu_by_phase':gpu,'gpu_note':'Whole-device NVML samples; arithmetic sample means, not process-only or continuous peaks. Inspect each observer start time for coverage.',
            'gpu_units':{'gpu_util':'percent','gpu_mem_used':'bytes','gpu_power':'watts','gpu_temp':'degrees Celsius'},
            'sampled_model_swap_peak_bytes':max(swap) if swap else None,
            'minimum_linux_available_bytes':min(available) if available else None}

def model_io(samples: list[Record]) -> Record:
    """Report observed cgroup read deltas only when the kernel supplies counters."""
    parsed: list[tuple[float, dict[str, int]]] = []
    for sample in samples:
        text = sample.get('model_io')
        if not isinstance(text, str):
            continue
        counters: dict[str, int] = {}
        for line in text.splitlines():
            fields = line.split()
            if not fields:
                continue
            values = dict(field.split('=', 1) for field in fields[1:] if '=' in field)
            if 'rbytes' in values:
                counters[fields[0]] = int(values['rbytes'])
        if counters:
            parsed.append((sample['at'], counters))
    deltas: dict[str, int] = {}
    if len(parsed) >= 2:
        before, after = parsed[0][1], parsed[-1][1]
        deltas = {device: after[device]-value for device, value in before.items()
                  if device in after and after[device] >= value}
    return {'available': bool(deltas), 'samples': len(parsed),
            'read_bytes_by_device': deltas,
            'single_device_read_bytes': next(iter(deltas.values())) if len(deltas) == 1 else None,
            'start': parsed[0][0] if parsed else None, 'end': parsed[-1][0] if parsed else None,
            'note': 'Observed model-cgroup interval only. Do not sum stacked block devices. Missing counters are unavailable, not zero.'}


def collect(path: Path) -> Record:
    """Combine complete and partial evidence; preserve absent counters as null."""
    warnings: list[str] = []
    driver = read_json(path/'case-summary.json')
    observed = rows(path/'metrics-samples.jsonl', warnings)
    samples = [s for s in observed if 'read_bytes' in s.get('disk', {})]
    scoped_io = model_io(observed)
    windows = [request for kind in ('repair', 'resume')
               for file in sorted(path.glob(f'*-{kind}.network.jsonl'))
               for request in request_windows(file, samples, warnings)]
    timed = [r['timings'] for r in windows if r.get('timings', {}).get('predicted_ms', 0) > 0]
    decoded = sum(int(t.get('predicted_n', 0)) for t in timed)
    decode_ms = sum(float(t['predicted_ms']) for t in timed)
    prefilled = sum(int(t.get('prompt_n', 0)) for t in timed)
    prefill_ms = sum(float(t.get('prompt_ms', 0)) for t in timed)
    targets = driver.get('targets', [])
    checkpoints = [collect_case(path, target) for target in targets]
    seen: dict[str, Record] = {}
    for request in rows(path/'metrics-requests.jsonl', warnings):
        seen[request['id']] = request
    facts = native_facts(path/'engine.log')
    return {'case': path.name, 'status': driver.get('status'), 'model': driver.get('model'), 'runtime_provenance': driver.get('runtime_provenance'), 'lane_calibration': driver.get('lane_calibration'), 'resident_experts': driver.get('resident_experts',False), 'routing_profile': driver.get('routing_profile'),
        'checkpoints': [{'target': target, 'prefill_tps': row['initial_prefill_tokens_per_second'],
             'decode_tps': row['weighted_native_decode_tokens_per_second'], 'new_input': row['initial_prefill_tokens'],
             'reused_input': row['initial_context_reused_tokens'], 'generated_tokens': row['generated_tokens'],
             'workflow_seconds': row['repair_resume_wall_seconds'], 'quality': row['quality'], 'tool_errors': row['tool_errors']}
            for target, row in zip(targets, checkpoints)],
        'aggregate_omp': {'generated_tokens': decoded, 'decode_ms': decode_ms,
            'decode_tps': 1000*decoded/decode_ms if decode_ms else None,
            'new_prefill_tokens': prefilled, 'prefill_ms': prefill_ms,
            'all_request_prefill_tps': 1000*prefilled/prefill_ms if prefill_ms else None,
            'note': 'Includes repair and resume API calls only; excludes the short warmup.'},
        'peak_cgroup_bytes': max((r['peak_observed_cgroup_bytes'] or 0 for r in checkpoints), default=None),
        'memory_events': checkpoints[0]['memory_events'] if checkpoints else {},
        'cleanup': {key: driver.get(key) for key in ('server_stopped','profile_restored','worker_cleanup','lane_profile_restored','observer_stopped','guard_stopped','active_state_restored','state_config_restored','normal_settings_unchanged','port_8899_closed')},
        'resources':sampled_resources(path,warnings),'native': facts, 'observer': read_json(path/'metrics-observer.json'),
        'exact_ram_cache_hits': None, 'exact_ram_cache_misses': None,
        'exact_miss_note': 'This collector does not infer misses from CPU-lane percentages. Exact counters remain unavailable unless exposed in saved native/API evidence.',
        'model_disk_read_bytes': scoped_io['single_device_read_bytes'],
        'model_io': scoped_io,
        'model_io_note': 'Cgroup counters are used when available. Shared root counters include other processes and cannot identify model-only reads.',
        'shared_root_observed': {'samples': len(samples), 'start': samples[0]['at'] if samples else None,
            'end': samples[-1]['at'] if samples else None,
            'read_bytes': samples[-1]['disk']['read_bytes']-samples[0]['disk']['read_bytes'] if len(samples)>1 else None},
        'request_windows': windows, 'completed_api_metrics': list(seen.values()), 'warnings': warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='*', type=Path)
    args = parser.parse_args()
    paths = args.cases or sorted(p.parent for p in HERE.glob('*/case-summary.json'))
    results = [collect(path if path.is_absolute() else HERE/path) for path in paths]
    (HERE/'cache-io-comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    for result in results:
        print(json.dumps({key: result[key] for key in ('case','status','checkpoints','aggregate_omp','peak_cgroup_bytes','memory_events','cleanup','shared_root_observed')}))


if __name__ == '__main__':
    main()
