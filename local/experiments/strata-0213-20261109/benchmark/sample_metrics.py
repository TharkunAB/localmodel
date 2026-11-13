"""Sample only this experiment's read-only metrics and shared Linux disk counters."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / 'state/active.json'
stop_requested = False


def append(path: Path, item: dict[str, Any]) -> None:
    with path.open('a') as stream:
        stream.write(json.dumps(item, separators=(',', ':')) + '\n')


def disks() -> dict[str, Any]:
    """Return counters for the Linux root block device; these include all users."""
    device = os.stat(ROOT).st_dev
    major, minor = os.major(device), os.minor(device)
    for line in Path('/proc/diskstats').read_text().splitlines():
        fields = line.split()
        if list(map(int, fields[:2])) == [major, minor]:
            values = list(map(int, fields[3:]))
            return {'device': fields[2], 'major': major, 'minor': minor,
                    'read_ios': values[0], 'read_sectors': values[2], 'read_bytes': values[2] * 512,
                    'read_ms': values[3], 'write_ios': values[4], 'write_bytes': values[6] * 512,
                    'scope': 'Shared Linux root block device, not attributed to this model'}
    return {'major': major, 'minor': minor, 'unavailable': True}


def sample(active: dict[str, Any], case: Path, seen: set[str]) -> None:
    """Read the known controller and endpoint without sending inference requests."""
    pid = int(active['controller_pid'])
    proc = Path('/proc') / str(pid)
    if not proc.exists() or proc.stat().st_uid != os.getuid():
        return
    before = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{active['port']}/metrics?requests=all", timeout=3) as response:
        metrics = json.load(response)
    current = json.loads(ACTIVE.read_text())
    if current.get('controller_pid') != pid or proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] != before:
        return
    if metrics.get('engine', {}).get('model') != active['model']:
        raise RuntimeError('Metrics model differs from the recorded experiment model.')
    group = next(line[3:] for line in Path('/proc', str(active['pid']), 'cgroup').read_text().splitlines() if line.startswith('0::'))
    scope = Path('/sys/fs/cgroup' + group)
    row = {'at': time.time(), 'server_time': metrics['time'], 'controller_pid': pid, 'controller_start_ticks': before,
           'live': metrics.get('live'), 'totals': metrics.get('totals'), 'hardware': metrics.get('hardware'),
           'disk': disks(), 'model_io_available': (scope / 'io.stat').exists()}
    row['extra_metrics'] = {key: value for key, value in metrics.items()
                            if key not in ('time', 'live', 'totals', 'hardware', 'requests')}
    if (scope / 'io.stat').exists():
        row['model_io'] = (scope / 'io.stat').read_text()
    append(case / 'metrics-samples.jsonl', row)
    metadata = case / 'metrics-observer.json'
    if not metadata.exists():
        metadata.write_text(json.dumps({'started_at': row['at'], 'controller_pid': pid, 'controller_start_ticks': before,
             'engine': metrics.get('engine'), 'cgroup_io_available': row['model_io_available'],
             'proc_self_io_available': Path('/proc/self/io').exists(),
             'note': 'Collection can start after the first prefill. Root disk counters include other processes. No inference is sent.'}, indent=2) + '\n')
    for request in metrics.get('requests', []):
        if request['id'] not in seen:
            append(case / 'metrics-requests.jsonl', {'observed_at': row['at'], **request})
            seen.add(request['id'])


def main() -> None:
    from benchmark_paths import HERE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=21600)
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    seen_by_case: dict[str, set[str]] = {}
    while not stop_requested and time.monotonic() - started < args.seconds:
        case: Path | None = None
        try:
            if ACTIVE.exists():
                active = json.loads(ACTIVE.read_text())
                case = Path(active.get('strata_slot_directory', '/')).resolve().parent
                if case.parent == HERE and case.is_dir():
                    seen = seen_by_case.setdefault(str(case), set())
                    sample(active, case, seen)
        except (OSError, ValueError, KeyError, StopIteration, RuntimeError) as error:
            if case is not None and case.parent == HERE and case.is_dir():
                append(case / 'metrics-observer-errors.jsonl', {'at': time.time(), 'error': str(error)})
        time.sleep(args.interval)


if __name__ == '__main__':
    def stop(_signal: int, _frame: Any) -> None:
        global stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main()
