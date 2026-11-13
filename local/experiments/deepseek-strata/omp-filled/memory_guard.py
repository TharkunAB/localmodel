"""Stop this benchmark below the configured host memory floors."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(sys.argv[1]).absolute()
ACTIVE = ROOT / 'state/active.json'
FLOOR = 8 * 2**30
WINDOWS_FLOOR = int(os.environ.get("STRATA_BENCH_WINDOWS_FLOOR_GIB", "8")) * 2**30
if not 1 <= WINDOWS_FLOOR <= FLOOR:
    raise ValueError("Use a Windows memory floor from 1 to 8 GiB.")
PS = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'


def available() -> dict[str, Any]:
    """Sample Linux available memory and Windows free physical memory."""
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    windows = int(subprocess.check_output([PS, '-NoProfile', '-Command',
        '[long](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024'], text=True, timeout=15).strip())
    return {'at': time.time(), 'linux_available_bytes': int(info['MemAvailable'].split()[0]) * 1024,
            'windows_available_bytes': windows}


def main() -> None:
    """Monitor only the model controller recorded at guard startup."""
    expected = json.loads(ACTIVE.read_text())['controller_pid']
    errors = 0
    while True:
        finished = OUTPUT / 'finished.json'
        if finished.exists() and json.loads(finished.read_text()).get('status') == 'completed':
            return
        current: dict[str, Any] = json.loads(ACTIVE.read_text()) if ACTIVE.exists() else {}
        if current.get('controller_pid') != expected:
            return
        reason = 'The benchmark driver stopped before completion.' if finished.exists() else ''
        sample: dict[str, Any]
        try:
            sample = available()
            errors = 0
            if sample['linux_available_bytes'] < FLOOR or sample['windows_available_bytes'] < WINDOWS_FLOOR:
                reason = f'Available RAM fell below its floor: Windows {WINDOWS_FLOOR / 2**30:g} GiB, Linux 8 GiB.'
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors += 1
            sample = {'at': time.time(), 'error': str(error)}
            if errors >= 3:
                reason = 'Three consecutive memory checks failed; available memory could not be verified.'
        try:
            group = next(line[3:] for line in (Path('/proc')/str(current['pid'])/'cgroup').read_text().splitlines()
                         if line.startswith('0::'))
            scope = Path('/sys/fs/cgroup'+group)
            sample['model_memory'] = {key:(scope/key).read_text().strip() for key in
                ('memory.current','memory.peak','memory.max','memory.swap.current','memory.swap.max','memory.events')}
        except (OSError, KeyError, StopIteration):
            pass
        with (OUTPUT / 'memory.jsonl').open('a') as stream:
            stream.write(json.dumps(sample) + '\n')
        (OUTPUT / 'memory-current.json').write_text(json.dumps(sample) + '\n')
        if reason:
            (OUTPUT / 'aborted.json').write_text(json.dumps({'reason': reason, 'sample': sample}, indent=2) + '\n')
            worker = OUTPUT / 'worker.json'
            if worker.exists():
                pid = int(json.loads(worker.read_text())['pid'])
                proc = Path('/proc') / str(pid)
                try:
                    if proc.stat().st_uid == os.getuid() and os.fsencode(str(OUTPUT)) in (proc / 'environ').read_bytes():
                        subprocess.run(['systemctl', '--user', 'stop', f'omp-local-{pid}.scope'], timeout=30, check=False)
                except (OSError, subprocess.SubprocessError):
                    pass
            subprocess.run([str(ROOT / 'bin/localmodel'), 'stop'], timeout=50, check=False)
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
