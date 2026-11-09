"""Stop this benchmark below the Linux memory floor."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from run import stop_worker

from benchmark_paths import ROOT
OUTPUT = Path(sys.argv[1]).absolute()
ACTIVE = ROOT / 'state/active.json'
FLOOR = 8 * 2**30


def available() -> dict[str, Any]:
    """Sample Linux available memory."""
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {'at': time.time(), 'linux_available_bytes': int(info['MemAvailable'].split()[0]) * 1024,
            'loadavg_1m': os.getloadavg()[0]}


def main() -> None:
    """Monitor only the model controller recorded at guard startup."""
    expected = json.loads(ACTIVE.read_text())['controller_pid']
    errors = 0
    while True:
        current: dict[str, Any] = json.loads(ACTIVE.read_text()) if ACTIVE.exists() else {}
        if current.get('controller_pid') != expected:
            return
        reason = ''
        sample: dict[str, Any]
        try:
            sample = available()
            errors = 0
            if sample['linux_available_bytes'] < FLOOR:
                reason = 'Available Linux RAM fell below its 8 GiB floor.'
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
                ('memory.current','memory.peak','memory.max','memory.swap.current','memory.swap.max','memory.events','memory.stat')}
            sample['model_cpu']=(scope/'cpu.stat').read_text()
            sample['model_io']=(scope/'io.stat').read_text()
        except (OSError, KeyError, StopIteration):
            pass
        with (OUTPUT / 'memory.jsonl').open('a') as stream:
            stream.write(json.dumps(sample) + '\n')
        (OUTPUT / 'memory-current.json').write_text(json.dumps(sample) + '\n')
        if int(sample.get('model_memory',{}).get('memory.swap.current','0'))>0:
            reason='The model scope used swap.'
        if reason:
            (OUTPUT / 'aborted.json').write_text(json.dumps({'reason': reason, 'sample': sample}, indent=2) + '\n')
            try:stop_worker(OUTPUT,dict(os.environ))
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError):pass
            latest: dict[str, Any] = json.loads(ACTIVE.read_text()) if ACTIVE.exists() else {}
            if latest.get('controller_pid') == expected:
                subprocess.run([str(ROOT / 'bin/localmodel'), 'stop'], timeout=50, check=False)
            return
        time.sleep(5)


if __name__ == '__main__':
    try:main()
    except BaseException as error:
        (OUTPUT/'aborted.json').write_text(json.dumps({'reason':'Linux guard failed: '+str(error),'at':time.time()})+'\n')
        raise
