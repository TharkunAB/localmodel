"""Score a public corpus with an isolated, capped DeepSeek engine."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import FrameType
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import deepseek_strata as strata

CAP = 64 * 2**30
FLOOR = 8 * 2**30


def digest(path: Path) -> str:
    """Hash one bounded artifact."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def available() -> int:
    """Read available Linux RAM in bytes."""
    row = next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    return int(row.split()[1]) * 1024


def save(path: Path, value: Any) -> None:
    """Retain a readable measurement record."""
    path.write_text(json.dumps(value, indent=2) + '\n')


def interrupted(signum: int, _frame: FrameType | None) -> None:
    """Enter normal cleanup when the controller is stopped."""
    raise RuntimeError(f'Score controller received signal {signum}.')


def main() -> int:
    """Validate the plan, then run only when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--lib-dir', action='append', default=[], type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--receipt-sha256')
    parser.add_argument('--ids', required=True, type=Path)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--route', choices=('off', '0.2', '0.2,0.3'), default='off')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    if digest(binary) != args.sha256:
        raise RuntimeError('Native binary differs from its pin.')
    if args.receipt:
        if digest(args.receipt) != args.receipt_sha256:
            raise RuntimeError('Build receipt differs from its pin.')
        receipt = json.loads(args.receipt.read_text())
        if Path(receipt['binary_path']).resolve() != binary or receipt['binary_sha256'] != args.sha256:
            raise RuntimeError('Build receipt names another binary.')
        if receipt['base_revision'] != '146df3c468739147f520ac3816de806f71232571':
            raise RuntimeError('Build receipt names another source base.')
    elif args.route != 'off':
        raise RuntimeError('Cache routing requires the reviewed experimental build receipt.')
    info = strata.manifest()
    if not args.receipt and args.sha256 != info['runtime']['binary_sha256']:
        raise RuntimeError('An unofficial engine requires a pinned build receipt.')
    if not strata.verified():
        raise RuntimeError('Target weight verification receipt changed.')
    strata.check_pack('none')
    ids = json.loads(args.ids.read_text())
    vocabulary = json.loads((strata.pack_directory() / 'tokenizer/vocab.json').read_text())
    segments: list[dict[str, Any]] = ids['segments']
    if ids['vocab_size'] != len(vocabulary) or len(segments) != 5:
        raise RuntimeError('Use the five-segment DeepSeek-tokenized corpus.')
    for segment in segments:
        if len(segment['ids']) != 1000 or any(not 0 <= token < len(vocabulary) for token in segment['ids']):
            raise RuntimeError('Invalid corpus tokens.')
    env = strata.environment()
    unit = f'strata-candidate-score-{os.getpid()}.scope'
    env.update(CAPPED_UNIT=unit, HEAVY_PEAKS=str(output.parent / 'peaks'),
               LD_LIBRARY_PATH=':'.join([*(str(p.resolve()) for p in args.lib_dir), str(binary.parent), '/usr/lib/wsl/lib']),
               DS4_ENGRAM_CACHE_MB='256', DS4_ENGRAM_THREADS='16', DS4_LANE_CAL='none',
               STRATA_DS4_DISK_SKIP_W='0', STRATA_DS4_SLOT_KEEP='0')
    if args.route != 'off':
        env['STRATA_DS4_CACHE_ROUTE'] = args.route
    command = [str(binary), '--pack', str(strata.pack_directory()), '--max-context', '8192',
               '--ssd-cache', str(48 * 2**30 / 10**9), '--cpu-threads', '12', '--reserve-mib', '700',
               '--prefill-chunk', '4096', '--dspark', 'none', '--experts', '6', '--vision', 'off',
               '--expert-profile', str(args.profile.resolve()), '--profile-readonly', '--window', '16',
               '--score', str(args.ids.resolve()), '--score-out', str(output / 'scores.tsv'), '--verbose']
    plan = {'command': command, 'binary_sha256': args.sha256, 'receipt_sha256': args.receipt_sha256,
            'ids_sha256': digest(args.ids), 'profile_sha256': digest(args.profile),
            'positions': sum(len(s['ids']) - 1 for s in segments), 'route': args.route,
            'context_capacity': 8192, 'memory_cap': CAP, 'memory_floor': FLOOR, 'unit': unit,
            'environment': {k: v for k, v in env.items() if k.startswith(('DS4_', 'STRATA_')) or k == 'LD_LIBRARY_PATH'}}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'plan.json', plan)
    with (ROOT / 'state/server.lock').open('a') as lock, strata.model_lock(shared=True):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'state/active.json').exists() or available() < 72 * 2**30 or os.getloadavg()[0] > 8:
            raise RuntimeError('A model is active or the machine lacks benchmark headroom.')
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', 8899)) == 0:
                raise RuntimeError('Model port is occupied.')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        used, load = [int(s.strip()) for s in gpu.strip().split(',')]
        if used > 2048 or load > 5:
            raise RuntimeError('GPU is busy.')
        compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        if compute.strip():
            raise RuntimeError('Another GPU compute process is running.')
        wrapped = [str(Path.home() / 'Code/heavy-management/bin/capped'), '--mem', '64G', '--time', '30m',
                   str(Path(info['runtime']['venv']) / 'bin/python'),
                   str(ROOT / 'experiments/deepseek-strata/network_guard.py'), *command]
        started = time.monotonic()
        status: dict[str, Any] = {'status': 'running', 'unit': unit}
        process: subprocess.Popen[str] | None = None
        group: Path | None = None
        verified = False
        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        try:
            with (output / 'native.log').open('w') as log, (output / 'resources.jsonl').open('w') as samples:
                process = subprocess.Popen(wrapped, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, text=True)
                while process.poll() is None:
                    free = available()
                    if free < FLOOR:
                        raise RuntimeError('Available Linux RAM fell below 8 GiB.')
                    if group is None:
                        name = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ControlGroup', '--value'],
                                              env=env, text=True, capture_output=True, timeout=10).stdout.strip()
                        if name:
                            group = Path('/sys/fs/cgroup' + name)
                    sample: dict[str, Any] = {'elapsed': time.monotonic() - started, 'available': free}
                    if group is not None and group.exists():
                        for key in ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current',
                                    'memory.swap.max', 'memory.events', 'memory.stat'):
                            sample[key] = (group / key).read_text().strip()
                        for key in ('io.stat', 'cpu.stat'):
                            try:
                                sample[key] = (group / key).read_text().strip()
                            except FileNotFoundError:
                                sample[key] = None
                        if int(sample['memory.max']) != CAP or int(sample['memory.swap.max']) != 0 or int(sample['memory.swap.current']) != 0:
                            raise RuntimeError('Memory or swap limits differ from the plan.')
                        for pid in (group / 'cgroup.procs').read_text().split():
                            proc = Path('/proc') / pid
                            try:
                                if (proc / 'exe').resolve() == binary:
                                    fields = dict(row.split(':', 1) for row in (proc / 'status').read_text().splitlines())
                                    if fields['Seccomp'].strip() != '2' or fields['NoNewPrivs'].strip() != '1':
                                        raise RuntimeError('Native network isolation is absent.')
                                    verified = True
                                    sample['native_pid'] = int(pid)
                            except FileNotFoundError:
                                pass
                    samples.write(json.dumps(sample) + '\n')
                    samples.flush()
                    if time.monotonic() - started > 1800:
                        raise RuntimeError('Score run exceeded its time limit.')
                    time.sleep(2)
                if process.returncode != 0 or not verified:
                    raise RuntimeError(f'Scorer exited {process.returncode}; isolation verified={verified}.')
            scored = json.loads((output / 'scores.tsv.seg.json').read_text())
            if sum(s['n'] for s in scored['segments']) != plan['positions']:
                raise RuntimeError('Scoring did not cover all planned positions.')
            native_log = (output / 'native.log').read_text()
            if args.route != 'off' and 'cache-aware routing (STRATA_DS4_CACHE_ROUTE, CHANGES THE ANSWERS)' not in native_log:
                raise RuntimeError('The native cache-route activation marker is absent.')
            status.update(status='completed', score_sha256=digest(output / 'scores.tsv'), segments=scored)
        except BaseException as error:
            status.update(status='failed', error=str(error))
            raise
        finally:
            for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            cleanup_errors: list[str] = []
            try:
                try:
                    subprocess.run(['systemctl', '--user', 'stop', unit], env=env, capture_output=True, timeout=15)
                except (OSError, subprocess.SubprocessError) as error:
                    cleanup_errors.append(str(error))
                if process is not None and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                    except ProcessLookupError:
                        pass
                # Recheck the unit after stopping its launcher, including an early-start race.
                probe = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ControlGroup', '--value'],
                                       env=env, text=True, capture_output=True, timeout=10).stdout.strip()
                if probe:
                    group = Path('/sys/fs/cgroup' + probe)
                if group is not None and group.exists() and (group / 'cgroup.procs').read_text().strip():
                    subprocess.run(['systemctl', '--user', 'kill', '--signal=KILL', '--kill-whom=all', unit],
                                   env=env, capture_output=True, timeout=10)
                    for _ in range(50):
                        if not group.exists() or not (group / 'cgroup.procs').read_text().strip():
                            break
                        time.sleep(.1)
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_errors.append(str(error))
            finally:
                try:
                    remaining = (group / 'cgroup.procs').read_text().split() if group is not None else []
                except FileNotFoundError:
                    remaining = []
                status.update(elapsed_seconds=time.monotonic() - started, native_isolation_verified=verified,
                              owned_process_stopped=process is None or process.poll() is not None,
                              owned_scope_empty=not remaining, remaining_pids=remaining, cleanup_errors=cleanup_errors,
                              profile_unchanged=digest(args.profile) == plan['profile_sha256'])
                if remaining or cleanup_errors:
                    status['status'] = 'cleanup_failed'
                save(output / 'summary.json', status)
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    print(json.dumps(status))
    return 0 if status['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
