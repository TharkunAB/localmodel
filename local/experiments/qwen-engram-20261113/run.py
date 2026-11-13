"""Compare Qwen weights, Engram precision and context with the measured OMP fixture."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any, cast
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'results/qwen-engram-20261113'
WORK = BASE / 'work'
FIXTURE = ROOT / 'experiments/deepseek-strata/filled_omp.py'
COMMON = ROOT / 'experiments/strata-0213-20261109/benchmark'
EXTRA = ROOT / 'results/qwen-unsloth-20261106/extended-corpus/expert_source.cpp'
EXTRA_SHA = '1683798c7e99dd7636b86c3dd5a040bc0176e6fd15f1afe90c66fc24bfa8857a'
ARMS = {'default192': ('default', 196608), 'q8192': ('q8_0', 196608),
        'default256': ('default', 262144), 'q8256': ('q8_0', 262144)}
Record = dict[str, Any]
sys.path.insert(0, str(ROOT))
import localmodel


def save(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(quant: str = 'iq1_m', expert_ram_gb: float = 36) -> Record:
    """Freeze current code, settings, corpus and runtime identities without hashing large weights."""
    files = [ROOT / name for name in ('localmodel.py', 'qwen_strata.py', 'qwen_engram.py',
             'omp_local.py', 'config/models.json', 'config/omp.json', 'experiments/qwen-strata/manifest.json')]
    files += list((ROOT / 'experiments/deepseek-strata/omp-filled').glob('*'))
    files += [FIXTURE, COMMON / 'collect.py', COMMON / 'sample_metrics.py', COMMON / 'benchmark_paths.py', EXTRA]
    files += list(ROOT.glob('state/sources/prism-e3de715-*.cpp')) + list(HERE.glob('*.py')) + [HERE / 'run-case.sh']
    info = localmodel.qwen_strata.manifest(quant)
    selected_manifest = localmodel.qwen_strata.manifest_path(quant)
    files += [selected_manifest]
    paths = [Path(info['model_directory']) / item['name'] for item in info['files']]
    paths += [Path(info['runtime']['binary']), localmodel.qwen_strata.qwen_engram.verify()]
    files += [Path(info['source_directory']) / 'data/expert-profile.bin']
    files += list(ROOT.glob('*.py')) + [localmodel.qwen_strata.qwen_engram.table_path().parent / 'verification.json', ROOT / 'config/omp-runtime.json']
    return {'quant': quant, 'expert_ram_gb': expert_ram_gb,
            'quant_manifest': str(selected_manifest.relative_to(ROOT)), 'quant_manifest_sha256': digest(selected_manifest),
            'model_revision': info['model_revision'],
            'files': {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files)) if p.is_file()},
            'runtime': info['runtime'], 'source_tag': info['source_tag'], 'source_revision': info['source_revision'],
            'large_files': {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths},
            'note': 'Large weight identities are size and mtime. Fetch receipts verify the managed SHA256 pins separately.'}


def memory() -> Record:
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {'at': time.time(), 'linux_available_bytes': int(values['MemAvailable'].split()[0]) * 1024,
            'loadavg_1m': os.getloadavg()[0]}


def scope_for(pid: int) -> Path:
    group = next(line[3:] for line in Path('/proc', str(pid), 'cgroup').read_text().splitlines() if line.startswith('0::'))
    return Path('/sys/fs/cgroup' + group)


def guard_reason(sample: Record) -> str | None:
    if sample['linux_available_bytes'] < 8 * 2**30:
        return 'Available Linux memory fell below 8 GiB.'
    model = sample.get('model_memory', {})
    if int(model.get('memory.swap.current', 0)):
        return 'The owned model cgroup used swap.'
    events = dict(line.split() for line in model.get('memory.events', '').splitlines())
    if int(events.get('oom', 0)) or int(events.get('oom_kill', 0)):
        return 'The owned model cgroup reported an OOM event.'
    return None


def query_gpu() -> Record:
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                     '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = [[int(value.strip()) for value in line.split(',')] for line in output.splitlines()]
    return {'gpu_util_percent': max(row[0] for row in rows), 'gpu_used_mib': sum(row[1] for row in rows)}


def admission(quiet_seconds: int) -> list[Record]:
    """Check resources without stopping any existing work."""
    if (ROOT / 'state/active.json').exists():
        raise RuntimeError('A managed model record exists. Do not interrupt it.')
    localmodel.check_free(8899)
    own = scope_for(os.getpid())
    if int((own / 'memory.max').read_text()) > 4 * 2**30 or (own / 'memory.swap.max').read_text().strip() != '0':
        raise RuntimeError('Run through run-case.sh with its 4 GiB controller cap and no swap.')
    samples: list[Record] = []
    deadline = time.monotonic() + quiet_seconds
    while True:
        compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10)
        sample = {**memory(), **query_gpu()}
        samples.append(sample)
        if compute.strip() or sample['gpu_util_percent'] > 5 or sample['gpu_used_mib'] > 2048:
            raise RuntimeError('GPU is not quiet: ' + json.dumps(sample))
        if sample['linux_available_bytes'] < 72 * 2**30 or sample['loadavg_1m'] > 8:
            raise RuntimeError('Need 72 GiB available Linux RAM and one-minute CPU load at most 8: ' + json.dumps(sample))
        if time.monotonic() >= deadline:
            return samples
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def alive(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def search_health() -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open('http://127.0.0.1:8888/healthz', timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def stop_process(process: subprocess.Popen[str] | None) -> None:
    """Stop only a process group started by this controller."""
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=65)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def stop_worker() -> None:
    path = WORK / 'worker.json'
    if not path.exists():
        return
    pid = int(json.loads(path.read_text())['pid'])
    proc = Path('/proc', str(pid))
    if proc.exists():
        marker = os.fsencode('OMP_BENCH_ROOT=' + str(WORK))
        if proc.stat().st_uid != os.getuid() or marker not in (proc / 'environ').read_bytes().split(b'\0'):
            raise RuntimeError('OMP worker ownership changed; refusing to signal it.')
    subprocess.run(['systemctl', '--user', 'stop', f'omp-local-{pid}.scope'], timeout=20,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if proc.exists():
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def stop_model(process: subprocess.Popen[str] | None, scope: Path | None) -> None:
    """Stop the owned controller and require its recorded model scope to be empty."""
    if scope is None and process is not None and (ROOT / 'state/active.json').exists():
        active = json.loads((ROOT / 'state/active.json').read_text())
        if active.get('controller_pid') == process.pid and Path('/proc', str(active['pid'])).exists():
            candidate = scope_for(int(active['pid']))
            if (candidate / 'memory.max').read_text().strip() == str(64 * 2**30):
                scope = candidate
    stop_process(process)
    if scope is not None and (scope / 'cgroup.events').exists():
        events = dict(line.split() for line in (scope / 'cgroup.events').read_text().splitlines())
        if events.get('populated') == '1':
            subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=SIGKILL', scope.name],
                           timeout=10, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 15
        while (scope / 'cgroup.events').exists():
            events = dict(line.split() for line in (scope / 'cgroup.events').read_text().splitlines())
            if events.get('populated') == '0':
                return
            if time.monotonic() > deadline:
                raise RuntimeError('Owned model scope still contains processes after cleanup.')
            time.sleep(0.1)


class Guard:
    """Watch memory continuously and collect existing native metrics."""
    def __init__(self, server: subprocess.Popen[str], observer: ModuleType) -> None:
        self.server = server
        self.observer = observer
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.scope: Path | None = None
        self.last: Record = {}
        self.ready = False

    def abort(self, reason: str) -> None:
        save(WORK / 'aborted.json', {'reason': reason, 'at': time.time()})
        os.kill(os.getpid(), signal.SIGTERM)

    def run(self) -> None:
        failures = 0
        seen: set[str] = set()
        while not self.done.is_set():
            try:
                row = memory()
                critical = guard_reason(row)
                if critical:
                    self.abort(critical)
                    return
                active_path = ROOT / 'state/active.json'
                if active_path.exists():
                    active = json.loads(active_path.read_text())
                    if active.get('controller_pid') != self.server.pid:
                        raise RuntimeError('Active controller changed during the benchmark.')
                    candidate = scope_for(int(active['pid']))
                    if (candidate / 'memory.max').read_text().strip() == str(64 * 2**30):
                        self.scope = candidate
                    if self.scope:
                        row['model_memory'] = {key: (self.scope / key).read_text().strip() for key in
                            ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current', 'memory.swap.max', 'memory.events', 'memory.stat')}
                        io_path = self.scope / 'io.stat'
                        row['model_io'] = io_path.read_text() if io_path.exists() else None
                    critical = guard_reason(row)
                    if critical:
                        self.abort(critical)
                        return
                    if self.ready:
                        self.observer.sample(active, WORK, seen)
                row.update(query_gpu())
                failures = 0
                reason = guard_reason(row)
                self.last = row
                with (WORK / 'memory.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                save(WORK / 'memory-current.json', row)
                with (WORK / 'resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
            except (OSError, ValueError, RuntimeError, KeyError, StopIteration, subprocess.SubprocessError) as error:
                failures += 1
                reason = str(error) if failures >= 3 else None
            if reason:
                self.abort(reason)
                return
            self.done.wait(2)


def verify_edit_schema(body: Record) -> None:
    edits = [item['function'] for item in body.get('tools', []) if item.get('function', {}).get('name') == 'edit']
    if len(edits) != 1:
        raise RuntimeError('Expected one guarded edit tool.')
    schema = edits[0].get('parameters', {})
    if not {'path', 'old_string', 'new_string'} <= set(schema.get('required', [])) or not set(schema.get('properties', {})) <= {'i', 'path', 'old_string', 'new_string', 'replace_all'}:
        raise RuntimeError('OMP edit schema is not the guarded replacement mode.')


def compaction_probe(benchmark: Any) -> Record:
    config_path = WORK / 'config/omp.json'
    config = json.loads(config_path.read_text())
    config['compaction']['enabled'] = True
    save(config_path, config)
    private_path = WORK / 'state/omp-local/agent/config.yml'
    if private_path.with_suffix('.yaml').exists():
        raise RuntimeError('The benchmark expects its JSON-formatted private config.yml, not config.yaml.')
    private_config = json.loads(private_path.read_text())
    previously_enabled = private_config.get('compaction', {}).get('enabled')
    private_config.setdefault('compaction', {})['enabled'] = True
    save(private_path, private_config)
    private_evidence = {'path': str(private_path), 'previously_enabled': previously_enabled,
                        'enabled': json.loads(private_path.read_text())['compaction']['enabled'],
                        'method_order': private_config['compaction'].get('methodOrder'),
                        'model_thresholds': json.loads((WORK / 'state/omp-local/launcher.yml').read_text())
                            .get('compaction', {}).get('modelThresholds', {})}
    save(WORK / 'compaction-effective-config.json', private_evidence)
    before = len(benchmark.session.read_text().splitlines())
    target = benchmark.targets[-1]
    prompt = ('Resume the last checkpoint after automatic maintenance. Do not use tools. Return only JSON with '
              'release_key, target_branch, deadline_tag for checkpoint ' + str(target) + ', and probe_value equal to clamp(6,2,8).')
    result = benchmark.run_omp(prompt, str(target) + '-compaction')
    records = [json.loads(line) for line in benchmark.session.read_text().splitlines()[before:]]
    events = [json.loads(line) for line in Path(result['artifacts'] + '.stdout.jsonl').read_text().splitlines() if line.startswith('{')]
    evidence = [event for event in events if 'compaction' in str(event.get('type', ''))]
    entries = [entry for entry in records if 'compaction' in str(entry.get('type', ''))]
    expected = {**benchmark.facts[str(target)], 'probe_value': 6}
    try:
        passed = json.loads(result['answer'].strip()) == expected and not result['tools']
    except ValueError:
        passed = False
    return {**result, 'attempted': True, 'observed': bool(entries), 'events': evidence,
            'effective_private_config': private_evidence,
            'entry_types': [entry['type'] for entry in entries], 'retention_passed': passed,
            'note': 'Automatic maintenance uses normal OMP thresholds. A successful recall without an observed compaction is not a compaction result.'}


def benchmark_case(args: argparse.Namespace, output: Path, context: int, engram: str, targets: list[int], provenance: Path) -> None:
    expected = json.loads(provenance.read_text())
    if snapshot(args.quant, args.expert_ram_gb) != expected:
        raise RuntimeError('Frozen provenance changed. Review changes before creating a new study freeze.')
    info = localmodel.qwen_strata.manifest(args.quant)
    preflight = admission(args.quiet_seconds)
    WORK.mkdir()
    save(WORK / 'provenance.json', expected)
    report: Record = {'status': 'starting', 'arm': args.arm, 'case': args.case, 'repeat': args.repeat,
        'seed': 20261113 + args.repeat - 1, 'engram': engram, 'context_capacity_tokens': context,
        'targets': targets, 'quant': args.quant, 'expert_ram_gb': args.expert_ram_gb,
        'model_revision': info['model_revision'], 'quant_manifest_sha256': expected['quant_manifest_sha256'],
        'memory_cap_gib': 64, 'speculative': 'mtp', 'draft_tokens': 4, 'preflight': preflight}
    state = ROOT / 'state/qwen-server.json'
    original = state.read_bytes() if state.exists() else None
    profile = Path(info['source_directory']) / 'data/expert-profile.bin'
    profile_before = profile.read_bytes()
    (WORK / 'expert-profile-before.bin').write_bytes(profile_before)
    protected = [Path.home() / '.omp/agent' / name for name in ('config.yml', 'config.yaml')]
    protected += [ROOT / 'state' / launcher / 'agent' / name for launcher in ('omp-local', 'omp-remote') for name in ('config.yml', 'config.yaml')]
    settings_before = {str(path): digest(path) if path.exists() else None for path in protected}
    save(WORK / 'normal-settings-before.json', settings_before)
    report.update(expert_profile_sha256=digest(profile), lane_profile='Not used by this Qwen configuration; expert_profile_save is absent.')
    engine_log = ROOT / 'state/qwen-engine.log'
    offset = engine_log.stat().st_size if engine_log.exists() else 0
    server: subprocess.Popen[str] | None = None
    search: subprocess.Popen[str] | None = None
    search_scope: Path | None = None
    guard: Guard | None = None
    started = time.monotonic()
    env = dict(os.environ, PI_EDIT_VARIANT='replace', OMP_BENCH_MAX_REQUESTS='16', STRATA_BENCH_RELEASE='v0.2.13')
    env.pop('CAPPED_UNIT', None)
    os.environ.update(env)
    sys.path.insert(0, str(COMMON))
    observer = load('qwen_metrics', COMMON / 'sample_metrics.py')
    collector = load('qwen_collect', COMMON / 'collect.py')
    command = [str(localmodel.qwen_strata.PYTHON), '-B', str(HERE / 'server.py'), '--controller-marker', str(ROOT / 'localmodel.py'),
        '--seed', str(report['seed']), '--', 'start', 'qwen3.8', '--quant', args.quant, '--engram', engram, '--context', str(context),
        '--mem', '64G', '--expert-ram-gb', str(args.expert_ram_gb), '--adaptive-fetch', '0', '--vram-reserve-mib', '1024',
        '--speculative', 'mtp', '--draft-tokens', '4', '--kv-vram-cells', '65536', '--ple-io', 'direct',
        '--no-prompt-lookup', '--reasoning-effort', 'xhigh']
    save(WORK / 'command.json', command)
    try:
        if alive(8888):
            if not search_health():
                raise RuntimeError('Port 8888 belongs to an unhealthy or unknown service. It was not changed.')
            report['search_owned'] = False
        else:
            with (WORK / 'search.log').open('w') as log:
                search = subprocess.Popen([str(ROOT / 'bin/searxng-local')], env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            report['search_owned'] = True
            deadline = time.monotonic() + 45
            while not search_health():
                if search.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Owned SearXNG did not become healthy.')
                time.sleep(1)
            candidate = scope_for(search.pid)
            if (candidate / 'memory.max').read_text().strip() != str(2**30):
                raise RuntimeError('Owned search service is not in its 1 GiB cgroup.')
            search_scope = candidate
        with (WORK / 'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        save(WORK / 'controller.json', {'controller_pid': server.pid})
        guard = Guard(server, observer)
        guard.thread.start()
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None or (WORK / 'aborted.json').exists() or time.monotonic() > deadline:
                raise RuntimeError('Qwen startup failed; inspect server.log and aborted.json.')
            try:
                status = localmodel.current(8899)
                break
            except (OSError, ValueError, RuntimeError):
                time.sleep(2)
        if (status['context'] != context or status.get('speculative') != 'mtp'
                or status.get('expert_ram_gb') != args.expert_ram_gb or status.get('quant') != args.quant):
            raise RuntimeError('Live model settings differ from the selected arm.')
        if guard.scope is None:
            raise RuntimeError('The owned model 64 GiB cgroup was not found.')
        limits = {key: (guard.scope / key).read_text().strip() for key in ('memory.max', 'memory.swap.max')}
        if limits != {'memory.max': str(64 * 2**30), 'memory.swap.max': '0'}:
            raise RuntimeError('Model cgroup limits differ from the plan.')
        current_log = engine_log.read_bytes()[offset:].decode(errors='replace')
        markers = [line for line in current_log.splitlines() if 'draft layer' in line or 'draft head' in line or 'PLE on, table' in line]
        if 'strata mtp: draft layer loaded,' not in current_log or 'without the draft layer' in current_log or 'without draft layer' in current_log:
            raise RuntimeError('Native startup does not verify an active MTP draft layer.')
        selected_table = localmodel.qwen_strata.qwen_engram.table_path() if engram == 'q8_0' else Path(info['ple_shard'])
        if 'PLE on, table 320001536 rows of ' + str(selected_table) not in current_log:
            raise RuntimeError('Native startup did not confirm the selected Engram artifact.')
        report.update(live=status, limits=limits, startup_seconds=time.monotonic() - started, mtp_startup_evidence=markers)
        launch_config = json.loads(state.read_text())
        if launch_config.get('expert_profile_save') or '--expert-profile-save' in launch_config['args']:
            raise RuntimeError('This study requires the frozen input expert profile without persistent learning.')
        save(WORK / 'launch-config.json', launch_config)
        guard.ready = True
        fixture = load('qwen_filled', FIXTURE)
        setattr(fixture, 'PYTHON', localmodel.qwen_strata.PYTHON)
        benchmark = fixture.Benchmark(WORK, targets, args.timeout, model='qwen3.8', context=context,
            source_directory=Path(info['source_directory']), tokenizer_directory=Path(info['pack_directory']) / 'tokenizer')
        benchmark.facts = {str(target): {'release_key': f'cedar-{target}-quartz', 'target_branch': f'release/qwen-{target}', 'deadline_tag': f'gate-{target}-amber'} for target in targets}
        if digest(EXTRA) != EXTRA_SHA:
            raise RuntimeError('The pinned public extension corpus changed.')
        benchmark.corpus += '\nPublic source file: strata-c3b3f5-expert_source.cpp\n' + EXTRA.read_text()
        benchmark.report['sources'].append({'path': str(EXTRA), 'sha256': EXTRA_SHA})
        benchmark.report.update(server_context_capacity=context, expert_cache_gib=args.expert_ram_gb,
            quant=args.quant, model_revision=info['model_revision'], quant_manifest_sha256=expected['quant_manifest_sha256'],
            engine_version=expected['runtime']['version'], speculative='mtp')
        original_run = benchmark.run_omp
        def run_omp(prompt: str, phase: str, capture: bool = False) -> Record:
            result: Record = original_run(prompt, phase, capture)
            if capture:
                verify_edit_schema(result)
                paired = {key: result.get(key) for key in ('model', 'messages', 'tools', 'tool_choice')}
                value = hashlib.sha256(json.dumps(paired, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                if phase.endswith('-final-size'):
                    save(WORK / (phase + '-identity.json'), {'sha256': value, 'phase': phase, 'input_tokens': benchmark.count(result)})
            return result
        benchmark.run_omp = run_omp
        benchmark.run()
        report['all_quality_checks_passed'] = benchmark.report['all_quality_checks_passed']
        if args.case == 'long':
            report['compaction'] = compaction_probe(benchmark)
        phases = [collector.phase(path, []) for path in sorted(WORK.glob('*.network.jsonl')) if '-capture-' not in path.name]
        requests = [request for phase in phases for request in phase['requests']]
        offered = sum(request.get('draft_offered_tokens') or 0 for request in requests)
        accepted = sum(request.get('draft_accepted_tokens') or 0 for request in requests)
        # Native metrics contain Qwen counters when the SSE extension omits them.
        native = [json.loads(line) for line in (WORK / 'metrics-requests.jsonl').read_text().splitlines()] if (WORK / 'metrics-requests.jsonl').exists() else []
        offered = max(offered, sum(row.get('drafts_offered') or 0 for row in native))
        accepted = max(accepted, sum(row.get('drafts_accepted') or 0 for row in native))
        report.update(drafts_offered=offered, drafts_accepted=accepted)
        if not offered or not accepted:
            raise RuntimeError('No positive native MTP offered/accepted counters were recorded.')
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if guard:
            guard.done.set()
            guard.thread.join(timeout=15)
            report['guard_stopped'] = not guard.thread.is_alive()
            report['memory'] = guard.last.get('model_memory', {})
        for name, action in [('worker', stop_worker), ('server', lambda: stop_model(server, guard.scope if guard else None)), ('search', lambda: stop_model(search, search_scope))]:
            try:
                action()
                report[name + '_stopped'] = True
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                report[name + '_cleanup_error'] = str(error)
        report['owned_search_port_closed'] = not alive(8888) if report.get('search_owned') else None
        if report.get('search_owned') and not report['owned_search_port_closed']:
            report['search_cleanup_error'] = 'The owned search port is still open.'
        active_path = ROOT / 'state/active.json'
        with (ROOT / 'state/server.lock').open('a') as server_lock:
            try:
                fcntl.flock(server_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if server is not None and server.poll() is not None and report.get('server_stopped') and active_path.exists():
                    remaining = json.loads(active_path.read_text())
                    if remaining.get('controller_pid') == server.pid and not Path('/proc', str(remaining['pid'])).exists():
                        active_path.unlink()
                report['port_8899_closed'] = not alive(8899)
                if not active_path.exists() and report['port_8899_closed'] and report.get('server_stopped'):
                    if original is None:
                        state.unlink(missing_ok=True)
                    else:
                        state.write_bytes(original)
                    report['state_config_restored'] = True
                    if profile.read_bytes() != profile_before:
                        with localmodel.qwen_strata.model_lock():
                            profile.write_bytes(profile_before)
                    report['profile_restored'] = profile.read_bytes() == profile_before
            except BlockingIOError:
                report['state_restore_skipped'] = 'Another server owns server.lock; its state was not changed.'
        settings_after = {str(path): digest(path) if path.exists() else None for path in protected}
        save(WORK / 'normal-settings-after.json', settings_after)
        report['normal_settings_unchanged'] = settings_after == settings_before
        if not report['normal_settings_unchanged']:
            report.update(status='failed', error='Normal OMP settings changed; no global overwrite was attempted.')
        if engine_log.exists():
            (WORK / 'engine.log').write_bytes(engine_log.read_bytes()[offset:])
        report['total_seconds'] = time.monotonic() - started
        if any(key.endswith('_cleanup_error') for key in report) or not report.get('server_stopped') or not report.get('worker_stopped') or not report.get('guard_stopped', True) or not report.get('port_8899_closed', False):
            report.update(status='failed', cleanup_incomplete=True)
        save(WORK / 'case-summary.json', report)
        measurements = [collector.collect_case(WORK, target) for target in targets]
        for index, measurement in enumerate(measurements):
            final = measurement['quality']['repair_final_json']
            try:
                value = json.loads(final.get('final_visible_text', ''))
            except ValueError:
                value = {}
            expected_function = ('clamp', 'clamp_index', 'clamp_many', 'clamp_span')[index]
            final.pop('fixed_function_is_clamp', None)
            final['expected_function'] = expected_function
            final['fixed_function_correct'] = isinstance(value, dict) and cast(Record, value).get('fixed_function') == expected_function
            final['passed'] = all(final.get(key) for key in ('whole_reply_is_json', 'exact_keys', 'facts_match', 'no_final_tool_calls', 'fixed_function_correct')) and not final.get('duplicate_keys')
        save(WORK / 'measurements.json', measurements)
        if report.get('server_stopped') and report.get('worker_stopped') and report.get('guard_stopped', True):
            WORK.rename(output)
        print(json.dumps({'output': str(output), **{key: report.get(key) for key in ('status', 'error', 'all_quality_checks_passed', 'total_seconds')}}), flush=True)
        if report.get('cleanup_incomplete'):
            raise RuntimeError('Owned benchmark cleanup is incomplete; inspect case-summary.json.')


def main() -> None:
    global WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quant', choices=localmodel.qwen_strata.QUANTS, default='iq1_m')
    parser.add_argument('--expert-ram-gb', type=float, default=36)
    parser.add_argument('--arm', choices=ARMS, default='default192')
    parser.add_argument('--case', choices=('matched64', 'long'), default='matched64')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--quiet-seconds', type=int, default=20)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        safe = {'linux_available_bytes': 9 * 2**30, 'model_memory': {'memory.swap.current': '0', 'memory.events': 'max 12\noom 0\noom_kill 0\n'}}
        assert guard_reason(safe) is None
        assert guard_reason({**safe, 'linux_available_bytes': 7 * 2**30})
        assert guard_reason({**safe, 'model_memory': {'memory.swap.current': '1'}})
        assert guard_reason({**safe, 'model_memory': {'memory.events': 'oom 1\n'}})
        assert ARMS['default192'][1] == 196608 and ARMS['q8256'][1] == 262144
        print('CPU-only guard and arm checks passed; no GPU or model was queried.')
        return
    if args.repeat < 1 or args.timeout < 60 or args.quiet_seconds < 10:
        parser.error('Use repeat >= 1, timeout >= 60, and quiet-seconds >= 10.')
    if not math.isfinite(args.expert_ram_gb) or not 0 <= args.expert_ram_gb < 64:
        parser.error('Use a finite expert RAM budget of at least 0 and below the 64 GiB model cap.')
    quant_study = args.quant != 'iq1_m' or args.expert_ram_gb != 36
    base = ROOT / 'results' / ('qwen-quants-20261113' if quant_study else 'qwen-engram-20261113')
    # Keep the workspace hook used by the shared guard and cleanup functions.
    WORK = base / 'work'  # pyright: ignore[reportConstantRedefinition]
    selection = f'{args.quant}-ram{args.expert_ram_gb:g}g'
    provenance = base / (f'provenance-{selection}.json' if quant_study else 'provenance.json')
    engram, context = ARMS[args.arm]
    targets = [16384, 32768, 49152, 65536] if args.case == 'matched64' else [16384, 65536, 131072, 184320 if context == 196608 else 245760]
    output = base / f'{selection + "-" if quant_study else ""}{args.arm}-{args.case}-r{args.repeat}'
    if not args.run and not args.freeze:
        info = localmodel.qwen_strata.manifest(args.quant)
        print(json.dumps({'arm': args.arm, 'quant': args.quant, 'expert_ram_gb': args.expert_ram_gb,
                          'engram': engram, 'context_capacity': context, 'filled_targets': targets,
                          'source_directory': info['source_directory'],
                          'tokenizer_directory': str(Path(info['pack_directory']) / 'tokenizer'),
                          'provenance': str(provenance), 'output': str(output), 'starts_inference': False}, indent=2))
        return
    base.mkdir(exist_ok=True, parents=True)
    with (base / 'benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.freeze:
            if provenance.exists():
                raise RuntimeError('Provenance already exists; preserve it with its results.')
            save(provenance, snapshot(args.quant, args.expert_ram_gb))
            print('Frozen current code and runtime identities without inference.')
            return
        if output.exists() or WORK.exists():
            raise RuntimeError('Case output or stable work directory exists; inspect it before another run.')
        signal.alarm(75 * 60)
        benchmark_case(args, output, context, engram, targets, provenance)


if __name__ == '__main__':
    def interrupt(_signal: int, _frame: Any) -> None:
        raise RuntimeError('Benchmark interrupted; stopping only its owned processes.')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    main()
