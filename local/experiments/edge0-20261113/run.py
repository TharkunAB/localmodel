"""Measure Edge0 through isolated OMP. Preview unless --run is present."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, cast

import server

ROOT, HERE = server.ROOT, server.HERE
shared = server.shared
BASE = ROOT / 'results/edge0-20261113'
WORK = BASE / 'work'
setattr(shared, 'WORK', WORK)
Record = dict[str, Any]


def capped() -> None:
    """Put the benchmark controller and CPU checks in a 4 GiB no-swap scope."""
    os.environ.setdefault('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
    os.environ.setdefault('DBUS_SESSION_BUS_ADDRESS', f'unix:path=/run/user/{os.getuid()}/bus')
    scope = shared.scope_for(os.getpid())
    maximum = (scope / 'memory.max').read_text().strip()
    if maximum != 'max' and int(maximum) <= 4 * 2**30 and (scope / 'memory.swap.max').read_text().strip() == '0':
        return
    env = dict(os.environ)
    env.pop('CAPPED_UNIT', None)
    env.update(XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',
               DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus',
               HEAVY_PEAKS=str(ROOT / 'state/edge0-benchmark-peaks'))
    command = [str(ROOT / 'libexec/capped'), '--mem', '4G', str(server.PYTHON), '-B', str(HERE / 'run.py'), *sys.argv[1:]]
    os.execve(command[0], command, env)


def measure(args: argparse.Namespace, output: Path, targets: list[int]) -> None:
    preflight = shared.admission(args.quiet_seconds)
    WORK.mkdir()
    report: Record = {'status': 'starting', 'model': server.NAME, 'case': args.case, 'seed': args.seed,
                     'context_capacity_tokens': args.context, 'targets': targets, 'memory_cap_gib': 64,
                     'expert_cache_gib': 40, 'speculative': 'mtp', 'draft_tokens': 3, 'preflight': preflight}
    protected = [Path.home() / '.omp/agent' / name for name in ('config.yml', 'config.yaml')]
    protected += [ROOT / 'state' / launcher / 'agent' / name
                  for launcher in ('omp-local', 'omp-remote') for name in ('config.yml', 'config.yaml')]
    protected += [ROOT / 'state' / name for name in ('qwen-server.json', 'deepseek-server.json')]
    before = {str(path): shared.digest(path) if path.exists() else None for path in protected}
    shared.save(WORK / 'normal-settings-before.json', before)
    sys.path.insert(0, str(shared.COMMON))
    observer = shared.load('edge_metrics', shared.COMMON / 'sample_metrics.py')
    collector = shared.load('edge_collect', shared.COMMON / 'collect.py')
    child: subprocess.Popen[str] | None = None
    search: subprocess.Popen[str] | None = None
    search_scope: Path | None = None
    guard: Any = None
    started = time.monotonic()
    env = dict(os.environ, PI_EDIT_VARIANT='replace', OMP_BENCH_MAX_REQUESTS='16')
    env.pop('CAPPED_UNIT', None)
    command = [str(server.PYTHON), '-B', str(HERE / 'server.py'), '--run', '--context', str(args.context),
               '--output', str(WORK), '--seed', str(args.seed), '--controller-marker', str(ROOT / 'localmodel.py'),
               '--parent-pid', str(os.getpid())]
    shared.save(WORK / 'command.json', command)
    try:
        if shared.alive(8888):
            if not shared.search_health():
                raise RuntimeError('Existing port 8888 is not healthy SearXNG; it was not changed.')
            report['search_owned'] = False
        else:
            with (WORK / 'search.log').open('w') as stream:
                search = subprocess.Popen([str(ROOT / 'bin/searxng-local')], env=env, stdout=stream,
                                          stderr=subprocess.STDOUT, text=True, start_new_session=True)
            report['search_owned'] = True
            deadline = time.monotonic() + 45
            while True:
                if search.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Owned SearXNG did not become healthy.')
                candidate = cast(Path, shared.scope_for(search.pid))
                if (candidate / 'memory.max').read_text().strip() == str(2**30):
                    search_scope = candidate
                if shared.search_health():
                    break
                time.sleep(1)
            if search_scope is None:
                raise RuntimeError('Owned SearXNG is not in its 1 GiB scope.')
        with (WORK / 'server.log').open('w') as stream:
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True)
        guard = shared.Guard(child, observer)
        guard.thread.start()
        deadline = time.monotonic() + 900
        while True:
            if child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Edge0 startup failed; inspect server.log and engine.log.')
            try:
                status = server.localmodel.current(8899)
                break
            except (OSError, ValueError, RuntimeError):
                time.sleep(2)
        if status['model'] != server.NAME or status['context'] != args.context or status.get('runtime') != 'strata-edge0':
            raise RuntimeError('Live model identity or context differs from the Edge0 case.')
        status['backend'] = 'strata'
        if guard.scope is None:
            raise RuntimeError('The Edge0 64 GiB model scope was not observed.')
        report['limits'] = {key: (guard.scope / key).read_text().strip() for key in ('memory.max', 'memory.swap.max')}
        if report['limits'] != {'memory.max': str(64 * 2**30), 'memory.swap.max': '0'}:
            raise RuntimeError('Edge0 model memory limits differ from the plan.')
        log = (WORK / 'engine.log').read_text()
        if 'strata mtp: draft layer loaded,' not in log or 'without draft layer' in log or 'without the draft layer' in log:
            raise RuntimeError('Edge0 did not load the requested MTP layer.')
        report.update(live=status, startup_seconds=time.monotonic() - started)
        guard.ready = True
        fixture = shared.load('edge_filled', shared.FIXTURE)
        fixture.PYTHON = server.PYTHON
        os.environ.update(env)
        benchmark = fixture.Benchmark(WORK, targets, args.timeout, model=server.NAME, context=args.context,
                                      source_directory=server.SOURCE, tokenizer_directory=server.MODEL / 'tokenizer')
        benchmark.facts = {str(target): {'release_key': f'cedar-{target}-quartz',
            'target_branch': f'release/edge0-{target}', 'deadline_tag': f'gate-{target}-amber'} for target in targets}
        if len(targets) > 1:
            if shared.digest(shared.EXTRA) != shared.EXTRA_SHA:
                raise RuntimeError('The public extension corpus changed.')
            benchmark.corpus += '\nPublic source: strata expert_source.cpp\n' + shared.EXTRA.read_text()
            benchmark.report['sources'].append({'path': str(shared.EXTRA), 'sha256': shared.EXTRA_SHA})
        benchmark.report.update(engine_version='edge0-da9cc526', expert_cache_gib=40, speculative='mtp')
        original = benchmark.run_omp
        def run_omp(prompt: str, phase: str, capture: bool = False) -> Record:
            result: Record = original(prompt, phase, capture)
            if capture:
                shared.verify_edit_schema(result)
            return result
        benchmark.run_omp = run_omp
        benchmark.run()
        report['all_quality_checks_passed'] = benchmark.report['all_quality_checks_passed']
        if args.case == 'long':
            report['compaction'] = shared.compaction_probe(benchmark)
        native = [json.loads(line) for line in (WORK / 'metrics-requests.jsonl').read_text().splitlines()]
        report.update(drafts_offered=sum(row.get('drafts_offered') or 0 for row in native),
                      drafts_accepted=sum(row.get('drafts_accepted') or 0 for row in native))
        if not report['drafts_offered'] or not report['drafts_accepted']:
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
            guard.thread.join(15)
            report['guard_stopped'] = not guard.thread.is_alive()
            report['memory'] = guard.last.get('model_memory', {})
        if search_scope is None and search is not None and search.poll() is None:
            try:
                candidate = cast(Path, shared.scope_for(search.pid))
                if (candidate / 'memory.max').read_text().strip() == str(2**30):
                    search_scope = candidate
            except FileNotFoundError:
                pass
        for name, action in [('worker', shared.stop_worker),
                             ('server', lambda: shared.stop_model(child, guard.scope if guard else None)),
                             ('search', lambda: shared.stop_model(search, search_scope))]:
            try:
                action()
                report[name + '_stopped'] = True
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                report[name + '_cleanup_error'] = str(error)
        after = {str(path): shared.digest(path) if path.exists() else None for path in protected}
        report['normal_settings_unchanged'] = before == after
        report['port_8899_closed'] = not shared.alive(8899)
        with (ROOT / 'state/server.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                active_path = ROOT / 'state/active.json'
                if child is not None and child.poll() is not None and report.get('server_stopped') and active_path.exists():
                    active = json.loads(active_path.read_text())
                    if active.get('controller_pid') == child.pid and not Path('/proc', str(active['pid'])).exists():
                        active_path.unlink()
                report['owned_active_record_removed'] = not active_path.exists() or json.loads(active_path.read_text()).get('controller_pid') != (child.pid if child else None)
            except BlockingIOError:
                report['owned_active_record_removed'] = False
        if report.get('search_owned') and shared.alive(8888):
            report['search_cleanup_error'] = 'The owned search port is still open.'
        report['total_seconds'] = time.monotonic() - started
        if (any(key.endswith('_cleanup_error') for key in report) or not report.get('guard_stopped', True)
                or not report['normal_settings_unchanged'] or not report['port_8899_closed']
                or not report['owned_active_record_removed']):
            report.update(status='failed', cleanup_incomplete=True)
        shared.save(WORK / 'normal-settings-after.json', after)
        shared.save(WORK / 'case-summary.json', report)
        measurements = [collector.collect_case(WORK, target) for target in targets]
        for index, measurement in enumerate(measurements):
            final = measurement['quality']['repair_final_json']
            try:
                answer = json.loads(final.get('final_visible_text', ''))
            except ValueError:
                answer = {}
            final.pop('fixed_function_is_clamp', None)
            final['expected_function'] = ('clamp', 'clamp_index', 'clamp_many', 'clamp_span')[index]
            final['fixed_function_correct'] = isinstance(answer, dict) and cast(Record, answer).get('fixed_function') == final['expected_function']
            final['passed'] = all(final.get(key) for key in ('whole_reply_is_json', 'exact_keys', 'facts_match', 'no_final_tool_calls', 'fixed_function_correct')) and not final.get('duplicate_keys')
        shared.save(WORK / 'measurements.json', measurements)
        if not report.get('cleanup_incomplete'):
            WORK.rename(output)
        print(json.dumps({'output': str(output), 'status': report['status'], 'error': report.get('error')}))
        if report.get('cleanup_incomplete'):
            raise RuntimeError('Edge0 cleanup is incomplete; inspect the retained work directory.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=server.localmodel.tokens, choices=(32768, 196608, 262144), default=32768)
    parser.add_argument('--case', choices=('short', 'matched64', 'long'), default='short')
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--quiet-seconds', type=int, default=20)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    targets = ([16384] if args.case == 'short' else [16384, 32768, 49152, 65536] if args.case == 'matched64'
               else [16384, 65536, 131072, 184320 if args.context == 196608 else 245760])
    if args.timeout < 60 or args.quiet_seconds < 10 or any(target + 4096 > args.context for target in targets):
        parser.error('Use a sufficient context, timeout >= 60 and quiet-seconds >= 10.')
    output = BASE / f'q3k-q4-{args.context}-{args.case}-{args.seed}'
    if not args.run and not args.selftest:
        print(json.dumps({'starts_inference': False, 'targets': targets, 'output': str(output),
                          'configuration': server.configuration(args.context, WORK, args.seed)}, indent=2))
        return
    capped()
    if args.selftest:
        assert shared.guard_reason({'linux_available_bytes': 7 * 2**30})
        config = server.configuration(32768, WORK, args.seed)
        assert config['model_name'] == 'edge0-145b' and config['args'][config['args'].index('--spec') + 1] == '4'
        assert config['env']['STRATA_LATENT_DENSE_FMT'] == '4' and config['api_monitor'] is False
        print('CPU-only plan and guard checks passed. No GPU or model was queried.')
        return
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if WORK.exists() or output.exists():
            raise RuntimeError('Edge0 case or work directory exists; preserve and inspect it before another run.')
        signal.alarm(75 * 60)
        measure(args, output, targets)


if __name__ == '__main__':
    def interrupt(_number: int, _frame: Any) -> None:
        raise RuntimeError('Edge0 benchmark interrupted; stopping owned processes.')
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(number, interrupt)
    main()
