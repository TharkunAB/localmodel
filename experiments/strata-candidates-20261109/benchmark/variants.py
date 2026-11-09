"""Verify the requested native variant and its live process evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request
from typing import Any, cast

from benchmark_paths import ROOT

SEED = 20261109
ENV_KEYS = ('DS4_BORROW', 'STRATA_DS4_CACHE_ROUTE')


def sha256(path: Path) -> str:
    """Hash file bytes without loading the whole file into Python memory."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose only the two reviewed environment controls and pinned engine override."""
    parser.add_argument('--borrow', type=int, choices=(0, 1))
    parser.add_argument('--cache-route', choices=('off', '0.2', '0.2,0.3'), default='off')
    parser.add_argument('--native-binary', type=Path)
    parser.add_argument('--native-sha256')
    parser.add_argument('--native-receipt', type=Path)
    parser.add_argument('--native-receipt-sha256')


def arguments(args: argparse.Namespace) -> list[str]:
    """Pass the same explicit variant to the isolated controller process."""
    result = ['--cache-route', args.cache_route]
    for key in ('borrow', 'native_binary', 'native_sha256', 'native_receipt', 'native_receipt_sha256'):
        value = getattr(args, key)
        if value is not None:
            result.extend(['--' + key.replace('_', '-'), str(value)])
    return result


def read_variant(args: argparse.Namespace, info: dict[str, Any]) -> dict[str, Any]:
    """Verify all override pins before a controller can start an engine."""
    environment: dict[str, str] = {}
    if args.borrow is not None:
        environment['DS4_BORROW'] = str(args.borrow)
    if args.cache_route != 'off':
        if args.native_binary is None:
            raise RuntimeError('Cache-aware routing requires the reviewed custom native binary.')
        environment['STRATA_DS4_CACHE_ROUTE'] = args.cache_route
    runtime = info['runtime']
    binary = (Path(runtime['directory']) / 'engine-dsv41' /
              f"{runtime['version']}-linux-x86_64-cuda-cuda13/strata-dsv41").resolve()
    expected = runtime['binary_sha256']
    custom: dict[str, Any] | None = None
    override = (args.native_binary, args.native_sha256, args.native_receipt, args.native_receipt_sha256)
    if any(value is not None for value in override):
        if any(value is None for value in override):
            raise RuntimeError('A custom engine requires its binary, SHA256, build receipt and receipt SHA256.')
        binary = args.native_binary.resolve(strict=True)
        receipt_path = args.native_receipt.resolve(strict=True)
        if not binary.is_relative_to(ROOT) or not receipt_path.is_relative_to(ROOT):
            raise RuntimeError('Custom artifacts must remain inside this checkout.')
        if not re.fullmatch('[0-9a-f]{64}', args.native_sha256) or not re.fullmatch('[0-9a-f]{64}', args.native_receipt_sha256):
            raise RuntimeError('Use full lowercase SHA256 pins for the binary and receipt.')
        if sha256(receipt_path) != args.native_receipt_sha256:
            raise RuntimeError('The custom build receipt differs from its pin.')
        receipt = json.loads(receipt_path.read_text())
        if Path(receipt['binary_path']).resolve() != binary or receipt['binary_sha256'] != args.native_sha256:
            raise RuntimeError('The custom engine differs from its build receipt.')
        if not re.fullmatch('[0-9a-f]{40}', receipt['base_revision']) or receipt['base_revision'] != info['source_revision']:
            raise RuntimeError('The custom engine must use the exact installed source base revision.')
        directories = receipt.get('lib_dirs')
        if not isinstance(directories, list) or not directories or any(
                not isinstance(path, str) or not Path(path).is_absolute() or not Path(path).is_dir()
                for path in cast(list[object], directories)):
            raise RuntimeError('The build receipt needs existing absolute library directories.')
        libraries = receipt.get('libraries')
        if not isinstance(libraries, list) or not libraries:
            raise RuntimeError('The build receipt needs pinned runtime libraries.')
        for library in cast(list[dict[str, Any]], libraries):
            if sha256(Path(library['path'])) != library['sha256']:
                raise RuntimeError(f"A custom runtime library differs from its pin: {library['path']}.")
        expected = args.native_sha256
        custom = {'receipt_path': str(receipt_path), 'receipt_sha256': args.native_receipt_sha256,
                  'receipt': receipt}
    if not binary.is_file() or not os.access(binary, os.X_OK) or sha256(binary) != expected:
        raise RuntimeError('The requested native executable differs from its pin or is not executable.')
    return {'borrow': args.borrow, 'cache_route': args.cache_route, 'environment': environment,
            'native_binary': str(binary), 'native_sha256': expected, 'custom': custom,
            'sampling': {'temperature': 1.0, 'top_p': 0.95, 'seed': SEED}, 'native_verbose': True}


def log_evidence(path: Path, offset: int = 0) -> dict[str, Any]:
    """Keep native feature markers separate from requested environment settings."""
    with path.open('rb') as stream:
        stream.seek(offset if path.stat().st_size >= offset else 0)
        lines = stream.read().decode('utf-8', errors='replace').rsplit('engine started:', 1)[-1].splitlines()
    borrowed = [line for line in lines if 'lends the prompt path up to' in line]
    fallback = [line for line in lines if 'keeps the prompt path\'s room for itself' in line or 'too few experts in VRAM' in line]
    returned = [line for line in lines if "the prompt path's room handed back:" in line]
    routing = [line for line in lines if 'cache-aware routing' in line.lower() or 'cache route' in line.lower()]
    return {'borrow_allocation': borrowed, 'borrow_fallback': fallback, 'borrow_returns': returned,
            'cache_route': routing}


def check_markers(variant: dict[str, Any], markers: dict[str, Any], require_route: bool = False) -> None:
    """Reject a disabled borrow arm that still lends prompt memory."""
    if variant['borrow'] == 0 and (markers['borrow_allocation'] or markers['borrow_returns']):
        raise RuntimeError('DS4_BORROW=0 still has native borrowed-room activation markers.')
    if variant['borrow'] != 0 and not (markers['borrow_allocation'] or markers['borrow_fallback']):
        raise RuntimeError('The native log does not establish borrowed-room activation or its allocation fallback.')
    if variant['cache_route'] == 'off' and markers['cache_route']:
        raise RuntimeError('The route-off arm has a native cache-aware routing activation marker.')
    if require_route and variant['cache_route'] != 'off' and not markers['cache_route']:
        raise RuntimeError('The native log does not establish cache-aware routing activation.')
    if markers['cache_route'] and variant['cache_route'] != 'off':
        values = [float(value) for value in variant['cache_route'].split(',')]
        ram, vram = values if len(values) == 2 else (values[0], values[0])
        expected = f"a held expert's selection bonus {ram:.3g} in RAM, {vram:.3g} in VRAM"
        if any(expected not in marker for marker in markers['cache_route']):
            raise RuntimeError('The native routing bonuses differ from the requested values.')


def live_evidence(active: dict[str, Any], variant: dict[str, Any], launch: dict[str, Any],
                  engine_log: Path, offset: int) -> dict[str, Any]:
    """Verify the owned engine bytes, inherited controls and reported sampling."""
    expected_path = Path(variant['native_binary'])
    if Path(launch['exe']).resolve() != expected_path or launch['sampling'] != variant['sampling']:
        raise RuntimeError('The saved launch config differs from the requested binary or sampling.')
    if '--verbose' not in launch['args']:
        raise RuntimeError('The native engine must log activation evidence for every arm.')
    for key in ENV_KEYS:
        if launch['env'].get(key) != variant['environment'].get(key):
            raise RuntimeError(f'Launch configuration lost the requested {key} control.')
    parents: dict[int, int] = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid == os.getuid():
                parents[int(proc.name)] = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    owned = {int(active['pid'])}
    while True:
        expanded = owned | {pid for pid, parent in parents.items() if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    matches: list[Path] = []
    for pid in owned:
        proc = Path('/proc') / str(pid)
        try:
            if (proc / 'exe').resolve(strict=True) == expected_path:
                matches.append(proc)
        except OSError:
            continue
    if len(matches) != 1:
        raise RuntimeError('Expected one matching native process below the owned model scope.')
    proc = matches[0]
    ticks = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    actual_sha = sha256(proc / 'exe')
    if actual_sha != variant['native_sha256']:
        raise RuntimeError('The running native executable differs from its requested hash.')
    environment = dict(item.split(b'=', 1) for item in (proc / 'environ').read_bytes().split(b'\0') if b'=' in item)
    actual_env = {key: environment[os.fsencode(key)].decode() for key in ENV_KEYS if os.fsencode(key) in environment}
    if actual_env != variant['environment']:
        raise RuntimeError('The running native environment differs from the requested variant.')
    argv = [value.decode() for value in (proc / 'cmdline').read_bytes().split(b'\0') if value]
    for flag in ('--max-context', '--ssd-cache', '--cpu-threads', '--prefill-chunk', '--dspark', '--experts'):
        if argv[argv.index(flag) + 1] != launch['args'][launch['args'].index(flag) + 1]:
            raise RuntimeError(f'The running native {flag} setting differs from the launch config.')
    if '--verbose' not in argv:
        raise RuntimeError('The running native command does not enable verbose evidence.')
    security = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines() if ':' in line)
    if security.get('Seccomp', '').strip() != '2':
        raise RuntimeError('The running native engine has no active seccomp filter.')
    loaded_libraries: dict[str, str] = {}
    for line in (proc / 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith(('libcudart.', 'libcublas.', 'libcublasLt.', 'libstrata_kq.')):
            path = Path(fields[5]).resolve(strict=True)
            loaded_libraries[str(path)] = ''
    expected_libraries = {str(Path(item['path']).resolve()): item['sha256']
                          for item in variant['custom']['receipt']['libraries']} if variant['custom'] else None
    for path in loaded_libraries:
        actual = sha256(Path(path))
        if expected_libraries is not None and expected_libraries.get(path) != actual:
            raise RuntimeError(f'A loaded custom runtime library differs from its receipt: {path}.')
        loaded_libraries[path] = actual
    if (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19] != ticks:
        raise RuntimeError('The native PID changed while its evidence was read.')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8899/props', timeout=10) as response:
        props = json.load(response)
    sampling = props['default_generation_settings']['params']
    if any(sampling.get(key) != value for key, value in variant['sampling'].items()):
        raise RuntimeError('The live API sampling defaults differ from the fixed comparison settings.')
    markers = log_evidence(engine_log, offset)
    check_markers(variant, markers)
    return {'pid': int(proc.name), 'start_ticks': ticks, 'native_binary': str(expected_path),
            'native_sha256': actual_sha, 'environment': actual_env,
            'native_argv': argv, 'seccomp': security['Seccomp'].strip(),
            'loaded_libraries': loaded_libraries, 'reported_sampling': sampling, 'markers': markers}
