"""Verify the official engine and its live process evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import urllib.request
from typing import Any

SEED = 20261109
ENV_KEYS = ('DS4_BORROW', 'STRATA_DS4_CACHE_ROUTE')


def sha256(path: Path) -> str:
    """Hash file bytes without loading the whole file into Python memory."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_variant(info: dict[str, Any]) -> dict[str, Any]:
    """Verify the installed official executable before a controller starts it."""
    runtime = info['runtime']
    binary = (Path(runtime['directory']) / 'engine-dsv41' /
              f"{runtime['version']}-linux-x86_64-cuda-cuda13/strata-dsv41").resolve()
    expected = runtime['binary_sha256']
    if not binary.is_file() or not os.access(binary, os.X_OK) or sha256(binary) != expected:
        raise RuntimeError('The official native executable differs from its pin or is not executable.')
    return {'borrow': None, 'cache_route': 'off', 'environment': {},
            'native_binary': str(binary), 'native_sha256': expected,
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


def check_markers(markers: dict[str, Any]) -> None:
    """Require official borrowing evidence and reject custom routing activation."""
    if not (markers['borrow_allocation'] or markers['borrow_fallback']):
        raise RuntimeError('The native log does not establish borrowed-room activation or its allocation fallback.')
    if markers['cache_route']:
        raise RuntimeError('The official engine reported custom cache-aware routing activation.')


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
    for path in loaded_libraries:
        loaded_libraries[path] = sha256(Path(path))
    if (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19] != ticks:
        raise RuntimeError('The native PID changed while its evidence was read.')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8899/props', timeout=10) as response:
        props = json.load(response)
    sampling = props['default_generation_settings']['params']
    if any(sampling.get(key) != value for key, value in variant['sampling'].items()):
        raise RuntimeError('The live API sampling defaults differ from the fixed comparison settings.')
    markers = log_evidence(engine_log, offset)
    check_markers(markers)
    return {'pid': int(proc.name), 'start_ticks': ticks, 'native_binary': str(expected_path),
            'native_sha256': actual_sha, 'environment': actual_env,
            'native_argv': argv, 'seccomp': security['Seccomp'].strip(),
            'loaded_libraries': loaded_libraries, 'reported_sampling': sampling, 'markers': markers}
