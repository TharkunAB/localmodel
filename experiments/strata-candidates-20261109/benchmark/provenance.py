"""Reject a different runtime or workload before a benchmark starts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from benchmark_paths import HERE, ROOT

def verify(info: dict[str, Any], source_tag: str, native_version: str) -> dict[str, Any]:
    """Check declared versions, source receipt, and the frozen workload inputs."""
    runtime = info['runtime']
    if info['source_tag'] != source_tag or runtime['version'] != native_version:
        raise RuntimeError(f"Expected {source_tag}/{native_version}; installed {info['source_tag']}/{runtime['version']}.")
    pins = json.loads((HERE / 'workload-pins.json').read_text())
    corpus={str(p.relative_to(ROOT)) for p in (ROOT/'state/sources').glob('prism-e3de715-*.cpp')}
    if corpus!={name for name in pins if name.startswith('state/sources/prism-e3de715-')}:
        raise RuntimeError('The frozen public source corpus file list changed.')
    for name, expected in pins.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Workload input changed: {name}.')
    harness = json.loads((HERE / 'harness-pins.json').read_text())
    for name, expected in harness.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Harness input changed: {name}.')
    frozen_runtime = json.loads((HERE / 'runtime-pins.json').read_text())
    for name, expected in frozen_runtime.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen runtime or launcher changed: {name}.')
    receipt = Path(info['source_directory']) / 'source-files.json'
    receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    if receipt_hash != info['source_receipt_sha256']:
        raise RuntimeError('Source receipt differs from its pin.')
    binary = Path(runtime['directory']) / 'engine-dsv41' / f"{native_version}-linux-x86_64-cuda-cuda13/strata-dsv41"
    stat = binary.stat()
    runtime_inputs={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('localmodel.py','deepseek_strata.py','experiments/deepseek-strata/runtime.py','experiments/deepseek-strata/manifest.json','experiments/deepseek-coder/manifest.json')}
    return {'runtime_launcher_sha256':runtime_inputs,'source_tag': source_tag, 'source_revision': info['source_revision'],
            'source_directory': info['source_directory'], 'source_receipt_sha256': receipt_hash,
            'native_version': native_version, 'native_binary': str(binary),
            'expected_native_sha256': runtime['binary_sha256'], 'native_size': stat.st_size,
            'native_inode': stat.st_ino, 'native_mtime_ns': stat.st_mtime_ns,
            'python': str(Path(runtime['venv']) / 'bin/python'), 'workload_sha256': pins, 'harness_sha256': harness,
            'note': 'Production check_ready verifies source contents, signed feed, native bytes and dependencies before launch.'}
