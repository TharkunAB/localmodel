"""Freeze this prepared harness and the current runtime before dry runs or inference."""
import hashlib
import argparse
import json
from pathlib import Path
from benchmark_paths import CODE, HERE, ROOT
import sys

sys.path.insert(0, str(ROOT))
import deepseek_strata
from variants import add_arguments, read_variant


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
add_arguments(parser)
args = parser.parse_args()
for name in ('runtime-pins.json', 'harness-pins.json'):
    if (HERE / name).exists():
        raise RuntimeError(f'{name} already exists; do not overwrite frozen evidence')
info = deepseek_strata.manifest()
assert info['source_tag'] == 'v0.2.12' and info['runtime']['version'] == '0.1.58'
variant = read_variant(args, info)
if args.borrow is not None or args.cache_route != 'off':
    parser.error('Freeze the shared engine inputs, without selecting a performance arm.')
runtime = [ROOT / name for name in ('localmodel.py', 'deepseek_strata.py', 'qwen_strata.py', 'config/models.json',
    'experiments/deepseek-strata/runtime.py', 'experiments/deepseek-strata/manifest.json',
    'experiments/deepseek-coder/manifest.json', 'experiments/deepseek-strata/network_guard.py',
    'libexec/capped', 'libexec/capped-record', '../heavy-management/bin/capped')]
runtime += [Path(info['source_directory']) / 'source-files.json',
    Path(info['runtime']['directory']) / 'engine-dsv41/0.1.58-linux-x86_64-cuda-cuda13/strata-dsv41']
runtime_hashes = {str(path.resolve()): sha(path) for path in runtime}
assert runtime_hashes[str(runtime[-1].resolve())] == info['runtime']['binary_sha256']
if variant['custom'] is not None:
    for path in [Path(variant['native_binary']), Path(variant['custom']['receipt_path']),
                 *[Path(item['path']) for item in variant['custom']['receipt']['libraries']]]:
        runtime_hashes[str(path.resolve())] = sha(path)
(HERE / 'runtime-pins.json').write_text(json.dumps(runtime_hashes, indent=2) + '\n')
own = [*CODE.glob('*.py'), *CODE.glob('*.ts'), CODE / 'run-case.sh', HERE / 'workload-pins.json',
    HERE / 'extended-corpus.json', HERE / 'runtime-pins.json', CODE / 'README.md', HERE / 'source-ancestry.json', *HERE.glob('routing-baseline/*'),
    *HERE.glob('extended-corpus/*')]
(HERE / 'harness-pins.json').write_text(json.dumps({str(path.resolve()): sha(path)
    for path in own if path.is_file()}, indent=2) + '\n')
print(json.dumps({'frozen_runtime_files': len(runtime_hashes), 'source_tag': info['source_tag'],
    'native_version': info['runtime']['version'], 'inference_started': False}))
