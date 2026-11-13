"""Freeze this prepared harness and the current runtime before dry runs or inference."""
import hashlib
import argparse
import json
from pathlib import Path
from benchmark_paths import CODE, HERE, RELEASE, ROOT, VERSIONS
import sys

sys.path.insert(0, str(ROOT))
import deepseek_strata
from variants import read_variant


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
args = parser.parse_args()
for name in ('runtime-pins.json', 'harness-pins.json'):
    if (HERE / name).exists():
        raise RuntimeError(f'{name} already exists; do not overwrite frozen evidence')
info = deepseek_strata.manifest()
assert info['source_tag'] == RELEASE and info['runtime']['version'] == VERSIONS[RELEASE]
variant = read_variant(info)
runtime = [ROOT / name for name in ('localmodel.py', 'deepseek_strata.py', 'qwen_strata.py', 'config/models.json',
    'experiments/deepseek-strata/runtime.py', 'experiments/deepseek-strata/manifest.json',
    'experiments/deepseek-coder/manifest.json', 'experiments/deepseek-strata/network_guard.py',
    'libexec/capped', 'libexec/capped-record', '../heavy-management/bin/capped')]
runtime += [Path(info['source_directory']) / 'source-files.json',
    Path(variant['native_binary'])]
runtime_hashes = {str(path.resolve()): sha(path) for path in runtime}
assert runtime_hashes[str(runtime[-1].resolve())] == info['runtime']['binary_sha256']
(HERE / 'runtime-pins.json').write_text(json.dumps(runtime_hashes, indent=2) + '\n')
own = [*CODE.glob('*.py'), *CODE.glob('*.ts'), CODE / 'run-case.sh', HERE / 'workload-pins.json',
    HERE / 'extended-corpus.json', HERE / 'runtime-pins.json', CODE / 'README.md', HERE / 'source-ancestry.json', *HERE.glob('routing-baseline/*'),
    *HERE.glob('extended-corpus/*')]
(HERE / 'harness-pins.json').write_text(json.dumps({str(path.resolve()): sha(path)
    for path in own if path.is_file()}, indent=2) + '\n')
print(json.dumps({'frozen_runtime_files': len(runtime_hashes), 'source_tag': info['source_tag'],
    'native_version': info['runtime']['version'], 'inference_started': False}))
