"""Run the reviewed Edge0 API under the normal model ownership and memory limits."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'state/sources/edge0-da9cc526/edge0-strata'
REVISION = 'da9cc526747759227714ddde0728763d95306c04'
BUILD = ROOT / 'state/builds/edge0-da9cc526'
BINARY = BUILD / 'native/strata'
MODEL = ROOT / 'models/edge0-145b-q3k'
NAME = 'edge0-145b'
MODES = {
    'plain': {'draft_tokens': 0, 'vram_reserve_mib': 800, 'kv_vram_cells': 32768},
    'mtp': {'draft_tokens': 3, 'vram_reserve_mib': 300, 'kv_vram_cells': 32768},
    'mtp-small': {'draft_tokens': 3, 'vram_reserve_mib': 550, 'kv_vram_cells': 20480},
}
PYTHON = ROOT / 'state/venvs/strata-official-0.2.8/bin/python'
sys.path.insert(0, str(ROOT))
import localmodel


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load('edge_guard', ROOT / 'experiments/qwen-engram-20261113/run.py')


def protect_parent(expected: int) -> None:
    """Stop this controller if its benchmark parent disappears."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot establish the benchmark parent-death signal.')
    if os.getppid() != expected:
        raise RuntimeError('The benchmark parent changed before Edge0 startup.')


def configuration(context: int, output: Path, seed: int, mode: str = 'plain') -> dict[str, Any]:
    """Use explicit reviewed flags; do not execute downloaded model configuration."""
    selected = MODES[mode]
    drafts = selected['draft_tokens']
    args = ['--pack', str(MODEL), '--native', str(MODEL / 'model-skeleton.gguf'),
            '--native-dense-gguf', str(MODEL / 'model-skeleton.gguf'),
            '--ple-gguf', str(MODEL / 'ple-fp8.gguf'), '--ple-io', 'direct',
            '--expert-profile', str(output / 'profile.bin'), '--expert-cache', 'auto',
            '--prefill', 'auto', '--spec', str(max(2, drafts + 1)),
            '--max-context', str(context), '--kv', 'int8', '--kv-resident', str(selected['kv_vram_cells']),
            '--resident-budget-gib', '40', '--vram-reserve-mib', str(selected['vram_reserve_mib']),
            '--suffix-draft', '0', '--lookup-chain', '0']
    if drafts:
        args += ['--spec-min-p', '0.5', '--mtp', str(MODEL / 'mtp')]
    if mode == 'mtp-small':
        args += ['--mtp-draft-vocab', str(SOURCE / 'data/draft_vocab_en.bin')]
    env = {'STRATA_LATENT_A': str(MODEL / 'experts.bin'), 'STRATA_LATENT_LABELS': str(MODEL / 'labels.bin'),
           'STRATA_LATENT_DENSE': str(MODEL / 'latent_dense_q4.bin'), 'STRATA_LATENT_DENSE_FMT': '4',
           'STRATA_EMBED_VRAM': '0', 'STRATA_LATENT_FUSED': '1', 'STRATA_LATENT_FUSED_DU': '1',
           'STRATA_LATENT_U_GPU': '1', 'STRATA_LAT_FUSED_Q3K': '1', 'STRATA_ROUTE_TAIL_SKIP': '0'}
    return {'exe': str(BINARY), 'args': args, 'cwd': str(SOURCE), 'tokenizer': str(MODEL / 'tokenizer'),
            'model_name': NAME, 'arch': 'qwen4exp', 'gpu': [0], 'drafting': 'mtp' if drafts else 'off',
            'mtp': bool(drafts), 'experiment_mode': mode,
            'log': str(output / 'engine.log'), 'env': env, 'sampling': {'temperature': 1, 'top_p': 0.95, 'seed': seed},
            'fit_max_tokens': True, 'api_monitor': False, 'open_browser': False}


def verify() -> dict[str, Any]:
    """Require the reviewed source, build receipt and all artifact receipts."""
    receipt = BUILD / 'build-receipt.json'
    if not receipt.is_file() or receipt.is_symlink() or not BINARY.is_file() or BINARY.is_symlink():
        raise RuntimeError('Build Edge0 and write its reviewed build-receipt.json before running this case.')
    proof: dict[str, Any] = json.loads(receipt.read_text())
    if (proof.get('source_revision') != REVISION or proof.get('binary') != str(BINARY)
            or proof.get('binary_sha256') != shared.digest(BINARY)):
        raise RuntimeError('The Edge0 binary differs from its build receipt.')
    library = BINARY.parent / 'libstrata_kq.so'
    libraries = proof.get('shared_libraries', [])
    if (len(libraries) != 1 or libraries[0].get('path') != str(library)
            or not library.is_file() or library.is_symlink()
            or libraries[0].get('size') != library.stat().st_size
            or libraries[0].get('sha256') != shared.digest(library)):
        raise RuntimeError('The Edge0 kernel library differs from its build receipt.')
    head = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if head != REVISION or dirty:
        raise RuntimeError('The Edge0 tracked source differs from the reviewed revision.')
    fetch = load('edge_artifacts', HERE / 'fetch.py')
    fetch.execute(fetch.pins(), check=True)
    return proof


def run(context: int, output: Path, seed: int, mode: str = 'plain') -> int:
    """Own one API process and its scope until cleanup finishes."""
    controller_scope = shared.scope_for(os.getpid())
    maximum = (controller_scope / 'memory.max').read_text().strip()
    if maximum == 'max' or int(maximum) > 4 * 2**30 or (controller_scope / 'memory.swap.max').read_text().strip() != '0':
        raise RuntimeError('Start Edge0 through run.py so its controller has a 4 GiB no-swap limit.')
    setattr(shared, 'WORK', output)
    selected = MODES[mode]
    active = ROOT / 'state/active.json'
    with (ROOT / 'state/server.lock').open('a') as lock, (ROOT / 'state/models-edge0.lock').open('a') as model_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(model_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if active.exists():
            raise RuntimeError('A managed model record already exists; it was not changed.')
        localmodel.check_free(8899)
        proof = verify()
        output.mkdir(parents=True, exist_ok=True)
        shared.save(output / 'build-receipt.json', proof)
        shutil.copyfile(MODEL / 'verification.json', output / 'artifact-receipt.json')
        shutil.copyfile(HERE / 'manifest.json', output / 'model-manifest.json')
        shutil.copyfile(MODEL / 'profile.bin', output / 'profile.bin')
        shared.save(output / 'provenance.json', {'source_revision': REVISION,
            'files': {str(path.relative_to(ROOT)): shared.digest(path) for path in
                      (HERE / 'run.py', HERE / 'server.py', shared.FIXTURE,
                       ROOT / 'experiments/qwen-engram-20261113/run.py', ROOT / 'config/omp-runtime.json')},
            'expert_profile_sha256': shared.digest(output / 'profile.bin')})
        shared.save(output / 'launch-config.json', configuration(context, output, seed, mode))
        env = {key: value for key, value in os.environ.items() if not key.startswith(('STRATA_', 'MAYA_', 'DS4_'))}
        env.pop('CAPPED_UNIT', None)
        env.pop('LD_PRELOAD', None)
        env.pop('LD_AUDIT', None)
        env.update(XDG_STATE_HOME=str(output / 'api-state'), HEAVY_PEAKS=str(output / 'peaks'),
                   LD_LIBRARY_PATH=f'{BINARY.parent}:{ROOT / "state/runtimes/strata-qwen-0.1.59/engine"}')
        command = [str(ROOT / 'libexec/capped'), '--mem', '64G', str(PYTHON), '-I', '-B',
                   str(ROOT / 'experiments/deepseek-strata/network_guard.py'), str(PYTHON), '-I', '-B',
                   str(SOURCE / 'serve/server.py'), '--engine', 'strata', '--config', str(output / 'launch-config.json'),
                   '--host', '127.0.0.1', '--port', '8899', '--offline']
        process: subprocess.Popen[str] | None = None
        scope: Path | None = None
        try:
            process = subprocess.Popen(command, cwd=SOURCE, env=env, text=True, start_new_session=True)
            deadline = time.monotonic() + 20
            while process.poll() is None:
                candidate = shared.scope_for(process.pid)
                if (candidate / 'memory.max').read_text().strip() == str(64 * 2**30):
                    scope = candidate
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError('The Edge0 API did not enter its 64 GiB scope.')
                time.sleep(0.1)
            if scope is None or (scope / 'memory.swap.max').read_text().strip() != '0':
                raise RuntimeError('The Edge0 API did not establish its memory and no-swap limits.')
            shared.save(active, {'pid': process.pid, 'controller_pid': os.getpid(), 'command': command,
                'model': NAME, 'profile': NAME, 'runtime': 'strata-edge0', 'port': 8899, 'context': context,
                'settings_profile': mode, 'memory_limit': '64G',
                'speculative': 'mtp' if selected['draft_tokens'] else 'none', 'kv_cache': 'int8',
                'expert_ram_gb': 40, 'expert_tail_skip': 0, **selected,
                'reasoning_effort': 'xhigh', 'started': time.time()})
            return process.wait()
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if scope is None and process is not None and process.poll() is None:
                try:
                    candidate = shared.scope_for(process.pid)
                    if (candidate / 'memory.max').read_text().strip() == str(64 * 2**30):
                        scope = candidate
                except FileNotFoundError:
                    pass
            try:
                shared.stop_model(process, scope)
            except ProcessLookupError:
                shared.stop_model(None, scope)
            if active.exists() and json.loads(active.read_text()).get('controller_pid') == os.getpid():
                active.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=localmodel.tokens, default=32768)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--mode', choices=MODES, default='plain')
    parser.add_argument('--controller-marker', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.controller_marker != ROOT / 'localmodel.py' or args.context not in (32768, 196608, 262144):
        parser.error('Use the real localmodel controller marker and context 32K, 192K or 256K.')
    if not args.run:
        print(json.dumps(configuration(args.context, args.output.absolute(), args.seed, args.mode), indent=2))
        return 0
    protect_parent(args.parent_pid)
    return run(args.context, args.output.absolute(), args.seed, args.mode)


if __name__ == '__main__':
    def interrupt(_number: int, _frame: Any) -> None:
        raise RuntimeError('Edge0 controller interrupted; cleaning up its API scope.')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    raise SystemExit(main())
