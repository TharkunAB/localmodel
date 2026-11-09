"""Isolate benchmark slots and optional resident expert copies for one launch."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from benchmark_paths import ROOT
sys.path.insert(0, str(ROOT))
import localmodel
from project_paths import read_manifest
from provenance import verify
from variants import add_arguments, read_variant


def configure(manifest_path: Path | None, model: str, slot_directory: Path, variant: dict[str, Any]) -> str:
    """Use the installed profile and change only experimental launch settings."""
    strata = localmodel.deepseek_strata
    if model != 'deepseek-full':
        raise RuntimeError('This comparison uses the full DeepSeek target only.')
    alias = strata.NAME
    info = strata.manifest(alias)
    if manifest_path is not None:
        expected = read_manifest(manifest_path)
        def pins(value: dict[str, Any]) -> list[tuple[str, int, str]]:
            return [(item['name'], item['size'], item['sha256']) for item in value['files']]
        if expected['model_revision'] != info['model_revision'] or pins(expected) != pins(info) or expected['dspark'] != info['dspark']:
            raise RuntimeError('Experiment weight pins differ from the installed profile.')
    original_settings = strata.settings
    def settings(args: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
        """Use isolated slots with one of the bounded cache budgets."""
        result = original_settings(args, item)
        if result['env']['STRATA_DS4_SLOT_KEEP'] == '1':
            result['env']['STRATA_DS4_SLOT_DIR'] = str(slot_directory.resolve())
        if args.expert_memory != 'cache' or result['expert_memory'] != 'cache' or args.expert_ram_gb != 48:
            raise RuntimeError('Use explicit cache mode and a 48 GiB expert cache.')
        native = result['args']
        if float(native[native.index('--ssd-cache')+1]) != args.expert_ram_gb * 2**30 / 10**9 or '--mmap-experts' in native:
            raise RuntimeError('The native cache budget or file-backed expert path differs from the plan.')
        for key in ('DS4_COPY_GB', 'DS4_STORE_GB', 'STRATA_DS4_RAM_TIER'):
            if key in result['env']:
                raise RuntimeError(f'Unexpected expert-memory override: {key}.')
        result['env'].update(variant['environment'])
        result['sampling'] = dict(variant['sampling'])
        result['args'].append('--verbose')
        result['exe'] = variant['native_binary']
        if variant['custom'] is not None:
            result['lib_dirs'] = variant['custom']['receipt']['lib_dirs']
        return result
    strata.settings = settings
    return alias


def main() -> int:
    """Run only the installed start command under the experimental settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-marker',type=Path,required=True)
    parser.add_argument('--expected-source-tag', required=True)
    parser.add_argument('--expected-native-version', required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--model', choices=('deepseek-full',), required=True)
    parser.add_argument('--slot-directory', type=Path, required=True)
    add_arguments(parser)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.controller_marker!=ROOT/'localmodel.py':parser.error('Use the production localmodel path as the controller marker.')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or command[0] != 'start':
        parser.error('This experiment wrapper supports only start, including --dry-run.')
    variant = read_variant(args, localmodel.deepseek_strata.manifest())
    alias = configure(args.manifest.resolve() if args.manifest else None, args.model, args.slot_directory, variant)
    verify(localmodel.deepseek_strata.manifest(alias), args.expected_source_tag, args.expected_native_version)
    if len(command) < 2 or localmodel.ALIASES.get(command[1], command[1]) != alias:
        parser.error('The start command must select the same explicit profile as --model.')
    sys.argv = [str(ROOT / 'localmodel.py'), *command]
    return localmodel.main()


if __name__ == '__main__':
    raise SystemExit(main())
