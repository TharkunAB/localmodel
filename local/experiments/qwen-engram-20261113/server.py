"""Add a repeatable sampling seed to the unchanged Qwen lifecycle."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Any
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import localmodel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-marker', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.controller_marker != ROOT / 'localmodel.py' or command[:2] != ['start', 'qwen3.8']:
        parser.error('Use the real localmodel path and start qwen3.8.')
    original = localmodel.qwen_strata.settings
    def settings(options: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
        value = original(options, item)
        value['sampling']['seed'] = args.seed
        return value
    localmodel.qwen_strata.settings = settings
    sys.argv = [str(ROOT / 'localmodel.py'), *command]
    return localmodel.main()


if __name__ == '__main__':
    raise SystemExit(main())
