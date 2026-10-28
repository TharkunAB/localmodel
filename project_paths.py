"""Resolve recorded project paths without changing pinned manifest files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent
CAPPED = ROOT / "libexec/capped"
RECORDED_ROOT = "/home/tritlo/Code/local-models"


def _relocate(value: Any) -> Any:
    """Resolve project paths in JSON values and embedded command arguments."""
    if isinstance(value, str):
        if value == RECORDED_ROOT:
            return str(ROOT)
        return value.replace(RECORDED_ROOT + "/", str(ROOT) + "/")
    if isinstance(value, list):
        return [_relocate(item) for item in cast(list[Any], value)]
    if isinstance(value, dict):
        return {key: _relocate(item) for key, item in cast(dict[str, Any], value).items()}
    return value


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest and resolve its recorded paths for this checkout."""
    return cast(dict[str, Any], _relocate(json.loads(path.read_text())))
