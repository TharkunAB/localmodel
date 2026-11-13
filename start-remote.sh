#!/usr/bin/env bash
# Fetch the current remote CLI or legacy launcher from TharkunAB/localmodel main.
# Requires Bash, Python 3.10+, and /dev/fd. Forward all arguments to the launcher.
set -Eeuo pipefail
command -v python3 >/dev/null 2>&1 || { echo 'ERROR: Python 3.10+ is required.' >&2; exit 1; }
# Keep standard input available to the fetched launcher.
exec python3 /dev/fd/3 "$@" 3<<'PYTHON_START_REMOTE'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys
import tempfile
import urllib.error
import urllib.request
from typing import Any, cast

REPO = 'TharkunAB/localmodel'
API = f'https://api.github.com/repos/{REPO}'
MAX_BYTES = 16 * 1024 * 1024
ENTRYPOINT = 'remote/bin/remotemodel'
MANIFEST = 'remote/package.json'


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    """Keep requests on the fixed GitHub HTTPS URLs."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        raise ValueError('GitHub redirected a request; refusing to continue.')


def fetch(url: str) -> bytes:
    """Read a bounded response without redirects or cached main resolution."""
    request = urllib.request.Request(url, headers={
        'User-Agent': 'localmodel-remote-bootstrap',
        'Accept': 'application/vnd.github+json' if url.startswith(API + '/') else 'text/plain',
        'Cache-Control': 'no-cache',
    })
    with urllib.request.build_opener(RejectRedirect()).open(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError('GitHub returned an unexpected response status.')
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('GitHub response exceeds the bootstrap size limit.')
    return data


def entry_for(entries: dict[str, Any], path: str) -> dict[str, Any]:
    """Accept regular files from the selected commit only."""
    entry = entries.get(path)
    if (not entry or entry.get('type') != 'blob' or entry.get('mode') not in ('100644', '100755')
            or not re.fullmatch(r'[0-9a-f]{40}', entry.get('sha', ''))):
        raise ValueError(f'The selected main commit has no valid {path}. Publish the remote package first.')
    return entry


def verify(payload: bytes, entry: dict[str, Any]) -> None:
    """Verify source bytes against the immutable Git tree."""
    blob = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
    if len(payload) > MAX_BYTES or blob != entry['sha'] or b'\0' in payload:
        raise ValueError('Remote source does not match its Git blob hash or text format.')


def check_package(directory: Path, entries: dict[str, Any]) -> None:
    """Reject altered cached files and symbolic links before execution."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('The cached remote package must be a regular directory.')
    directories = {str(parent) for name in entries for parent in Path(name).parents if str(parent) != '.'}
    for path in directory.rglob('*'):
        name = str(path.relative_to(directory))
        if path.is_symlink() or (name not in directories if path.is_dir() else name not in entries):
            raise ValueError(f'The cached remote package contains an unexpected entry: {path}')
    for name in entries:
        path = directory / name
        if any(parent.is_symlink() for parent in path.parents if parent != directory.parent):
            raise ValueError('The cached remote package contains a directory link.')
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError(f'The cached remote package is incomplete or linked: {path}')
        verify(path.read_bytes(), entries[name])
    if (directory / ENTRYPOINT).read_bytes().split(b'\n', 1)[0] not in (b'#!/usr/bin/env bash', b'#!/bin/bash'):
        raise ValueError('The remote package entry point is not a Bash script.')


def package(revision: str, entries: dict[str, Any]) -> Path:
    """Cache a complete verified package; keep weights and builds separate."""
    manifest_entry = entry_for(entries, MANIFEST)
    manifest = fetch(f'https://raw.githubusercontent.com/{REPO}/{revision}/{MANIFEST}')
    verify(manifest, manifest_entry)
    names_value: Any = json.loads(manifest)['files']
    if not isinstance(names_value, list):
        raise ValueError('The remote package manifest has an invalid file list.')
    items = cast(list[object], names_value)
    if not 1 <= len(items) <= 128 or not all(isinstance(name, str) for name in items):
        raise ValueError('The remote package manifest has an invalid file list.')
    names = cast(list[str], items)
    if ENTRYPOINT not in names or MANIFEST in names or len(names) != len(set(names)):
        raise ValueError('The remote package manifest has a missing entry point or duplicate files.')
    if any(not name.startswith(('remote/', 'local/')) or '..' in PurePosixPath(name).parts
           or str(PurePosixPath(name)) != name or '\\' in name for name in names):
        raise ValueError('The remote package manifest contains an unsafe path.')
    selected = {MANIFEST: manifest_entry, **{name: entry_for(entries, name) for name in names}}
    cache = Path(os.environ.get('REMOTEMODEL_CODE_DIR', '/workspace/remote-models/launcher')).expanduser().absolute()
    if cache.is_symlink():
        raise ValueError('The remote code cache must not be a symbolic link.')
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = cache / revision
    if not destination.exists() and not destination.is_symlink():
        with tempfile.TemporaryDirectory(prefix='.fetch-', dir=cache) as temporary:
            stage = Path(temporary) / 'package'
            for name, entry in selected.items():
                payload = manifest if name == MANIFEST else fetch(f'https://raw.githubusercontent.com/{REPO}/{revision}/{name}')
                verify(payload, entry)
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.write_bytes(payload)
                target.chmod(0o700 if entry['mode'] == '100755' else 0o600)
            try:
                stage.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
                # Another bootstrap may have published the same complete commit.
    check_package(destination, selected)
    return destination / ENTRYPOINT


def main() -> None:
    """Verify the selected source, then replace this process with Bash."""
    os.close(3)
    ref = json.loads(fetch(f'{API}/git/ref/heads/main'))
    revision = ref['object']['sha']
    if ref['object']['type'] != 'commit' or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('GitHub returned an invalid main revision.')
    tree = json.loads(fetch(f'{API}/git/trees/{revision}?recursive=1'))
    if tree.get('truncated') is not False or not isinstance(tree.get('tree'), list):
        raise ValueError('GitHub did not return a complete commit tree.')
    entries = {entry['path']: entry for entry in tree['tree']}
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    args = sys.argv[1:]
    if not args or args[0] in ('start', 'fetch', 'list', 'status', 'stop', 'help', '-h', '--help', '--backend-help', '--version'):
        launcher = package(revision, entries)
        print(f'Launching {REPO}@{revision}:{ENTRYPOINT}', file=sys.stderr, flush=True)
        os.execvp('bash', ['bash', str(launcher), *args])
    path = 'remote/start-remote-model.sh'
    if path not in entries:
        path = 'bin/start-remote-model.sh'
    entry = entry_for(entries, path)
    payload = fetch(f'https://raw.githubusercontent.com/{REPO}/{revision}/{path}')
    verify(payload, entry)
    if payload.split(b'\n', 1)[0] not in (b'#!/usr/bin/env bash', b'#!/bin/bash') or b'\0' in payload:
        raise ValueError('The remote launcher is not a Bash script.')
    # TemporaryFile is private and unlinked. The inherited descriptor survives exec.
    with tempfile.TemporaryFile() as script:
        script.write(payload)
        script.seek(0)
        fd = script.fileno()
        os.set_inheritable(fd, True)
        print(f'Launching {REPO}@{revision}:{path}', file=sys.stderr, flush=True)
        os.execvp('bash', ['bash', f'/dev/fd/{fd}', *sys.argv[1:]])


try:
    main()
except urllib.error.HTTPError as error:
    sys.exit(f'ERROR: GitHub HTTP {error.code}; no launcher was started. Check access and API limits.')
except (urllib.error.URLError, OSError):
    sys.exit('ERROR: Could not fetch or start the remote launcher. No cached launcher was used.')
except (KeyError, TypeError, AttributeError):
    sys.exit('ERROR: GitHub returned invalid launcher metadata.')
except ValueError as error:
    sys.exit(f'ERROR: {error}')
except KeyboardInterrupt:
    sys.exit(130)
PYTHON_START_REMOTE
