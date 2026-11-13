#!/usr/bin/env bash
# Fetch the current remote launcher from TharkunAB/localmodel main.
# Requires Bash, Python 3.10+, and /dev/fd. Forward all arguments to the launcher.
set -Eeuo pipefail
command -v python3 >/dev/null 2>&1 || { echo 'ERROR: Python 3.10+ is required.' >&2; exit 1; }
# Keep standard input available to the fetched launcher.
exec python3 /dev/fd/3 "$@" 3<<'PYTHON_START_REMOTE'
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import sys
import tempfile
import urllib.error
import urllib.request

REPO = 'TharkunAB/localmodel'
API = f'https://api.github.com/repos/{REPO}'
MAX_BYTES = 16 * 1024 * 1024


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


def main() -> None:
    """Verify the selected Git blob, then replace this process with Bash."""
    os.close(3)
    ref = json.loads(fetch(f'{API}/git/ref/heads/main'))
    revision = ref['object']['sha']
    if ref['object']['type'] != 'commit' or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('GitHub returned an invalid main revision.')
    tree = json.loads(fetch(f'{API}/git/trees/{revision}?recursive=1'))
    if tree.get('truncated') is not False or not isinstance(tree.get('tree'), list):
        raise ValueError('GitHub did not return a complete commit tree.')
    entries = {entry['path']: entry for entry in tree['tree']}
    path = 'remote/start-remote-model.sh'
    if path not in entries:
        path = 'bin/start-remote-model.sh'
    entry = entries.get(path)
    if (not entry or entry.get('type') != 'blob' or entry.get('mode') not in ('100644', '100755')
            or not re.fullmatch(r'[0-9a-f]{40}', entry.get('sha', ''))):
        raise ValueError('The selected commit has no valid remote launcher.')
    payload = fetch(f'https://raw.githubusercontent.com/{REPO}/{revision}/{path}')
    blob = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
    if blob != entry['sha']:
        raise ValueError('The remote launcher does not match its Git blob hash.')
    if payload.split(b'\n', 1)[0] not in (b'#!/usr/bin/env bash', b'#!/bin/bash') or b'\0' in payload:
        raise ValueError('The remote launcher is not a Bash script.')
    # TemporaryFile is private and unlinked. The inherited descriptor survives exec.
    with tempfile.TemporaryFile() as script:
        script.write(payload)
        script.seek(0)
        fd = script.fileno()
        os.set_inheritable(fd, True)
        print(f'Launching {REPO}@{revision}:{path}', file=sys.stderr, flush=True)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
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
