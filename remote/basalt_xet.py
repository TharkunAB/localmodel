"""Download a pinned Basalt pack and verify its bytes in one pass."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
import os
from pathlib import Path
import sys
import time


def write_verified(path: Path, chunks: Iterable[bytes], size: int, sha: str) -> None:
    """Leave a sequential prefix on interruption. Reject incomplete or incorrect streams."""
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600), "wb") as dest:
        if os.fstat(dest.fileno()).st_size:
            raise ValueError(f"Refusing to overwrite a partial transfer at {path}")
        digest = hashlib.sha256()
        received = 0
        started = time.monotonic()
        next_update = started + 10
        for chunk in chunks:
            if received + len(chunk) > size:
                raise ValueError("The Xet stream exceeds the pinned size.")
            dest.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            now = time.monotonic()
            if received and now >= next_update:
                rate = received / (now - started)
                print(f"Download + SHA-256: {received/size:.1%}, {rate/2**20:.0f} MiB/s, "
                      f"about {(size-received)/rate/60:.1f} min left", flush=True)
                next_update = now + 10
        if received != size:
            raise ValueError(f"Xet delivered {received} of {size} bytes; retained the partial download.")
        if digest.hexdigest() != sha:
            path.unlink()
            raise ValueError("The Xet stream differs from the pinned SHA-256; discarded the download.")
        dest.flush()
        os.fsync(dest.fileno())


def main() -> None:
    from hf_xet import XetFileInfo, XetSession
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    repo, rev, name, partial, expected_size, sha = sys.argv[1:]
    size = int(expected_size)
    metadata = get_hf_file_metadata(hf_hub_url(repo, name, revision=rev), token=False)
    if (metadata.commit_hash, metadata.size, metadata.etag) != (rev, size, sha):
        raise ValueError("Hub file metadata differs from the pinned commit, size or SHA-256.")
    xet = metadata.xet_file_data
    if xet is None:
        raise ValueError("The Hub did not advertise Xet for this file. Use --download-method curl to retry.")
    session = XetSession()
    group = session.new_download_stream_group(token_refresh_url=xet.refresh_route)
    chunks = group.download_stream(XetFileInfo(xet.file_hash, size))
    write_verified(Path(partial), chunks, size, sha)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        sys.exit(f"Basalt Xet download: {exc}")
