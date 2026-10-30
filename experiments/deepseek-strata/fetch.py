"""Download the pinned UD-IQ1_S files. Verify every file before use."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, cast
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from project_paths import read_manifest  # noqa: E402


def main() -> int:
    """Fetch only the selected backbone and shared Engram file."""
    info = read_manifest(Path(__file__).with_name("manifest.json"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(info["model_directory"]))
    parser.add_argument("--run", action="store_true", help="Download files. Omit to print the plan.")
    parser.add_argument("--jobs", type=int, default=3, choices=range(1, 5), help="Parallel files (default: 3).")
    args = parser.parse_args()
    files = cast(list[dict[str, Any]], info["files"])
    print(json.dumps({"directory": str(args.directory), "bytes": sum(int(f["size"]) for f in files),
                      "revision": info["model_revision"], "files": [f["name"] for f in files]}, indent=2), flush=True)
    if not args.run:
        return 0
    args.directory.mkdir(parents=True, exist_ok=True)
    needed = sum(max(0, int(f["size"]) - (args.directory / f["name"]).stat().st_size)
                 if (args.directory / f["name"]).exists() else
                 max(0, int(f["size"]) - (args.directory / (f["name"] + ".part")).stat().st_size)
                 if (args.directory / (f["name"] + ".part")).exists() else int(f["size"]) for f in files)
    if shutil.disk_usage(args.directory).free < needed + 10 * 1024**3:
        raise RuntimeError("The selected SSD needs the remaining download size plus 10 GiB free.")
    def download(item: dict[str, Any]) -> None:
        """Resume one file and verify its pinned checksum."""
        name = str(item["name"])
        if Path(name).name != name:
            raise ValueError("Use file names without directories.")
        target = args.directory / name
        partial = args.directory / (name + ".part")
        if target.is_symlink() or partial.is_symlink():
            raise ValueError(f"Refusing a symbolic link: {name}.")
        started = time.time()
        if not target.exists():
            for attempt in range(4):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == item["size"]:
                    break
                if offset > item["size"]:
                    raise RuntimeError(f"Oversized partial file: {partial}.")
                try:
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    request = urllib.request.Request(str(item["url"]), headers=headers)
                    with urllib.request.urlopen(request, timeout=90) as response:
                        if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                            raise RuntimeError("The server did not honor the resume range.")
                        with partial.open("ab", buffering=8 * 1024**2) as stream:
                            while chunk := response.read(min(8 * 1024**2, int(item["size"]) - offset + 1)):
                                offset += len(chunk)
                                if offset > item["size"]:
                                    raise RuntimeError("The download exceeds its pinned size.")
                                stream.write(chunk)
                    if offset != item["size"]:
                        raise OSError("The transfer ended early.")
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            candidate = partial
        else:
            candidate = target
        if candidate.stat().st_size != item["size"]:
            raise RuntimeError(f"Wrong file size: {candidate}.")
        print(f"Verifying {name}", flush=True)
        with candidate.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"Wrong SHA256: {candidate}.")
        if candidate == partial:
            partial.rename(target)
        print(json.dumps({"verified": name, "sha256": digest, "seconds": time.time() - started}), flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as workers:
        for _ in workers.map(download, sorted(files, key=lambda f: int(f["size"]))):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
