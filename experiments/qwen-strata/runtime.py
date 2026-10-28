"""Fetch and prepare pinned Qwen files without loading a model."""

from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from project_paths import read_manifest  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "stock_strata_runtime", ROOT / "experiments/deepseek-strata/runtime.py"
)
assert SPEC and SPEC.loader
STOCK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STOCK)


def digest(path: Path) -> str:
    """Calculate the complete file digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path: Path) -> dict[str, int]:
    """Record the identity of a verified regular file."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Expected an ordinary file: {path}")
    st = path.stat()
    return dict(
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        ctime_ns=st.st_ctime_ns,
        inode=st.st_ino,
        device=st.st_dev,
    )


def download(item: dict[str, Any], path: Path) -> None:
    """Resume an HTTPS download and verify its pinned size and digest."""
    expected = str(item["sha256"])
    if path.exists():
        if identity(path)["size"] != item["size"] or digest(path) != expected:
            raise RuntimeError(f"Existing file differs from its pin: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.is_symlink():
        raise RuntimeError(f"Refusing a symlink: {partial}")
    print(f"Downloading {path.name} ({item['size']} bytes)", flush=True)
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "8",
            "--retry-delay",
            "3",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            str(item["url"]),
        ],
        check=True,
    )
    if partial.stat().st_size != item["size"] or digest(partial) != expected:
        raise RuntimeError(f"Download differs from its pin: {partial}")
    partial.rename(path)


def engine(info: dict[str, Any], install: bool) -> None:
    """Check the signed engine package without executing the engine."""
    s = info["runtime"]
    rt = Path(s["directory"])
    source = Path(info["source_directory"])
    for name, key in [
        ("strata-feed.json", "feed_sha256"),
        ("strata-feed.json.sig", "feed_signature_sha256"),
    ]:
        path = rt / name
        if install:
            STOCK.checked_download(
                s["feed_url"] + (".sig" if name.endswith(".sig") else ""), path, s[key]
            )
        if not path.is_file() or digest(path) != s[key]:
            raise RuntimeError("Pinned signed feed is missing or changed")
    sys.path.insert(0, str(source))
    signing = importlib.import_module("strata_control.signing")
    feed = importlib.import_module("strata_control.feed")
    component = importlib.import_module("strata_control.engine_component")
    keys = signing.load_trusted_keys({})
    key = signing.verify_bytes(
        keys,
        (rt / "strata-feed.json").read_bytes(),
        json.loads((rt / "strata-feed.json.sig").read_text()),
    )
    index = feed.parse(
        (rt / "strata-feed.json").read_bytes(), source=s["feed_url"], key_id=key
    )
    package = next(
        (
            x
            for x in index.engine
            if x.version == s["version"]
            and x.platform == "linux-x86_64"
            and x.variant == "cuda13"
            and x.backend == "cuda"
        ),
        None,
    )
    if (
        package is None
        or package.sha256 != s["archive_sha256"]
        or package.manifest_sha256 != s["manifest_sha256"]
    ):
        raise RuntimeError("Pinned engine does not match the signed feed")
    target = Path(s["binary"]).parent
    if install and not target.exists():
        archive = rt / Path(s["archive_url"]).name
        download(
            dict(
                url=s["archive_url"], size=s["archive_size"], sha256=s["archive_sha256"]
            ),
            archive,
        )
        with tempfile.TemporaryDirectory(prefix="engine-", dir=rt) as stage:
            stage_path = Path(stage)
            total = 0
            with tarfile.open(archive) as tf:
                for member in tf:
                    if (
                        member.name.startswith("/")
                        or ".." in Path(member.name).parts
                        or not (member.isfile() or member.isdir())
                    ):
                        raise RuntimeError("Unsafe engine archive member")
                    total += member.size
                    if total > 2 * 1024**3:
                        raise RuntimeError("Engine archive exceeds extraction budget")
                    tf.extract(member, stage_path, filter="data")
            candidates = list(stage_path.rglob(component.MANIFEST))
            if len(candidates) != 1:
                raise RuntimeError("Ambiguous engine manifest")
            candidate = candidates[0].parent
            if digest(candidates[0]) != s["manifest_sha256"]:
                raise RuntimeError("Engine manifest digest mismatch")
            c = component.read_manifest(candidate)
            signing.install_component_verifier(keys)
            component.verify_signature(c, {component.ENV_REQUIRE_SIGNED: "1"})
            component.verify(c, full=True)
            candidate.rename(target)
    if digest(target / component.MANIFEST) != s["manifest_sha256"]:
        raise RuntimeError("Engine manifest changed")
    c = component.read_manifest(target)
    signing.install_component_verifier(keys)
    component.verify_signature(c, {component.ENV_REQUIRE_SIGNED: "1"})
    component.verify(c, full=True)
    if not Path(s["binary"]).is_file():
        raise RuntimeError("Engine executable missing")


def prepare(info: dict[str, Any]) -> None:
    """Prepare the native weight index and the pinned MTP tensors."""
    model = Path(info["model_directory"])
    source = Path(info["source_directory"])
    rt = Path(info["runtime"]["directory"])
    venv = rt / "prepare-venv"
    python = venv / "bin/python"
    if not python.exists():
        subprocess.run(["uv", "venv", str(venv), "--python", "/usr/bin/python3"], check=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), *info["prepare_packages"]],
        check=True,
    )
    library = rt / "gguf-py"
    archive = rt / "llama-source.zip"
    download(info["gguf_source"], archive)
    library_files: set[str] = set()
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            parts = Path(entry.filename).parts
            if len(parts) > 2 and parts[1] == "gguf-py" and not entry.is_dir():
                if ".." in parts or entry.filename.startswith("/"):
                    raise RuntimeError("Unsafe library path")
                dest = rt / Path(*parts[1:])
                library_files.add(str(dest.relative_to(library)))
                data = z.read(entry)
                if dest.exists():
                    if dest.is_symlink() or dest.read_bytes() != data:
                        raise RuntimeError(
                            f"Pinned preparation library changed: {dest}"
                        )
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
    if any(
        str(p.relative_to(library)) not in library_files for p in library.rglob("*.py")
    ):
        raise RuntimeError("Preparation library contains unpinned Python files")
    env = dict(
        os.environ,
        STRATA_GGUF_PY=str(library),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_ENDPOINT="https://huggingface.co",
        STRATA_MTP_REVISION=info["mtp_revision"],
    )
    for key in ["STRATA_SOURCE", "STRATA_ALLOW_UNPINNED", "PYTHONPATH"]:
        env.pop(key, None)

    def run(tool: str, *args: str) -> None:
        """Run a verified upstream preparation tool."""
        subprocess.run(
            [str(python), str(source / "tools" / tool), *args], env=env, check=True
        )

    pack = Path(info["pack_directory"])
    mtp = Path(info["mtp_directory"]).parent
    run("iq_pack.py", "--gguf", info["first_shard"], "--out", str(pack))
    run("mtp_fetch.py", "fetch", "--out", str(mtp))
    run("mtp_fetch.py", "verify", "--out", str(mtp))
    run(
        "mtp_pack.py",
        "--src",
        str(mtp),
        "--experts",
        "q2_0",
        "--out",
        str(mtp / "mtp-q2_0.gguf"),
    )
    run(
        "mtp_rt.py",
        "--gguf",
        str(mtp / "mtp-q2_0.gguf"),
        "--out",
        info["mtp_directory"],
    )
    shutil.copyfile(
        source / "data/draft_vocab.bin", Path(info["mtp_directory"]) / "draft_vocab.bin"
    )
    files = {
        str(p.relative_to(model)): dict(sha256=digest(p), **identity(p))
        for p in sorted(model.rglob("*"))
        if p.is_file() and p.name != "artifacts.json"
    }
    receipt = dict(
        schema=1, manifest_sha256=digest(HERE / "manifest.json"), files=files
    )
    (model / "artifacts.json").write_text(json.dumps(receipt, indent=2) + "\n")


def check(info: dict[str, Any]) -> None:
    """Check recorded identities without rereading the large weight files."""
    model = Path(info["model_directory"])
    if model.is_symlink() or any(p.is_symlink() for p in model.rglob("*")):
        raise RuntimeError("Model files and directories must not be symlinks")
    receipt = json.loads((model / "artifacts.json").read_text())
    if receipt["manifest_sha256"] != digest(HERE / "manifest.json"):
        raise RuntimeError("Artifact receipt uses another manifest")
    actual = {str(p.relative_to(model)) for p in model.rglob("*") if p.is_file()}
    if actual != set(receipt["files"]) | {"artifacts.json"}:
        raise RuntimeError("Model directory has missing or unrecorded files")
    for name, item in receipt["files"].items():
        path = model / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("Unsafe receipt path")
        if identity(path) != {k: v for k, v in item.items() if k != "sha256"}:
            raise RuntimeError(f"Artifact changed since verification: {path}")
    for f in info["files"]:
        if receipt["files"][f["name"]]["sha256"] != f["sha256"]:
            raise RuntimeError("Model digest differs from its pin")
    for path in [
        Path(info["pack_directory"]) / "native_experts.txt",
        Path(info["pack_directory"]) / "index.txt",
        Path(info["mtp_directory"]) / "experts.bin",
    ]:
        if not path.is_file():
            raise RuntimeError(f"Missing prepared artifact: {path}")
    sys.path.insert(0, str(Path(info["source_directory"]) / "tools"))
    reader = importlib.import_module("gguf_reader")
    ple = reader.GGUFFile(Path(info["ple_shard"]))
    if not any(t.name == "per_layer_token_embd.weight" for t in ple.tensors):
        raise RuntimeError("The configured PLE shard has no embedding table")


def main() -> None:
    """Preview, prepare, or verify the pinned artifacts."""
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--inventory", action="store_true")
    a = ap.parse_args()
    info = read_manifest(HERE / "manifest.json")
    model = Path(info["model_directory"])
    if a.inventory:
        receipt = model / "artifacts.json"
        files: dict[str, Any] = (
            json.loads(receipt.read_text())["files"] if receipt.exists() else {}
        )
        print(
            json.dumps(
                sorted(
                    set(files)
                    | {f["name"] for f in info["files"]}
                    | {f["name"] + ".part" for f in info["files"]}
                    | {"artifacts.json"}
                )
            )
        )
        return
    if not a.run and not a.check:
        print(json.dumps(info, indent=2))
        return
    STOCK.verify_source(Path(info["source_directory"]), info)
    if a.run and not (model / "artifacts.json").exists():
        required = 20 * 1024**3
        for f in info["files"]:
            final = model / f["name"]
            partial = final.with_name(final.name + ".part")
            have = (
                final.stat().st_size
                if final.is_file()
                else partial.stat().st_size
                if partial.is_file()
                else 0
            )
            required += max(0, f["size"] - have)
        if shutil.disk_usage(ROOT).free < required:
            raise RuntimeError(
                f"Preparation requires {required / 1024**3:.1f} GiB free, including 10 GiB headroom"
            )
    engine(info, a.run)
    if a.run:
        if (model / "artifacts.json").exists():
            check(info)
        else:
            for f in info["files"]:
                download(f, model / f["name"])
            prepare(info)
    check(info)
    print(
        json.dumps(
            dict(
                verified=True,
                loads_model=False,
                binary=info["runtime"]["binary"],
                model_directory=str(model),
            )
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as e:
        raise SystemExit(str(e)) from e
