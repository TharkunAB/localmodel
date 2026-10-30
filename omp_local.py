"""Start isolated OMP with the current inference server."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, cast
from urllib.parse import urlsplit
import urllib.request

from localmodel import NoMetadataRedirect, current
from opencode_local import memory_limit
from opencode_remote import current as remote_current


ROOT = Path(__file__).resolve().parent


def require_searxng(endpoint: str) -> None:
    """Require a successful local search health check before OMP starts."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoMetadataRedirect())
    try:
        with opener.open(endpoint.rstrip("/") + "/healthz", timeout=2) as response:
            ready = response.status == 200 and response.read(16).strip() == b"OK"
    except OSError:
        ready = False
    if not ready:
        raise RuntimeError(f"SearXNG is not ready at {endpoint}. Start searxng-local first, then retry.")


def runtime() -> Path:
    """Prepare the pinned OMP bundle with local network routing changes."""
    manifest = cast(dict[str, Any], json.loads((ROOT / "config/omp-runtime.json").read_text()))
    managed = ROOT / "state/omp-package" / manifest["version"]
    package = managed / "node_modules/@oh-my-pi/pi-coding-agent"
    if managed.exists():
        source = package / "dist/cli.js"
    else:
        executable = shutil.which("omp")
        if executable is None:
            raise RuntimeError(f"OMP is not installed. Run {ROOT / 'bin/setup-harness'} --run first.")
        source = Path(executable).resolve()
        package = source.parent.parent
    version = json.loads((package / "package.json").read_text())["version"]
    original = source.read_bytes()
    if version != manifest["version"] or hashlib.sha256(original).hexdigest() != manifest["input_sha256"]:
        raise RuntimeError("The installed OMP differs from the reviewed runtime. Review config/omp-runtime.json before upgrading.")
    directory = ROOT / "state/omp-runtime" / version
    if managed.exists():
        directory /= "managed"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["PI_PACKAGE_DIR"] = str(package)
    target = directory / "cli.js"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == manifest["output_sha256"]:
        return target
    content = original.decode()
    for patch in cast(list[dict[str, str]], manifest["patches"]):
        if content.count(patch["before"]) != 1:
            raise RuntimeError(f"The OMP patch {patch['id']} does not match exactly once.")
        content = content.replace(patch["before"], patch["after"])
    if hashlib.sha256(content.encode()).hexdigest() != manifest["output_sha256"]:
        raise RuntimeError("The patched OMP checksum does not match the reviewed runtime.")
    for sibling in source.parent.iterdir():
        destination = directory / sibling.name
        if sibling != source and not destination.exists():
            destination.symlink_to(sibling, target_is_directory=sibling.is_dir())
    modules = directory / "node_modules"
    if not modules.exists():
        modules.symlink_to(package.parents[1], target_is_directory=True)
    temporary = directory / "cli.js.tmp"
    temporary.write_text(content)
    temporary.chmod(0o755)
    temporary.replace(target)
    return target


def collaboration() -> dict[str, Any]:
    """Set the allowed collaboration origins from the launcher configuration."""
    os.environ["OMP_LOCAL_COLLAB_ORIGINS"] = "[]"
    os.environ.pop("OMP_LOCAL_COLLAB_CA", None)
    os.environ.pop("OMP_LOCAL_COLLAB_CA_FILE", None)
    source = ROOT / "config/omp-collab.json"
    if not source.exists():
        return {"relayUrl": "", "webUrl": "", "autoStart": "off"}
    config = cast(dict[str, Any], json.loads(source.read_text()))
    origins = cast(list[str], config["allowedOrigins"])
    allowed: list[str] = []
    for origin in origins:
        url = urlsplit(origin)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.path not in ("", "/") or url.query or url.fragment):
            raise RuntimeError("Collaboration allowedOrigins must contain HTTPS origins.")
        allowed.append(f"wss://{url.netloc.lower()}")
    for key, scheme in (("relayUrl", "wss"), ("webUrl", "https")):
        url = urlsplit(str(config[key]))
        if (url.scheme != scheme or f"wss://{url.netloc.lower()}" not in allowed
                or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment):
            raise RuntimeError(f"Collaboration {key} must use an allowed {scheme.upper()} origin.")
    certificate = ROOT / str(config["caFile"])
    if not certificate.is_file():
        raise RuntimeError(f"The collaboration CA file is missing: {certificate}")
    os.environ["OMP_LOCAL_COLLAB_CA_FILE"] = str(certificate)
    os.environ["OMP_LOCAL_COLLAB_ORIGINS"] = json.dumps(allowed)
    return {"relayUrl": config["relayUrl"], "webUrl": config["webUrl"], "autoStart": "off"}


def model_provider(status: dict[str, Any]) -> dict[str, Any]:
    """Describe one live OpenAI-compatible model to OMP."""
    context = int(status["context"])
    output = min(int(status["output"]), context - 1)
    name = str(status["model"]).lower().split("/")[-1]
    deepseek = any(name == family or name.startswith(family + "-") for family in ("deepseek-v4.1", "deepseek-gsq"))
    mistral = name == "mistral-large-4" or name.startswith("mistral-large-4-")
    return {
        "baseUrl": status["base_url"],
        "api": "openai-completions",
        "auth": "none",
        "models": [{
            "id": status["model"],
            "name": status["model"],
            "reasoning": True,
            "thinking": {
                "mode": "effort", "efforts": ["minimal", "low", "medium", "high", "xhigh", "max"],
                "defaultLevel": "xhigh",
            },
            "input": ["text"], "supportsTools": True,
            "contextWindow": context, "maxTokens": output,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "compat": {
                "maxTokensField": "max_tokens", "supportsDeveloperRole": False,
                "supportsStore": False, "supportsReasoningEffort": True,
                **({"reasoningEffortMap": {"minimal": "low", "xhigh": "max"}} if deepseek else {}),
                **({"reasoningEffortMap": {"minimal": "low", "xhigh": "high", "max": "high"}} if mistral else {}),
                "reasoningContentField": status["reasoning_field"],
                "streamFirstEventTimeoutMs": 3600000, "streamIdleTimeoutMs": 3600000,
                "extraBody": {"temperature": status["temperature"], "top_p": status["top_p"]},
            },
        }],
    }


def profile(status: dict[str, Any], bonsai: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a settings overlay with live model roles and context budgets."""
    settings: dict[str, Any] = {}
    providers = {"model": model_provider(status)}
    primary = f"model/{status['model']}"
    task = primary
    if bonsai is not None:
        providers["bonsai"] = model_provider(bonsai)
        task = f"bonsai/{bonsai['model']}"
    settings["modelRoles"] = {name: primary for name in ("default", "smol", "slow", "plan", "title")}
    settings["modelRoles"].update({"task": task, "web": "web/searxng"})
    settings["task"] = {
        "agentModelOverrides": {name: "@task" for name in ("task", "scout", "reviewer", "security-reviewer", "sonic")},
        "completionProbe": False,
    }
    thresholds = {
        f"{provider}/{model['id']}": max(
            1, min(int(model["contextWindow"] * 0.8),
                   model["contextWindow"] - model["maxTokens"] - min(16384, model["contextWindow"] // 8)),
        )
        for provider, info in providers.items() for model in info["models"]
    }
    settings["compaction"] = {"modelThresholds": thresholds}
    return settings, {"providers": providers}


def main(*, remote: bool = False) -> None:
    """Discover the model, isolate OMP state, and start the requested command."""
    launcher = "omp-remote" if remote else "omp-local"
    state = ROOT / "state" / launcher
    args, _limit = memory_limit(sys.argv[1:], launcher=launcher, state=state)
    if args[:1] in (["--help"], ["-h"]):
        print(
            f"{launcher} [--mem SIZE] [OMP arguments]\n"
            "  Bare memory sizes use GiB. The cap includes child processes.\n"
            + ("  Discover the main model on localhost:8898; context defaults to 1M.\n"
               "  Discover the task model on localhost:8899, including Qwen3.8-Flash-Next. No cached fallback.\n"
               "  Resume accepts different deepseek-v4.1 profile suffixes.\n" if remote
               else "  Discover the current localmodel endpoint and its context.\n")
            + "  Start searxng-local first. OMP sessions require its health check on port 8888.\n"
            "  URL reads use direct fetching.\n"
            "  Collaboration uses the configured LAN service. Session uploads are disabled.\n"
            "  Settings changes persist separately for this profile. Model routing stays managed.\n"
            "  Goals: /goal OBJECTIVE, /goal pause, /goal resume. Skills: /skill:NAME.\n",
            flush=True,
        )
    agent = state / "agent"
    agent.mkdir(parents=True, exist_ok=True, mode=0o700)
    instructions = agent / "AGENTS.md"
    if not instructions.exists() and (Path.home() / "AGENTS.md").is_file():
        instructions.symlink_to(Path.home() / "AGENTS.md")
    for variable, name in (("XDG_DATA_HOME", "data"), ("XDG_STATE_HOME", "state"), ("XDG_CACHE_HOME", "cache")):
        directory = state / name / "omp"
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory.parent)
    os.environ["PI_CODING_AGENT_DIR"] = str(agent)
    os.environ["PI_CONFIG_DIR"] = os.path.relpath(state, Path.home())
    os.environ["OMP_PROFILE"] = "default"
    os.environ.pop("PI_PROFILE", None)
    for variable in ("PI_SMOL_MODEL", "PI_SLOW_MODEL", "PI_PLAN_MODEL"):
        os.environ.pop(variable, None)
    os.environ["OMP_SKIP_SETUP"] = "1"
    if remote:
        os.environ["OMP_LOCAL_REMOTE_MODEL_FAMILY"] = "deepseek-v4.1"
    else:
        os.environ.pop("OMP_LOCAL_REMOTE_MODEL_FAMILY", None)
    os.environ["PI_AUTH_NO_BORROW"] = "1"
    os.environ["PI_NO_TITLE"] = "1"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    options = args[:args.index("--")] if "--" in args else args
    if any(value == "--profile" or value.startswith("--profile=") for value in options):
        sys.exit(f"{launcher} uses its own isolated profile; omit --profile.")
    administrative = any(value in ("--help", "-h", "--version", "-v") for value in options) or args[:1] in (
        ["config"], ["agents"], ["completions"],
    )
    try:
        defaults = json.loads((ROOT / "config/omp.json").read_text())
        if not administrative:
            require_searxng(str(defaults["searxng"]["endpoint"]))
        cli = runtime()
        collab = collaboration()
        overlay = state / "launcher.yml"
        base: dict[str, Any] = json.loads(overlay.read_text()) if administrative and overlay.exists() else {}
        if not administrative:
            status = remote_current() if remote else current()
            task_model = current(8899) if remote else None
            base, models = profile(status, task_model)
            (agent / "models.yml").write_text(json.dumps(models, indent=2) + "\n")
        base["collab"] = collab
        for key in ("providers", "searxng", "retry", "disabledProviders", "mcp", "lsp", "launch",
                    "telemetry", "dev", "startup", "marketplace"):
            base[key] = defaults[key]
        overlay.write_text(json.dumps(base, indent=2) + "\n")
        os.environ["PI_CONFIG_FILES"] = str(overlay)
        if not (agent / "config.yaml").exists():
            try:
                with (agent / "config.yml").open("x") as stream:
                    stream.write(json.dumps(defaults, indent=2) + "\n")
            except FileExistsError:
                pass
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"Cannot configure OMP: {error}")
    bun = shutil.which("bun")
    if bun is None:
        sys.exit("Bun is required to run the installed OMP package.")
    os.execv(bun, [bun, str(cli), *args])


if __name__ == "__main__":
    main()
