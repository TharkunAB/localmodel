"""Start isolated OpenCode with the current local inference server."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, cast

from localmodel import current, request


ROOT = Path(__file__).resolve().parent


def process_memory_limit(pid: int) -> int | None:
    """Read the RAM limit of a process in a harness memory scope."""
    try:
        groups = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        group = next(line[3:] for line in groups if line.startswith("0::"))
        if not Path(group).name.startswith(("opencode-local-", "opencode-remote-", "omp-local-", "omp-remote-")):
            return None
        directory = Path("/sys/fs/cgroup") / group.lstrip("/")
        if (directory / "memory.swap.max").read_text().strip() != "0":
            return None
        return int((directory / "memory.max").read_text().strip())
    except (OSError, ValueError, StopIteration):
        return None


def memory_limit(
    args: list[str], *, launcher: str = "opencode-local", state: Path = ROOT / "state/opencode",
) -> tuple[list[str], int | None]:
    """Apply a shared RAM cap before the harness can start child processes."""
    if args and args[0] == "--mem":
        if len(args) < 2:
            sys.exit("--mem requires a positive size, such as 32 or 512M.")
        size, remaining = args[1], args[2:]
    elif args and args[0].startswith("--mem="):
        size, remaining = args[0].split("=", 1)[1], args[1:]
    else:
        return args, None
    if not re.fullmatch(r"[1-9][0-9]*[KMGT]?", size, re.IGNORECASE):
        sys.exit("--mem requires a positive size, such as 32 or 512M. Bare numbers use GiB.")
    size = size.upper()
    if size[-1].isdigit():
        size += "G"
    limit = int(size[:-1]) * 1024 ** ("KMGT".index(size[-1]) + 1)
    options = remaining[:remaining.index("--")] if "--" in remaining else remaining
    if any(value == "--server" or value.startswith("--server=") for value in options):
        sys.exit("--mem cannot limit an external --server. Start the local service with --mem first.")
    if process_memory_limit(os.getpid()) == limit:
        return remaining, limit
    recorder = ROOT / "libexec/capped-record"
    if not recorder.is_file():
        sys.exit(f"The memory limit requires {recorder}.")
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={os.environ['XDG_RUNTIME_DIR']}/bus")
    os.environ.setdefault("HEAVY_PEAKS", str(state / "memory-peaks"))
    os.environ["CAPPED_UNIT"] = f"{launcher}-{os.getpid()}"
    os.environ["CAPPED_LABEL"] = launcher
    os.environ["CAPPED_MEM"] = size
    os.execvp("systemd-run", [
        "systemd-run", "--user", "--scope", "--quiet", f"--unit={os.environ['CAPPED_UNIT']}",
        "-p", f"MemoryMax={size}", "-p", f"MemoryHigh={size}",
        "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue",
        str(recorder), sys.executable, str(Path(sys.argv[0]).resolve()), *args,
    ])


def plugins(state: Path = ROOT / "state/opencode") -> list[str]:
    """Register the reviewed goal entry and return configured local plugins."""
    options: object = json.loads((ROOT / "config/opencode-goal.json").read_text())
    if not isinstance(options, dict):
        raise ValueError("config/opencode-goal.json must contain an object.")
    goal_state = state / "goals"
    if goal_state.is_symlink():
        raise ValueError("The private goal directory must not be a symbolic link.")
    goal_state.mkdir(parents=True, exist_ok=True, mode=0o700)
    goal_state.chmod(0o700)
    plugin_dir = state / "config/opencode/plugins"
    if plugin_dir.is_symlink():
        raise ValueError("The private plugin directory must not be a symbolic link.")
    plugin_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    entry = plugin_dir / "goal.ts"
    if entry.is_symlink():
        raise ValueError("The goal entry must not be a symbolic link.")
    source = ROOT / "state/plugins/opencode-goal-plugin-0.11.1/src/goal-plugin.js"
    if not source.is_file():
        raise RuntimeError(
            f"The reviewed OpenCode goal plugin is missing. Run {ROOT / 'bin/setup-harness'} "
            "--run --with-opencode-goal first."
        )
    settings = {**cast(dict[str, Any], options), "stateFilePath": str(goal_state / "state.json")}
    content = (
        f"import plugin from {json.dumps(source.as_uri())};\n"
        "export default { ...plugin, setup(ctx: { options: Record<string, unknown> }) {\n"
        f"  ctx.options = {json.dumps(settings)};\n"
        "  return plugin.setup(ctx);\n"
        "} };\n"
    )
    if not entry.exists() or entry.read_text() != content:
        entry.write_text(content)
        entry.chmod(0o600)
    return [
        str(ROOT / "config/searxng-plugin"),
    ]


def web_settings() -> dict[str, Any]:
    """Read persistent web search selection and tool permissions."""
    value: object = json.loads((ROOT / "config/opencode-web.json").read_text())
    if not isinstance(value, dict):
        raise ValueError("config/opencode-web.json must contain an object.")
    settings = cast(dict[str, Any], value)
    if settings.keys() - {"websearch", "permissions"}:
        raise ValueError("config/opencode-web.json supports only websearch and permissions.")
    search: dict[str, Any] | bool | None = settings.get("websearch")
    if search is not False and not (
        isinstance(search, dict) and isinstance(search.get("provider"), str) and search["provider"]
    ):
        raise ValueError("Set websearch to false or an object with a provider in config/opencode-web.json.")
    return settings


def bonsai_status(state: Path) -> dict[str, Any] | None:
    """Discover the Bonsai subagent model or reuse its saved metadata."""
    cache = state / "bonsai-subagent.json"
    saved: dict[str, Any] = {}
    for path in (cache, ROOT / "state/active.json"):
        try:
            value: object = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(value, dict):
            info = cast(dict[str, Any], value)
            if info.get("port") == 8899 and isinstance(info.get("model"), str) and info["model"].strip():
                saved = info
                break
    try:
        inventory = request(8899, "/v1/models")
        entries: object = inventory["data"]
        if not isinstance(entries, list):
            raise ValueError("Expected a model list at port 8899.")
        models = cast(list[object], entries)
        if len(models) != 1 or not isinstance(models[0], dict):
            raise ValueError("Expected one Bonsai subagent model at port 8899.")
        info = cast(dict[str, Any], models[0])
        model = info.get("id")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("The Bonsai subagent server did not report a model ID.")
        capacity = info.get("max_model_len", info.get("max_context_length", info.get("context_length")))
        if capacity is None:
            try:
                capacity = request(8899, "/props")["default_generation_settings"]["n_ctx"]
            except (OSError, ValueError, KeyError, TypeError):
                capacity = saved.get("context", 262144)
        if type(capacity) is not int or capacity < 2:
            raise ValueError("The Bonsai subagent server reported an invalid context capacity.")
        saved = {
            "model": model, "context": capacity, "port": 8899,
            "reasoning_effort": saved.get("reasoning_effort") or "xhigh",
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(saved, indent=2) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not saved:
            print(f"Bonsai subagents unavailable: {error}. Start the model on port 8899.", file=sys.stderr)
            return None
    context = saved.get("context", 262144)
    if type(context) is not int or context < 2:
        context = 262144
    return {
        "model": saved["model"], "context": context, "output": min(16000, context - 1),
        "base_url": "http://127.0.0.1:8899/v1", "port": 8899,
        "temperature": 1.0, "top_p": 0.95, "reasoning_field": "reasoning_content",
        "reasoning_effort": saved.get("reasoning_effort") or "xhigh",
    }


def provider(status: dict[str, Any], provider_id: str, name: str) -> dict[str, Any]:
    """Build a provider from one served model and its context capacity."""
    model = str(status["model"])
    context = int(status["context"])
    if context < 2:
        raise ValueError("The model must allow at least two context tokens.")
    body = {"temperature": status["temperature"], "top_p": status["top_p"]}
    if status.get("reasoning_effort") and status["reasoning_effort"] != "default":
        body["reasoning_effort"] = status["reasoning_effort"]
    return {
        "name": name,
        "package": "@opencode/ai/providers/openai-compatible",
        "settings": {
            "baseURL": status["base_url"], "apiKey": "local", "provider": provider_id,
            "timeout": 3600000, "headerTimeout": 3600000, "chunkTimeout": 3600000,
        },
        "body": body,
        "models": {
            model: {
                "name": f"{model} ({status['preset']})" if status.get("preset") else model,
                "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
                "compatibility": {"reasoningField": status["reasoning_field"]},
                "limit": {"context": context, "output": min(int(status["output"]), context - 1)},
            },
        },
    }


def profile(
    status: dict[str, Any], *, state: Path = ROOT / "state/opencode", provider_id: str = "local",
) -> dict[str, Any]:
    """Build an OpenCode profile with a separate Bonsai subagent route."""
    result: dict[str, Any] = {
        "model": f"{provider_id}/{status['model']}",
        "update": "disable",
        "share": "disabled",
        "warming": False,
        **web_settings(),
        "agents": {},
        "plugins": plugins(state),
        "providers": {provider_id: provider(status, provider_id, f"Current {provider_id} model")},
        "experimental": {
            "policies": [
                {"action": "provider.use", "resource": "*", "effect": "deny"},
                {"action": "provider.use", "resource": provider_id, "effect": "allow"},
            ],
        },
    }
    bonsai = bonsai_status(state)
    if bonsai is not None:
        result["providers"]["bonsai"] = provider(bonsai, "bonsai", "Bonsai 2 27B subagents")
        result["experimental"]["policies"].append(
            {"action": "provider.use", "resource": "bonsai", "effect": "allow"},
        )
        result["agents"] = {
            name: {"model": {"providerID": "bonsai", "model": bonsai["model"]}}
            for name in ("general", "explore")
        }
    return result


def main(
    *, get_status: Callable[[], dict[str, Any]] = current, launcher: str = "opencode-local",
) -> None:
    """Read the loaded model and start a separate OpenCode process."""
    state = ROOT / "state" / ("opencode" if launcher == "opencode-local" else launcher)
    args, limit = memory_limit(sys.argv[1:], launcher=launcher, state=state)
    if args[:1] in (["--help"], ["-h"]):
        print(
            f"{launcher} --mem SIZE [OpenCode arguments]\n"
            "  Cap OpenCode and its child processes together. Bare sizes use GiB.\n"
            f"  Example: {launcher} --mem 32 service start\n"
            + (
                "  Discover one model at localhost:8898/v1/models (IPv4 or IPv6).\n"
                "  Context: 1048576 tokens, reduced to a smaller reported server limit.\n"
                "  Output: up to 262144 tokens within that context budget.\n"
                "  Service commands also work without a model.\n"
                if launcher == "opencode-remote" else ""
            ),
            flush=True,
        )
    executable = Path.home() / ".opencode/bin/opencode"
    if not executable.is_file():
        sys.exit(f"OpenCode is not installed at {executable}.")

    for variable, name in (
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_STATE_HOME", "state"),
        ("TMPDIR", "tmp"),
    ):
        directory = state / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    config_dir = state / "config/opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENCODE_CONFIG_DIR"] = str(config_dir)
    os.environ["OPENCODE_CONFIG_PROJECT_DISABLE"] = "1"
    os.environ["OPENCODE_DISABLE_MODELS_FETCH"] = "1"
    for variable in ("OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT", "OPENCODE_CLI_CONFIG_CONTENT"):
        os.environ.pop(variable, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    options = args[:args.index("--")] if "--" in args else args
    command_args = options
    while command_args and command_args[0].startswith("-"):
        takes_value = command_args[0] in ("--log-level", "--server", "--session", "-s", "--prompt", "--model", "-m")
        command_args = command_args[2 if takes_value else 1:]
    administrative = bool(command_args and command_args[0] in ("service", "serve", "reload")) or any(
        value in ("--help", "-h", "--version", "-v", "--completions")
        or value.startswith("--completions=") for value in options
    )
    config_path = config_dir / "opencode.json"
    if administrative:
        if (
            command_args[:2] in (["service", "start"], ["service", "restart"])
            or command_args[:1] in (["serve"], ["reload"])
        ) and not any(value in ("--help", "-h") for value in options):
            try:
                config = profile(
                    get_status(), state=state,
                    provider_id="remote" if launcher == "opencode-remote" else "local",
                )
            except (OSError, RuntimeError, ValueError) as error:
                print(f"Model unavailable; keeping the saved configuration: {error}", file=sys.stderr)
            else:
                config_path.write_text(json.dumps(config, indent=2) + "\n")
        if not config_path.exists():
            config_path.write_text(json.dumps({
                "update": "disable", "share": "disabled", "warming": False,
                **web_settings(),
                "plugins": plugins(state),
                "providers": {},
                "experimental": {"policies": [
                    {"action": "provider.use", "resource": "*", "effect": "deny"},
                ]},
            }, indent=2) + "\n")
        if (
            limit is not None
            and command_args[:2] in (["service", "start"], ["service", "restart"])
            and not any(value in ("--help", "-h", "--version", "-v") for value in options)
        ):
            result = subprocess.run([str(executable), *args], check=False)
            if result.returncode:
                sys.exit(result.returncode)
            try:
                registration = json.loads((state / "state/opencode/service.json").read_text())
                actual = process_memory_limit(int(registration["pid"]))
            except (OSError, ValueError, KeyError, TypeError):
                actual = None
            if actual is None or actual > limit:
                sys.exit(
                    "The existing service is outside the requested memory limit. "
                    f"Use {launcher} --mem SIZE service restart to restart it with the cap. "
                    "Restart interrupts its running jobs."
                )
            sys.exit(0)
        os.execv(str(executable), [str(executable), *args])

    try:
        config = profile(
            get_status(), state=state, provider_id="remote" if launcher == "opencode-remote" else "local",
        )
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"Cannot configure OpenCode: {error}")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    cli_path = config_dir / "cli.json"
    cli: dict[str, Any] = json.loads(cli_path.read_text()) if cli_path.exists() else {}
    cli.setdefault("session", {})["tps"] = True
    cli_path.write_text(json.dumps(cli, indent=2) + "\n")

    if args and args[0] in ("run", "mini"):
        options = args[1:args.index("--")] if "--" in args else args[1:]
        if not any(value == "--model" or value.startswith("--model=") or value.startswith("-m") for value in options):
            args = [args[0], "--model", str(config["model"]), *args[1:]]
    if not args or args[0].startswith("-"):
        args = ["--standalone", *args]
    elif args[0] in ("run", "mini", "models", "api", "acp"):
        args = [args[0], "--standalone", *args[1:]]
    os.execv(str(executable), [str(executable), *args])


if __name__ == "__main__":
    main()
