"""Start isolated OpenCode with the model served on port 8898."""

from __future__ import annotations

import json
import sys
from typing import Any, cast
import urllib.request

from localmodel import NoMetadataRedirect
from opencode_local import main


PORT = 8898
CONTEXT = 1048576
OUTPUT = 262144


def request(base_url: str, route: str) -> Any:
    """Read loopback metadata without a proxy or redirect."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoMetadataRedirect())
    with opener.open(base_url + route, timeout=2) as response:
        return json.load(response)


def magnitude_model(base_url: str, models: list[object]) -> list[object]:
    """Get the active Magnitude model and its allocated context."""
    entries = [cast(dict[str, Any], row) for row in models if isinstance(row, dict)]
    if not any(row.get("owned_by") == "magnitude" for row in entries):
        return models
    offered = {row["id"]: row for row in entries}
    state = request(base_url, "/api/v1/instances")
    if not isinstance(state, dict) or not isinstance(cast(dict[str, Any], state).get("instances"), list):
        raise ValueError("Magnitude did not report its model instances.")
    ready: list[object] = []
    for item in cast(list[dict[str, Any]], state["instances"]):
        lifecycle = item.get("lifecycle", {})
        if lifecycle.get("_tag") != "Ready" or item.get("modelId") not in offered:
            continue
        capacity = lifecycle.get("allocation", {}).get("contextWindowTokens")
        if type(capacity) is not int or capacity < 2:
            raise ValueError("Magnitude reported an invalid loaded context capacity.")
        ready.append({**offered[item["modelId"]], "context_length": capacity})
    # A single idle catalog model can load on its next request.
    return ready or models


def current() -> dict[str, Any]:
    """Read the remote model ID and its available context capacity."""
    failure: Exception | None = None
    data: object = None
    for base_url in (f"http://127.0.0.1:{PORT}", f"http://[::1]:{PORT}"):
        try:
            data = request(base_url, "/v1/models")
        except (OSError, ValueError) as error:
            failure = error
        else:
            break
    else:
        raise RuntimeError(
            f"No model is ready on localhost:{PORT}/v1/models (IPv4 or IPv6). "
            "Start the remote server or its SSH tunnel first."
        ) from failure
    if not isinstance(data, dict):
        raise ValueError("The remote /v1/models response must contain an object.")
    entries: object = cast(dict[str, Any], data).get("data")
    if not isinstance(entries, list):
        raise ValueError("The remote /v1/models response must contain a data list.")
    models = magnitude_model(base_url, cast(list[object], entries))
    if len(models) != 1:
        raise ValueError(f"Expected one served model at port {PORT}.")
    entry = models[0]
    if not isinstance(entry, dict):
        raise ValueError("The remote model entry must contain an object.")
    info = cast(dict[str, Any], entry)
    model = info.get("id")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("The remote server did not report a model ID.")

    context = CONTEXT
    capacity = info.get("max_model_len", info.get("max_context_length", info.get("context_length")))
    if capacity is None and info.get("owned_by") == "basalt" and isinstance(info.get("meta"), dict):
        capacity = info["meta"].get("n_ctx")
    if capacity is None:
        try:
            props = request(base_url, "/props")
            capacity = props["default_generation_settings"]["n_ctx"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if capacity is None and info.get("owned_by") == "basalt":
        raise ValueError("Basalt did not report the loaded context capacity.")
    if capacity is not None:
        if type(capacity) is not int or capacity < 2:
            raise ValueError("The remote server reported an invalid context capacity.")
        context = min(context, capacity)
    if context < CONTEXT:
        print(
            f"Remote server reports {context} context tokens; using that limit instead of {CONTEXT}.",
            file=sys.stderr,
        )
    name = model.lower().split("/")[-1]
    mistral = name == "mistral-large-4" or name.startswith("mistral-large-4-")
    return {
        "model": model,
        "context": context,
        "output": min(OUTPUT, max(1, context // 4)),
        "base_url": base_url + "/v1",
        "temperature": 1.0,
        "top_p": 0.95 if mistral else 1.0,
        "reasoning_field": "reasoning_content",
    }


if __name__ == "__main__":
    main(get_status=current, launcher="opencode-remote")
