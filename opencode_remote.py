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
    models = cast(list[object], entries)
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
    if capacity is None:
        try:
            props = request(base_url, "/props")
            capacity = props["default_generation_settings"]["n_ctx"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if capacity is not None:
        if type(capacity) is not int or capacity < 2:
            raise ValueError("The remote server reported an invalid context capacity.")
        context = min(context, capacity)
    if context < CONTEXT:
        print(
            f"Remote server reports {context} context tokens; using that limit instead of {CONTEXT}.",
            file=sys.stderr,
        )
    return {
        "model": model,
        "context": context,
        "output": min(OUTPUT, max(1, context // 4)),
        "base_url": base_url + "/v1",
        "temperature": 1.0,
        "top_p": 1.0,
        "reasoning_field": "reasoning_content",
    }


if __name__ == "__main__":
    main(get_status=current, launcher="opencode-remote")
