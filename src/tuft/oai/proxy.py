"""HTTP proxy for forwarding OpenAI-compatible requests to vLLM backends."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse


logger = logging.getLogger(__name__)


async def proxy_request(
    client: httpx.AsyncClient,
    backend_url: str,
    path: str,
    body: dict[str, Any],
    stream: bool,
    *,
    user_model_name: str,
    response_id_prefix: str,
) -> JSONResponse | StreamingResponse:
    """Proxy an OpenAI API request to a vLLM backend.

    Args:
        client: The shared httpx.AsyncClient instance.
        backend_url: Base URL of the vLLM OpenAI server (e.g. "http://10.0.0.1:8000").
        path: API path to append (e.g. "/v1/completions").
        body: The request body dict (already modified with backend model name).
        stream: Whether to use streaming mode.
        user_model_name: Original model name from user request, used for rewriting.
        response_id_prefix: Prefix for the response ID
            (e.g. ``"{session}:sample"``).  The final ID becomes
            ``"{prefix}:{vllm_original_id}"``.

    Returns:
        A JSONResponse (non-streaming) or StreamingResponse (streaming).
    """
    url = f"{backend_url}{path}"
    headers = {"Authorization": "Bearer EMPTY"}

    if stream:
        req = client.build_request("POST", url, json=body, headers=headers, timeout=300.0)
        resp = await client.send(req, stream=True)
        if resp.status_code != 200:
            error_body = await resp.aread()
            await resp.aclose()
            try:
                error_json = json.loads(error_body)
            except json.JSONDecodeError:
                error_json = {"error": {"message": error_body.decode()}}
            return JSONResponse(content=error_json, status_code=resp.status_code)
        return StreamingResponse(
            _stream_proxy(resp, user_model_name, response_id_prefix),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream_proxy(
            client, url, headers, body, user_model_name, response_id_prefix
        )


async def _non_stream_proxy(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    user_model_name: str,
    response_id_prefix: str,
) -> JSONResponse:
    """Forward a non-streaming request and rewrite the response."""
    resp = await client.post(url, json=body, headers=headers, timeout=300.0)
    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    data = resp.json()
    _rewrite_response(data, user_model_name, response_id_prefix)
    return JSONResponse(content=data)


async def _stream_proxy(
    resp: httpx.Response,
    user_model_name: str,
    response_id_prefix: str,
) -> AsyncIterator[str]:
    """Forward a streaming response, rewriting each SSE chunk."""
    try:
        async for line in resp.aiter_lines():
            if not line.strip():
                yield "\n"
                continue
            if line.startswith("data: "):
                payload = line[6:].strip()
                if payload == "[DONE]":
                    yield "data: [DONE]\n\n"
                    continue
                try:
                    chunk = json.loads(payload)
                    _rewrite_response(chunk, user_model_name, response_id_prefix)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                except json.JSONDecodeError:
                    yield f"{line}\n"
            else:
                yield f"{line}\n"
    finally:
        await resp.aclose()


def _rewrite_response(
    data: dict[str, Any],
    user_model_name: str,
    response_id_prefix: str,
) -> None:
    """Rewrite ``model`` and ``id`` fields in-place.

    - ``model`` → the original model name the user sent
    - ``id`` → ``{response_id_prefix}:{vllm_original_id}``
    """
    if "model" in data:
        data["model"] = user_model_name
    if "id" in data:
        vllm_id = data["id"]
        data["id"] = f"{response_id_prefix}:{vllm_id}"
