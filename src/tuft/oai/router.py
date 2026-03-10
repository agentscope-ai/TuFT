"""OpenAI-compatible API router for TuFT.

Endpoints:
- POST /oai/api/v1/completions
- POST /oai/api/v1/chat/completions
- GET  /oai/api/v1/models
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import User
from ..exceptions import (
    InvalidRequestException,
    ServerException,
    ServiceUnavailableException,
    TuFTException,
    UnknownModelException,
)
from ..state import ServerState
from ..telemetry.tracing import get_tracer
from .model_resolver import resolve_model
from .proxy import proxy_request


logger = logging.getLogger(__name__)
_get_tracer = lambda: get_tracer("tuft.oai")  # noqa: E731


def _get_state(request: Request) -> ServerState:
    state = getattr(request.app.state, "server_state", None)
    if state is None:
        raise RuntimeError("Server state has not been initialized")
    return state


def _get_httpx_client(request: Request) -> httpx.AsyncClient:
    client = getattr(request.app.state, "httpx_client", None)
    if client is None:
        raise RuntimeError("httpx client has not been initialized")
    return client


async def _get_user_oai(request: Request) -> User:
    """Authenticate via ``Authorization: Bearer`` or ``X-API-Key``.

    Supports both header styles for OpenAI SDK compatibility.
    """
    state = _get_state(request)

    # Try Authorization: Bearer first (OpenAI SDK standard)
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        api_key = auth_header[7:].strip()
    else:
        # Fall back to X-API-Key header
        api_key = request.headers.get("x-api-key", "")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide Authorization: Bearer <key> or X-API-Key header.",
        )

    user = state.get_user(api_key)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return user


_user_dep = Depends(_get_user_oai)


def create_oai_router() -> APIRouter:
    """Create the OpenAI-compatible API router."""
    router = APIRouter(prefix="/oai/api/v1", tags=["OpenAI Compatible API"])

    @router.post("/completions", response_model=None)
    async def completions(
        request: Request,
        user: User = _user_dep,
    ):
        return await _proxy_inference(request, user, "/v1/completions")

    @router.post("/chat/completions", response_model=None)
    async def chat_completions(
        request: Request,
        user: User = _user_dep,
    ):
        return await _proxy_inference(request, user, "/v1/chat/completions")

    @router.get("/models")
    async def list_models(
        request: Request,
        user: User = _user_dep,
    ) -> JSONResponse:
        """List models visible to the current user.

        Returns:
        - All shared base models
        - LoRA adapters belonging to this user (as tinker:// paths)
        """
        state = _get_state(request)
        models_data: list[dict[str, Any]] = []
        now_ts = int(time.time())

        # Add base models (shared by all users)
        for model_cfg in state.config.supported_models:
            models_data.append(
                {
                    "id": model_cfg.model_name,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": "system",
                    "is_public": True,
                }
            )

        # Add user-specific LoRA adapters (from sampling sessions)
        for session_id, record in state.sampling.sampling_sessions.items():
            if record.user_id != user.user_id:
                continue
            if record.model_path is None:
                continue
            # Present as tinker:// path if available
            tinker_path = f"tinker://{record.session_id}/sampler_weights/{session_id}"
            models_data.append(
                {
                    "id": tinker_path,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": user.user_id,
                    "parent": record.base_model,
                    "is_public": False,
                }
            )

        return JSONResponse(
            {
                "object": "list",
                "data": models_data,
            }
        )

    # Track which LoRA adapters have been loaded via the OpenAI API
    _loaded_loras: set[str] = set()

    async def _ensure_lora_loaded(
        client: httpx.AsyncClient,
        backend_url: str,
        lora_name: str,
        lora_path: str,
    ) -> None:
        """Load a LoRA adapter into vLLM's OpenAI serving layer if not already loaded."""
        if lora_name in _loaded_loras:
            return

        url = f"{backend_url}/v1/load_lora_adapter"
        resp = await client.post(
            url,
            json={"lora_name": lora_name, "lora_path": lora_path},
            headers={"Authorization": "Bearer EMPTY"},
            timeout=60.0,
        )
        if resp.status_code == 200:
            _loaded_loras.add(lora_name)
            logger.info("Loaded LoRA '%s' via vLLM OpenAI API", lora_name)
        else:
            # If 400 "already exists", that's fine
            body = resp.json() if resp.status_code < 500 else {}
            msg = body.get("message", "") if isinstance(body, dict) else str(body)
            if "already" in msg.lower():
                _loaded_loras.add(lora_name)
                return
            raise RuntimeError(
                f"Failed to load LoRA adapter '{lora_name}': {resp.status_code} {resp.text}"
            )

    async def _proxy_inference(
        request: Request,
        user: User,
        vllm_path: str,
    ) -> JSONResponse | StreamingResponse:
        """Core logic for proxying completions and chat/completions."""
        state = _get_state(request)
        client = _get_httpx_client(request)
        tracer = _get_tracer()

        body = await request.json()
        model_field = body.get("model")

        try:
            if not model_field:
                raise InvalidRequestException("Missing 'model' field in request body.")

            with tracer.start_as_current_span("oai.proxy_inference") as span:
                span.set_attribute("oai.endpoint", vllm_path)
                span.set_attribute("oai.model", model_field)
                span.set_attribute("oai.user_id", user.user_id)

                # Resolve model
                try:
                    resolved = resolve_model(model_field, state.config)
                except ValueError as exc:
                    raise UnknownModelException(model_name=model_field) from exc

                span.set_attribute("oai.base_model", resolved.base_model)
                if resolved.lora_id:
                    span.set_attribute("oai.lora_id", resolved.lora_id)

                # Get backend OpenAI URL
                backend = state.sampling._base_backends.get(resolved.base_model)
                if backend is None:
                    raise UnknownModelException(model_name=resolved.base_model)

                backend_url = backend.get_openai_api_url()
                if backend_url is None:
                    raise ServiceUnavailableException(
                        f"OpenAI API not available for model: {resolved.base_model}"
                    )

                # Ensure LoRA is loaded via the vLLM OpenAI server's API
                if resolved.lora_adapter_path and resolved.lora_id:
                    try:
                        await _ensure_lora_loaded(
                            client=client,
                            backend_url=backend_url,
                            lora_name=resolved.lora_id,
                            lora_path=str(resolved.lora_adapter_path),
                        )
                    except Exception as exc:
                        raise ServerException(f"Failed to load LoRA adapter: {exc}") from exc

                # Replace model name with backend model name for vLLM
                user_model_name = body["model"]
                body["model"] = resolved.backend_model_name
                stream = body.get("stream", False)

                # Build the response-id prefix.
                # Final id = "{session_part}:sample:{vllm_original_id}"
                # session_part is the sampling session id (from the
                # SamplingController) when a LoRA adapter is used, otherwise
                # the base model backend identifier.
                session_part = resolved.lora_id or "base"
                response_id_prefix = f"{session_part}:sample"

                return await proxy_request(
                    client=client,
                    backend_url=backend_url,
                    path=vllm_path,
                    body=body,
                    stream=stream,
                    user_model_name=user_model_name,
                    response_id_prefix=response_id_prefix,
                )
        except TuFTException as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return router
