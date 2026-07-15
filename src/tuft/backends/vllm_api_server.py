"""In-process OpenAI-compatible API server for the TuFT vLLM engine actor.

Adapted from Trinity-RFT (``trinity/common/models/vllm_patch/api_patch_v17.py``,
Apache-2.0, https://github.com/agentscope-ai/Trinity-RFT).

vLLM's stock ``run_server`` entrypoint always constructs its own engine; there
is no supported way to point it at a pre-existing ``AsyncLLM``. This module
reuses vLLM's own building blocks (``make_arg_parser``, ``build_app``,
``init_app_state``, ``serve_http``) to serve the engine that the TuFT actor
already created, so the direct ``generate()`` path (Tinker API) and the
OpenAI-compatible HTTP path share one engine and one set of LoRA adapters.

Two accommodations are required to run inside a Ray actor:

1. ``loop.add_signal_handler`` is replaced with a no-op -- uvicorn installs
   signal handlers on startup, which raises when not on the main thread.
2. The listening socket is created and bound *before* the server starts to
   avoid port-assignment races (https://github.com/vllm-project/vllm/issues/8204).

MAINTENANCE NOTE: this embedding recipe touches vLLM internals that have
historically churned across minor releases (0.12 / 0.13 / 0.17 / 0.22 / 0.23
all moved or renamed pieces of it). The imports below are valid for the vLLM
range pinned in pyproject.toml (>=0.19.1,<=0.23.0); revisit this module when
bumping the pin. See docs/sphinx_doc/source/development/vllm-backend.md.
"""

import asyncio
import functools
import logging
from typing import Optional

import vllm
import vllm.envs as envs
from packaging.version import InvalidVersion, parse as parse_version
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
    validate_api_server_args,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import is_valid_ipv6_address
from vllm.utils.system_utils import set_ulimit
from vllm.version import __version__ as VLLM_VERSION


def _get_vllm_version():
    try:
        return parse_version(vllm.__version__)
    except InvalidVersion:
        return parse_version("0.19.1")


def _patch_reasoning_content_alias(logger: logging.Logger) -> None:
    """Map ``reasoning_content`` -> ``reasoning`` in assistant chat messages.

    vLLM's reasoning parsers *emit* ``reasoning_content`` but (before 0.22)
    reject it when it is sent back in a follow-up request. Fixed upstream in
    vLLM 0.22.0 (#42664); only applied for older versions.
    """
    import vllm.entrypoints.chat_utils as chat_utils

    current = getattr(chat_utils, "_parse_chat_message_content", None)
    if current is None:
        raise RuntimeError("vLLM patch failed: _parse_chat_message_content not found")

    if getattr(current, "__patched_reasoning_content_alias__", False):
        return

    @functools.wraps(current)
    def _patched_parse_chat_message_content(
        message,
        mm_tracker,
        content_format,
        interleave_strings,
        mm_processor_kwargs=None,
    ):
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and message.get("reasoning") is None
            and message.get("reasoning_content") is not None
        ):
            message["reasoning"] = message.pop("reasoning_content")

        return current(
            message,
            mm_tracker,
            content_format,
            interleave_strings,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    _patched_parse_chat_message_content.__patched_reasoning_content_alias__ = True  # type: ignore[attr-defined]
    chat_utils._parse_chat_message_content = _patched_parse_chat_message_content

    logger.info("Patched vLLM chat_utils to map reasoning_content -> reasoning")


def _dummy_add_signal_handler(self, *args, **kwargs):
    # uvicorn installs signal handlers on startup; that raises when the event
    # loop is not on the main thread (as in a Ray actor). Do nothing instead.
    pass


async def run_api_server(
    async_llm,
    host: str,
    port: int,
    model_path: str,
    logger: logging.Logger,
    chat_template: Optional[str] = None,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: Optional[str] = None,
    reasoning_parser: Optional[str] = None,
    enable_log_requests: bool = False,
) -> None:
    """Serve vLLM's OpenAI-compatible API on ``host:port`` using ``async_llm``.

    Runs until cancelled (the engine actor wraps this in an ``asyncio.Task``
    and cancels it on shutdown).
    """
    logger.info("vLLM API server version %s", VLLM_VERSION)

    parser = FlexibleArgumentParser(description="Run the OpenAI API server.")
    args = make_arg_parser(parser)
    cli_args = [
        "--host",
        str(host),
        "--port",
        str(port),
        "--model",
        model_path,
        "--enable-server-load-tracking",  # enable tracking for load balancing
    ]
    if enable_log_requests:
        cli_args.append("--enable-log-requests")
    if enable_auto_tool_choice:
        cli_args.append("--enable-auto-tool-choice")
    if tool_call_parser:
        cli_args.extend(["--tool-call-parser", tool_call_parser])
    if reasoning_parser:
        cli_args.extend(["--reasoning-parser", reasoning_parser])
    if chat_template:
        cli_args.extend(["--chat-template", chat_template])
    args = parser.parse_args(cli_args)
    args.structured_outputs_config.reasoning_parser = reasoning_parser
    logger.info("Starting vLLM OpenAI API server with args: %s", args)

    validate_api_server_args(args)

    # Bind the port before serving to avoid race conditions with Ray
    # (https://github.com/vllm-project/vllm/issues/8204).
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # Avoid uvicorn dropping requests when many are active concurrently.
    set_ulimit()

    is_ssl = args.ssl_keyfile and args.ssl_certfile
    host_part = (
        f"[{sock_addr[0]}]" if is_valid_ipv6_address(sock_addr[0]) else sock_addr[0] or "0.0.0.0"
    )  # noqa: E501
    listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{sock_addr[1]}"
    logger.info("vLLM API server listening on %s", listen_address)

    if _get_vllm_version() < parse_version("0.22.0"):
        _patch_reasoning_content_alias(logger)

    assert args is not None
    app = build_app(args)
    await init_app_state(async_llm, app.state, args)

    loop = asyncio.get_event_loop()
    loop.add_signal_handler = functools.partial(  # type: ignore[method-assign, assignment]
        _dummy_add_signal_handler, loop
    )

    shutdown_task = await serve_http(
        app,
        sock=sock,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        # When 'disable_uvicorn_access_log' is True, no access log is emitted.
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
        h11_max_header_count=args.h11_max_header_count,
    )

    # Await server shutdown only after the backend context is exited.
    try:
        await shutdown_task
    finally:
        sock.close()
