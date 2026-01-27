"""FastAPI application exposing a local-compatible Tinker API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import timezone
from functools import partial
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from tinker import types

from .auth import User
from .config import AppConfig
from .exceptions import TuFTException
from .persistence import get_redis_store
from .state import ServerState


logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _get_user(
    request: Request,
    api_key: str = Depends(api_key_header),
) -> User:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )
    user = request.app.state.server_state.get_user(api_key)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return user


class WeightsInfoBody(BaseModel):
    tinker_path: str


def _normalize_checkpoint_id(raw: str) -> str:
    if "/" not in raw:
        return raw
    prefix, remainder = raw.split("/", 1)
    if prefix not in {"weights", "sampler_weights"} or not remainder:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid checkpoint reference",
        )
    return remainder


def _get_state(request: Request) -> ServerState:
    state = getattr(request.app.state, "server_state", None)
    if state is None:
        raise RuntimeError("Server state has not been initialized")
    return state


def _instrument_fastapi(app: FastAPI) -> None:
    """Instrument FastAPI app with OpenTelemetry if available."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.debug("FastAPI instrumentation enabled")
    except ImportError:
        logger.debug("FastAPI instrumentation not available")
    except Exception as e:
        logger.warning("Failed to instrument FastAPI: %s", e)


def create_root_app(config: AppConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await app.state.server_state.async_init()
            logger.info("Server initialized successfully")
            yield
        finally:
            logger.info("Server shutting down")
            await app.state.server_state.future_store.shutdown()
            store = get_redis_store()
            if store.is_enabled:
                store.close()
            try:
                from .telemetry import shutdown_telemetry

                shutdown_telemetry()
            except ImportError as e:
                logger.warning("Failed to shutdown telemetry: %s", e)

    def require_user_dependency(route):
        if not any(dep.dependency == _get_user for dep in getattr(route, "dependencies", [])):
            route.dependencies = getattr(route, "dependencies", []) + [Depends(_get_user)]
        return route

    resolved_config = config or AppConfig()
    if resolved_config.persistence.enabled:
        store = get_redis_store()
        store.configure(resolved_config.persistence)

    app = FastAPI(
        title="TuFT",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.server_state = ServerState(resolved_config)

    # Instrument FastAPI with OpenTelemetry if enabled
    if resolved_config.telemetry.enabled:
        _instrument_fastapi(app)

    @app.get("/api/v1/healthz", response_model=types.HealthResponse)
    async def healthz() -> types.HealthResponse:
        return types.HealthResponse(status="ok")

    @app.get(
        "/api/v1/get_server_capabilities",
        response_model=types.GetServerCapabilitiesResponse,
    )
    async def get_server_capabilities(
        state: ServerState = Depends(_get_state),
    ) -> types.GetServerCapabilitiesResponse:
        return types.GetServerCapabilitiesResponse(supported_models=state.build_supported_models())

    @app.post(
        "/api/v1/create_session",
        response_model=types.CreateSessionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session(
        request: types.CreateSessionRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.CreateSessionResponse:
        record = state.create_session(request, user)
        return types.CreateSessionResponse(session_id=record.session_id)

    @app.post(
        "/api/v1/session_heartbeat",
        response_model=types.SessionHeartbeatResponse,
    )
    async def session_heartbeat(
        request: types.SessionHeartbeatRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.SessionHeartbeatResponse:
        state.heartbeat(request.session_id, user_id=user.user_id)
        return types.SessionHeartbeatResponse()

    @app.post(
        "/api/v1/create_sampling_session",
        response_model=types.CreateSamplingSessionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_sampling_session(
        request: types.CreateSamplingSessionRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.CreateSamplingSessionResponse:
        try:
            sampling_session_id = await state.create_sampling_session(
                session_id=request.session_id,
                user_id=user.user_id,
                base_model=request.base_model,
                model_path=request.model_path,
                session_seq_id=request.sampling_session_seq_id,
            )
        except TuFTException as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create sampling session: {exc.detail}",
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create sampling session: {str(exc)}",
            ) from exc
        return types.CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)

    @app.post(
        "/api/v1/create_model",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_model(
        request: types.CreateModelRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.UntypedAPIFuture:
        if request.lora_config is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Missing LoRA config"
            )
        try:
            training_record = await state.create_model(
                session_id=request.session_id,
                base_model=request.base_model,
                lora_config=request.lora_config,
                model_owner=user.user_id,
                user_metadata=request.user_metadata,
            )
        except TuFTException as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create model: {exc.detail}",
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create model: {str(exc)}",
            ) from exc
        response = types.CreateModelResponse(model_id=training_record.training_run_id)
        return await state.future_store.create_ready_future(
            response,
            model_id=training_record.training_run_id,
            user_id=user.user_id,
        )

    @app.post(
        "/api/v1/get_info",
        response_model=types.GetInfoResponse,
    )
    async def get_info(
        request: types.GetInfoRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.GetInfoResponse:
        return state.get_model_info(request.model_id, user_id=user.user_id)

    @app.post(
        "/api/v1/unload_model",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def unload_model(
        request: types.UnloadModelRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.UntypedAPIFuture:
        try:
            await state.unload_model(request.model_id, user_id=user.user_id)
        except TuFTException as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to unload model: {exc.detail}",
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to unload model: {str(exc)}",
            ) from exc
        response = types.UnloadModelResponse(model_id=request.model_id)
        return await state.future_store.create_ready_future(
            response, model_id=request.model_id, user_id=user.user_id
        )

    async def _queue_future(
        operation: Callable[[], Any],
        state: ServerState,
        user_id: str,
        *,
        model_id: str | None = None,
        operation_type: str | None = None,
        operation_args: dict[str, Any] | None = None,
    ) -> types.UntypedAPIFuture:
        return await state.future_store.enqueue(
            operation,
            model_id=model_id,
            user_id=user_id,
            operation_type=operation_type,  # type: ignore[arg-type]
            operation_args=operation_args,
        )

    @app.post(
        "/api/v1/forward",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def forward(
        request: types.ForwardRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.UntypedAPIFuture:
        inp = request.forward_input

        async def _operation() -> types.ForwardBackwardOutput:
            return await state.run_forward(
                request.model_id,
                user.user_id,
                inp.data,
                inp.loss_fn,
                inp.loss_fn_config,
                request.seq_id,
                backward=False,
            )

        return await _queue_future(
            _operation,
            state,
            model_id=request.model_id,
            user_id=user.user_id,
            operation_type="forward",
            operation_args={
                "model_id": request.model_id,
                "user_id": user.user_id,
                "data": inp.data,
                "loss_fn": inp.loss_fn,
                "loss_fn_config": inp.loss_fn_config,
                "seq_id": request.seq_id,
                "backward": False,
            },
        )

    @app.post(
        "/api/v1/forward_backward",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def forward_backward(
        request: types.ForwardBackwardRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.UntypedAPIFuture:
        inp = request.forward_backward_input

        async def _operation() -> types.ForwardBackwardOutput:
            return await state.run_forward(
                request.model_id,
                user.user_id,
                inp.data,
                inp.loss_fn,
                inp.loss_fn_config,
                request.seq_id,
                backward=True,
            )

        return await _queue_future(
            _operation,
            state,
            model_id=request.model_id,
            user_id=user.user_id,
            operation_type="forward_backward",
            operation_args={
                "model_id": request.model_id,
                "user_id": user.user_id,
                "data": inp.data,
                "loss_fn": inp.loss_fn,
                "loss_fn_config": inp.loss_fn_config,
                "seq_id": request.seq_id,
                "backward": True,
            },
        )

    @app.post(
        "/api/v1/optim_step",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def optim_step(
        request: types.OptimStepRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.UntypedAPIFuture:
        async def _operation() -> types.OptimStepResponse:
            return await state.run_optim_step(
                request.model_id,
                user.user_id,
                request.adam_params,
                request.seq_id,
            )

        return await _queue_future(
            _operation,
            state,
            model_id=request.model_id,
            user_id=user.user_id,
            operation_type="optim_step",
            operation_args={
                "model_id": request.model_id,
                "user_id": user.user_id,
                "params": request.adam_params,
                "seq_id": request.seq_id,
            },
        )

    @app.post(
        "/api/v1/save_weights",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def save_weights(
        request: types.SaveWeightsRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.UntypedAPIFuture:
        async def _operation() -> types.SaveWeightsResponse:
            checkpoint = await state.save_checkpoint(
                request.model_id,
                user.user_id,
                request.path,
                "training",
                seq_id=request.seq_id,
            )
            return types.SaveWeightsResponse(path=checkpoint.tinker_checkpoint.tinker_path)

        return await _queue_future(
            _operation,
            state,
            model_id=request.model_id,
            user_id=user.user_id,
            operation_type="save_weights",
            operation_args={
                "model_id": request.model_id,
                "user_id": user.user_id,
                "name": request.path,
                "checkpoint_type": "training",
            },
        )

    @app.post(
        "/api/v1/save_weights_for_sampler",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def save_weights_for_sampler(
        request: types.SaveWeightsForSamplerRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.UntypedAPIFuture:
        async def _operation() -> types.SaveWeightsForSamplerResponse:
            checkpoint = await state.save_checkpoint(
                request.model_id,
                user.user_id,
                request.path,
                "sampler",
                seq_id=request.seq_id,
            )
            return types.SaveWeightsForSamplerResponse(
                path=checkpoint.tinker_checkpoint.tinker_path
            )

        return await _queue_future(
            _operation,
            state,
            model_id=request.model_id,
            user_id=user.user_id,
            operation_type="save_weights_for_sampler",
            operation_args={
                "model_id": request.model_id,
                "user_id": user.user_id,
                "name": request.path,
                "checkpoint_type": "sampler",
            },
        )

    @app.post(
        "/api/v1/load_weights",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def load_weights(
        request: types.LoadWeightsRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.UntypedAPIFuture:
        async def _operation() -> types.LoadWeightsResponse:
            await state.load_checkpoint(
                model_id=request.model_id,
                user_id=user.user_id,
                path=request.path,
                optimizer=request.optimizer,
                seq_id=request.seq_id,
            )
            return types.LoadWeightsResponse(path=request.path)

        return await _queue_future(
            _operation,
            state,
            model_id=request.model_id,
            user_id=user.user_id,
            operation_type="load_weights",
            operation_args={
                "model_id": request.model_id,
                "user_id": user.user_id,
                "path": request.path,
                "optimizer": request.optimizer,
            },
        )

    @app.post(
        "/api/v1/asample",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def asample(
        request: types.SampleRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.UntypedAPIFuture:
        return await _queue_future(
            partial(state.run_sample, request=request, user_id=user.user_id),
            state=state,
            user_id=user.user_id,
            operation_type="sample",
            operation_args={
                "request": request,
                "user_id": user.user_id,
            },
        )

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(
        request: types.FutureRetrieveRequest,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> Any:
        try:
            payload = await state.future_store.retrieve(
                request_id=request.request_id, user_id=user.user_id
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Unknown request_id"
            ) from exc
        return payload  # FastAPI will serialize the stored Tinker type

    @app.get(
        "/api/v1/training_runs",
        response_model=types.TrainingRunsResponse,
    )
    async def list_training_runs(
        limit: int = Query(20, ge=1, le=500),
        offset: int = Query(0, ge=0),
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.TrainingRunsResponse:
        return state.list_training_runs(user_id=user.user_id, limit=limit, offset=offset)

    @app.get(
        "/api/v1/training_runs/{model_id}",
        response_model=types.TrainingRun,
    )
    async def get_training_run(
        model_id: str,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.TrainingRun:
        return state.get_training_run_view(model_id=model_id, user_id=user.user_id)

    def _build_checkpoint_cursor(total: int, limit: int, offset: int) -> types.Cursor:
        return types.Cursor(offset=offset, limit=limit, total_count=total)

    @app.get(
        "/api/v1/training_runs/{model_id}/checkpoints",
        response_model=types.CheckpointsListResponse,
    )
    async def list_training_run_checkpoints(
        model_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.CheckpointsListResponse:
        checkpoints = state.list_checkpoints(model_id, user_id=user.user_id)
        total = len(checkpoints)
        start = min(offset, total)
        end = min(start + limit, total)
        subset = checkpoints[start:end]
        cursor = _build_checkpoint_cursor(total, limit, offset)
        return types.CheckpointsListResponse(checkpoints=subset, cursor=cursor)

    @app.delete(
        "/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_path:path}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_checkpoint(
        model_id: str,
        checkpoint_path: str,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> None:
        state.delete_checkpoint(model_id, user.user_id, _normalize_checkpoint_id(checkpoint_path))

    @app.post(
        "/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_path:path}/publish",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def publish_checkpoint(
        model_id: str,
        checkpoint_path: str,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> None:
        state.set_checkpoint_visibility(
            model_id,
            user.user_id,
            _normalize_checkpoint_id(checkpoint_path),
            public=True,
        )

    @app.delete(
        "/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_path:path}/publish",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def unpublish_checkpoint(
        model_id: str,
        checkpoint_path: str,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> None:
        state.set_checkpoint_visibility(
            model_id,
            user.user_id,
            _normalize_checkpoint_id(checkpoint_path),
            public=False,
        )

    @app.get(
        "/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_path:path}/archive",
        status_code=status.HTTP_302_FOUND,
    )
    async def checkpoint_archive(
        model_id: str,
        checkpoint_path: str,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> Response:
        archive = state.build_archive_url(
            model_id,
            user_id=user.user_id,
            checkpoint_id=_normalize_checkpoint_id(checkpoint_path),
        )
        expires = archive.expires.astimezone(timezone.utc)
        return Response(
            status_code=status.HTTP_302_FOUND,
            headers={
                "Location": archive.url,
                "Expires": expires.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            },
        )

    @app.get(
        "/api/v1/checkpoints",
        response_model=types.CheckpointsListResponse,
    )
    async def list_user_checkpoints(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.CheckpointsListResponse:
        checkpoints = state.list_user_checkpoints(user.user_id)
        total = len(checkpoints)
        start = min(offset, total)
        end = min(start + limit, total)
        subset = checkpoints[start:end]
        cursor = _build_checkpoint_cursor(total, limit, offset)
        return types.CheckpointsListResponse(checkpoints=subset, cursor=cursor)

    @app.post(
        "/api/v1/weights_info",
        response_model=types.WeightsInfoResponse,
    )
    async def weights_info(
        body: WeightsInfoBody,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.WeightsInfoResponse:
        return state.get_weights_info(body.tinker_path, user.user_id)

    @app.get(
        "/api/v1/sessions/{session_id}",
        response_model=types.GetSessionResponse,
    )
    async def get_session(
        session_id: str,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.GetSessionResponse:
        return state.get_session_overview(session_id, user.user_id)

    @app.get(
        "/api/v1/sessions",
        response_model=types.ListSessionsResponse,
    )
    async def list_sessions(
        limit: int = Query(20, ge=1, le=500),
        offset: int = Query(0, ge=0),
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.ListSessionsResponse:
        return state.list_sessions(user_id=user.user_id, limit=limit, offset=offset)

    @app.post(
        "/api/v1/telemetry",
        response_model=types.TelemetryResponse,
    )
    async def send_telemetry(
        body: types.TelemetrySendRequest,
    ) -> types.TelemetryResponse:
        # We currently accept telemetry events for protocol compatibility but do not persist them.
        return types.TelemetryResponse(status="accepted")

    @app.get(
        "/api/v1/samplers/{sampler_id}",
        response_model=types.GetSamplerResponse,
    )
    async def get_sampler(
        sampler_id: str,
        state: ServerState = Depends(_get_state),
        user: User = Depends(_get_user),
    ) -> types.GetSamplerResponse:
        return state.get_sampler_info(sampler_id, user.user_id)

    for route in app.routes:
        if getattr(route, "path", None) != "/api/v1/healthz" and hasattr(route, "dependencies"):
            require_user_dependency(route)
    return app
