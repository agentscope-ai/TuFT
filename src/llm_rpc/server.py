"""FastAPI application exposing a local-compatible Tinker API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timezone
from functools import partial
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import Response
from pydantic import BaseModel

from tinker import types

from .config import AppConfig
from .state import ServerState


class WeightsInfoBody(BaseModel):
    tinker_path: str


def _normalize_checkpoint_id(raw: str) -> str:
    if "/" not in raw:
        return raw
    prefix, remainder = raw.split("/", 1)
    if prefix not in {"weights", "sampler_weights"} or not remainder:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid checkpoint reference"
        )
    return remainder


def _get_state(request: Request) -> ServerState:
    state = getattr(request.app.state, "server_state", None)
    if state is None:
        raise RuntimeError("Server state has not been initialized")
    return state


def create_root_app(config: AppConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.server_state.async_init()
        yield

    app = FastAPI(title="LLM-RPC", version="0.1.0", lifespan=lifespan)
    app.state.server_state = ServerState(config)

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
    ) -> types.CreateSessionResponse:
        record = state.create_session(request)
        return types.CreateSessionResponse(session_id=record.session_id)

    @app.post(
        "/api/v1/session_heartbeat",
        response_model=types.SessionHeartbeatResponse,
    )
    async def session_heartbeat(
        request: types.SessionHeartbeatRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.SessionHeartbeatResponse:
        state.heartbeat(request.session_id)
        return types.SessionHeartbeatResponse()

    @app.post(
        "/api/v1/create_sampling_session",
        response_model=types.CreateSamplingSessionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_sampling_session(
        request: types.CreateSamplingSessionRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.CreateSamplingSessionResponse:
        sampling_session_id = state.create_sampling_session(
            session_id=request.session_id,
            base_model=request.base_model,
            model_path=request.model_path,
            session_seq_id=request.sampling_session_seq_id,
        )
        return types.CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)

    @app.post(
        "/api/v1/create_model",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_model(
        request: types.CreateModelRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.UntypedAPIFuture:
        if request.lora_config is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Missing LoRA config"
            )
        training_record = state.create_model(
            session_id=request.session_id,
            base_model=request.base_model,
            lora_config=request.lora_config,
            user_metadata=request.user_metadata,
        )
        response = types.CreateModelResponse(model_id=training_record.training_run_id)
        return state.future_store.create_ready_future(
            response, model_id=training_record.training_run_id
        )

    @app.post(
        "/api/v1/get_info",
        response_model=types.GetInfoResponse,
    )
    async def get_info(
        request: types.GetInfoRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.GetInfoResponse:
        return state.get_model_info(request.model_id)

    @app.post(
        "/api/v1/unload_model",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def unload_model(
        request: types.UnloadModelRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.UntypedAPIFuture:
        state.unload_model(request.model_id)
        response = types.UnloadModelResponse(model_id=request.model_id)
        return state.future_store.create_ready_future(response, model_id=request.model_id)

    def _queue_future(
        operation: Callable[[], Any],
        state: ServerState,
        *,
        model_id: str | None = None,
    ) -> types.UntypedAPIFuture:
        return state.future_store.enqueue(operation, model_id=model_id)

    @app.post(
        "/api/v1/forward",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def forward(
        request: types.ForwardRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.UntypedAPIFuture:
        inp = request.forward_input

        async def _operation() -> types.ForwardBackwardOutput:
            return await state.run_forward(
                request.model_id,
                inp.data,
                inp.loss_fn,
                inp.loss_fn_config,
                request.seq_id,
                backward=False,
            )

        return _queue_future(_operation, state, model_id=request.model_id)

    @app.post(
        "/api/v1/forward_backward",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def forward_backward(
        request: types.ForwardBackwardRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.UntypedAPIFuture:
        inp = request.forward_backward_input

        async def _operation() -> types.ForwardBackwardOutput:
            return await state.run_forward(
                request.model_id,
                inp.data,
                inp.loss_fn,
                inp.loss_fn_config,
                request.seq_id,
                backward=True,
            )

        return _queue_future(_operation, state, model_id=request.model_id)

    @app.post(
        "/api/v1/optim_step",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def optim_step(
        request: types.OptimStepRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.UntypedAPIFuture:
        async def _operation() -> types.OptimStepResponse:
            return await state.run_optim_step(request.model_id, request.adam_params, request.seq_id)

        return _queue_future(
            _operation,
            state,
            model_id=request.model_id,
        )

    @app.post(
        "/api/v1/save_weights",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def save_weights(
        request: types.SaveWeightsRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.UntypedAPIFuture:
        async def _operation() -> types.SaveWeightsResponse:
            checkpoint = await state.save_checkpoint(request.model_id, request.path, "training")
            return types.SaveWeightsResponse(path=checkpoint.to_api(request.model_id).tinker_path)

        return _queue_future(_operation, state, model_id=request.model_id)

    @app.post(
        "/api/v1/save_weights_for_sampler",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def save_weights_for_sampler(
        request: types.SaveWeightsForSamplerRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.UntypedAPIFuture:
        async def _operation() -> types.SaveWeightsForSamplerResponse:
            checkpoint = await state.save_checkpoint(request.model_id, request.path, "sampler")
            return types.SaveWeightsForSamplerResponse(
                path=checkpoint.to_api(request.model_id).tinker_path
            )

        return _queue_future(_operation, state, model_id=request.model_id)

    @app.post(
        "/api/v1/load_weights",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def load_weights(
        request: types.LoadWeightsRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.UntypedAPIFuture:
        async def _operation() -> types.LoadWeightsResponse:
            await state.load_checkpoint(request.model_id, request.path, request.optimizer)
            return types.LoadWeightsResponse(path=request.path)

        return _queue_future(_operation, state, model_id=request.model_id)

    @app.post(
        "/api/v1/asample",
        response_model=types.UntypedAPIFuture,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def asample(
        request: types.SampleRequest,
        state: ServerState = Depends(_get_state),
    ) -> types.UntypedAPIFuture:
        return _queue_future(partial(state.run_sample, request=request), state)

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(
        request: types.FutureRetrieveRequest,
        state: ServerState = Depends(_get_state),
    ) -> Any:
        try:
            payload = state.future_store.retrieve(request.request_id)
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
    ) -> types.TrainingRunsResponse:
        return state.list_training_runs(limit=limit, offset=offset)

    @app.get(
        "/api/v1/training_runs/{model_id}",
        response_model=types.TrainingRun,
    )
    async def get_training_run(
        model_id: str, state: ServerState = Depends(_get_state)
    ) -> types.TrainingRun:
        return state.get_training_run_view(model_id)

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
    ) -> types.CheckpointsListResponse:
        checkpoints = state.list_checkpoints(model_id)
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
    ) -> None:
        state.delete_checkpoint(model_id, _normalize_checkpoint_id(checkpoint_path))

    @app.post(
        "/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_path:path}/publish",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def publish_checkpoint(
        model_id: str,
        checkpoint_path: str,
        state: ServerState = Depends(_get_state),
    ) -> None:
        state.set_checkpoint_visibility(
            model_id, _normalize_checkpoint_id(checkpoint_path), public=True
        )

    @app.delete(
        "/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_path:path}/publish",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def unpublish_checkpoint(
        model_id: str,
        checkpoint_path: str,
        state: ServerState = Depends(_get_state),
    ) -> None:
        state.set_checkpoint_visibility(
            model_id, _normalize_checkpoint_id(checkpoint_path), public=False
        )

    @app.get(
        "/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_path:path}/archive",
        status_code=status.HTTP_302_FOUND,
    )
    async def checkpoint_archive(
        model_id: str,
        checkpoint_path: str,
        state: ServerState = Depends(_get_state),
    ) -> Response:
        archive = state.build_archive_url(model_id, _normalize_checkpoint_id(checkpoint_path))
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
    ) -> types.CheckpointsListResponse:
        checkpoints = state.list_user_checkpoints()
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
    ) -> types.WeightsInfoResponse:
        return state.get_weights_info(body.tinker_path)

    @app.get(
        "/api/v1/sessions/{session_id}",
        response_model=types.GetSessionResponse,
    )
    async def get_session(
        session_id: str, state: ServerState = Depends(_get_state)
    ) -> types.GetSessionResponse:
        return state.get_session_overview(session_id)

    @app.get(
        "/api/v1/sessions",
        response_model=types.ListSessionsResponse,
    )
    async def list_sessions(
        limit: int = Query(20, ge=1, le=500),
        offset: int = Query(0, ge=0),
        state: ServerState = Depends(_get_state),
    ) -> types.ListSessionsResponse:
        return state.list_sessions(limit=limit, offset=offset)

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
        sampler_id: str, state: ServerState = Depends(_get_state)
    ) -> types.GetSamplerResponse:
        return state.get_sampler_info(sampler_id)

    return app
