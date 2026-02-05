from fastapi import APIRouter, Depends, HTTPException, status

from ..api.deps import get_api_key, get_data_proxy
from ..models.schemas import (
    ErrorResponse,
    GetTrainingRunDetailResponse,
    ListCheckpointsResponse,
    ListModelsResponse,
    ListRunsResponse,
    SampleRequest,
    SampleResponse,
)
from ..services.base_proxy import DataProxy


API_KEY_DEP = Depends(get_api_key)
PROXY_DEP = Depends(get_data_proxy)

router = APIRouter()


@router.get("/runs", response_model=ListRunsResponse, responses={400: {"model": ErrorResponse}})
def get_runs(
    api_key: str = API_KEY_DEP,
    proxy: DataProxy = PROXY_DEP,
):
    runs = proxy.list_training_runs(api_key)
    return {"runs": runs}


@router.get(
    "/checkpoints",
    response_model=ListCheckpointsResponse,
    responses={400: {"model": ErrorResponse}},
)
def get_checkpoints(
    api_key: str = API_KEY_DEP,
    proxy: DataProxy = PROXY_DEP,
):
    ckpts = proxy.list_checkpoints(api_key)
    return {"checkpoints": ckpts}


@router.get("/models", response_model=ListModelsResponse, responses={400: {"model": ErrorResponse}})
def get_models(
    api_key: str = API_KEY_DEP,
    proxy: DataProxy = PROXY_DEP,
):
    model_names = proxy.list_models(api_key)
    models = [{"model_name": name} for name in model_names]
    return {"models": models}


@router.post(
    "/sample",
    response_model=SampleResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def run_sample(
    request: SampleRequest,
    api_key: str = API_KEY_DEP,
    proxy: DataProxy = PROXY_DEP,
):
    if not request.model_path and not request.base_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'model_path' or 'base_model' must be provided.",
        )

    try:
        results = proxy.sample(
            api_key=api_key,
            data_list=request.data_list,
            model_path=request.model_path or "",
            base_model=request.base_model or "",
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            seed=request.seed,
            stop=request.stop,
        )
        return SampleResponse(samples=results)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sampling failed: {str(e)}",
        ) from e


@router.get(
    "/runs/{run_id}",
    response_model=GetTrainingRunDetailResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def get_run_detail(
    run_id: str,
    api_key: str = API_KEY_DEP,
    proxy: DataProxy = PROXY_DEP,
):
    if not run_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Run ID cannot be empty."
        )

    detail = proxy.get_training_detail(api_key=api_key, run_id=run_id.strip())
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Training run '{run_id}' not found."
        )
    return GetTrainingRunDetailResponse(run_detail=detail)


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
