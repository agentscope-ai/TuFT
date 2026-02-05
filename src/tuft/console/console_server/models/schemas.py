from typing import List, Optional, Sequence, Union

from pydantic import BaseModel, Field


class TrainingRunItem(BaseModel):
    id: str
    base_model: str
    lora_rank: str
    last_request_time: str


class CheckpointItem(BaseModel):
    id: str
    type: str
    path: str
    size: int
    visibility: bool
    created: str


class ModelItem(BaseModel):
    model_name: str


class SampleRequest(BaseModel):
    data_list: List[str] = Field(..., description="List of input prompts")
    model_path: Optional[str] = None
    base_model: Optional[str] = None

    # Sampling parameters
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, Sequence[str], Sequence[int]]] = None


class TrainingRunDetail(BaseModel):
    id: str
    base_model: str
    model_owner: str
    is_lora: bool
    corrupted: bool
    lora_rank: str
    last_request_time: str
    last_checkpoint: str
    last_sampler_checkpoint: str
    user_metadata: dict[str, str]


class SampleResponse(BaseModel):
    samples: List[str]


class ListRunsResponse(BaseModel):
    runs: List[TrainingRunItem]


class ListCheckpointsResponse(BaseModel):
    checkpoints: List[CheckpointItem]


class ListModelsResponse(BaseModel):
    models: List[ModelItem]


class GetTrainingRunDetailResponse(BaseModel):
    run_detail: TrainingRunDetail


class ErrorResponse(BaseModel):
    detail: str
