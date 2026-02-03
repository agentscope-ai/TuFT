import logging
import os
from typing import List, Optional, Sequence, Union, cast

import tinker
from tinker import types

from ..models.schemas import CheckpointItem, TrainingRunDetail, TrainingRunItem


logger = logging.getLogger(__name__)


def get_tinker_client(api_key: str, base_url: Optional[str] = None):
    if base_url is None:
        base_url = os.getenv("TUFT_SERVER_URL", "http://localhost:10610/")
    logger.debug(f"Using TuFT server URL: {base_url}")
    return tinker.ServiceClient(base_url=base_url, api_key=api_key)


def list_all_sessions(rest_client):
    sessions = []
    offset = 0
    retries = 0
    max_retries = 3
    page_size = 5
    while retries < max_retries:
        try:
            future = rest_client.list_sessions(limit=page_size, offset=offset)
            response = future.result()

            if not response.sessions:
                break

            sessions.extend(response.sessions)
            if len(response.sessions) < 5:
                break

            offset += page_size
            retries = 0

        except Exception as e:
            retries += 1
            if retries >= max_retries:
                raise RuntimeError(f"Failed to fetch sessions after {max_retries} retries") from e
    return sessions


def list_training_runs_internal(client) -> List[TrainingRunItem]:
    rest = client.create_rest_client()
    runs = []
    runs_info = []
    offset = 0
    retries = 0
    max_retries = 3
    page_size = 50
    while retries < max_retries:
        try:
            future = rest.list_training_runs(limit=page_size, offset=offset)
            response = future.result()

            if not response.training_runs:
                break
            runs.extend(response.training_runs)

            if len(response.training_runs) < page_size:
                break

            offset += page_size
            retries = 0

        except Exception as e:
            retries += 1
            if retries >= max_retries:
                raise RuntimeError(f"Failed to fetch sessions after {max_retries} retries") from e

    for run in runs:
        runs_info.append(
            TrainingRunItem(
                id=run.training_run_id,
                base_model=run.base_model,
                last_request_time=(
                    run.last_request_time.isoformat() if run.last_request_time else "-"
                ),
                lora_rank=str(run.lora_rank) if run.lora_rank is not None else "-",
            )
        )
    return runs_info


def list_training_runs(api_key: str) -> List[TrainingRunItem]:
    client = get_tinker_client(api_key)
    print("Listing training runs...")
    return list_training_runs_internal(client)


# list training ckpt only now
def list_checkpoints(api_key: str) -> List[CheckpointItem]:
    client = get_tinker_client(api_key)
    rest = client.create_rest_client()
    runs_info = list_training_runs_internal(client)
    ckpts = []
    for run_info in runs_info:
        ckpts.extend(rest.list_checkpoints(run_info.id).result().checkpoints)

    ckpts_info = []
    for cp in ckpts:
        ckpts_info.append(
            CheckpointItem(
                id=cp.checkpoint_id,
                type=cp.checkpoint_type,
                path=cp.tinker_path,
                size=cp.size_bytes,
                visibility=cp.public,
                created=cp.time.isoformat() if cp.time else "-",
            )
        )

    return ckpts_info


def list_models(api_key: str) -> List[str]:
    client = get_tinker_client(api_key)
    capabilities = client.get_server_capabilities()
    return [
        model.model_name for model in capabilities.supported_models if model.model_name is not None
    ]


def get_training_detail(api_key: str, run_id: str) -> TrainingRunDetail:
    client = get_tinker_client(api_key)
    rest = client.create_rest_client()
    training_run = rest.get_training_run(run_id).result(timeout=30)
    return TrainingRunDetail(
        id=training_run.training_run_id,
        base_model=training_run.base_model,
        model_owner=training_run.model_owner,
        is_lora="Yes" if training_run.is_lora else "No",
        corrupted="Yes" if training_run.corrupted else "No",
        lora_rank=(str(training_run.lora_rank) if training_run.lora_rank is not None else "-"),
        last_request_time=(
            training_run.last_request_time.isoformat() if training_run.last_request_time else "-"
        ),
        last_checkpoint=(
            training_run.last_checkpoint.checkpoint_id if training_run.last_checkpoint else "-"
        ),
        last_sampler_checkpoint=(
            training_run.last_sampler_checkpoint.checkpoint_id
            if training_run.last_sampler_checkpoint
            else "-"
        ),
        user_metadata=training_run.user_metadata if training_run.user_metadata else {},
    )


def sample(
    api_key: str,
    data_list: List[str],
    model_path: str,
    base_model: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, Sequence[str], Sequence[int]]] = None,
) -> List[str]:
    if os.getenv("TUFT_CPU_TEST") == "1":
        return [
            "Mock sample result\n",
            f"data list: {data_list}\n",
            f"temperature={temperature}, top_p={top_p}, top_k={top_k}, \
                max_tokens={max_tokens}, seed={seed}, stop={stop}\n",
        ]

    client = get_tinker_client(api_key)
    if model_path:
        sample_client = client.create_sampling_client(model_path=model_path)
    else:
        sample_client = client.create_sampling_client(base_model=base_model)

    tokenizer = sample_client.get_tokenizer()

    sampling_param = types.SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
    )

    prompt_list = []
    for text in data_list:
        messages = [{"role": "user", "content": text}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_text = cast(str, prompt_text)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_list.append(prompt_tokens)

    results_token_list = [
        sample_client.sample(
            prompt=types.ModelInput.from_ints(prompt),
            num_samples=1,
            sampling_params=sampling_param,
        ).result()
        for prompt in prompt_list
    ]

    return [
        tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        for result in results_token_list
    ]
