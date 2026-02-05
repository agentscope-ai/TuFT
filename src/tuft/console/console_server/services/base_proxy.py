from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

from ..models.schemas import (
    CheckpointItem,
    TrainingRunDetail,
    TrainingRunItem,
)


class DataProxy(ABC):
    @abstractmethod
    def list_training_runs(self, api_key: str) -> List[TrainingRunItem]:
        pass

    @abstractmethod
    def list_checkpoints(self, api_key: str) -> List[CheckpointItem]:
        pass

    @abstractmethod
    def list_models(self, api_key: str) -> List[str]:
        pass

    @abstractmethod
    def get_training_detail(self, api_key: str, run_id: str) -> Optional[TrainingRunDetail]:
        pass

    @abstractmethod
    def sample(
        self,
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
        pass
