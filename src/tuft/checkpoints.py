"""Module for managing checkpoints on disk."""

import contextlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_serializer
from tinker import types

from .exceptions import CheckpointIncompatibleException, CheckpointMetadataReadException


def compute_tree_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


class CheckpointMetadata(BaseModel):
    """A representation of checkpoint metadata."""

    model_id: str
    name: str
    base_model: str
    checkpoint_type: types.CheckpointType
    created_at: str
    session_id: str
    tinker_path: str
    owner_name: str
    size_bytes: int = 0
    lora_rank: int | None = None
    # Effective peft ``lora_alpha`` the adapter was trained with. None for
    # checkpoints written before this field existed; those fall back to the peft
    # ``adapter_config.json`` (see ``CheckpointRecord.saved_lora_alpha``).
    lora_alpha: int | None = None
    # LoRA target-module selection. Together with lora_rank these define the
    # adapter geometry a checkpoint was written from. Older checkpoints may omit
    # these fields, but must provide target_modules in adapter_config.json to load.
    train_attn: bool | None = None
    train_mlp: bool | None = None
    train_unembed: bool | None = None
    target_modules: list[str] | None = None
    public: bool = False
    future_id: int = 0
    seq_id: int | None = None


class CheckpointRecord(BaseModel):
    """A record representing a checkpoint on disk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoint_id: str
    owner_name: str
    checkpoint_type: types.CheckpointType
    training_run_id: str
    path: Path
    size_bytes: int = 0
    public: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    future_id: int = 0
    seq_id: int | None = None

    @field_serializer("path")
    def serialize_path(self, path: Path) -> str:
        """Serialize Path to string for JSON."""
        return str(path)

    @property
    def tinker_checkpoint(self) -> types.Checkpoint:
        """Get a Tinker Checkpoint instance representing this record."""
        return types.Checkpoint(
            checkpoint_id=self.checkpoint_id,
            checkpoint_type=self.checkpoint_type,
            time=self.created_at,
            tinker_path=self.tinker_path,
            size_bytes=self.size_bytes,
            public=self.public,
        )

    @property
    def metadata(self) -> CheckpointMetadata:
        """Get the checkpoint metadata.

        Raises:
            CheckpointMetadataReadException: If the metadata file does not
                exist or is invalid.
        """
        try:
            return CheckpointMetadata.model_validate_json(
                self.metadata_path.read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            raise CheckpointMetadataReadException(checkpoint_id=self.checkpoint_id) from exc
        except ValidationError as exc:
            raise CheckpointMetadataReadException(checkpoint_id=self.checkpoint_id) from exc

    @property
    def tinker_path(self) -> str:
        """Get the tinker style path for this checkpoint."""
        folder = "weights" if self.checkpoint_type == "training" else "sampler_weights"
        return f"tinker://{self.training_run_id}/{folder}/{self.checkpoint_id}"

    @property
    def adapter_path(self) -> Path:
        """Get the path to the adapter weights file."""
        return self.path / "adapter"

    @property
    def optimizer_path(self) -> Path:
        """Get the path to the optimizer state file."""
        return self.path / "optimizer"

    @property
    def metadata_path(self) -> Path:
        """Get the path to the metadata JSON file."""
        return self.path / "metadata.json"

    @property
    def saved_lora_alpha(self) -> int | None:
        """Effective peft ``lora_alpha`` this checkpoint was trained with.

        Prefers ``metadata.json``, then falls back to the peft
        ``adapter_config.json`` in the adapter directory, which makes checkpoints
        written before ``metadata.lora_alpha`` existed self-describing. Returns
        None when neither source records an alpha, in which case the alpha cannot
        be checked and the load proceeds.
        """
        with contextlib.suppress(CheckpointMetadataReadException):
            recorded = self.metadata.lora_alpha
            if recorded is not None:
                return int(recorded)
        try:
            adapter_config = json.loads(
                (self.adapter_path / "adapter_config.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        if not isinstance(adapter_config, dict):
            return None
        alpha = adapter_config.get("lora_alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            return None
        return int(alpha)

    def validate_lora_alpha(self, expected_lora_alpha: int) -> None:
        """Reject loading this checkpoint into an adapter with a different alpha.

        LoRA update scale is proportional to ``lora_alpha / rank``, so replaying
        weights trained at one alpha into an adapter built with another silently
        rescales every update. The FSDP backend is the dangerous case: its slots
        are built from ``ModelConfig.lora_alpha_ratio`` up front and loading only
        copies weights into them.

        Raises:
            CheckpointIncompatibleException: If the checkpoint records an alpha
                that differs from ``expected_lora_alpha``.
        """
        saved = self.saved_lora_alpha
        if saved is None or saved == expected_lora_alpha:
            return
        raise CheckpointIncompatibleException(
            checkpoint_id=self.checkpoint_id,
            detail=(
                f"Checkpoint {self.checkpoint_id} was trained with lora_alpha={saved} but this "
                f"server would give the adapter lora_alpha={expected_lora_alpha}. Loading it "
                "would rescale every LoRA update. Set the model's lora_alpha_ratio so that "
                "rank * lora_alpha_ratio matches the checkpoint (the 'hf' backend used "
                "lora_alpha_ratio: 1 before the setting existed), or start a new training run."
            ),
        )

    def set_visibility(self, public: bool) -> None:
        """Set the visibility of the checkpoint."""
        self.public = public
        metadata = self.metadata
        metadata.public = public
        self.save_metadata(
            base_model=metadata.base_model,
            session_id=metadata.session_id,
            lora_rank=metadata.lora_rank,
            lora_alpha=metadata.lora_alpha,
            train_attn=metadata.train_attn,
            train_mlp=metadata.train_mlp,
            train_unembed=metadata.train_unembed,
            target_modules=metadata.target_modules,
        )

    def save_metadata(
        self,
        base_model: str,
        session_id: str,
        lora_rank: int | None,
        lora_alpha: int | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
        target_modules: list[str] | None = None,
    ) -> None:
        """Save the checkpoint metadata to disk."""
        # check the format of metadata
        try:
            metadata = CheckpointMetadata(
                model_id=self.training_run_id,
                name=self.checkpoint_id,
                base_model=base_model,
                checkpoint_type=self.checkpoint_type,
                created_at=self.created_at.isoformat(),
                session_id=session_id,
                tinker_path=self.tinker_path,
                owner_name=self.owner_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
                target_modules=target_modules,
                public=self.public,
                size_bytes=self.size_bytes,
                future_id=self.future_id,
                seq_id=self.seq_id,
            )
        except Exception as e:
            raise ValueError(f"Invalid checkpoint metadata: {e}") from e
        self.metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_tinker_path(cls, path: str, checkpoint_root_dir: Path) -> "CheckpointRecord":
        """Create a CheckpointRecord from a Tinker path.

        Raises:
            FileNotFoundError: If the checkpoint directory or metadata.json is missing.
            json.JSONDecodeError: If metadata.json cannot be parsed as JSON.
        """
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(path)
        checkpoint_path = (
            checkpoint_root_dir / parsed.training_run_id / parsed.checkpoint_id.split("/", 1)[-1]
        )
        record = cls(
            checkpoint_id=parsed.checkpoint_id.split("/", 1)[-1],
            checkpoint_type=parsed.checkpoint_type,
            training_run_id=parsed.training_run_id,
            path=checkpoint_path,
            owner_name="",  # Will be filled from metadata later
            size_bytes=0,  # Will be filled from metadata later
        )
        metadata = record.metadata  # This may raise FileNotFoundError or JSONDecodeError
        record.owner_name = metadata.owner_name
        record.size_bytes = metadata.size_bytes
        record.public = metadata.public
        record.created_at = datetime.fromisoformat(metadata.created_at)
        record.future_id = metadata.future_id
        record.seq_id = metadata.seq_id
        return record

    def delete(self) -> None:
        """Delete the checkpoint from disk."""
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self.path)

    @classmethod
    def from_training_run(
        cls,
        training_run_id: str,
        checkpoint_name: str,
        owner_name: str,
        checkpoint_type: types.CheckpointType,
        checkpoint_root_dir: Path,
        exist_ok: bool = True,
    ) -> "CheckpointRecord":
        """Create a CheckpointRecord from a training run."""
        checkpoint_dir = checkpoint_root_dir / training_run_id / checkpoint_name
        if not exist_ok and checkpoint_dir.exists():
            raise FileExistsError(f"Checkpoint directory already exists: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)
        return cls(
            checkpoint_id=checkpoint_name,
            owner_name=owner_name,
            checkpoint_type=checkpoint_type,
            training_run_id=training_run_id,
            path=checkpoint_dir,
            size_bytes=0,
        )
