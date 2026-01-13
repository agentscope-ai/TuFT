"""Module for managing checkpoints on disk."""

import contextlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from pydantic import BaseModel

from tinker import types


class CheckpointMetadata(BaseModel):
    """A representation of checkpoint metadata."""

    model_id: str
    name: str
    base_model: str
    checkpoint_type: types.CheckpointType
    created_at: str
    session_id: str
    tinker_path: str
    owner_name: str = "default"
    lora_rank: int | None = None
    public: bool = False
    size_bytes: int | None = None


@dataclass
class CheckpointRecord:
    """A record representing a checkpoint on disk."""

    checkpoint_id: str
    checkpoint_type: types.CheckpointType
    training_run_id: str
    path: Path
    size_bytes: int
    public: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    owner_name: str = "default"  # TODO: implement user management

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
    def metadata(self) -> Dict:
        """Get the checkpoint metadata as a dictionary."""
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

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

    def set_visibility(self, public: bool) -> None:
        """Set the visibility of the checkpoint."""
        self.public = public
        metadata = self.metadata
        metadata["public"] = public
        self.save_metadata(
            base_model=metadata["base_model"],
            session_id=metadata["session_id"],
            lora_rank=metadata["lora_rank"],
        )

    def save_metadata(self, base_model: str, session_id: str, lora_rank: int) -> None:
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
                public=self.public,
                size_bytes=self.size_bytes,
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
        metadata_raw = (checkpoint_path / "metadata.json").read_text(encoding="utf-8")
        metadata = json.loads(metadata_raw)

        return cls(
            checkpoint_id=parsed.checkpoint_id.split("/", 1)[-1],
            checkpoint_type=parsed.checkpoint_type,
            training_run_id=parsed.training_run_id,
            path=checkpoint_path,
            size_bytes=metadata.get("size_bytes", 0),
            public=metadata.get("public", False),
            created_at=(
                datetime.fromisoformat(metadata.get("created_at"))
                if "created_at" in metadata
                else datetime.now(timezone.utc)
            ),
            owner_name=metadata.get("owner_name", "default"),
        )

    def delete(self) -> None:
        """Delete the checkpoint from disk."""
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self.path)

    @classmethod
    def from_training_run(
        cls,
        training_run_id: str,
        checkpoint_name: str,
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
            checkpoint_type=checkpoint_type,
            training_run_id=training_run_id,
            path=checkpoint_dir,
            size_bytes=0,
        )
