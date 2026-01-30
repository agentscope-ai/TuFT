"""Some custom exceptions."""

from typing import Any


class TuFTException(Exception):
    """Base exception for TuFT errors."""

    def __init__(self, detail: str = ""):
        super().__init__(detail)
        self.detail = detail


class ModelException(TuFTException):
    """Base exception for Model related errors."""


class CheckpointException(TuFTException):
    """Base exception for Checkpoint related errors."""


class FutureException(TuFTException):
    """Base exception for Future related errors."""


class SessionException(TuFTException):
    """Base exception for Session related errors."""


class AuthenticationException(TuFTException):
    """Base exception for Authentication related errors."""


class LossFunctionException(TuFTException):
    """Base exception for Loss Function related errors."""


class UnknownModelException(ModelException):
    """A model was requested that is not known."""

    def __init__(self, model_name: str | None):
        detail = f"Unknown model: {model_name}"
        super().__init__(detail)
        self.model_name = model_name


class CheckpointNotFoundException(CheckpointException):
    """Checkpoint not found."""

    def __init__(self, checkpoint_id: str):
        detail = f"Checkpoint {checkpoint_id} not found."
        super().__init__(detail)
        self.checkpoint_id = checkpoint_id


class CheckpointAccessDeniedException(CheckpointException):
    """Access to the checkpoint is denied."""

    def __init__(self, checkpoint_id: str):
        detail = f"Access to checkpoint {checkpoint_id} is denied."
        super().__init__(detail)
        self.checkpoint_id = checkpoint_id


class CheckpointMetadataReadException(CheckpointException):
    """Failed to read checkpoint metadata."""

    def __init__(self, checkpoint_id: str):
        detail = f"Failed to read metadata for checkpoint {checkpoint_id}."
        super().__init__(detail)
        self.checkpoint_id = checkpoint_id


class SequenceConflictException(FutureException):
    """A sequence conflict occurred."""

    def __init__(self, expected: int, got: int):
        detail = f"Sequence conflict: expected {expected}, got {got}."
        super().__init__(detail)
        self.expected = expected
        self.got = got


class MissingSequenceIDException(FutureException):
    """Missing sequence ID in the request."""

    def __init__(self):
        detail = "Missing sequence ID in the request."
        super().__init__(detail)


class FutureNotFoundException(FutureException):
    """Future not found."""

    def __init__(self, request_id: str):
        detail = f"Future with request ID {request_id} not found."
        super().__init__(detail)
        self.request_id = request_id


class SessionNotFoundException(SessionException):
    """Session not found."""

    def __init__(self, session_id: str):
        detail = f"Session {session_id} not found."
        super().__init__(detail)
        self.session_id = session_id


class UserMismatchException(AuthenticationException):
    """User ID does not match the owner of the resource.
    Do not expose user IDs in the detail message for security reasons.
    """

    def __init__(self):
        detail = "You do not have permission to access this resource."
        super().__init__(detail)


class LossFunctionNotFoundException(LossFunctionException):
    """Loss function not found."""

    def __init__(self, loss_function_name: str):
        detail = f"Loss function {loss_function_name} not found."
        super().__init__(detail)
        self.loss_function_name = loss_function_name


class LossFunctionMissingInputException(LossFunctionException):
    def __init__(self, missing_input_name: str):
        detail = f"Missing '{missing_input_name}' in loss_fn_inputs."
        super().__init__(detail)
        self.input_name = missing_input_name


class LossFunctionInputShapeMismatchException(LossFunctionException):
    def __init__(self, shapes: list):
        detail = f"Input tensors must have the same shape. Got shapes: {shapes}"
        super().__init__(detail)
        self.shapes = shapes


class PersistenceException(TuFTException):
    """Base exception for Persistence related errors."""


class ConfigMismatchError(PersistenceException):
    """Raised when current config doesn't match the stored config in Redis.

    This error occurs during server startup when persistence is enabled and
    the configuration has changed since the last run. This can cause data
    corruption when restoring persisted state.
    """

    def __init__(
        self,
        diff: dict[str, dict[str, Any]],
    ):
        self.diff = diff

        # Build detailed diff message
        diff_parts = []
        for field_name, field_diff in diff.items():
            # Handle list fields (added/removed)
            added = field_diff.get("added")
            removed = field_diff.get("removed")
            # Handle scalar fields (current/stored)
            current = field_diff.get("current")
            stored = field_diff.get("stored")

            parts = []
            if added is not None:
                parts.append(f"added: {added}")
            if removed is not None:
                parts.append(f"removed: {removed}")
            if current is not None or stored is not None:
                parts.append(f"current: {current}, stored: {stored}")

            if parts:
                diff_parts.append(f"{field_name} ({', '.join(parts)})")

        diff_str = "; ".join(diff_parts) if diff_parts else "unknown difference"

        message = (
            f"Configuration mismatch detected: {diff_str}.\n"
            "The current configuration does not match the stored configuration in Redis.\n"
            "This can cause data corruption when restoring persisted state.\n\n"
            "Options:\n"
            "  1. Use a different Redis database (change redis_url in config)\n"
            "  2. Use --refresh-persistence to clear existing data and start fresh\n"
            "     (WARNING: This will delete all persisted sessions, training runs, etc.)\n"
            "  3. Restore the original configuration that matches the stored data"
        )
        super().__init__(message)
