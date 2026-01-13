"""Some custom exceptions."""


class LLMRPCException(Exception):
    """Base exception for LLM-RPC errors."""

    def __init__(self, detail: str = ""):
        super().__init__(detail)
        self.detail = detail


class ModelException(LLMRPCException):
    """Base exception for Model related errors."""


class CheckpointException(LLMRPCException):
    """Base exception for Checkpoint related errors."""


class FutureException(LLMRPCException):
    """Base exception for Future related errors."""


class SessionException(LLMRPCException):
    """Base exception for Session related errors."""


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


class SessionNotFoundException(SessionException):
    """Session not found."""

    def __init__(self, session_id: str):
        detail = f"Session {session_id} not found."
        super().__init__(detail)
        self.session_id = session_id
