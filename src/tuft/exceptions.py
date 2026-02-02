"""Some custom exceptions."""


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

class LossFunctionUnknownMetricReductionException(LossFunctionException):
    def __init__(self, reduction_type: str):
        detail = f"Unknown metric reduction type: {reduction_type}"
        super().__init__(detail)
        self.reduction_type = reduction_type