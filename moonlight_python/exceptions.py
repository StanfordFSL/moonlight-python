"""Exception hierarchy for moonlight-python."""


class MoonlightError(Exception):
    """Base exception for all moonlight-python errors."""


class DiscoveryError(MoonlightError):
    """Failed to discover servers."""


class ConnectionError(MoonlightError):
    """Failed to connect to server."""


class PairingError(MoonlightError):
    """Pairing protocol failure."""


class PairingAlreadyInProgress(PairingError):
    """Server is already in a pairing session."""


class WrongPinError(PairingError):
    """Incorrect PIN entered."""


class HttpResponseError(MoonlightError):
    """Server returned a non-200 status code."""

    def __init__(self, status_code: int, status_message: str):
        self.status_code = status_code
        self.status_message = status_message
        super().__init__(f"HTTP {status_code}: {status_message}")


class StreamingError(MoonlightError):
    """Error during streaming session."""


class StreamStartError(StreamingError):
    """Failed to start streaming session."""

    def __init__(self, stage: int, error_code: int):
        self.stage = stage
        self.error_code = error_code
        super().__init__(f"Stream start failed at stage {stage} (error {error_code})")


class StreamTerminatedError(StreamingError):
    """Stream was terminated unexpectedly."""

    def __init__(self, error_code: int):
        self.error_code = error_code
        super().__init__(f"Stream terminated (error {error_code})")


class StreamNotActiveError(StreamingError):
    """Operation requires an active stream but none is running."""


class DecoderError(MoonlightError):
    """Video decoding error."""
