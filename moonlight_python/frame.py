"""Frame dataclass wrapping decoded video data with metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Frame:
    """A decoded video frame with metadata from the streaming session.

    Attributes:
        data: Decoded pixel data as (H, W, 3) uint8 numpy array.
        frame_number: Sequential frame number from the encoder.
        frame_type: 1 for IDR/keyframe, 0 for P-frame.
        timestamp_us: Presentation timestamp in microseconds.
        receive_time_us: Time frame was received (microseconds).
        enqueue_time_us: Time frame was enqueued for decoding (microseconds).
        rtp_timestamp: RTP timestamp from the network stream.
        host_processing_latency_us: Host-side processing latency (microseconds).
    """

    data: np.ndarray
    frame_number: int = 0
    frame_type: int = 0
    timestamp_us: int = 0
    receive_time_us: int = 0
    enqueue_time_us: int = 0
    rtp_timestamp: int = 0
    host_processing_latency_us: int = 0

    @property
    def is_keyframe(self) -> bool:
        """True if this is an IDR/keyframe."""
        return self.frame_type == 1

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the frame data array."""
        return self.data.shape

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self.data.shape[1]
