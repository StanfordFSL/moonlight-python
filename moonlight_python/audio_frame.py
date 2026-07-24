"""AudioChunk dataclass wrapping decoded PCM audio with metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AudioChunk:
    """A chunk of decoded PCM audio from the streaming session.

    Attributes:
        data: Decoded PCM samples as (samples, channels) float32 in [-1.0, 1.0].
        sample_rate: Sample rate in Hz (48000 for Moonlight/Sunshine).
        channels: Number of audio channels.
        timestamp_us: Presentation timestamp in microseconds, on the same
            monotonic clock as video frames (for A/V sync).
        receive_time_us: Wall-clock time the packet was received (microseconds).
        frame_index: Sequential index of the source Opus packet.
    """

    data: np.ndarray
    sample_rate: int = 48000
    channels: int = 2
    timestamp_us: int = 0
    receive_time_us: int = 0
    frame_index: int = 0

    @property
    def num_samples(self) -> int:
        """Number of samples per channel in this chunk."""
        return self.data.shape[0]

    @property
    def duration_us(self) -> int:
        """Duration of this chunk in microseconds."""
        if self.sample_rate <= 0:
            return 0
        return self.num_samples * 1_000_000 // self.sample_rate
