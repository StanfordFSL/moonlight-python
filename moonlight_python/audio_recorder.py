"""Recording support for saving decoded audio to a WAV file."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .audio_frame import AudioChunk


class WavRecorder:
    """Writes decoded PCM audio to a 16-bit WAV file.

    Used as the companion audio track when recording video as a directory of
    image frames. Input chunks are float32 in [-1, 1]; they are converted to
    16-bit signed PCM.

    Usage::

        with WavRecorder("./captures/audio.wav") as rec:
            rec.write(chunk)
    """

    def __init__(self, output_path: str | Path, sample_rate: int = 48000,
                 channels: int = 2) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._channels = channels
        self._wave = wave.open(str(self._path), "wb")
        self._wave.setnchannels(channels)
        self._wave.setsampwidth(2)  # 16-bit
        self._wave.setframerate(sample_rate)
        self._closed = False

    def write(self, chunk: AudioChunk) -> None:
        """Append an audio chunk to the WAV file.

        Args:
            chunk: AudioChunk with (samples, channels) float32 data.
        """
        if self._closed:
            raise RuntimeError("WavRecorder is closed")
        data = chunk.data
        if data.ndim == 1:
            data = data[:, np.newaxis]
        # float32 [-1, 1] -> int16 PCM. (samples, channels) C-contiguous
        # flattens to interleaved samples, which WAV expects.
        pcm = np.clip(data, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        self._wave.writeframes(np.ascontiguousarray(pcm).tobytes())

    def close(self) -> None:
        """Finalize and close the WAV file. Must be called."""
        if not self._closed:
            self._closed = True
            self._wave.close()

    def __enter__(self) -> WavRecorder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
