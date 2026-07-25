"""Recording support for saving decoded audio to a WAV file."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

from .audio_frame import AudioChunk

log = logging.getLogger(__name__)

# Matches VideoRecorder: never bridge more than this in one go.
_MAX_GAP_SECONDS = 10.0


class WavRecorder:
    """Writes decoded PCM audio to a 16-bit WAV file.

    Used as the companion audio track when recording video as a directory of
    image frames. Input chunks are float32 in [-1, 1]; they are converted to
    16-bit signed PCM.

    Like :class:`~moonlight_python.recorder.VideoRecorder`, each chunk is placed
    at ``frame_index * samples_per_frame`` rather than simply appended, so audio
    lost upstream leaves a gap of silence instead of pulling everything after it
    earlier and desynchronising the file from the image sequence.

    Usage::

        with WavRecorder("./captures/audio.wav") as rec:
            rec.write(chunk)
    """

    def __init__(self, output_path: str | Path, sample_rate: int = 48000,
                 channels: int = 2, samples_per_frame: int = 0) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._channels = channels
        # 0 means "infer from the first chunk".
        self._samples_per_frame = samples_per_frame
        self._wave = wave.open(str(self._path), "wb")
        self._wave.setnchannels(channels)
        self._wave.setsampwidth(2)  # 16-bit
        self._wave.setframerate(sample_rate)
        self._closed = False
        self._epoch_index: int | None = None
        self._last_index = -1
        self._written = 0
        self._silence_samples = 0

    @property
    def silence_samples(self) -> int:
        """Total samples of silence inserted to bridge gaps."""
        return self._silence_samples

    def write(self, chunk: AudioChunk) -> None:
        """Append an audio chunk to the WAV file, bridging any gap before it.

        Args:
            chunk: AudioChunk with (samples, channels) float32 data.
        """
        if self._closed:
            raise RuntimeError("WavRecorder is closed")

        if chunk.num_samples == 0:
            return

        if self._epoch_index is None:
            self._epoch_index = chunk.frame_index
            self._last_index = chunk.frame_index - 1
            if not self._samples_per_frame:
                self._samples_per_frame = chunk.num_samples

        # A frame_index that doesn't advance means the caller isn't populating it
        # (it defaults to 0); fall back to plain appending.
        indexed = chunk.frame_index > self._last_index
        self._last_index = max(self._last_index, chunk.frame_index)

        if indexed:
            expected = (chunk.frame_index - self._epoch_index) * self._samples_per_frame
            gap = expected - self._written
            if gap >= self._samples_per_frame:
                max_gap = int(_MAX_GAP_SECONDS * self._sample_rate)
                if gap > max_gap:
                    log.warning("Audio gap of %.1fs exceeds the %.0fs limit; "
                                "re-anchoring", gap / self._sample_rate,
                                _MAX_GAP_SECONDS)
                    self._epoch_index += (gap - max_gap) // self._samples_per_frame
                    gap = max_gap
                self._write_silence(gap)

        data = chunk.data
        if data.ndim == 1:
            data = data[:, np.newaxis]
        # float32 [-1, 1] -> int16 PCM. (samples, channels) C-contiguous
        # flattens to interleaved samples, which WAV expects.
        pcm = np.clip(data, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        self._wave.writeframes(np.ascontiguousarray(pcm).tobytes())
        self._written += data.shape[0]

    def _write_silence(self, samples: int) -> None:
        """Write ``samples`` of silence, chunked to bound the allocation."""
        if samples <= 0:
            return
        self._silence_samples += samples
        remaining = samples
        while remaining > 0:
            n = min(remaining, self._sample_rate)
            silence = np.zeros((n, self._channels), dtype="<i2")
            self._wave.writeframes(silence.tobytes())
            remaining -= n
        self._written += samples

    def close(self) -> None:
        """Finalize and close the WAV file. Must be called."""
        if not self._closed:
            self._closed = True
            self._wave.close()

    def __enter__(self) -> WavRecorder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
