"""PyAV-based H.264/HEVC/AV1 decoder producing numpy arrays.

Reference: moonlight-qt/app/streaming/video/ffmpeg.cpp

Receives Annex B NAL unit data from moonlight-common-c and decodes
to numpy arrays via PyAV (FFmpeg).
"""

from __future__ import annotations

import av
import numpy as np

from .exceptions import DecoderError

CODEC_NAMES = {
    "h264": "h264",
    "hevc": "hevc",
    "h265": "hevc",
    "av1": "av1",
}


class Decoder:
    """Video decoder using PyAV."""

    def __init__(self, codec: str = "h264", output_format: str = "bgr24"):
        """Initialize the decoder.

        Args:
            codec: Video codec name ("h264", "hevc", "av1")
            output_format: Output pixel format for numpy conversion ("bgr24" or "rgb24")
        """
        codec_name = CODEC_NAMES.get(codec.lower())
        if codec_name is None:
            raise DecoderError(f"Unsupported codec: {codec}")

        valid_formats = {"bgr24", "rgb24"}
        if output_format not in valid_formats:
            raise DecoderError(
                f"Unsupported output_format: {output_format!r}. "
                f"Must be one of: {', '.join(sorted(valid_formats))}"
            )

        self._codec_ctx = av.CodecContext.create(codec_name, "r")
        self._codec_ctx.thread_type = "AUTO"
        self._codec_ctx.thread_count = 0  # auto
        self._output_format = output_format
        self._open = True

    def decode(self, annex_b_data: bytes) -> list[np.ndarray]:
        """Decode Annex B data into numpy frame(s).

        Args:
            annex_b_data: Raw Annex B formatted bitstream data

        Returns:
            List of numpy arrays (H, W, 3) uint8. Usually 0 or 1 frames.
        """
        if not self._open:
            raise DecoderError("Decoder is closed")

        packet = av.Packet(annex_b_data)
        frames: list[np.ndarray] = []
        try:
            decoded = self._codec_ctx.decode(packet)
            for frame in decoded:
                arr = frame.to_ndarray(format=self._output_format)
                frames.append(arr)
        except av.error.InvalidDataError:
            # Corrupted frame, skip
            pass
        return frames

    def close(self) -> None:
        """Flush and close the decoder."""
        if self._open:
            self._open = False

    def __enter__(self) -> "Decoder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
