"""Recording support for saving frames as images or video."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from .frame import Frame


class ImageRecorder:
    """Saves frames as timestamped image files.

    Usage::

        recorder = ImageRecorder("./captures/")
        recorder.write(frame)  # saves captures/frame_000001.png
    """

    def __init__(self, output_dir: str | Path, format: str = "png",
                 prefix: str = "frame") -> None:
        self._output_dir = Path(output_dir)
        self._format = format
        self._prefix = prefix
        self._count = 0
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame: Frame) -> Path:
        """Save a frame as an image file.

        Args:
            frame: Frame to save.

        Returns:
            Path to the saved image file.
        """
        from PIL import Image

        self._count += 1
        filename = f"{self._prefix}_{self._count:06d}.{self._format}"
        path = self._output_dir / filename

        # Convert BGR to RGB for saving
        rgb = frame.data[:, :, ::-1]
        Image.fromarray(rgb).save(path)
        return path

    def __enter__(self) -> ImageRecorder:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class VideoRecorder:
    """Encodes frames to a video file via PyAV.

    Usage::

        with VideoRecorder("output.mp4", 1920, 1080) as rec:
            rec.write(frame)
    """

    def __init__(self, output_path: str | Path, width: int, height: int,
                 fps: int = 30, codec: str = "libx264") -> None:
        import av

        self._path = Path(output_path)
        self._container = av.open(str(self._path), mode="w")
        self._stream = self._container.add_stream(codec, rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._codec_tb = self._stream.codec_context.time_base
        self._closed = False

    def write(self, frame: Frame, pts: int | None = None) -> None:
        """Encode and write a frame to the video file.

        Args:
            frame: Frame to write.
            pts: Presentation timestamp in time_base units (milliseconds).
                 When provided, sets the frame PTS for variable-framerate output.
        """
        import av

        if self._closed:
            raise RuntimeError("VideoRecorder is closed")

        # Convert BGR to RGB
        rgb = frame.data[:, :, ::-1]
        video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        if pts is not None:
            # Convert milliseconds to codec time_base units
            video_frame.pts = int(pts * Fraction(1, 1000) / self._codec_tb)
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def close(self) -> None:
        """Flush the encoder and close the output file. Must be called."""
        if not self._closed:
            self._closed = True
            # Flush encoder
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
