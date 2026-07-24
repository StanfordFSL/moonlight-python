"""Recording support for saving frames as images or video."""

from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path

import numpy as np

from .audio_frame import AudioChunk
from .frame import Frame

# Channel-count -> FFmpeg layout name
_CHANNEL_LAYOUTS = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}


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

    def close(self) -> None:
        """No-op for API compatibility with VideoRecorder."""

    def __enter__(self) -> ImageRecorder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class VideoRecorder:
    """Encodes frames to a video file via PyAV, optionally with an audio track.

    When ``audio=True`` an AAC audio stream is added to the same container and
    fed via :meth:`write_audio`. Video and audio are typically written from two
    different threads, so all encode/mux operations are guarded by a lock.

    Usage::

        with VideoRecorder("output.mp4", 1920, 1080) as rec:
            rec.write(frame)

        with VideoRecorder("output.mp4", 1920, 1080, audio=True) as rec:
            rec.write(frame)         # from the video thread
            rec.write_audio(chunk)   # from the audio thread
    """

    def __init__(self, output_path: str | Path, width: int, height: int,
                 fps: int = 30, codec: str = "libx264",
                 audio: bool = False, sample_rate: int = 48000,
                 channels: int = 2, audio_codec: str = "aac") -> None:
        import av

        self._path = Path(output_path)
        self._container = av.open(str(self._path), mode="w")
        self._stream = self._container.add_stream(codec, rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.codec_context.max_b_frames = 0
        self._fps = fps
        self._last_pts = -1
        self._closed = False

        # Serializes all container/encoder access (video + audio threads).
        self._mux_lock = threading.Lock()

        self._has_audio = audio
        if audio:
            self._sample_rate = sample_rate
            self._channels = channels
            self._astream = self._container.add_stream(audio_codec, rate=sample_rate)
            layout = _CHANNEL_LAYOUTS.get(channels, "stereo")
            try:
                self._astream.layout = layout
            except Exception:
                pass
            self._afifo = av.AudioFifo()
            self._audio_pts = 0  # running sample count for PTS
            self._audio_time_base = Fraction(1, sample_rate)

    @property
    def has_audio(self) -> bool:
        return self._has_audio

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
            # Convert milliseconds to frame-count units (codec time_base is 1/fps)
            frame_pts = pts * self._fps // 1000
            if frame_pts <= self._last_pts:
                frame_pts = self._last_pts + 1
            self._last_pts = frame_pts
            video_frame.pts = frame_pts
        with self._mux_lock:
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)

    def write_audio(self, chunk: AudioChunk) -> None:
        """Encode and write an audio chunk to the audio track.

        Audio is encoded gaplessly with a running sample-count PTS. Combined
        with the video's wall-clock PTS anchored to the same recording start,
        this keeps the two tracks in sync.

        Args:
            chunk: AudioChunk with (samples, channels) float32 data.
        """
        import av

        if not self._has_audio or self._closed:
            return

        data = chunk.data
        if data.ndim == 1:
            data = data[:, np.newaxis]
        # Build a planar-float frame: (channels, samples)
        planar = np.ascontiguousarray(data.T, dtype=np.float32)
        layout = _CHANNEL_LAYOUTS.get(self._channels, "stereo")
        af = av.AudioFrame.from_ndarray(planar, format="fltp", layout=layout)
        af.sample_rate = self._sample_rate

        with self._mux_lock:
            self._afifo.write(af)
            self._drain_audio_locked(flush=False)

    def _drain_audio_locked(self, flush: bool) -> None:
        """Pull fixed-size frames from the FIFO and encode them. Lock held."""
        frame_size = self._astream.codec_context.frame_size or 1024
        while True:
            frame = self._afifo.read(frame_size, partial=flush)
            if frame is None:
                break
            frame.pts = self._audio_pts
            frame.time_base = self._audio_time_base
            frame.sample_rate = self._sample_rate
            self._audio_pts += frame.samples
            for packet in self._astream.encode(frame):
                self._container.mux(packet)
            if not flush and self._afifo.samples < frame_size:
                break

    def close(self) -> None:
        """Flush the encoders and close the output file. Must be called."""
        if not self._closed:
            self._closed = True
            with self._mux_lock:
                # Flush video encoder
                for packet in self._stream.encode():
                    self._container.mux(packet)
                # Flush audio: remaining FIFO samples, then the encoder
                if self._has_audio:
                    self._drain_audio_locked(flush=True)
                    for packet in self._astream.encode():
                        self._container.mux(packet)
                self._container.close()

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
