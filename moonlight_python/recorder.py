"""Recording support for saving frames as images or video."""

from __future__ import annotations

import logging
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np

from .audio_frame import AudioChunk
from .frame import Frame

log = logging.getLogger(__name__)

# Channel-count -> FFmpeg layout name
_CHANNEL_LAYOUTS = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}

# Video PTS are expressed in 90 kHz ticks (the RTP-native rate, and the
# conventional choice for variable-framerate mp4). At this resolution the
# monotonic clamp in write() is a safety net rather than a routine correction:
# even a burst of frames sharing one capture timestamp costs ~11 us each rather
# than a full frame period, so the video timeline cannot ratchet ahead of real
# time the way it does with a 1/fps time base.
_VIDEO_TIME_BASE = Fraction(1, 90000)
_VIDEO_TICKS_PER_SECOND = 90000

# Largest gap we will bridge with silence in one go (10 s). Anything beyond this
# means the audio timeline has gone wrong rather than that audio was lost, so we
# re-anchor instead of writing minutes of silence.
_MAX_GAP_SECONDS = 10.0


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

    **Synchronisation model.** Both tracks are timestamped from the *host's*
    clock, so they cannot drift relative to one another over a long recording:

    * video PTS come from the capture timestamp passed to :meth:`write`
      (moonlight's ``presentationTimeUs``), in a 90 kHz time base;
    * audio PTS come from ``chunk.frame_index * samples_per_frame`` — a count of
      the Opus packets the host generated. Because that is a *position* rather
      than a running total, audio lost anywhere in the pipeline leaves a gap
      that is filled with silence instead of pulling all later audio earlier.

    The only thing derived from the local clock is the one-time alignment of the
    two epochs, performed by the caller.

    Usage::

        with VideoRecorder("output.mp4", 1920, 1080) as rec:
            rec.write(frame)

        with VideoRecorder("output.mp4", 1920, 1080, audio=True) as rec:
            rec.write(frame, pts_us=...)   # from the video thread
            rec.write_audio(chunk)         # from the audio thread
    """

    def __init__(self, output_path: str | Path, width: int, height: int,
                 fps: int = 30, codec: str = "libx264",
                 audio: bool = False, sample_rate: int = 48000,
                 channels: int = 2, audio_codec: str = "aac",
                 samples_per_frame: int = 0) -> None:
        import av

        self._path = Path(output_path)
        self._container = av.open(str(self._path), mode="w")
        self._stream = self._container.add_stream(codec, rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.codec_context.max_b_frames = 0
        # Timestamp in 90 kHz ticks rather than in 1/fps units. A 1/fps time base
        # cannot represent a frame arriving early, so bursts used to be pushed
        # forward a whole frame period each, stretching the video timeline past
        # wall clock and leaving the audio track short at the end.
        self._stream.codec_context.time_base = _VIDEO_TIME_BASE
        self._stream.codec_context.framerate = Fraction(fps, 1)
        self._stream.time_base = _VIDEO_TIME_BASE
        self._fps = fps
        self._last_pts = -1
        self._last_video_us = 0
        self._pts_clamps = 0
        self._closed = False

        # Serializes all container/encoder access (video + audio threads).
        self._mux_lock = threading.Lock()

        self._has_audio = audio
        if audio:
            self._sample_rate = sample_rate
            self._channels = channels
            # 0 means "infer from the first chunk".
            self._samples_per_frame = samples_per_frame
            self._astream = self._container.add_stream(audio_codec, rate=sample_rate)
            layout = _CHANNEL_LAYOUTS.get(channels, "stereo")
            try:
                self._astream.layout = layout
            except Exception:
                log.warning("Could not set audio layout %r; using the codec default",
                            layout, exc_info=True)
            self._afifo = av.AudioFifo()
            self._audio_pts = 0       # samples pulled out of the FIFO and encoded
            self._audio_written = 0   # samples pushed into the FIFO
            self._audio_epoch_index: int | None = None
            self._last_index = -1
            self._silence_samples = 0
            self._gap_events = 0
            self._audio_time_base = Fraction(1, sample_rate)

    @property
    def has_audio(self) -> bool:
        return self._has_audio

    @property
    def audio_position_us(self) -> int:
        """Length of audio written so far, in microseconds."""
        if not self._has_audio:
            return 0
        return self._audio_written * 1_000_000 // self._sample_rate

    @property
    def last_video_us(self) -> int:
        """Capture timestamp of the most recent video frame, in microseconds."""
        return self._last_video_us

    def write(self, frame: Frame, pts_us: int | None = None) -> None:
        """Encode and write a frame to the video file.

        Args:
            frame: Frame to write.
            pts_us: Capture timestamp in microseconds, relative to the start of
                the recording. When provided, sets the frame PTS for
                variable-framerate output. Dropped frames become real gaps.
        """
        import av

        if self._closed:
            raise RuntimeError("VideoRecorder is closed")

        # Convert BGR to RGB
        rgb = frame.data[:, :, ::-1]
        video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        if pts_us is not None:
            frame_pts = (pts_us * _VIDEO_TICKS_PER_SECOND) // 1_000_000
            if frame_pts <= self._last_pts:
                # Non-monotonic capture timestamp — nudge forward by the minimum
                # representable step rather than reordering or dropping.
                frame_pts = self._last_pts + 1
                self._pts_clamps += 1
            self._last_pts = frame_pts
            self._last_video_us = max(
                self._last_video_us,
                frame_pts * 1_000_000 // _VIDEO_TICKS_PER_SECOND,
            )
            video_frame.pts = frame_pts
            video_frame.time_base = _VIDEO_TIME_BASE
        with self._mux_lock:
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)

    def write_audio(self, chunk: AudioChunk) -> None:
        """Encode and write an audio chunk to the audio track.

        The chunk's position on the timeline is ``frame_index * samples_per_frame``,
        counted from the first chunk of the recording. Any shortfall against that
        position — a packet lost on the network, or one dropped by a full queue —
        is bridged with silence, so the audio track stays aligned with wall clock
        instead of sliding earlier by the amount of audio that went missing.

        Args:
            chunk: AudioChunk with (samples, channels) float32 data.
        """
        if not self._has_audio or self._closed:
            return

        if chunk.num_samples == 0:
            return

        with self._mux_lock:
            if self._closed:
                return

            if self._audio_epoch_index is None:
                self._audio_epoch_index = chunk.frame_index
                self._last_index = chunk.frame_index - 1
                if not self._samples_per_frame:
                    # No negotiated value; packet duration is fixed for the life
                    # of the stream, so the first chunk is a reliable stand-in.
                    self._samples_per_frame = chunk.num_samples

            # A frame_index that doesn't advance means the caller isn't populating
            # it (it defaults to 0). Fall back to plain appending rather than
            # treating every chunk as a duplicate.
            indexed = chunk.frame_index > self._last_index
            self._last_index = max(self._last_index, chunk.frame_index)

            if indexed:
                expected = ((chunk.frame_index - self._audio_epoch_index)
                            * self._samples_per_frame)
                gap = expected - self._audio_written
                # Only react to a whole missing packet; anything smaller is noise
                # in the samples-per-packet estimate.
                if gap >= self._samples_per_frame:
                    self._gap_events += 1
                    max_gap = int(_MAX_GAP_SECONDS * self._sample_rate)
                    if gap > max_gap:
                        log.warning(
                            "Audio gap of %.1fs exceeds the %.0fs limit; re-anchoring "
                            "the audio timeline (recording may be misaligned here)",
                            gap / self._sample_rate, _MAX_GAP_SECONDS,
                        )
                        # Re-anchor so subsequent chunks are measured from here.
                        self._audio_epoch_index += (
                            (gap - max_gap) // self._samples_per_frame
                        )
                        gap = max_gap
                    self._write_silence_locked(gap)

            self._afifo.write(self._to_audio_frame(chunk.data))
            self._audio_written += chunk.num_samples
            self._drain_audio_locked(flush=False)

    def _to_audio_frame(self, data: np.ndarray) -> "object":
        """Build a planar-float AudioFrame from (samples, channels) PCM."""
        import av

        if data.ndim == 1:
            data = data[:, np.newaxis]
        # Build a planar-float frame: (channels, samples)
        planar = np.ascontiguousarray(data.T, dtype=np.float32)
        layout = _CHANNEL_LAYOUTS.get(self._channels, "stereo")
        af = av.AudioFrame.from_ndarray(planar, format="fltp", layout=layout)
        af.sample_rate = self._sample_rate
        return af

    def _write_silence_locked(self, samples: int) -> None:
        """Push ``samples`` of silence into the FIFO. Lock held."""
        if samples <= 0:
            return
        self._silence_samples += samples
        # Chunked so a long gap doesn't allocate one huge array.
        block = self._sample_rate  # 1 s
        remaining = samples
        while remaining > 0:
            n = min(remaining, block)
            silence = np.zeros((n, self._channels), dtype=np.float32)
            self._afifo.write(self._to_audio_frame(silence))
            remaining -= n
        self._audio_written += samples

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

    @property
    def audio_stats(self) -> dict[str, int]:
        """Counters describing how the audio timeline was reconstructed."""
        if not self._has_audio:
            return {"pts_clamps": self._pts_clamps}
        return {
            "samples_written": self._audio_written,
            "silence_samples": self._silence_samples,
            "gap_events": self._gap_events,
            "pts_clamps": self._pts_clamps,
        }

    def _pad_audio_tail_locked(self) -> None:
        """Extend the audio track with silence so it ends with the video track.

        The audio pipeline runs behind video, so even after draining there is
        usually a small deficit. Without this the file ends with video that has
        no audio under it.
        """
        if self._last_video_us <= 0:
            return
        target = self._last_video_us * self._sample_rate // 1_000_000
        pad = target - self._audio_written
        if pad <= 0:
            return
        max_pad = int(_MAX_GAP_SECONDS * self._sample_rate)
        if pad > max_pad:
            log.error("Audio track is %.1fs short of the video track; padding only "
                      "%.0fs. Audio was lost somewhere in the pipeline.",
                      pad / self._sample_rate, _MAX_GAP_SECONDS)
            pad = max_pad
        else:
            log.info("Padding audio tail with %.3fs of silence",
                     pad / self._sample_rate)
        self._write_silence_locked(pad)

    def close(self) -> None:
        """Flush the encoders and close the output file. Must be called."""
        if not self._closed:
            self._closed = True
            with self._mux_lock:
                # Flush video encoder
                for packet in self._stream.encode():
                    self._container.mux(packet)
                # Flush audio: pad to the video duration, then the FIFO remainder,
                # then the encoder.
                if self._has_audio:
                    self._pad_audio_tail_locked()
                    self._drain_audio_locked(flush=True)
                    for packet in self._astream.encode():
                        self._container.mux(packet)
                    if self._silence_samples:
                        log.info(
                            "Audio track: %.2fs total, %.3fs inserted as silence "
                            "across %d gap(s)",
                            self._audio_written / self._sample_rate,
                            self._silence_samples / self._sample_rate,
                            self._gap_events,
                        )
                self._container.close()

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
