"""moonlight-python: Python client for Moonlight/Sunshine game streaming.

Receive decoded video frames as numpy arrays for CV research pipelines.

Usage::

    from moonlight_python import MoonlightClient

    client = MoonlightClient()
    servers = client.discover()
    server = client.connect("192.168.1.X")
    client.pair()

    with client.stream(app="Desktop", width=1920, height=1080, fps=30):
        for frame in client.frames():
            # frame.data is numpy array (H, W, 3) uint8 BGR
            result = my_cv_model(frame.data)
"""

from __future__ import annotations

import atexit
import logging
import random
import secrets
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import requests

from .audio_decoder import AudioDecoder
from .audio_frame import AudioChunk
from .audio_recorder import WavRecorder
from .buffer import LatestFrameBuffer
from .config import CODEC_MAP, StreamConfig, VIDEO_FORMAT_H264
from .decoder import Decoder
from .discovery import connect_to_server, discover_servers
from .exceptions import (
    ConnectionError,
    HttpResponseError,
    MoonlightError,
    PairingError,
    StreamingError,
    StreamNotActiveError,
)
from .frame import Frame
from .http_client import NvHTTP
from .identity import Identity
from .pairing import pair as do_pair
from .recorder import ImageRecorder, VideoRecorder
from .server import AppInfo, ServerInfo
from ._stream_manager import StreamManager
from .stream import StreamingSession

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# How long to keep pulling in-flight audio after the video loop ends. The audio
# pipeline (jitter buffer -> raw queue -> decode thread -> subscription queue)
# runs behind video, so stopping both at once truncates the audio track.
_AUDIO_DRAIN_TIMEOUT = 2.0
# Treat the queue as drained after this long with nothing arriving.
_AUDIO_DRAIN_IDLE = 0.25
# Consecutive write failures tolerated before the feeder gives up.
_AUDIO_MAX_ERRORS = 10

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("moonlight-python")
except PackageNotFoundError:  # running from a bare source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "MoonlightClient",
    "ServerInfo",
    "AppInfo",
    "StreamConfig",
    "Frame",
    "AudioChunk",
    "LatestFrameBuffer",
    "ImageRecorder",
    "VideoRecorder",
    "WavRecorder",
    "AudioDecoder",
    "MoonlightError",
    "PairingError",
    "StreamingError",
    "ConnectionError",
    "StreamNotActiveError",
]


class _SharedLatestFrameBuffer:
    """Lightweight adapter that reads latest_frame from a StreamManager.

    Provides the same interface as LatestFrameBuffer for use as a drop-in
    replacement when a shared stream is active.
    """

    def __init__(self, manager: StreamManager) -> None:
        self._manager = manager

    def start(self) -> None:
        """No-op — stream is already running."""

    def stop(self) -> None:
        """No-op — stream lifecycle managed by stop_stream()."""

    def get(self, timeout: float | None = None) -> Frame | None:
        """Poll the shared stream's latest frame.

        Args:
            timeout: Max seconds to wait for a frame.

        Returns:
            The most recent Frame, or None if timeout expired.
        """
        if timeout is None:
            timeout = 30.0
        if timeout == 0:
            return self._manager.latest_frame

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self._manager.latest_frame
            if frame is not None:
                return frame
            time.sleep(0.01)
        return self._manager.latest_frame

    @property
    def stats(self) -> dict[str, int]:
        return {"frames_received": 0, "frames_dropped": 0}

    def __enter__(self) -> _SharedLatestFrameBuffer:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


class _SharedLatestAudioBuffer:
    """Latest-audio-chunk reader over a StreamManager (drops stale chunks)."""

    def __init__(self, manager: StreamManager) -> None:
        self._manager = manager

    def start(self) -> None:
        """No-op — stream is already running."""

    def stop(self) -> None:
        """No-op — stream lifecycle managed by stop_stream()."""

    def get(self, timeout: float | None = None) -> "AudioChunk | None":
        """Poll the shared stream's latest audio chunk.

        Args:
            timeout: Max seconds to wait for a chunk.

        Returns:
            The most recent AudioChunk, or None if timeout expired.
        """
        if timeout is None:
            timeout = 30.0
        if timeout == 0:
            return self._manager.latest_audio

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            chunk = self._manager.latest_audio
            if chunk is not None:
                return chunk
            time.sleep(0.005)
        return self._manager.latest_audio

    def __enter__(self) -> _SharedLatestAudioBuffer:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


class MoonlightClient:
    """High-level client for Moonlight/Sunshine streaming."""

    def __init__(self, config_dir: str | Path = "~/.moonlight-python"):
        self._identity = Identity(config_dir)
        self._http: NvHTTP | None = None
        self._server: ServerInfo | None = None
        self._session: StreamingSession | None = None
        self._decoder: Decoder | None = None
        self._stream_manager: StreamManager | None = None
        self._recording_thread: threading.Thread | None = None
        self._recording_stop: threading.Event | None = None
        self._recording_sub = None
        self._recording_audio_sub = None
        self._recording_error: BaseException | None = None

        # Register cleanup
        atexit.register(self._cleanup)

    def discover(self, timeout: float = 5.0) -> list[ServerInfo]:
        """Discover Moonlight/Sunshine servers via mDNS.

        Args:
            timeout: How long to wait for mDNS responses (seconds)

        Returns:
            List of discovered ServerInfo objects
        """
        return discover_servers(self._identity, timeout)

    def connect(self, host: str, port: int = 47989) -> ServerInfo:
        """Connect to a specific host, auto-pairing if needed.

        If this client has not been paired with the server before, a random
        PIN is generated and printed for the user to enter in the Sunshine
        web UI. The method blocks until pairing completes.

        Args:
            host: IP address or hostname
            port: HTTP port (default 47989)

        Returns:
            ServerInfo for the connected host
        """
        server = connect_to_server(host, self._identity, port)
        self._server = server

        # Set up HTTP client for subsequent operations
        self._http = NvHTTP(
            host, self._identity,
            http_port=port,
            https_port=server.https_port,
            server_cert_pem=server.server_cert_pem,
        )

        # Check if we're paired by trying an authenticated HTTPS request.
        # get_app_list() uses mutual TLS — if the server recognizes our
        # client cert from a previous pairing, this succeeds silently.
        try:
            self._http.get_app_list()
        except (HttpResponseError, requests.RequestException):
            self.pair()

        return server

    def pair(self, server: ServerInfo | None = None,
             pin: str | None = None) -> None:
        """Pair with a server.

        Args:
            server: ServerInfo to pair with (uses last connected if None)
            pin: 4-digit PIN. If None, generates a random PIN and prints it
                 for the user to enter in the Sunshine web UI.
        """
        if server is None:
            server = self._server
        if server is None:
            raise ConnectionError("Not connected to a server")

        if pin is None:
            pin = f"{random.randint(0, 9999):04d}"
            # web_port = server.https_port + 1
            print(f"PIN: {pin}")
            print(f"Enter this PIN in the Sunshine web UI.")

        http = NvHTTP(
            server.address, self._identity,
            http_port=server.http_port,
            https_port=server.https_port,
        )

        cert_pem = do_pair(http, self._identity, pin, server)

        # Update server with pinned cert
        server.server_cert_pem = cert_pem
        server.paired = True
        self._server = server

        # Re-create HTTP client with the pinned cert
        self._http = NvHTTP(
            server.address, self._identity,
            http_port=server.http_port,
            https_port=server.https_port,
            server_cert_pem=cert_pem,
        )

    def apps(self, server: ServerInfo | None = None) -> list[AppInfo]:
        """Get the list of available apps on the server.

        Args:
            server: Server to query (uses last connected if None)

        Returns:
            List of AppInfo objects
        """
        http = self._get_http(server)
        return http.get_app_list()

    def frames(self) -> Iterator[Frame]:
        """Yield decoded video frames from the active stream.

        Requires an active stream via start_stream().

        Yields:
            Frame objects with .data as numpy array (H, W, 3) uint8
        """
        self._require_stream("frames")
        sub = self._stream_manager.subscribe()
        try:
            yield from sub
        finally:
            self._stream_manager.unsubscribe(sub)

    def latest_frame(self) -> _SharedLatestFrameBuffer:
        """Return a latest-frame reader for the active stream.

        Requires an active stream via start_stream().

        Usage::

            with client.latest_frame() as buf:
                while True:
                    frame = buf.get(timeout=1.0)
                    if frame:
                        result = my_model(frame.data)

        Returns:
            _SharedLatestFrameBuffer (use as context manager)
        """
        self._require_stream("latest_frame")
        return _SharedLatestFrameBuffer(self._stream_manager)

    def audio(self) -> Iterator[AudioChunk]:
        """Yield decoded PCM audio chunks from the active stream.

        Requires an active stream started with ``record_audio=True`` (the
        default). If audio capture is disabled, this yields nothing.

        Yields:
            AudioChunk objects with .data as (samples, channels) float32 PCM.
        """
        self._require_stream("audio")
        if not self._stream_manager.has_audio:
            return
        sub = self._stream_manager.subscribe_audio()
        try:
            yield from sub
        finally:
            self._stream_manager.unsubscribe_audio(sub)

    def latest_audio(self) -> _SharedLatestAudioBuffer:
        """Return a latest-audio-chunk reader for the active stream.

        Requires an active stream started with ``record_audio=True``.

        Usage::

            with client.latest_audio() as buf:
                while True:
                    chunk = buf.get(timeout=1.0)
                    if chunk is not None:
                        process(chunk.data)

        Returns:
            _SharedLatestAudioBuffer (use as context manager)
        """
        self._require_stream("latest_audio")
        return _SharedLatestAudioBuffer(self._stream_manager)

    def record(self, output: str | Path,
               duration: float | None = None,
               max_frames: int | None = None,
               with_audio: bool | None = None) -> None:
        """Record from the active stream to a video file or directory of images.

        Requires an active stream via start_stream(). Auto-detects mode
        by the output path:
        - File with video extension (.mp4, .mkv, .avi) → VideoRecorder
        - Directory or path without video extension → ImageRecorder

        Recordings use wall-clock timestamps — dropped frames create real
        time gaps rather than being silently compressed.

        If the active stream was started with ``record_audio=True`` (the
        default), audio is included automatically: baked into video files as a
        synced AAC track, or written to ``<dir>/audio.wav`` for image dirs.

        Args:
            output: Output file path or directory.
            duration: Max recording duration in seconds.
            max_frames: Max number of frames to record.
            with_audio: Override audio inclusion. None (default) includes audio
                iff the stream captured it; False disables it explicitly.

        At least one of duration or max_frames must be provided.
        Use start_recording() / stop_recording() for open-ended recording.
        """
        if duration is None and max_frames is None:
            raise ValueError(
                "record() requires duration or max_frames. "
                "Use start_recording() / stop_recording() for open-ended recording."
            )
        self._require_stream("record")
        fps = self._stream_manager.fps

        output_path = Path(output)
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        is_video = output_path.suffix.lower() in video_extensions

        use_audio = self._resolve_with_audio(with_audio)
        first_frame = self._stream_manager.latest_frame
        sub = self._stream_manager.subscribe()
        audio_sub = self._stream_manager.subscribe_audio() if use_audio else None
        try:
            self._record_from_frames(
                sub, output_path, is_video, fps,
                duration, max_frames, first_frame=first_frame,
                audio_sub=audio_sub,
                sample_rate=self._stream_manager.audio_sample_rate,
                channels=self._stream_manager.audio_channels,
                samples_per_frame=self._stream_manager.audio_samples_per_frame,
            )
        finally:
            self._stream_manager.unsubscribe(sub)
            if audio_sub is not None:
                self._stream_manager.unsubscribe_audio(audio_sub)

    def _record_from_frames(
        self,
        frames: Iterator[Frame],
        output_path: Path,
        is_video: bool,
        fps: int,
        duration: float | None,
        max_frames: int | None,
        first_frame: Frame | None = None,
        stop_event: threading.Event | None = None,
        audio_sub=None,
        sample_rate: int = 48000,
        channels: int = 2,
        samples_per_frame: int = 0,
    ) -> None:
        """Record frames from any iterator to the given output path.

        When ``audio_sub`` is provided, a companion feeder thread drains
        decoded audio chunks in parallel: into the video container as an AAC
        track (video mode), or into ``<dir>/audio.wav`` (image mode).

        Video is timestamped from each frame's capture time and audio from the
        host's Opus packet count, so the two tracks share the host's clock and
        cannot drift apart. When the video loop ends, the feeder is given a
        bounded window to flush audio that is still in flight rather than being
        killed outright.
        """
        count = 0
        start_time = time.perf_counter()
        video_epoch_us: int | None = None
        recorder = None
        wav = None
        audio_stop = threading.Event()   # hard kill
        audio_drain = threading.Event()  # video finished; flush what is left
        audio_thread: threading.Thread | None = None
        use_audio = audio_sub is not None

        def _open_recorder(frame: Frame):
            if is_video:
                h, w = frame.data.shape[:2]
                return VideoRecorder(output_path, w, h, fps, audio=use_audio,
                                     sample_rate=sample_rate, channels=channels,
                                     samples_per_frame=samples_per_frame)
            return ImageRecorder(output_path)

        def _audio_feed(target, is_wav: bool) -> None:
            errors = 0
            idle_since: float | None = None
            deadline: float | None = None
            while not audio_stop.is_set():
                if audio_drain.is_set() and deadline is None:
                    deadline = time.perf_counter() + _AUDIO_DRAIN_TIMEOUT
                chunk = audio_sub.get(timeout=0.1)
                if chunk is None:
                    if not audio_drain.is_set():
                        continue
                    if getattr(audio_sub, "is_closed", False):
                        return
                    now = time.perf_counter()
                    if idle_since is None:
                        idle_since = now
                    elif now - idle_since >= _AUDIO_DRAIN_IDLE or now > deadline:
                        return
                    continue
                idle_since = None
                try:
                    if is_wav:
                        target.write(chunk)
                    else:
                        target.write_audio(chunk)
                    errors = 0
                except Exception:
                    errors += 1
                    log.exception("Audio write failed at frame_index=%s",
                                  getattr(chunk, "frame_index", "?"))
                    if errors >= _AUDIO_MAX_ERRORS:
                        log.error("Giving up on audio after %d consecutive failures",
                                  errors)
                        return
                # Once the video has stopped, there is no point pulling audio
                # past the end of the video timeline.
                if (audio_drain.is_set() and not is_wav
                        and target.audio_position_us >= target.last_video_us):
                    return

        def _start_audio_feeder() -> None:
            nonlocal audio_thread, wav
            if not use_audio or audio_thread is not None:
                return
            # Discard anything the subscription buffered before the recording
            # started — up to a second of stale audio that would otherwise be
            # laid down at PTS 0 and make audio lead video for the whole file.
            while audio_sub.get(timeout=0.0) is not None:
                pass
            if is_video:
                audio_thread = threading.Thread(
                    target=_audio_feed, args=(recorder, False), daemon=True)
            else:
                wav = WavRecorder(output_path / "audio.wav", sample_rate, channels,
                                  samples_per_frame=samples_per_frame)
                audio_thread = threading.Thread(
                    target=_audio_feed, args=(wav, True), daemon=True)
            audio_thread.start()

        def _video_pts_us(frame: Frame) -> int:
            """Capture-relative PTS in microseconds, from the host's clock.

            ``Frame.timestamp_us`` is moonlight's ``presentationTimeUs``, whose
            epoch is the first captured frame of the session. Using it instead of
            the time this thread happened to dequeue the frame keeps encoder
            backlog from showing up as a stretched video timeline.
            """
            nonlocal video_epoch_us
            ts = frame.timestamp_us
            if video_epoch_us is None:
                video_epoch_us = ts
                return 0
            if ts > video_epoch_us:
                return ts - video_epoch_us
            # No usable host PTS for this frame (moonlight leaves it zero when it
            # has nothing to synthesize from) — fall back to the local clock.
            return int((time.perf_counter() - start_time) * 1_000_000)

        try:
            # Write the latest frame as the first frame (avoids black start)
            if first_frame is not None:
                recorder = _open_recorder(first_frame)
                if is_video:
                    # Anchor the video timeline here; later frames are measured
                    # against this frame's capture time.
                    video_epoch_us = first_frame.timestamp_us
                    recorder.write(first_frame, pts_us=0)
                else:
                    recorder.write(first_frame)
                count += 1
                _start_audio_feeder()

            for frame in frames:
                if stop_event is not None and stop_event.is_set():
                    break
                if recorder is None:
                    recorder = _open_recorder(frame)
                    _start_audio_feeder()
                if is_video:
                    recorder.write(frame, pts_us=_video_pts_us(frame))
                else:
                    recorder.write(frame)
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
                if duration is not None and (time.perf_counter() - start_time) >= duration:
                    break
        finally:
            # Let the feeder flush in-flight audio before tearing down, so the
            # recording does not end with video that has no audio under it.
            if audio_thread is not None:
                audio_drain.set()
                audio_thread.join(timeout=_AUDIO_DRAIN_TIMEOUT + 1.0)
                if audio_thread.is_alive():
                    log.warning("Audio feeder did not drain within %.1fs; "
                                "the tail will be padded with silence",
                                _AUDIO_DRAIN_TIMEOUT)
                    audio_stop.set()
                    audio_thread.join(timeout=2.0)
            if wav is not None:
                wav.close()
            if recorder is not None:
                recorder.close()

    def start_stream(self, app: str = "Desktop", width: int = 1920,
                     height: int = 1080, fps: int = 30,
                     bitrate_kbps: int = 10000, codec: str = "h264",
                     output_format: str = "bgr24",
                     ready_timeout: float = 10.0,
                     black_frame_threshold: float = 5.0,
                     record_audio: bool = True) -> None:
        """Start a persistent shared stream. Blocks until real frames flow.

        After calling this, stream(), record(), and latest_frame() will all
        tap into this shared connection instead of creating their own.

        Args:
            app: Application name to stream (default "Desktop")
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            bitrate_kbps: Bitrate in kbps
            codec: Video codec ("h264", "hevc", "av1")
            output_format: Pixel format — "bgr24" (default) or "rgb24"
            ready_timeout: Max seconds to wait for non-black frames
            black_frame_threshold: Mean pixel value threshold for real frames
            record_audio: Capture the audio stream (default True). Enables
                audio() / latest_audio() and audio in record().
        """
        if self._stream_manager is not None and self._stream_manager.is_running:
            raise StreamingError("Stream already active. Call stop_stream() first.")

        session, decoder = self._setup_stream(
            app, width, height, fps, bitrate_kbps, codec, output_format,
            record_audio=record_audio,
        )

        manager = StreamManager(session, decoder, fps, capture_audio=record_audio)
        try:
            manager.start(ready_timeout=ready_timeout,
                          black_frame_threshold=black_frame_threshold)
        except Exception:
            self._stop_streaming()
            raise

        self._stream_manager = manager

    def capture(self, output: str | Path) -> Path:
        """Capture a single screenshot from the active stream.

        Grabs the latest frame immediately (no waiting for a new one).
        Requires an active stream via start_stream().

        Args:
            output: Output image file path (e.g. "screenshot.png").

        Returns:
            Path to the saved image file.
        """
        self._require_stream("capture")
        frame = self._stream_manager.latest_frame
        if frame is None:
            raise StreamingError("No frame available yet")

        from PIL import Image

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb = frame.data[:, :, ::-1]
        Image.fromarray(rgb).save(output_path)
        return output_path

    def start_recording(self, output: str | Path,
                         with_audio: bool | None = None) -> None:
        """Start background recording to a video file or image directory.

        Requires an active stream via start_stream(). Recording runs in
        a background thread until stop_recording() is called. Audio is included
        automatically when the stream captured it (see record()).

        Args:
            output: Output file path (.mp4, .mkv, etc.) or directory for images.
            with_audio: Override audio inclusion. None (default) includes audio
                iff the stream captured it; False disables it explicitly.
        """
        self._require_stream("start_recording")
        if self._recording_thread is not None:
            raise StreamingError(
                "Recording already in progress. Call stop_recording() first."
            )

        fps = self._stream_manager.fps
        output_path = Path(output)
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        is_video = output_path.suffix.lower() in video_extensions

        use_audio = self._resolve_with_audio(with_audio)
        first_frame = self._stream_manager.latest_frame
        sub = self._stream_manager.subscribe()
        audio_sub = self._stream_manager.subscribe_audio() if use_audio else None
        sample_rate = self._stream_manager.audio_sample_rate
        channels = self._stream_manager.audio_channels
        samples_per_frame = self._stream_manager.audio_samples_per_frame
        self._recording_stop = threading.Event()
        self._recording_sub = sub
        self._recording_audio_sub = audio_sub
        self._recording_error: BaseException | None = None

        def _record() -> None:
            try:
                self._record_from_frames(
                    sub, output_path, is_video, fps,
                    duration=None, max_frames=None,
                    first_frame=first_frame,
                    stop_event=self._recording_stop,
                    audio_sub=audio_sub,
                    sample_rate=sample_rate,
                    channels=channels,
                    samples_per_frame=samples_per_frame,
                )
            except Exception as exc:
                self._recording_error = exc

        self._recording_thread = threading.Thread(target=_record, daemon=True)
        self._recording_thread.start()

    def stop_recording(self) -> None:
        """Stop the background recording started by start_recording()."""
        if self._recording_thread is None:
            return
        self._recording_stop.set()
        self._recording_thread.join(timeout=10.0)
        if self._recording_sub is not None:
            self._stream_manager.unsubscribe(self._recording_sub)
            self._recording_sub = None
        if self._recording_audio_sub is not None:
            self._stream_manager.unsubscribe_audio(self._recording_audio_sub)
            self._recording_audio_sub = None
        self._recording_thread = None
        if self._recording_error is not None:
            err = self._recording_error
            self._recording_error = None
            raise StreamingError(f"Recording failed: {err}") from err

    def stop_stream(self) -> None:
        """Stop the persistent shared stream and clean up."""
        self.stop_recording()
        if self._stream_manager is not None:
            self._stream_manager.stop()
            self._stream_manager = None
        self._stop_streaming()

    def quit_app(self) -> None:
        """Quit the currently running app on the server."""
        http = self._get_http()
        http.quit_app()

    def _setup_stream(self, app: str, width: int, height: int, fps: int,
                      bitrate_kbps: int, codec: str,
                      output_format: str = "bgr24",
                      record_audio: bool = True,
                      ) -> tuple[StreamingSession, Decoder]:
        """Set up a streaming session and decoder.

        Returns:
            (session, decoder) tuple ready for stream_frames().
        """
        http = self._get_http()

        if self._server is None:
            raise ConnectionError("Not connected to a server")

        # Find the app ID
        apps = http.get_app_list()
        app_info = None
        for a in apps:
            if a.name.lower() == app.lower():
                app_info = a
                break
        if app_info is None:
            available = [a.name for a in apps]
            raise MoonlightError(
                f"App '{app}' not found. Available: {available}"
            )

        # Set up stream config
        video_format = CODEC_MAP.get(codec.lower(), VIDEO_FORMAT_H264)
        config = StreamConfig(
            width=width,
            height=height,
            fps=fps,
            bitrate_kbps=bitrate_kbps,
            supported_video_formats=video_format,
            codec=codec,
        )

        # Generate random AES key/IV for remote input encryption
        ri_aes_key = secrets.token_bytes(16)
        ri_aes_iv = secrets.token_bytes(16)

        # Create streaming session to get launch query params
        session = StreamingSession()
        launch_params = session.get_launch_query_params()

        # Check if the app is already running
        server_info_xml = http.get_server_info(use_https=True)
        current_game = http.parse_server_info(server_info_xml).current_game

        if current_game != 0:
            if current_game == app_info.id:
                # Resume existing session
                rtsp_url = http.resume_app(
                    ri_aes_key, ri_aes_iv,
                    config.surroundaudioinfo,
                    launch_params,
                )
            else:
                # Different app running — quit it first, then launch
                http.quit_app()
                rtsp_url = http.launch_app(
                    app_info.id, width, height, fps,
                    bitrate_kbps, video_format,
                    ri_aes_key, ri_aes_iv,
                    config.surroundaudioinfo,
                    config.sops, config.local_audio,
                    launch_params,
                )
        else:
            # Launch the app
            rtsp_url = http.launch_app(
                app_info.id, width, height, fps,
                bitrate_kbps, video_format,
                ri_aes_key, ri_aes_iv,
                config.surroundaudioinfo,
                config.sops, config.local_audio,
                launch_params,
            )

        # Start the native streaming connection
        session.start(
            address=self._server.address,
            app_version=self._server.app_version,
            gfe_version=self._server.gfe_version,
            server_codec_mode_support=self._server.server_codec_mode_support,
            rtsp_session_url=rtsp_url or "",
            config=config,
            ri_aes_key=ri_aes_key,
            ri_aes_iv=ri_aes_iv,
            capture_audio=record_audio,
        )

        self._session = session

        # Set up decoder
        decoder = Decoder(codec=codec, output_format=output_format)
        self._decoder = decoder

        return session, decoder

    def stream(self, app: str = "Desktop", width: int = 1920,
               height: int = 1080, fps: int = 30,
               bitrate_kbps: int = 10000, codec: str = "h264",
               output_format: str = "bgr24",
               ready_timeout: float = 10.0,
               black_frame_threshold: float = 5.0,
               record_audio: bool = True):
        """Context manager for start_stream() / stop_stream().

        Usage::

            with client.stream(app="Desktop", width=1920, height=1080, fps=30):
                client.capture("shot.png")
                client.record("clip.mp4", duration=5)

        Args:
            Same as start_stream().
        """
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            self.start_stream(
                app=app, width=width, height=height, fps=fps,
                bitrate_kbps=bitrate_kbps, codec=codec,
                output_format=output_format,
                ready_timeout=ready_timeout,
                black_frame_threshold=black_frame_threshold,
                record_audio=record_audio,
            )
            try:
                yield self
            finally:
                self.stop_stream()

        return _ctx()

    def _require_stream(self, method: str) -> None:
        """Raise StreamNotActiveError if no shared stream is active."""
        if self._stream_manager is None or not self._stream_manager.is_running:
            raise StreamNotActiveError(
                f"{method}() requires an active stream. Call start_stream() first."
            )

    def _resolve_with_audio(self, with_audio: bool | None) -> bool:
        """Decide whether a recording should include audio.

        Audio can only be recorded if the stream captured it. When with_audio
        is None, include audio iff available; otherwise honor the request but
        still clamp to availability.
        """
        available = self._stream_manager.has_audio
        if with_audio is None:
            return available
        return with_audio and available

    def _get_http(self, server: ServerInfo | None = None) -> NvHTTP:
        if server is not None:
            return NvHTTP(
                server.address, self._identity,
                http_port=server.http_port,
                https_port=server.https_port,
                server_cert_pem=server.server_cert_pem,
            )
        if self._http is not None:
            return self._http
        raise ConnectionError("Not connected to a server. Call connect() first.")

    def _stop_streaming(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None

    def _cleanup(self) -> None:
        self.stop_stream()
