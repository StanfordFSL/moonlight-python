"""moonlight-python: Python client for Moonlight/Sunshine game streaming.

Receive decoded video frames as numpy arrays for CV research pipelines.

Usage::

    from moonlight_python import MoonlightClient

    client = MoonlightClient()
    servers = client.discover()
    server = client.connect("192.168.1.X")
    client.pair()

    for frame in client.stream(app="Desktop", width=1920, height=1080, fps=30):
        # frame.data is numpy array (H, W, 3) uint8 BGR
        result = my_cv_model(frame.data)
"""

from __future__ import annotations

import atexit
import random
import secrets
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from .buffer import LatestFrameBuffer
from .config import CODEC_MAP, StreamConfig, VIDEO_FORMAT_H264
from .decoder import Decoder
from .discovery import connect_to_server, discover_servers
from .exceptions import (
    ConnectionError,
    MoonlightError,
    PairingError,
    StreamingError,
)
from .frame import Frame
from .http_client import NvHTTP
from .identity import Identity
from .pairing import pair as do_pair
from .recorder import ImageRecorder, VideoRecorder
from .server import AppInfo, ServerInfo
from .stream import StreamingSession, stream_frames

__all__ = [
    "MoonlightClient",
    "ServerInfo",
    "AppInfo",
    "StreamConfig",
    "Frame",
    "LatestFrameBuffer",
    "ImageRecorder",
    "VideoRecorder",
    "MoonlightError",
    "PairingError",
    "StreamingError",
    "ConnectionError",
]


class MoonlightClient:
    """High-level client for Moonlight/Sunshine streaming."""

    def __init__(self, config_dir: str | Path = "~/.moonlight-python"):
        self._identity = Identity(config_dir)
        self._http: NvHTTP | None = None
        self._server: ServerInfo | None = None
        self._session: StreamingSession | None = None
        self._decoder: Decoder | None = None

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
        """Connect to a specific host and fetch server info.

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
            print(f"PIN: {pin}")
            print(f"Enter this PIN in the Sunshine web UI at "
                  f"https://{server.address}:47990")

        http = NvHTTP(
            server.address, self._identity,
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

    def stream(self, app: str = "Desktop", width: int = 1920, height: int = 1080,
               fps: int = 30, bitrate_kbps: int = 10000, codec: str = "h264",
               output_format: str = "bgr24") -> Iterator[Frame]:
        """Stream an application and yield decoded video frames.

        Args:
            app: Application name to stream (default "Desktop")
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            bitrate_kbps: Bitrate in kbps
            codec: Video codec ("h264", "hevc", "av1")
            output_format: Pixel format — "bgr24" (default) or "rgb24"

        Yields:
            Frame objects with .data as numpy array (H, W, 3) uint8
        """
        session, decoder = self._setup_stream(
            app, width, height, fps, bitrate_kbps, codec, output_format,
        )

        try:
            yield from stream_frames(session, decoder)
        finally:
            self._stop_streaming()

    def latest_frame(self, app: str = "Desktop", width: int = 1920,
                     height: int = 1080, fps: int = 30,
                     bitrate_kbps: int = 10000, codec: str = "h264",
                     output_format: str = "bgr24") -> LatestFrameBuffer:
        """Start streaming and return a LatestFrameBuffer context manager.

        The buffer runs the stream in a background thread and always holds
        the most recent frame, dropping older ones. Ideal for CV pipelines
        that process at variable rates.

        Usage::

            with client.latest_frame(app="Desktop") as buf:
                while True:
                    frame = buf.get(timeout=1.0)
                    if frame:
                        result = my_model(frame.data)

        Args:
            app: Application name to stream (default "Desktop")
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            bitrate_kbps: Bitrate in kbps
            codec: Video codec ("h264", "hevc", "av1")
            output_format: Pixel format — "bgr24" (default) or "rgb24"

        Returns:
            LatestFrameBuffer (use as context manager)
        """
        session, decoder = self._setup_stream(
            app, width, height, fps, bitrate_kbps, codec, output_format,
        )
        return LatestFrameBuffer(session, decoder)

    def record(self, output: str | Path, app: str = "Desktop",
               width: int = 1920, height: int = 1080, fps: int = 30,
               bitrate_kbps: int = 10000, codec: str = "h264",
               duration: float | None = None,
               max_frames: int | None = None) -> None:
        """Record a stream to a video file or directory of images.

        Auto-detects mode by the output path:
        - File with video extension (.mp4, .mkv, .avi) → VideoRecorder
        - Directory or path without video extension → ImageRecorder

        Args:
            output: Output file path or directory.
            app: Application name to stream.
            width: Video width in pixels.
            height: Video height in pixels.
            fps: Frames per second.
            bitrate_kbps: Bitrate in kbps.
            codec: Video codec ("h264", "hevc", "av1").
            duration: Max recording duration in seconds (None = unlimited).
            max_frames: Max number of frames to record (None = unlimited).
        """
        output_path = Path(output)
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        is_video = output_path.suffix.lower() in video_extensions

        count = 0
        start_time: float | None = None

        if is_video:
            with VideoRecorder(output_path, width, height, fps) as recorder:
                for frame in self.stream(app, width, height, fps,
                                         bitrate_kbps, codec):
                    if start_time is None:
                        start_time = time.monotonic()
                    recorder.write(frame)
                    count += 1
                    if max_frames is not None and count >= max_frames:
                        break
                    if duration is not None and (time.monotonic() - start_time) >= duration:
                        break
        else:
            with ImageRecorder(output_path) as recorder:
                for frame in self.stream(app, width, height, fps,
                                         bitrate_kbps, codec):
                    if start_time is None:
                        start_time = time.monotonic()
                    recorder.write(frame)
                    count += 1
                    if max_frames is not None and count >= max_frames:
                        break
                    if duration is not None and (time.monotonic() - start_time) >= duration:
                        break

    def quit_app(self) -> None:
        """Quit the currently running app on the server."""
        http = self._get_http()
        http.quit_app()

    def _setup_stream(self, app: str, width: int, height: int, fps: int,
                      bitrate_kbps: int, codec: str,
                      output_format: str = "bgr24",
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
        )

        self._session = session

        # Set up decoder
        decoder = Decoder(codec=codec, output_format=output_format)
        self._decoder = decoder

        return session, decoder

    def _get_http(self, server: ServerInfo | None = None) -> NvHTTP:
        if server is not None:
            return NvHTTP(
                server.address, self._identity,
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
        self._stop_streaming()
