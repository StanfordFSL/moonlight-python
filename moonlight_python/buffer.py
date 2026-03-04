"""Thread-safe latest-frame buffer for CV pipelines.

Keeps only the most recent frame, dropping older frames so that
consumers always get the freshest data regardless of processing speed.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from .frame import Frame

if TYPE_CHECKING:
    from .stream import StreamingSession, Decoder


class LatestFrameBuffer:
    """Runs the stream in a background thread, keeping only the latest frame.

    Usage::

        with LatestFrameBuffer(session, decoder) as buf:
            buf.start()
            while True:
                frame = buf.get(timeout=1.0)
                if frame is not None:
                    result = my_model(frame.data)
    """

    def __init__(self, session: StreamingSession, decoder: Decoder) -> None:
        self._session = session
        self._decoder = decoder

        self._lock = threading.Lock()
        self._event = threading.Event()  # signals that a frame is available
        self._latest: Frame | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._error: BaseException | None = None

        self._frames_received = 0
        self._frames_dropped = 0

    def start(self) -> None:
        """Launch the background thread that consumes the stream."""
        if self._thread is not None:
            raise RuntimeError("Buffer already started")
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Background thread: pull frames and keep the latest."""
        from .stream import stream_frames

        try:
            for frame in stream_frames(self._session, self._decoder):
                if not self._running:
                    break
                with self._lock:
                    if self._latest is not None:
                        self._frames_dropped += 1
                    self._latest = frame
                    self._frames_received += 1
                self._event.set()
        except Exception as exc:
            self._error = exc
            self._event.set()  # wake up any waiting get()

    def get(self, timeout: float | None = None) -> Frame | None:
        """Get the latest frame, blocking until one is available.

        Args:
            timeout: Max seconds to wait for the first frame.
                     None waits forever, 0 returns immediately.

        Returns:
            The most recent Frame, or None if timeout expired.

        Raises:
            Re-raises any exception from the stream thread.
        """
        if self._error is not None:
            raise self._error

        # Wait for at least one frame
        if not self._event.wait(timeout=timeout):
            return None

        if self._error is not None:
            raise self._error

        with self._lock:
            return self._latest

    def stop(self) -> None:
        """Stop the background thread and clean up."""
        self._running = False
        # Wake up LiWaitForNextVideoFrame so the pull thread can exit
        self._session.wake()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def stats(self) -> dict[str, int]:
        """Return stream statistics."""
        with self._lock:
            return {
                "frames_received": self._frames_received,
                "frames_dropped": self._frames_dropped,
            }

    def __enter__(self) -> LatestFrameBuffer:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
