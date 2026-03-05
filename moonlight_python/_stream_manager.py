"""Persistent shared stream with background pull thread and fan-out to subscribers.

Allows start_stream() to establish a single connection that stream(), record(),
and latest_frame() all tap into, avoiding duplicate RTSP connections.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator

import numpy as np

from .frame import Frame
from .stream import StreamingSession, stream_frames
from .decoder import Decoder
from .exceptions import StreamingError


def _is_real_frame(frame: Frame, threshold: float = 5.0) -> bool:
    """Check if a frame contains real content (not a black frame).

    Args:
        frame: Decoded video frame.
        threshold: Mean pixel value above which the frame is considered real.

    Returns:
        True if the frame is non-black.
    """
    return float(np.mean(frame.data)) > threshold


class FrameSubscription:
    """Per-consumer queue with drop-oldest overflow for frame fan-out."""

    def __init__(self, maxsize: int = 30) -> None:
        self._queue: queue.Queue[Frame | None] = queue.Queue(maxsize=maxsize)
        self._closed = False

    def put(self, frame: Frame) -> None:
        """Called by StreamManager to deliver a frame. Drops oldest on overflow."""
        if self._closed:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest frame to make room
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def get(self, timeout: float | None = None) -> Frame | None:
        """Blocking get. Returns None on timeout or when stream ends."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        """Signal end-of-stream with a None sentinel."""
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Force sentinel in by dropping oldest
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames until the subscription is closed."""
        while True:
            frame = self.get(timeout=1.0)
            if frame is None:
                if self._closed:
                    return
                continue
            yield frame


class StreamManager:
    """Owns a background pull thread and fans out frames to subscribers.

    Blocks on start() until real (non-black) frames are flowing.
    """

    def __init__(self, session: StreamingSession, decoder: Decoder, fps: int) -> None:
        self._session = session
        self._decoder = decoder
        self._fps = fps

        self._running = False
        self._ready_event = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._subscribers: list[FrameSubscription] = []
        self._latest_frame: Frame | None = None

    def start(self, ready_timeout: float = 10.0, black_frame_threshold: float = 5.0) -> None:
        """Start the background pull thread and block until real frames flow.

        Args:
            ready_timeout: Max seconds to wait for the first real frame.
            black_frame_threshold: Mean pixel value threshold for _is_real_frame().

        Raises:
            StreamingError: If timeout expires or the stream errors during startup.
        """
        self._running = True
        self._black_frame_threshold = black_frame_threshold
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        if not self._ready_event.wait(timeout=ready_timeout):
            if self._error is not None:
                raise StreamingError(f"Stream failed during startup: {self._error}")
            raise StreamingError(
                f"Timed out waiting {ready_timeout}s for non-black frames"
            )

        if self._error is not None:
            raise StreamingError(f"Stream failed during startup: {self._error}")

    def stop(self) -> None:
        """Stop the background thread and close all subscriptions."""
        self._running = False
        self._session.wake()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            for sub in self._subscribers:
                sub.close()
            self._subscribers.clear()

    def subscribe(self) -> FrameSubscription:
        """Create and register a new subscription."""
        sub = FrameSubscription()
        with self._lock:
            self._subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: FrameSubscription) -> None:
        """Remove a subscription and close it."""
        sub.close()
        with self._lock:
            try:
                self._subscribers.remove(sub)
            except ValueError:
                pass

    @property
    def latest_frame(self) -> Frame | None:
        """The most recently received frame (atomically updated)."""
        with self._lock:
            return self._latest_frame

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Background thread: pull frames, gate on black frames, fan out."""
        ready = False
        idr_stop = threading.Event()

        def _request_idr_periodically() -> None:
            """Send IDR requests and mouse nudges every 500ms until real frames arrive."""
            while not idr_stop.wait(0.5):
                self._session.request_idr()
                self._session.nudge_mouse()

        idr_thread = threading.Thread(target=_request_idr_periodically, daemon=True)
        self._session.request_idr()
        self._session.nudge_mouse()
        idr_thread.start()

        try:
            for frame in stream_frames(self._session, self._decoder):
                if not self._running:
                    break

                if not ready:
                    if not _is_real_frame(frame, self._black_frame_threshold):
                        continue
                    ready = True
                    idr_stop.set()
                    self._ready_event.set()

                with self._lock:
                    self._latest_frame = frame
                    for sub in self._subscribers:
                        sub.put(frame)
        except Exception as exc:
            self._error = exc
            self._ready_event.set()  # Unblock start() if still waiting
        finally:
            idr_stop.set()
            self._running = False
            with self._lock:
                for sub in self._subscribers:
                    sub.close()
