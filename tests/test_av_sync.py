"""Tests for audio/video synchronisation in recorded containers.

These cover the failure mode where the audio track ends short of the video
track: audio PTS used to be a running sample count, so anything lost in the
pipeline shortened the audio timeline permanently instead of leaving a gap.
"""

import queue
from pathlib import Path

import av
import numpy as np
import pytest

from moonlight_python.audio_frame import AudioChunk
from moonlight_python.frame import Frame
from moonlight_python.recorder import VideoRecorder
from moonlight_python.stream import StreamingSession

SAMPLE_RATE = 48000
SPF = 240  # samples per Opus packet (5 ms @ 48 kHz), moonlight's default


def _frame(width: int = 160, height: int = 120) -> Frame:
    data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Frame(data=data)


def _chunk(frame_index: int, samples: int = SPF, channels: int = 2) -> AudioChunk:
    data = (0.1 * np.random.randn(samples, channels)).astype(np.float32)
    return AudioChunk(data=data, sample_rate=SAMPLE_RATE, channels=channels,
                      frame_index=frame_index)


def _durations(path: Path) -> tuple[float, float]:
    """Return (video_duration, audio_duration) in seconds, decoded from the file."""
    container = av.open(str(path))
    try:
        vstream = next(s for s in container.streams if s.type == "video")
        astream = next(s for s in container.streams if s.type == "audio")
        vdur = float(vstream.duration * vstream.time_base) if vstream.duration else 0.0
        n_samples = sum(f.samples for f in container.decode(astream))
        return vdur, n_samples / astream.codec_context.sample_rate
    finally:
        container.close()


def _record(path: Path, n_packets: int, *, skip=(), fps: int = 30,
            frame_interval_us: int = 33_333, stop_audio_at: int | None = None):
    """Write matched A/V covering ``n_packets`` Opus packets of wall time."""
    duration_us = n_packets * SPF * 1_000_000 // SAMPLE_RATE
    n_frames = duration_us // frame_interval_us + 1
    rec = VideoRecorder(path, 160, 120, fps=fps, audio=True,
                        sample_rate=SAMPLE_RATE, channels=2,
                        samples_per_frame=SPF)
    try:
        for i in range(n_frames):
            rec.write(_frame(), pts_us=i * frame_interval_us)
        for i in range(n_packets):
            if i in skip:
                continue
            if stop_audio_at is not None and i >= stop_audio_at:
                break
            rec.write_audio(_chunk(i))
    finally:
        rec.close()
    # Read after close() so tail padding is included.
    return dict(rec.audio_stats)


class TestTrackDurations:
    def test_durations_agree(self, tmp_path: Path):
        """The headline invariant: both tracks span the same wall time."""
        out = tmp_path / "sync.mp4"
        _record(out, n_packets=1000)  # 5 s
        vdur, adur = _durations(out)
        assert abs(vdur - adur) < 0.15, f"video={vdur:.3f}s audio={adur:.3f}s"

    def test_variable_framerate_duration(self, tmp_path: Path):
        """Irregular frame arrival must not stretch the video timeline.

        The old 1/fps time base forced every colliding PTS a whole frame period
        into the future, so a stream delivering faster than the nominal rate
        ended up longer than wall clock with a silent tail.
        """
        out = tmp_path / "vfr.mp4"
        pts_us = 0
        with VideoRecorder(out, 160, 120, fps=30, audio=True,
                           sample_rate=SAMPLE_RATE, channels=2,
                           samples_per_frame=SPF) as rec:
            i = 0
            while pts_us < 3_000_000:
                rec.write(_frame(), pts_us=pts_us)
                # Alternate 20 ms / 50 ms — averages 30 fps but is never regular,
                # and the 20 ms steps are faster than the nominal frame period.
                pts_us += 20_000 if i % 2 == 0 else 50_000
                i += 1
            for k in range(3_000_000 * SAMPLE_RATE // (SPF * 1_000_000)):
                rec.write_audio(_chunk(k))
        vdur, adur = _durations(out)
        assert abs(vdur - 3.0) < 0.15, f"video timeline stretched to {vdur:.3f}s"
        assert abs(vdur - adur) < 0.15

    def test_pts_survives_burst(self, tmp_path: Path):
        """A burst of frames sharing a timestamp must not push the timeline far.

        At 90 kHz each clamped frame costs ~11 us; under the old 1/fps time base
        100 clamped frames cost 100/30 = 3.3 s.
        """
        out = tmp_path / "burst.mp4"
        with VideoRecorder(out, 160, 120, fps=30) as rec:
            for _ in range(100):
                rec.write(_frame(), pts_us=1_000_000)
        container = av.open(str(out))
        try:
            vstream = container.streams.video[0]
            overshoot = float(vstream.duration * vstream.time_base) - 1.0
        finally:
            container.close()
        assert overshoot < 0.05, f"burst pushed the timeline {overshoot:.3f}s"


class TestGapHandling:
    def test_dropped_packets_preserve_timeline(self, tmp_path: Path):
        """Losing a second of audio must leave a hole, not shorten the track."""
        out = tmp_path / "gap.mp4"
        skip = set(range(200, 400))  # 200 packets x 5 ms = 1.0 s
        stats = _record(out, n_packets=1000, skip=skip)
        vdur, adur = _durations(out)
        assert abs(vdur - adur) < 0.15, f"video={vdur:.3f}s audio={adur:.3f}s"
        # The hole is filled exactly; close() may add a little more for the tail.
        assert stats["silence_samples"] >= len(skip) * SPF
        assert stats["gap_events"] == 1

    def test_scattered_loss_does_not_accumulate(self, tmp_path: Path):
        """Many small losses must not compound into drift."""
        out = tmp_path / "scattered.mp4"
        skip = set(range(0, 1000, 10))  # 10% loss, spread out
        _record(out, n_packets=1000, skip=skip)
        vdur, adur = _durations(out)
        assert abs(vdur - adur) < 0.15, f"video={vdur:.3f}s audio={adur:.3f}s"

    def test_tail_padding(self, tmp_path: Path):
        """Audio cut off early is padded so the tracks still end together.

        This is the reported symptom: in-flight audio discarded at shutdown.
        """
        out = tmp_path / "tail.mp4"
        stats = _record(out, n_packets=1000, stop_audio_at=600)  # audio ends 2 s early
        vdur, adur = _durations(out)
        assert abs(vdur - adur) < 0.15, f"video={vdur:.3f}s audio={adur:.3f}s"
        assert stats["silence_samples"] >= 1.9 * SAMPLE_RATE

    def test_no_gaps_when_nothing_is_lost(self, tmp_path: Path):
        """Chunks arriving exactly on the grid need no correction."""
        out = tmp_path / "clean.mp4"
        stats = _record(out, n_packets=200)
        assert stats["gap_events"] == 0

    def test_unindexed_chunks_append(self, tmp_path: Path):
        """Callers that never set frame_index get plain append semantics.

        frame_index defaults to 0, so a non-advancing index means "not indexed"
        rather than "duplicate" — otherwise direct users of the public recorder
        API would silently lose all but the first chunk.
        """
        out = tmp_path / "unindexed.mp4"
        with VideoRecorder(out, 160, 120, fps=30, audio=True,
                           sample_rate=SAMPLE_RATE, channels=2,
                           samples_per_frame=SPF) as rec:
            rec.write(_frame(), pts_us=0)
            for _ in range(10):
                rec.write_audio(AudioChunk(
                    data=np.zeros((SPF, 2), dtype=np.float32),
                    sample_rate=SAMPLE_RATE, channels=2))
            written = rec.audio_stats["samples_written"]
            gaps = rec.audio_stats["gap_events"]
        assert written == 10 * SPF
        assert gaps == 0

    def test_absurd_gap_is_capped(self, tmp_path: Path, caplog):
        """A nonsensical index jump re-anchors instead of writing endless silence."""
        out = tmp_path / "absurd.mp4"
        with VideoRecorder(out, 160, 120, fps=30, audio=True,
                           sample_rate=SAMPLE_RATE, channels=2,
                           samples_per_frame=SPF) as rec:
            rec.write(_frame(), pts_us=0)
            rec.write_audio(_chunk(0))
            rec.write_audio(_chunk(10_000_000))  # ~14 hours ahead
            silence = rec.audio_stats["silence_samples"]
        # Capped at the 10 s limit rather than filling the whole bogus span.
        assert silence <= 10 * SAMPLE_RATE


class TestPacketLossSignal:
    """moonlight-common-c calls decodeAndPlaySample(NULL, 0) for missing packets.

    Ignoring that signal without advancing the frame index silently deletes the
    lost packet's duration from the audio timeline — the root cause of the drift.
    """

    def _session(self) -> StreamingSession:
        # Bypass __init__ so the test needs no shared library.
        s = object.__new__(StreamingSession)
        s._audio_queue = queue.Queue(maxsize=512)
        s._audio_frame_index = 0
        s._audio_lost_packets = 0
        s._audio_dropped_packets = 0
        s._audio_callback_errors = 0
        return s

    def test_loss_marker_advances_index(self):
        s = self._session()
        s._on_audio_sample(b"\x01\x02")
        s._on_audio_sample(None)          # lost
        s._on_audio_sample(None)          # lost
        s._on_audio_sample(b"\x03\x04")

        assert s._audio_frame_index == 4
        assert s._audio_lost_packets == 2

        first = s._audio_queue.get_nowait()
        second = s._audio_queue.get_nowait()
        assert s._audio_queue.empty(), "loss markers must not enqueue PCM"
        assert first.frame_index == 0
        # The gap in frame_index is what tells the recorder to insert silence.
        assert second.frame_index == 3

    def test_loss_gap_becomes_silence(self, tmp_path: Path):
        """End to end: an index gap from packet loss becomes silence."""
        out = tmp_path / "loss.mp4"
        s = self._session()
        for i in range(400):
            s._on_audio_sample(None if 100 <= i < 300 else b"\x01")

        with VideoRecorder(out, 160, 120, fps=30, audio=True,
                           sample_rate=SAMPLE_RATE, channels=2,
                           samples_per_frame=SPF) as rec:
            rec.write(_frame(), pts_us=0)
            while not s._audio_queue.empty():
                pkt = s._audio_queue.get_nowait()
                rec.write_audio(_chunk(pkt.frame_index))
            stats = dict(rec.audio_stats)

        assert stats["silence_samples"] == 200 * SPF

    def test_queue_overflow_is_counted(self):
        s = self._session()
        s._audio_queue = queue.Queue(maxsize=4)
        for _ in range(10):
            s._on_audio_sample(b"\x01")
        assert s._audio_dropped_packets == 6
        # Index still advanced for every packet, so the timeline stays honest.
        assert s._audio_frame_index == 10

    def test_loss_stats_reported(self):
        s = self._session()
        for i in range(10):
            s._on_audio_sample(None if i % 2 else b"\x01")
        assert s.audio_loss_stats == {
            "lost": 5, "dropped": 0, "callback_errors": 0,
        }


class TestDrainOnStop:
    """The reported symptom: audio still in flight was discarded at shutdown.

    The audio pipeline runs behind video, so ending both at the same instant
    leaves the last stretch of the file with video but no audio.
    """

    def test_tail_contains_real_audio(self, tmp_path: Path):
        from moonlight_python import MoonlightClient
        from moonlight_python._stream_manager import AudioSubscription

        sub = AudioSubscription(maxsize=4000)
        out = tmp_path / "drain.mp4"
        n_frames = 61  # ~2 s at 30 fps
        index = iter(range(100000))

        def frames():
            for i in range(n_frames):
                # ~6.7 packets of audio per 33.3 ms frame; the feeder cannot
                # keep up perfectly, so a backlog builds up behind the video.
                for _ in range(7):
                    sub.put(_chunk(next(index)))
                yield Frame(data=np.random.randint(0, 255, (120, 160, 3),
                                                   dtype=np.uint8),
                            timestamp_us=i * 33_333)

        MoonlightClient._record_from_frames(
            None, frames(), out, True, 30, None, None,
            audio_sub=sub, sample_rate=SAMPLE_RATE, channels=2,
            samples_per_frame=SPF,
        )

        vdur, adur = _durations(out)
        assert abs(vdur - adur) < 0.2, f"video={vdur:.3f}s audio={adur:.3f}s"

        # The last half second must be real audio, not padding silence.
        container = av.open(str(out))
        try:
            astream = next(s for s in container.streams if s.type == "audio")
            pcm = np.concatenate([f.to_ndarray().ravel()
                                  for f in container.decode(astream)])
        finally:
            container.close()
        tail = pcm[-(SAMPLE_RATE // 2) * 2:]  # last 0.5 s, both channels
        assert np.abs(tail).max() > 0.001, "the tail of the recording is silent"


class TestWavPath:
    """Image-directory recordings write a sibling audio.wav with the same rules."""

    def test_gap_becomes_silence(self, tmp_path: Path):
        import wave

        from moonlight_python.audio_recorder import WavRecorder

        out = tmp_path / "audio.wav"
        with WavRecorder(out, SAMPLE_RATE, 2, samples_per_frame=SPF) as rec:
            for i in range(400):
                if 100 <= i < 300:  # 1 s lost
                    continue
                rec.write(_chunk(i))
            silence = rec.silence_samples
        assert silence == 200 * SPF
        wf = wave.open(str(out))
        try:
            # 400 packets x 5 ms = 2 s, gap included.
            assert wf.getnframes() == 400 * SPF
        finally:
            wf.close()

    def test_unindexed_chunks_append(self, tmp_path: Path):
        import wave

        from moonlight_python.audio_recorder import WavRecorder

        out = tmp_path / "audio.wav"
        with WavRecorder(out, SAMPLE_RATE, 2) as rec:
            for _ in range(3):
                rec.write(AudioChunk(
                    data=np.zeros((960, 2), dtype=np.float32),
                    sample_rate=SAMPLE_RATE, channels=2))
        wf = wave.open(str(out))
        try:
            assert wf.getnframes() == 3 * 960
        finally:
            wf.close()


class TestBackwardCompatibility:
    def test_video_only_still_works(self, tmp_path: Path):
        out = tmp_path / "v.mp4"
        with VideoRecorder(out, 160, 120, fps=30) as rec:
            for i in range(10):
                rec.write(_frame(), pts_us=i * 33_333)
        container = av.open(str(out))
        try:
            assert [s.type for s in container.streams] == ["video"]
        finally:
            container.close()

    def test_write_without_pts(self, tmp_path: Path):
        """Callers that pass no timestamp still produce a valid file."""
        out = tmp_path / "nopts.mp4"
        with VideoRecorder(out, 160, 120, fps=30) as rec:
            for _ in range(5):
                rec.write(_frame())
        assert out.stat().st_size > 0
