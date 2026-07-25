"""Tests for audio: AudioChunk, AudioDecoder, WavRecorder, VideoRecorder audio."""

import wave
from pathlib import Path

import av
import numpy as np

from moonlight_python.audio_decoder import AudioDecoder, build_opus_head
from moonlight_python.audio_frame import AudioChunk
from moonlight_python.audio_recorder import WavRecorder
from moonlight_python.frame import Frame
from moonlight_python.recorder import VideoRecorder


def _make_opus_packets(n_frames: int = 3, samples: int = 960,
                       channels: int = 2) -> list[bytes]:
    """Encode a sine tone to raw Opus packets for decoder tests."""
    enc = av.CodecContext.create("libopus", "w")
    enc.sample_rate = 48000
    enc.format = "s16"
    layout = "stereo" if channels == 2 else "mono"
    enc.layout = layout

    packets: list[bytes] = []
    for k in range(n_frames):
        t = (k * samples + np.arange(samples)) / 48000
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        chans = np.stack([tone] * channels, axis=0)
        s16 = (chans * 32767).astype(np.int16)
        packed = s16.T.reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(packed, format="s16", layout=layout)
        frame.sample_rate = 48000
        frame.pts = k * samples
        for p in enc.encode(frame):
            packets.append(bytes(p))
    for p in enc.encode(None):
        packets.append(bytes(p))
    return packets


def _make_chunk(samples: int = 960, channels: int = 2,
                frame_index: int = 0) -> AudioChunk:
    data = (0.1 * np.random.randn(samples, channels)).astype(np.float32)
    return AudioChunk(data=data, sample_rate=48000, channels=channels,
                      frame_index=frame_index)


def _make_frame(width: int = 320, height: int = 240) -> Frame:
    data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Frame(data=data)


class TestAudioChunk:
    def test_fields_and_properties(self):
        data = np.zeros((480, 2), dtype=np.float32)
        chunk = AudioChunk(data=data, sample_rate=48000, channels=2,
                           timestamp_us=1000, frame_index=7)
        assert chunk.num_samples == 480
        assert chunk.channels == 2
        assert chunk.frame_index == 7
        # 480 samples @ 48 kHz = 10 ms
        assert chunk.duration_us == 10_000

    def test_duration_zero_rate(self):
        chunk = AudioChunk(data=np.zeros((100, 2), dtype=np.float32), sample_rate=0)
        assert chunk.duration_us == 0


class TestOpusHead:
    def test_stereo_family0_length(self):
        head = build_opus_head(48000, 2, 1, 1, [0, 1])
        assert head[:8] == b"OpusHead"
        assert head[9] == 2         # channel count
        assert head[18] == 0        # mapping family 0
        assert len(head) == 19

    def test_surround_family1(self):
        head = build_opus_head(48000, 6, 4, 2, [0, 1, 2, 3, 4, 5])
        assert head[9] == 6
        assert head[18] == 1        # mapping family 1
        # 19 base + stream count + coupled + 6 mapping bytes
        assert len(head) == 19 + 2 + 6


class TestAudioDecoder:
    def test_decode_stereo(self):
        packets = _make_opus_packets(n_frames=3, samples=960, channels=2)
        dec = AudioDecoder(sample_rate=48000, channels=2,
                           streams=1, coupled_streams=1, mapping=[0, 1])
        out = []
        for raw in packets:
            out.extend(dec.decode(raw))
        assert len(out) >= 1
        arr = out[0]
        assert arr.ndim == 2
        assert arr.shape[1] == 2          # (samples, channels)
        assert arr.dtype == np.float32
        assert float(np.abs(arr).max()) <= 1.0

    def test_decode_after_close_raises(self):
        dec = AudioDecoder()
        dec.close()
        try:
            dec.decode(b"\x00")
            assert False, "Should have raised"
        except Exception:
            pass


class TestWavRecorder:
    def test_roundtrip(self, tmp_path: Path):
        out = tmp_path / "audio.wav"
        with WavRecorder(out, sample_rate=48000, channels=2) as rec:
            for _ in range(3):
                rec.write(_make_chunk(samples=960, channels=2))
        assert out.exists()
        wf = wave.open(str(out))
        try:
            assert wf.getnchannels() == 2
            assert wf.getframerate() == 48000
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 3 * 960
        finally:
            wf.close()

    def test_creates_parent_dir(self, tmp_path: Path):
        out = tmp_path / "sub" / "audio.wav"
        with WavRecorder(out) as rec:
            rec.write(_make_chunk())
        assert out.exists()


class TestVideoRecorderAudio:
    def test_two_streams(self, tmp_path: Path):
        out = tmp_path / "av.mp4"
        # 6 frames at 10 fps = 600 ms of video, matched by 6 x 100 ms of audio.
        with VideoRecorder(out, 320, 240, fps=10, audio=True,
                           sample_rate=48000, channels=2,
                           samples_per_frame=4800) as rec:
            for i in range(6):
                rec.write(_make_frame(), pts_us=i * 100_000)
                rec.write_audio(_make_chunk(samples=4800, channels=2,
                                            frame_index=i))
        assert out.exists() and out.stat().st_size > 0

        c = av.open(str(out))
        try:
            types = sorted(s.type for s in c.streams)
            assert types == ["audio", "video"]
            astream = next(s for s in c.streams if s.type == "audio")
            assert astream.codec_context.name == "aac"
        finally:
            c.close()

    def test_video_only_unaffected(self, tmp_path: Path):
        out = tmp_path / "v.mp4"
        with VideoRecorder(out, 320, 240, fps=10) as rec:
            for _ in range(3):
                rec.write(_make_frame())
        c = av.open(str(out))
        try:
            assert [s.type for s in c.streams] == ["video"]
        finally:
            c.close()
