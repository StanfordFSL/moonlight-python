"""Tests for ImageRecorder and VideoRecorder."""

import tempfile
from pathlib import Path

import numpy as np

from moonlight_python.frame import Frame
from moonlight_python.recorder import ImageRecorder, VideoRecorder


def _make_frame(width: int = 320, height: int = 240,
                frame_number: int = 1) -> Frame:
    """Create a synthetic test frame with some non-trivial pixel data."""
    data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Frame(data=data, frame_number=frame_number)


class TestImageRecorder:
    def test_write_creates_png(self, tmp_path: Path):
        recorder = ImageRecorder(tmp_path, format="png")
        frame = _make_frame()
        path = recorder.write(frame)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.parent == tmp_path

    def test_sequential_numbering(self, tmp_path: Path):
        recorder = ImageRecorder(tmp_path)
        for i in range(3):
            path = recorder.write(_make_frame(frame_number=i))
        # Should have 3 files
        files = sorted(tmp_path.glob("*.png"))
        assert len(files) == 3
        assert "000001" in files[0].name
        assert "000003" in files[2].name

    def test_custom_prefix(self, tmp_path: Path):
        recorder = ImageRecorder(tmp_path, prefix="capture")
        path = recorder.write(_make_frame())
        assert "capture_" in path.name

    def test_context_manager(self, tmp_path: Path):
        with ImageRecorder(tmp_path) as recorder:
            recorder.write(_make_frame())
        files = list(tmp_path.glob("*.png"))
        assert len(files) == 1

    def test_creates_output_dir(self, tmp_path: Path):
        out_dir = tmp_path / "sub" / "dir"
        recorder = ImageRecorder(out_dir)
        recorder.write(_make_frame())
        assert out_dir.exists()


class TestVideoRecorder:
    def test_write_creates_video(self, tmp_path: Path):
        output = tmp_path / "test.mp4"
        with VideoRecorder(output, 320, 240, fps=10) as recorder:
            for i in range(5):
                recorder.write(_make_frame(frame_number=i))
        assert output.exists()
        assert output.stat().st_size > 0

    def test_close_required(self, tmp_path: Path):
        output = tmp_path / "test.mp4"
        recorder = VideoRecorder(output, 320, 240, fps=10)
        recorder.write(_make_frame())
        recorder.close()
        assert output.exists()

    def test_write_after_close_raises(self, tmp_path: Path):
        output = tmp_path / "test.mp4"
        recorder = VideoRecorder(output, 320, 240)
        recorder.close()
        try:
            recorder.write(_make_frame())
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_context_manager(self, tmp_path: Path):
        output = tmp_path / "test.mp4"
        with VideoRecorder(output, 320, 240, fps=10) as recorder:
            for i in range(3):
                recorder.write(_make_frame(frame_number=i))
        assert output.exists()
        assert output.stat().st_size > 0
