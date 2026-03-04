"""Tests for Frame dataclass."""

import numpy as np

from moonlight_python.frame import Frame


def test_frame_creation():
    data = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = Frame(data=data, frame_number=42, frame_type=1, timestamp_us=1000)
    assert frame.frame_number == 42
    assert frame.frame_type == 1
    assert frame.timestamp_us == 1000
    assert frame.data is data


def test_frame_defaults():
    data = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(data=data)
    assert frame.frame_number == 0
    assert frame.frame_type == 0
    assert frame.timestamp_us == 0
    assert frame.receive_time_us == 0
    assert frame.enqueue_time_us == 0
    assert frame.rtp_timestamp == 0
    assert frame.host_processing_latency_us == 0


def test_is_keyframe():
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    assert Frame(data=data, frame_type=1).is_keyframe is True
    assert Frame(data=data, frame_type=0).is_keyframe is False


def test_shape_properties():
    data = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = Frame(data=data)
    assert frame.shape == (720, 1280, 3)
    assert frame.height == 720
    assert frame.width == 1280


def test_all_metadata_fields():
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    frame = Frame(
        data=data,
        frame_number=100,
        frame_type=1,
        timestamp_us=5000,
        receive_time_us=5100,
        enqueue_time_us=5200,
        rtp_timestamp=12345,
        host_processing_latency_us=50,
    )
    assert frame.frame_number == 100
    assert frame.timestamp_us == 5000
    assert frame.receive_time_us == 5100
    assert frame.enqueue_time_us == 5200
    assert frame.rtp_timestamp == 12345
    assert frame.host_processing_latency_us == 50
