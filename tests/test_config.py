"""Tests for StreamConfig."""

from moonlight_python.config import StreamConfig, AUDIO_CONFIGURATION_STEREO, _surroundaudioinfo


def test_default_config():
    cfg = StreamConfig()
    assert cfg.width == 1920
    assert cfg.height == 1080
    assert cfg.fps == 30
    assert cfg.bitrate_kbps == 10000
    assert cfg.codec == "h264"


def test_surroundaudioinfo_stereo():
    """SURROUNDAUDIOINFO_FROM_AUDIO_CONFIGURATION(stereo) should give correct value."""
    # Stereo: channel_count=2, channel_mask=0x3
    # Result: (0x3 << 16) | 2 = 0x30002 = 196610
    result = _surroundaudioinfo(AUDIO_CONFIGURATION_STEREO)
    assert result == (0x3 << 16) | 2
