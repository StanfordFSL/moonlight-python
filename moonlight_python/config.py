"""Stream configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass

# Video format constants (from Limelight.h)
VIDEO_FORMAT_H264 = 0x0001
VIDEO_FORMAT_H265 = 0x0100
VIDEO_FORMAT_H265_MAIN10 = 0x0200
VIDEO_FORMAT_AV1_MAIN8 = 0x1000
VIDEO_FORMAT_AV1_MAIN10 = 0x2000

VIDEO_FORMAT_MASK_H264 = 0x000F
VIDEO_FORMAT_MASK_H265 = 0x0F00
VIDEO_FORMAT_MASK_AV1 = 0xF000
VIDEO_FORMAT_MASK_10BIT = 0xAA00

# Audio configuration
AUDIO_CONFIGURATION_STEREO = (0x3 << 16) | (2 << 8) | 0xCA


def _surroundaudioinfo(audio_config: int) -> int:
    channel_count = (audio_config >> 8) & 0xFF
    channel_mask = (audio_config >> 16) & 0xFFFF
    return (channel_mask << 16) | channel_count


CODEC_MAP = {
    "h264": VIDEO_FORMAT_H264,
    "hevc": VIDEO_FORMAT_H265,
    "h265": VIDEO_FORMAT_H265,
    "av1": VIDEO_FORMAT_AV1_MAIN8,
}


@dataclass
class StreamConfig:
    """Configuration for a streaming session."""

    width: int = 1920
    height: int = 1080
    fps: int = 30
    bitrate_kbps: int = 10000
    packet_size: int = 1392
    streaming_remotely: int = 0  # STREAM_CFG_LOCAL
    audio_configuration: int = AUDIO_CONFIGURATION_STEREO
    supported_video_formats: int = VIDEO_FORMAT_H264
    codec: str = "h264"
    sops: bool = True
    local_audio: bool = False

    @property
    def surroundaudioinfo(self) -> int:
        return _surroundaudioinfo(self.audio_configuration)
