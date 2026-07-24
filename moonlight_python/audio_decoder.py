"""PyAV-based Opus decoder producing float32 PCM numpy arrays.

moonlight-common-c delivers already-decrypted raw Opus packets (multistream,
48 kHz) to our audio callback. This module decodes them to PCM using PyAV
(FFmpeg + bundled libopus), mirroring the video ``Decoder``.

FFmpeg's Opus decoder is configured via ``OpusHead`` extradata (RFC 7845),
which we synthesize from the negotiated OPUS_MULTISTREAM_CONFIGURATION so that
both stereo (channel mapping family 0) and surround (family 1) decode correctly.
"""

from __future__ import annotations

import struct

import av
import numpy as np

from .exceptions import DecoderError


def build_opus_head(sample_rate: int, channels: int, streams: int,
                    coupled_streams: int, mapping: list[int],
                    pre_skip: int = 0) -> bytes:
    """Build an OpusHead identification header (RFC 7845, section 5.1).

    Channel mapping family 0 is used for mono/stereo; family 1 (which carries
    the stream/coupled counts and the channel mapping table) is used for
    multichannel/surround configurations.
    """
    family = 0 if channels <= 2 else 1
    head = bytearray()
    head += b"OpusHead"
    head += bytes([1])                       # version
    head += bytes([channels & 0xFF])         # output channel count
    head += struct.pack("<H", pre_skip)      # pre-skip
    head += struct.pack("<I", sample_rate)   # input sample rate (informational)
    head += struct.pack("<h", 0)             # output gain (Q7.8)
    head += bytes([family])                  # channel mapping family
    if family != 0:
        head += bytes([streams & 0xFF])
        head += bytes([coupled_streams & 0xFF])
        head += bytes(bytearray(mapping[:channels]))
    return bytes(head)


class AudioDecoder:
    """Opus decoder using PyAV, producing (samples, channels) float32 PCM."""

    def __init__(self, sample_rate: int = 48000, channels: int = 2,
                 streams: int = 1, coupled_streams: int = 1,
                 mapping: list[int] | None = None) -> None:
        """Initialize the decoder from an OPUS_MULTISTREAM_CONFIGURATION.

        Args:
            sample_rate: Opus sample rate in Hz (always 48000 for Moonlight).
            channels: Number of output channels.
            streams: Number of Opus streams (multistream).
            coupled_streams: Number of coupled (stereo) Opus streams.
            mapping: Channel mapping table (one byte per channel).
        """
        if mapping is None:
            mapping = list(range(channels))

        self._sample_rate = sample_rate
        self._channels = channels

        self._codec_ctx = av.CodecContext.create("opus", "r")
        self._codec_ctx.extradata = build_opus_head(
            sample_rate, channels, streams, coupled_streams, mapping
        )
        self._open = True

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def decode(self, opus_data: bytes) -> list[np.ndarray]:
        """Decode a raw Opus packet into PCM numpy array(s).

        Args:
            opus_data: Raw (decrypted) Opus packet bytes.

        Returns:
            List of numpy arrays, each (samples, channels) float32 in [-1, 1].
            Usually 0 or 1 arrays per packet.
        """
        if not self._open:
            raise DecoderError("AudioDecoder is closed")

        packet = av.Packet(opus_data)
        out: list[np.ndarray] = []
        try:
            for frame in self._codec_ctx.decode(packet):
                # Opus decodes to planar float (fltp): shape (channels, samples)
                arr = frame.to_ndarray()
                # Transpose to (samples, channels), contiguous float32
                out.append(np.ascontiguousarray(arr.T, dtype=np.float32))
        except av.error.InvalidDataError:
            # Corrupted packet — skip
            pass
        return out

    def close(self) -> None:
        """Close the decoder."""
        self._open = False

    def __enter__(self) -> "AudioDecoder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
