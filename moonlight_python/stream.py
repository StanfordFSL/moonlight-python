"""Streaming session using CFFI bindings to moonlight-common-c.

Uses ABI-mode (dlopen) to avoid compile-time dependency.
Pull-based frame model via CAPABILITY_PULL_RENDERER.

Reference: moonlight-common-c/src/Limelight.h
"""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cffi
import numpy as np

from .config import StreamConfig
from .decoder import Decoder
from .exceptions import StreamStartError, StreamTerminatedError
from .frame import Frame

# CFFI definitions matching Limelight.h
CDEF = """
typedef struct _STREAM_CONFIGURATION {
    int width;
    int height;
    int fps;
    int bitrate;
    int packetSize;
    int streamingRemotely;
    int audioConfiguration;
    int supportedVideoFormats;
    int clientRefreshRateX100;
    int colorSpace;
    int colorRange;
    int encryptionFlags;
    char remoteInputAesKey[16];
    char remoteInputAesIv[16];
} STREAM_CONFIGURATION;

typedef struct _LENTRY {
    struct _LENTRY* next;
    char* data;
    int length;
    int bufferType;
} LENTRY;

typedef struct _DECODE_UNIT {
    int frameNumber;
    int frameType;
    uint16_t frameHostProcessingLatency;
    uint64_t receiveTimeUs;
    uint64_t enqueueTimeUs;
    uint64_t presentationTimeUs;
    uint32_t rtpTimestamp;
    int fullLength;
    LENTRY* bufferList;
    bool hdrActive;
    uint8_t colorspace;
} DECODE_UNIT;

typedef struct _OPUS_MULTISTREAM_CONFIGURATION {
    int sampleRate;
    int channelCount;
    int streams;
    int coupledStreams;
    int samplesPerFrame;
    unsigned char mapping[8];
} OPUS_MULTISTREAM_CONFIGURATION;

typedef struct _DECODER_RENDERER_CALLBACKS {
    void* setup;
    void* start;
    void* stop;
    void* cleanup;
    void* submitDecodeUnit;
    int capabilities;
} DECODER_RENDERER_CALLBACKS;

typedef struct _AUDIO_RENDERER_CALLBACKS {
    void* init;
    void* start;
    void* stop;
    void* cleanup;
    void* decodeAndPlaySample;
    int capabilities;
} AUDIO_RENDERER_CALLBACKS;

typedef struct _CONNECTION_LISTENER_CALLBACKS {
    void (*stageStarting)(int stage);
    void (*stageComplete)(int stage);
    void (*stageFailed)(int stage, int errorCode);
    void (*connectionStarted)(void);
    void (*connectionTerminated)(int errorCode);
    void (*logMessage)(const char* format, ...);
    void (*rumble)(unsigned short controllerNumber, unsigned short lowFreqMotor, unsigned short highFreqMotor);
    void (*connectionStatusUpdate)(int connectionStatus);
    void (*setHdrMode)(bool hdrEnabled);
    void (*rumbleTriggers)(uint16_t controllerNumber, uint16_t leftTriggerMotor, uint16_t rightTriggerMotor);
    void (*setMotionEventState)(uint16_t controllerNumber, uint8_t motionType, uint16_t reportRateHz);
    void (*setControllerLED)(uint16_t controllerNumber, uint8_t r, uint8_t g, uint8_t b);
    void (*setAdaptiveTriggers)(uint16_t controllerNumber, uint8_t eventFlags, uint8_t typeLeft, uint8_t typeRight, uint8_t *left, uint8_t *right);
} CONNECTION_LISTENER_CALLBACKS;

typedef struct _SERVER_INFORMATION {
    const char* address;
    const char* serverInfoAppVersion;
    const char* serverInfoGfeVersion;
    const char* rtspSessionUrl;
    int serverCodecModeSupport;
} SERVER_INFORMATION;

void LiInitializeStreamConfiguration(STREAM_CONFIGURATION* streamConfig);
void LiInitializeVideoCallbacks(DECODER_RENDERER_CALLBACKS* drCallbacks);
void LiInitializeAudioCallbacks(AUDIO_RENDERER_CALLBACKS* arCallbacks);
void LiInitializeConnectionCallbacks(CONNECTION_LISTENER_CALLBACKS* clCallbacks);
void LiInitializeServerInformation(SERVER_INFORMATION* serverInfo);

int LiStartConnection(SERVER_INFORMATION* serverInfo, STREAM_CONFIGURATION* streamConfig,
    CONNECTION_LISTENER_CALLBACKS* clCallbacks, DECODER_RENDERER_CALLBACKS* drCallbacks,
    AUDIO_RENDERER_CALLBACKS* arCallbacks, void* renderContext, int drFlags,
    void* audioContext, int arFlags);

void LiStopConnection(void);
void LiInterruptConnection(void);

const char* LiGetStageName(int stage);
const char* LiGetLaunchUrlQueryParameters(void);

typedef void* VIDEO_FRAME_HANDLE;
bool LiWaitForNextVideoFrame(VIDEO_FRAME_HANDLE* frameHandle, DECODE_UNIT** decodeUnit);
bool LiPollNextVideoFrame(VIDEO_FRAME_HANDLE* frameHandle, DECODE_UNIT** decodeUnit);
void LiWakeWaitForVideoFrame(void);
void LiCompleteVideoFrame(VIDEO_FRAME_HANDLE handle, int drStatus);
"""

# Capability flags
CAPABILITY_PULL_RENDERER = 0x20
CAPABILITY_DIRECT_SUBMIT = 0x01

# Video format constants
VIDEO_FORMAT_H264 = 0x0001
VIDEO_FORMAT_H265 = 0x0100
VIDEO_FORMAT_AV1_MAIN8 = 0x1000

# DR status
DR_OK = 0
DR_NEED_IDR = -1

# Encryption flags
ENCFLG_ALL = -1  # 0xFFFFFFFF as signed int32

# Audio config: stereo
AUDIO_CONFIGURATION_STEREO = (0x3 << 16) | (2 << 8) | 0xCA


@dataclass(slots=True)
class RawFrame:
    """Raw frame data with metadata extracted from DECODE_UNIT."""

    annex_b_data: bytes
    frame_number: int
    frame_type: int
    timestamp_us: int
    receive_time_us: int
    enqueue_time_us: int
    rtp_timestamp: int
    host_processing_latency_us: int


def _find_shared_lib() -> str:
    """Find the moonlight-common-c shared library."""
    # Check alongside this module first (installed via scikit-build)
    module_dir = Path(__file__).parent
    for name in ["libmoonlight-common-c.so", "libmoonlight-common-c.dylib", "moonlight-common-c.dll"]:
        lib_path = module_dir / name
        if lib_path.exists():
            return str(lib_path)

    # Check in build directory
    build_dirs = [
        module_dir.parent / "build",
        module_dir.parent / "build" / "Release",
        module_dir.parent / "build" / "Debug",
    ]
    for d in build_dirs:
        if d.exists():
            for lib in d.rglob("libmoonlight-common-c*"):
                if lib.suffix in (".so", ".dylib", ".dll"):
                    return str(lib)

    raise FileNotFoundError(
        "Cannot find libmoonlight-common-c shared library. "
        "Build it with: pip install -e . (or cmake --build build)"
    )


class StreamingSession:
    """Manages a streaming session with moonlight-common-c."""

    def __init__(self) -> None:
        self._ffi = cffi.FFI()
        self._ffi.cdef(CDEF)
        lib_path = _find_shared_lib()
        # On Windows, add the DLL's directory to the search path so that
        # transitive dependencies (e.g. OpenSSL) next to it can be found.
        if os.name == "nt":
            os.add_dll_directory(str(Path(lib_path).parent))
        self._lib = self._ffi.dlopen(lib_path)

        # Connection state
        self._connected = False
        self._connection_started = threading.Event()
        self._connection_terminated = threading.Event()
        self._stage_failed = threading.Event()
        self._failed_stage = 0
        self._failed_error_code = 0
        self._termination_error_code = 0

        # Keep references to callbacks to prevent GC
        self._callbacks: list = []

    def _make_connection_callbacks(self) -> "cffi.FFI.CData":
        """Create CONNECTION_LISTENER_CALLBACKS with our handlers."""
        cl = self._ffi.new("CONNECTION_LISTENER_CALLBACKS*")
        self._lib.LiInitializeConnectionCallbacks(cl)

        @self._ffi.callback("void(int)")
        def stage_starting(stage: int) -> None:
            pass

        @self._ffi.callback("void(int)")
        def stage_complete(stage: int) -> None:
            pass

        @self._ffi.callback("void(int, int)")
        def stage_failed(stage: int, error_code: int) -> None:
            self._failed_stage = stage
            self._failed_error_code = error_code
            self._stage_failed.set()

        @self._ffi.callback("void()")
        def connection_started() -> None:
            self._connection_started.set()

        @self._ffi.callback("void(int)")
        def connection_terminated(error_code: int) -> None:
            self._termination_error_code = error_code
            self._connection_terminated.set()

        self._callbacks.extend([stage_starting, stage_complete, stage_failed,
                                connection_started, connection_terminated])

        cl.stageStarting = stage_starting
        cl.stageComplete = stage_complete
        cl.stageFailed = stage_failed
        cl.connectionStarted = connection_started
        cl.connectionTerminated = connection_terminated

        return cl

    def _make_video_callbacks(self) -> "cffi.FFI.CData":
        """Create DECODER_RENDERER_CALLBACKS for pull-based rendering."""
        dr = self._ffi.new("DECODER_RENDERER_CALLBACKS*")
        self._lib.LiInitializeVideoCallbacks(dr)
        dr.capabilities = CAPABILITY_PULL_RENDERER
        # All function pointers stay NULL — pull mode doesn't use them
        return dr

    def _make_audio_callbacks(self) -> "cffi.FFI.CData":
        """Create no-op AUDIO_RENDERER_CALLBACKS (discard audio)."""
        ar = self._ffi.new("AUDIO_RENDERER_CALLBACKS*")
        self._lib.LiInitializeAudioCallbacks(ar)
        ar.capabilities = CAPABILITY_DIRECT_SUBMIT  # Direct submit to avoid queuing
        return ar

    def start(self, address: str, app_version: str, gfe_version: str,
              server_codec_mode_support: int, rtsp_session_url: str,
              config: StreamConfig, ri_aes_key: bytes, ri_aes_iv: bytes) -> None:
        """Start the streaming connection."""
        if self._connected:
            raise RuntimeError("Already connected")

        # Reset state
        self._connection_started.clear()
        self._connection_terminated.clear()
        self._stage_failed.clear()

        # Server information
        si = self._ffi.new("SERVER_INFORMATION*")
        self._lib.LiInitializeServerInformation(si)

        # Keep byte strings alive
        addr_buf = address.encode("utf-8") + b"\x00"
        ver_buf = app_version.encode("utf-8") + b"\x00"
        self._addr_buf = addr_buf
        self._ver_buf = ver_buf

        si.address = self._ffi.from_buffer(addr_buf)
        si.serverInfoAppVersion = self._ffi.from_buffer(ver_buf)
        si.serverCodecModeSupport = server_codec_mode_support

        if gfe_version:
            gfe_buf = gfe_version.encode("utf-8") + b"\x00"
            self._gfe_buf = gfe_buf
            si.serverInfoGfeVersion = self._ffi.from_buffer(gfe_buf)

        if rtsp_session_url:
            rtsp_buf = rtsp_session_url.encode("utf-8") + b"\x00"
            self._rtsp_buf = rtsp_buf
            si.rtspSessionUrl = self._ffi.from_buffer(rtsp_buf)

        # Stream configuration
        sc = self._ffi.new("STREAM_CONFIGURATION*")
        self._lib.LiInitializeStreamConfiguration(sc)
        sc.width = config.width
        sc.height = config.height
        sc.fps = config.fps
        sc.bitrate = config.bitrate_kbps
        sc.packetSize = config.packet_size
        sc.streamingRemotely = config.streaming_remotely
        sc.audioConfiguration = config.audio_configuration
        sc.supportedVideoFormats = config.supported_video_formats
        sc.encryptionFlags = ENCFLG_ALL

        # Copy AES key and IV
        self._ffi.memmove(sc.remoteInputAesKey, ri_aes_key, 16)
        self._ffi.memmove(sc.remoteInputAesIv, ri_aes_iv, 16)

        # Callbacks
        cl = self._make_connection_callbacks()
        dr = self._make_video_callbacks()
        ar = self._make_audio_callbacks()

        # Keep structs alive
        self._si = si
        self._sc = sc
        self._cl = cl
        self._dr = dr
        self._ar = ar

        err = self._lib.LiStartConnection(si, sc, cl, dr, ar,
                                           self._ffi.NULL, 0,
                                           self._ffi.NULL, 0)
        if err != 0:
            if self._stage_failed.is_set():
                raise StreamStartError(self._failed_stage, self._failed_error_code)
            raise StreamStartError(0, err)

        self._connected = True

    def stop(self) -> None:
        """Stop the streaming connection."""
        if self._connected:
            self._lib.LiStopConnection()
            self._connected = False

    def get_launch_query_params(self) -> str:
        """Get the launch URL query parameters from moonlight-common-c."""
        result = self._lib.LiGetLaunchUrlQueryParameters()
        if result == self._ffi.NULL:
            return ""
        return self._ffi.string(result).decode("utf-8")

    def pull_frame(self) -> RawFrame | None:
        """Pull the next video frame (blocking).

        Returns:
            RawFrame with Annex B data and metadata, or None if terminated.
        """
        if not self._connected:
            return None

        if self._connection_terminated.is_set():
            return None

        frame_handle = self._ffi.new("VIDEO_FRAME_HANDLE*")
        decode_unit = self._ffi.new("DECODE_UNIT**")

        if not self._lib.LiWaitForNextVideoFrame(frame_handle, decode_unit):
            return None

        du = decode_unit[0]

        # Walk the buffer linked list and concatenate data
        data = bytearray()
        entry = du.bufferList
        while entry != self._ffi.NULL:
            chunk = self._ffi.buffer(entry.data, entry.length)
            data.extend(chunk)
            entry = entry.next

        raw = RawFrame(
            annex_b_data=bytes(data),
            frame_number=du.frameNumber,
            frame_type=du.frameType,
            timestamp_us=du.presentationTimeUs,
            receive_time_us=du.receiveTimeUs,
            enqueue_time_us=du.enqueueTimeUs,
            rtp_timestamp=du.rtpTimestamp,
            host_processing_latency_us=du.frameHostProcessingLatency,
        )

        # Complete the frame
        self._lib.LiCompleteVideoFrame(frame_handle[0], DR_OK)

        return raw

    def drain_frames(self) -> None:
        """Drain any pending frames without processing them."""
        frame_handle = self._ffi.new("VIDEO_FRAME_HANDLE*")
        decode_unit = self._ffi.new("DECODE_UNIT**")
        while self._lib.LiPollNextVideoFrame(frame_handle, decode_unit):
            self._lib.LiCompleteVideoFrame(frame_handle[0], DR_OK)

    def wake(self) -> None:
        """Wake up LiWaitForNextVideoFrame() so a blocked pull thread can exit."""
        self._lib.LiWakeWaitForVideoFrame()

    @property
    def is_connected(self) -> bool:
        return self._connected and not self._connection_terminated.is_set()

    @property
    def is_terminated(self) -> bool:
        return self._connection_terminated.is_set()

    @property
    def termination_error(self) -> int:
        return self._termination_error_code


def stream_frames(session: StreamingSession, decoder: Decoder) -> Iterator[Frame]:
    """Generator that yields decoded video frames as Frame objects.

    Args:
        session: Active streaming session
        decoder: Video decoder instance

    Yields:
        Frame objects containing decoded numpy array and metadata
    """
    try:
        while session.is_connected:
            raw = session.pull_frame()
            if raw is None:
                if session.is_terminated:
                    error = session.termination_error
                    if error != 0:
                        raise StreamTerminatedError(error)
                    return  # Graceful termination
                continue

            decoded_frames = decoder.decode(raw.annex_b_data)
            for arr in decoded_frames:
                yield Frame(
                    data=arr,
                    frame_number=raw.frame_number,
                    frame_type=raw.frame_type,
                    timestamp_us=raw.timestamp_us,
                    receive_time_us=raw.receive_time_us,
                    enqueue_time_us=raw.enqueue_time_us,
                    rtp_timestamp=raw.rtp_timestamp,
                    host_processing_latency_us=raw.host_processing_latency_us,
                )
    finally:
        session.stop()
