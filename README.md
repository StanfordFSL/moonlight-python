# moonlight-python

[![Tests](https://github.com/StanfordFSL/moonlight-python/actions/workflows/tests.yml/badge.svg)](https://github.com/StanfordFSL/moonlight-python/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/moonlight-python?v=1)](https://pypi.org/project/moonlight-python/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

Python client for [Moonlight](https://moonlight-stream.org/) / [Sunshine](https://app.lizardbyte.dev/Sunshine/) game streaming. Receives decoded video frames as numpy arrays for computer vision and robotics pipelines, with optional synced audio capture.

Built on [moonlight-common-c](https://github.com/moonlight-stream/moonlight-common-c) via CFFI and [PyAV](https://pyav.org/) (FFmpeg) for decoding.

## Requirements

- Python 3.10+
- A running [Sunshine](https://app.lizardbyte.dev/Sunshine/) server (or NVIDIA GameStream host)

## Installation

```bash
pip install moonlight-python
```

Pre-built wheels are available — no compiler or system dependencies needed.

> [!NOTE]
> **Supported platforms for pre-built wheels (Python 3.10–3.14):**
> | OS | Architecture |
> |---|---|
> | Linux | x86_64, aarch64 |
> | Windows | amd64 |
> | macOS | arm64 |
>
> If your platform or architecture is not listed above, you can still install from source (see below).

### From Source

Building from source requires CMake 3.15+, a C/C++ compiler, and OpenSSL development headers.

```bash
git clone --recursive https://github.com/StanfordFSL/moonlight-python.git
cd moonlight-python
pip install -e ".[dev]"
```

The `--recursive` flag pulls in the `moonlight-common-c` submodule. The pip install step uses scikit-build-core to compile the native library automatically.

### Building Wheels Locally

You can build manylinux wheels locally using [cibuildwheel](https://cibuildwheel.readthedocs.io/) (requires Docker):

```bash
./build_wheels.sh
```

The resulting `.whl` files will be in `wheelhouse/`.

## Quick Start

```python
from moonlight_python import MoonlightClient

client = MoonlightClient()
client.connect("192.168.1.100")  # auto-pairs on first connection

with client.stream(app="Desktop", width=1920, height=1080, fps=30):
    # Capture a screenshot
    client.capture("screenshot.png")

    # Record 5 seconds of video
    client.record("clip.mp4", duration=5)

    # Process frames in real time
    for frame in client.frames():
        # frame.data is a numpy array (H, W, 3) uint8 BGR
        result = my_model(frame.data)
        if done:
            break

    # Process audio in real time (48 kHz float32 PCM)
    for chunk in client.audio():
        # chunk.data is a numpy array (samples, channels) float32
        process_audio(chunk.data)
        break

    # Always get the most recent frame (drops old ones)
    with client.latest_frame() as buf:
        frame = buf.get(timeout=1.0)
        if frame is not None:
            result = my_model(frame.data)
```

## Usage

### Streaming

All operations require an active stream. Use `client.stream()` as a context manager, or `start_stream()` / `stop_stream()` for explicit control.

```python
# Context manager (recommended)
with client.stream(app="Desktop", width=1920, height=1080, fps=30):
    # ... use frames(), record(), capture(), latest_frame() ...
    pass

# Explicit start/stop
client.start_stream(app="Desktop", width=1920, height=1080, fps=30)
# ... use frames(), record(), capture(), latest_frame() ...
client.stop_stream()
```

`start_stream()` blocks until real (non-black) frames are flowing. It also requests an IDR frame and sends a mouse nudge to force Sunshine to produce frames immediately, even when joining an existing session.

**Parameters** (same for `stream()` and `start_stream()`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `app` | `"Desktop"` | Application name to stream |
| `width` | `1920` | Video width in pixels |
| `height` | `1080` | Video height in pixels |
| `fps` | `30` | Frames per second |
| `bitrate_kbps` | `10000` | Bitrate in kbps |
| `codec` | `"h264"` | Video codec: `"h264"`, `"hevc"`, or `"av1"` |
| `output_format` | `"bgr24"` | Pixel format: `"bgr24"` or `"rgb24"` |
| `ready_timeout` | `10.0` | Max seconds to wait for non-black frames |
| `black_frame_threshold` | `5.0` | Mean pixel value above which a frame is considered real |
| `record_audio` | `True` | Capture the audio stream. Enables `audio()` / `latest_audio()` and audio in `record()` |

### Iterating Frames

`frames()` is a generator that yields `Frame` objects:

```python
for frame in client.frames():
    # frame.data  — numpy array (H, W, 3) uint8, BGR format by default
    do_something(frame.data)
```

### Screenshot

Capture a single screenshot (grabs the latest frame immediately):

```python
client.capture("screenshot.png")
```

### Recording

Record to a video file or a directory of images. The output format is auto-detected from the file extension (`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm` → video, otherwise → image directory). At least one of `duration` or `max_frames` is required.

Recordings are timestamped from the host's capture clock — dropped frames create real time gaps rather than being silently compressed, so a variable frame rate is recorded faithfully. Video resolution matches the stream automatically.

**Audio** is included automatically when the stream was started with `record_audio=True` (the default):
- **Video files** get a synced AAC audio track baked into the same container.
- **Image directories** get a separate `audio.wav` (16-bit PCM) written alongside the frames.

Pass `with_audio=False` to `record()` / `start_recording()` to skip audio for a specific recording.

#### How A/V sync is maintained

Both tracks are placed on the **host's** clock, so they cannot drift apart over a long recording:

- **Video** PTS come from each frame's capture timestamp (moonlight's `presentationTimeUs`), in a 90 kHz time base.
- **Audio** is positioned at `frame_index × samples_per_frame` — a count of the Opus packets the host generated, not a running total of the samples that happened to arrive.

Because audio is positioned rather than appended, anything lost upstream — a packet dropped by the network, a full queue, a corrupt packet — leaves a **gap filled with silence** instead of pulling all subsequent audio earlier. On stop, the recorder drains audio still in flight (the audio pipeline runs behind video) and pads any remainder with silence so both tracks end together.

Losses are reported at `INFO` level; enable logging to see them:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

```python
# Record 60 seconds of video
client.record("capture.mp4", duration=60)

# Record exactly 100 frames of video
client.record("capture.mp4", max_frames=100)

# Record 30 seconds but stop early if 500 frames are reached
client.record("capture.mp4", duration=30, max_frames=500)

# Record 100 frames as PNGs
client.record("./frames/", max_frames=100)

# Record 10 seconds of PNGs
client.record("./frames/", duration=10)
```

### Background Recording

Use `start_recording()` / `stop_recording()` for open-ended recording that runs in the background while you do other work:

```python
# Record video in the background
client.start_recording("capture.mp4")
# ... process frames, run models, etc. ...
client.stop_recording()  # finalizes the video file

# Record images in the background
client.start_recording("./frames/")
# ... do other work ...
client.stop_recording()
```

### Latest Frame (for CV Pipelines)

If your model processes frames slower than the stream produces them, use `latest_frame()`. It always gives you the most recent frame, dropping old ones.

```python
with client.latest_frame() as buf:
    while True:
        frame = buf.get(timeout=1.0)
        if frame is None:
            continue
        result = my_model(frame.data)
```

### Frame Object

Every frame from `frames()` and `latest_frame()` is a `Frame` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `data` | `np.ndarray` | Pixel data, shape `(H, W, 3)`, dtype `uint8` |
| `frame_number` | `int` | Sequential frame number from encoder |
| `frame_type` | `int` | `1` = IDR/keyframe, `0` = P-frame |
| `is_keyframe` | `bool` | Property: `True` if IDR frame |
| `timestamp_us` | `int` | Presentation timestamp (microseconds) |
| `receive_time_us` | `int` | Network receive time (microseconds) |
| `enqueue_time_us` | `int` | Decode queue time (microseconds) |
| `rtp_timestamp` | `int` | RTP timestamp |
| `host_processing_latency_us` | `int` | Server-side processing latency |
| `width` | `int` | Property: frame width in pixels |
| `height` | `int` | Property: frame height in pixels |
| `shape` | `tuple` | Property: `data.shape` |

## Audio

When the stream is started with `record_audio=True` (the default), the Opus audio
stream from Sunshine is decoded to PCM and exposed in real time, mirroring the video
frame APIs. Audio is 48 kHz; the numpy data is `float32` in the range `[-1.0, 1.0]`.

### Iterating Audio

`audio()` is a generator that yields `AudioChunk` objects:

```python
with client.stream(app="Desktop"):
    for chunk in client.audio():
        # chunk.data — numpy array (samples, channels) float32, 48 kHz
        process(chunk.data)
```

### Latest Audio

Like `latest_frame()`, `latest_audio()` always returns the most recent chunk
(dropping stale ones) — useful when your processing runs slower than real time:

```python
with client.latest_audio() as buf:
    while True:
        chunk = buf.get(timeout=1.0)
        if chunk is not None:
            process(chunk.data)
```

Disable audio entirely with `client.stream(..., record_audio=False)`; then `audio()`
yields nothing and recordings contain no audio.

### AudioChunk Object

Every chunk from `audio()` and `latest_audio()` is an `AudioChunk` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `data` | `np.ndarray` | PCM data, shape `(samples, channels)`, dtype `float32` in `[-1, 1]` |
| `sample_rate` | `int` | Sample rate in Hz (48000) |
| `channels` | `int` | Number of audio channels |
| `timestamp_us` | `int` | Presentation timestamp (microseconds), shared clock with video |
| `receive_time_us` | `int` | Network receive time (microseconds) |
| `frame_index` | `int` | Sequential index of the source Opus packet |
| `num_samples` | `int` | Property: samples per channel |
| `duration_us` | `int` | Property: chunk duration in microseconds |

## Setup Reference

### Discover Servers

```python
servers = client.discover(timeout=5.0)
for s in servers:
    print(f"{s.hostname} @ {s.address}")
```

Or connect directly by IP (optionally specify the HTTP port, default `47989`):

```python
server = client.connect("192.168.1.100")
server = client.connect("192.168.1.100", port=48089)  # custom port
```

### Pairing

Pairing is required once per client identity. The identity (RSA key + certificate) is persisted in `~/.moonlight-python/`.

`connect()` automatically detects whether the client is paired and triggers pairing if needed — a random PIN is printed for you to enter in the Sunshine web UI. You can also pair explicitly or with a specific PIN:

```python
# Explicit pairing (not usually needed — connect() handles this)
client.pair()

# Pair with a specific PIN
client.pair(pin="1234")
```

### List Apps

```python
for app in client.apps():
    print(f"[{app.id}] {app.name}")
```

### Quit App

```python
client.quit_app()
```

### Low-Level Recorder Classes

You can use the recorder classes directly for full control over the encoding:

```python
from moonlight_python import VideoRecorder, ImageRecorder, WavRecorder

# Video
with VideoRecorder("output.mp4", 1920, 1080, fps=30) as rec:
    for frame in client.frames():
        rec.write(frame)
        if some_condition:
            break

# Video with a synced AAC audio track
with VideoRecorder("output.mp4", 1920, 1080, fps=30, audio=True) as rec:
    # pts_us is the capture time in microseconds, relative to the recording
    # start. Pass frame.timestamp_us - <first frame's timestamp_us>.
    rec.write(frame, pts_us=elapsed_us)   # from the video thread
    rec.write_audio(chunk)                # from the audio thread (thread-safe)
```

`write_audio()` positions each chunk using `chunk.frame_index`, so gaps caused by
lost audio are filled with silence. If you build `AudioChunk`s yourself and leave
`frame_index` at its default, chunks are simply appended in order.

```python

# Images
with ImageRecorder("./captures/", format="png") as rec:
    for frame in client.frames():
        path = rec.write(frame)  # returns Path to saved file

# Audio to WAV
with WavRecorder("./captures/audio.wav") as rec:
    for chunk in client.audio():
        rec.write(chunk)
```

## Examples

### Object Detection Pipeline

```python
from moonlight_python import MoonlightClient

client = MoonlightClient()
client.connect("192.168.1.100")

with client.stream(app="Desktop", width=1280, height=720, fps=30):
    with client.latest_frame() as buf:
        while True:
            frame = buf.get(timeout=2.0)
            if frame is None:
                print("Waiting for frames...")
                continue

            detections = yolo_model(frame.data)
            for det in detections:
                print(f"[Frame {frame.frame_number}] {det.label}: {det.confidence:.2f}")
```

### Record While Processing

```python
from moonlight_python import MoonlightClient

client = MoonlightClient()
client.connect("192.168.1.100")

with client.stream(app="Desktop", width=1920, height=1080, fps=30):
    client.start_recording("session.mp4")
    for frame in client.frames():
        result = my_model(frame.data)
        if done:
            break
    client.stop_recording()
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Architecture

```
MoonlightClient (high-level API)
├── Identity          — RSA keypair + X.509 cert (~/.moonlight-python/)
├── NvHTTP            — GameStream HTTP/HTTPS protocol
├── discovery          — mDNS server discovery (zeroconf)
├── pairing            — 5-stage cryptographic pairing protocol
├── StreamingSession  — CFFI bindings to moonlight-common-c (video + audio callbacks)
├── Decoder            — PyAV (FFmpeg) H.264/HEVC/AV1 decoding
├── AudioDecoder      — PyAV (FFmpeg) Opus → float32 PCM decoding
├── StreamManager     — Persistent shared stream, video + audio fan-out to subscribers
├── LatestFrameBuffer — Thread-safe latest-frame buffer
└── ImageRecorder / VideoRecorder / WavRecorder — Recording support (video, AAC track, WAV)
```
