# moonlight-python

[![Tests](https://github.com/StanfordFSL/moonlight-python/actions/workflows/tests.yml/badge.svg)](https://github.com/StanfordFSL/moonlight-python/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/moonlight-python?v=1)](https://pypi.org/project/moonlight-python/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

Python client for [Moonlight](https://moonlight-stream.org/) / [Sunshine](https://app.lizardbyte.dev/Sunshine/) game streaming. Receives decoded video frames as numpy arrays for computer vision and robotics pipelines.

Built on [moonlight-common-c](https://github.com/moonlight-stream/moonlight-common-c) via CFFI and [PyAV](https://pyav.org/) (FFmpeg) for decoding.

## Requirements

- Python 3.10+
- A running [Sunshine](https://app.lizardbyte.dev/Sunshine/) server (or NVIDIA GameStream host)

## Installation

```bash
pip install moonlight-python
```

Pre-built wheels are available for Python 3.10–3.14 on Linux (x86_64), Windows (amd64), and macOS (arm64, x86_64) — no compiler or system dependencies needed.

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

# Connect to your Sunshine server
server = client.connect("192.168.1.100")

# Pair (first time only — generates a PIN, enter it in Sunshine web UI)
client.pair()

# Stream frames
for frame in client.stream(app="Desktop", width=1920, height=1080, fps=30):
    print(f"Frame {frame.frame_number}: {frame.width}x{frame.height}")
    rgb_array = frame.data[:, :, ::-1]  # BGR to RGB
    break  # just one frame for demo
```

## Usage

### Discover Servers

```python
servers = client.discover(timeout=5.0)
for s in servers:
    print(f"{s.hostname} @ {s.address}")
```

Or connect directly by IP:

```python
server = client.connect("192.168.1.100")
```

### Pairing

Pairing is required once per client identity. The identity (RSA key + certificate) is persisted in `~/.moonlight-python/`.

```python
# Auto-generates a random PIN and prints it
client.pair()
# Output: PIN: 4829
#         Enter this PIN in the Sunshine web UI at https://192.168.1.100:47990

# Or specify a PIN explicitly
client.pair(pin="1234")
```

### List Apps

```python
for app in client.apps():
    print(f"[{app.id}] {app.name}")
```

### Stream Frames

`stream()` is a generator that yields `Frame` objects. Each frame contains a numpy array and metadata from the encoder.

```python
for frame in client.stream(app="Desktop", width=1280, height=720, fps=30):
    # frame.data  — numpy array (H, W, 3) uint8, BGR format by default
    # frame.frame_number, frame.timestamp_us, frame.is_keyframe, etc.
    do_something(frame.data)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `app` | `"Desktop"` | Application name to stream |
| `width` | `1920` | Video width in pixels |
| `height` | `1080` | Video height in pixels |
| `fps` | `30` | Frames per second |
| `bitrate_kbps` | `10000` | Bitrate in kbps |
| `codec` | `"h264"` | Video codec: `"h264"`, `"hevc"`, or `"av1"` |
| `output_format` | `"bgr24"` | Pixel format: `"bgr24"` or `"rgb24"` |

### Persistent Shared Stream

Use `start_stream()` to establish a single persistent connection. `record()` requires an active stream via `start_stream()`. `stream()` and `latest_frame()` can also tap into the shared connection when one is active.

```python
client.connect("192.168.1.100")

# Start a persistent stream — blocks until real (non-black) frames are flowing
client.start_stream(app="Desktop", width=1920, height=1080, fps=30)

# Record from the active stream (no new connection, accurate duration)
client.record("capture.mp4", duration=5)       # exactly 5 seconds

# Iterate frames from the active stream
for frame in client.stream():
    process(frame.data)

# Get the latest frame for CV pipelines
with client.latest_frame() as buf:
    frame = buf.get(timeout=1.0)

# Explicit cleanup (also auto-cleans on exit)
client.stop_stream()
```

`start_stream()` solves three problems compared to letting each method create its own connection:

1. **No duplicate connections** — a second RTSP connection can disrupt an existing Sunshine session. With a shared stream, `record()` and `stream()` tap into the same connection.
2. **Accurate recording duration** — recordings use wall-clock timestamps, so dropped frames create real time gaps rather than being silently compressed.
3. **No black frames** — `start_stream()` waits for real (non-black) frames before returning, so consumers only see actual content.
4. **Immediate frames** — an IDR frame is requested on start, so frames arrive immediately even when joining an existing Sunshine session.

**`start_stream()` parameters:**

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

**Note:** `record()` requires `start_stream()` to be called first. `stream()` and `latest_frame()` can still create their own connections if no shared stream is active.

### Latest Frame Buffer (for CV Pipelines)

If your model processes frames slower than the stream produces them, use `latest_frame()`. It runs the stream in a background thread and always gives you the most recent frame, dropping old ones.

```python
with client.latest_frame(app="Desktop", width=1280, height=720, fps=30) as buf:
    while True:
        frame = buf.get(timeout=1.0)
        if frame is None:
            continue
        # frame.data is always the freshest available frame
        result = my_model(frame.data)
        print(f"Processed frame {frame.frame_number}, "
              f"dropped {buf.stats['frames_dropped']} frames")
```

### Recording

Record to a video file or a directory of images. The output format is auto-detected from the path. **Requires `start_stream()` first.**

Recordings use wall-clock timestamps — dropped frames create real time gaps rather than being silently compressed.

```python
client.start_stream(app="Desktop", width=1920, height=1080, fps=30)

# Record 60 seconds of video
client.record("capture.mp4", duration=60)

# Record 100 frames as PNGs
client.record("./frames/", max_frames=100)

client.stop_stream()
```

You can also use the recorder classes directly for more control:

```python
from moonlight_python import VideoRecorder, ImageRecorder

# Video
with VideoRecorder("output.mp4", 1920, 1080, fps=30) as rec:
    for frame in client.stream(app="Desktop"):
        rec.write(frame)
        if some_condition:
            break

# Images
with ImageRecorder("./captures/", format="png") as rec:
    for frame in client.stream(app="Desktop"):
        path = rec.write(frame)  # returns Path to saved file
```

### Frame Object

Every frame from `stream()` and `latest_frame()` is a `Frame` dataclass:

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

### Quit App

```python
client.quit_app()
```

## Full Example: Object Detection Pipeline

```python
from moonlight_python import MoonlightClient

client = MoonlightClient()
client.connect("192.168.1.100")

with client.latest_frame(app="Desktop", width=1280, height=720, fps=30) as buf:
    while True:
        frame = buf.get(timeout=2.0)
        if frame is None:
            print("Waiting for frames...")
            continue

        # Your CV model here
        detections = yolo_model(frame.data)

        for det in detections:
            print(f"[Frame {frame.frame_number}] {det.label}: {det.confidence:.2f}")
```

## Full Example: Shared Stream with Recording

```python
from moonlight_python import MoonlightClient

client = MoonlightClient()
client.connect("192.168.1.100")

# Start a persistent stream (blocks until real frames arrive)
client.start_stream(app="Desktop", width=1920, height=1080, fps=30)

# Record exactly 10 seconds — duration is accurate because the stream is warm
client.record("clip.mp4", duration=10)

# Record 50 frames as images
client.record("./frames/", max_frames=50)

# Also iterate frames for real-time processing
for frame in client.stream():
    result = my_model(frame.data)
    if done:
        break

client.stop_stream()
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
├── StreamingSession  — CFFI bindings to moonlight-common-c
├── Decoder            — PyAV (FFmpeg) H.264/HEVC/AV1 decoding
├── StreamManager     — Persistent shared stream with fan-out to subscribers
├── LatestFrameBuffer — Thread-safe latest-frame buffer
└── ImageRecorder / VideoRecorder — Recording support
```
