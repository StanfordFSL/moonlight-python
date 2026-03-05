# moonlight-python

Python client for [Moonlight](https://moonlight-stream.org/) / [Sunshine](https://app.lizardbyte.dev/Sunshine/) game streaming. Receives decoded video frames as numpy arrays for computer vision and robotics pipelines.

## Installation

```bash
pip install moonlight-python
```

## Quick Start

```python
from moonlight_python import MoonlightClient

client = MoonlightClient()
client.connect("192.168.1.100")
client.pair()

for frame in client.stream(app="Desktop", width=1920, height=1080, fps=30):
    print(f"Frame {frame.frame_number}: {frame.width}x{frame.height}")
    break
```

## Persistent Shared Stream

Use `start_stream()` to establish a single connection shared by `stream()`, `record()`, and `latest_frame()` — avoiding duplicate connections, ensuring accurate recording duration, and skipping black startup frames.

```python
client.start_stream(app="Desktop", width=1920, height=1080, fps=30)
client.record("capture.mp4", duration=5)
client.stop_stream()
```

For full documentation, see the [GitHub repository](https://github.com/StanfordFSL/moonlight-python).
