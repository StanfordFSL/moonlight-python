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

with client.streaming(app="Desktop", width=1920, height=1080, fps=30):
    client.capture("screenshot.png")          # single screenshot
    client.record("capture.mp4", duration=5)  # record 5 seconds
    client.start_recording("long.mp4")        # background recording
    # ... do other work ...
    client.stop_recording()
```

For full documentation, see the [GitHub repository](https://github.com/StanfordFSL/moonlight-python).
