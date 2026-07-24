# moonlight-python

Python client for [Moonlight](https://moonlight-stream.org/) / [Sunshine](https://app.lizardbyte.dev/Sunshine/) game streaming. Receives decoded video frames as numpy arrays for computer vision and robotics pipelines, with optional synced audio capture.

## Installation

```bash
pip install moonlight-python
```

## Quick Start

```python
from moonlight_python import MoonlightClient

client = MoonlightClient()
client.connect("192.168.1.100")  # auto-pairs on first connection

with client.stream(app="Desktop", width=1920, height=1080, fps=30):
    client.capture("screenshot.png")          # single screenshot
    client.record("capture.mp4", duration=5)  # record 5s video + synced audio
    client.start_recording("long.mp4")        # background recording
    # ... do other work ...
    client.stop_recording()

    for chunk in client.audio():              # real-time PCM audio (float32)
        process(chunk.data)
        break
```

Audio capture is on by default (`record_audio=True`); recordings bake a synced
AAC track into video files, or write `audio.wav` alongside image-directory captures.

For full documentation, see the [GitHub repository](https://github.com/StanfordFSL/moonlight-python).
