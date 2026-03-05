"""Server and application data models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ServerInfo:
    """Information about a Moonlight/Sunshine host."""

    address: str
    hostname: str = ""
    mac: str = ""
    unique_id: str = ""
    http_port: int = 47989
    https_port: int = 47984
    app_version: str = ""
    gfe_version: str = ""
    server_codec_mode_support: int = 0
    current_game: int = 0
    state: str = ""
    paired: bool = False
    gpu_type: str = ""
    max_luma_pixels_hevc: int = 0
    is_nvidia: bool = False

    # Pinned server certificate (PEM bytes), set after pairing
    server_cert_pem: bytes = b""


@dataclass
class AppInfo:
    """An application available on the host."""

    id: int
    name: str
    hdr_supported: bool = False
    is_app_collector_game: bool = False
