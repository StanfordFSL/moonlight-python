"""GameStream HTTP API client.

Reference: moonlight-qt/app/backend/nvhttp.cpp
"""

from __future__ import annotations

import ssl
import struct
import tempfile
import uuid
from typing import Any
from xml.etree import ElementTree

import requests
import urllib3

from .identity import Identity

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from .server import AppInfo, ServerInfo
from .exceptions import HttpResponseError

DEFAULT_HTTP_PORT = 47989
DEFAULT_HTTPS_PORT = 47984
HARDCODED_UNIQUE_ID = "0123456789ABCDEF"
REQUEST_TIMEOUT = 5
LAUNCH_TIMEOUT = 120


class NvHTTP:
    """HTTP/HTTPS client for the GameStream protocol."""

    def __init__(self, address: str, identity: Identity,
                 http_port: int = DEFAULT_HTTP_PORT,
                 https_port: int = DEFAULT_HTTPS_PORT,
                 server_cert_pem: bytes = b""):
        self.address = address
        self.identity = identity
        self.http_port = http_port
        self.https_port = https_port
        self.server_cert_pem = server_cert_pem

        self._session = requests.Session()
        # We'll configure TLS per-request since we need client certs

    @property
    def base_url_http(self) -> str:
        return f"http://{self.address}:{self.http_port}"

    @property
    def base_url_https(self) -> str:
        return f"https://{self.address}:{self.https_port}"

    def _build_url(self, base_url: str, command: str, arguments: str | None = None) -> str:
        query = f"uniqueid={HARDCODED_UNIQUE_ID}&uuid={uuid.uuid4().hex}"
        if arguments:
            query += "&" + arguments
        return f"{base_url}/{command}?{query}"

    def _make_ssl_context(self) -> ssl.SSLContext:
        """Create an SSL context with client cert, no hostname verification.

        Sunshine's self-signed certs lack SubjectAltName fields, so we must
        disable hostname checking entirely. The moonlight-qt client does the
        same — it pins the server cert but doesn't check the hostname.
        """
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # Load client identity for mutual TLS
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as cert_f:
            cert_f.write(self.identity.cert_pem)
            cert_f.flush()
            cert_path = cert_f.name
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as key_f:
            key_f.write(self.identity.key_pem)
            key_f.flush()
            key_path = key_f.name

        ctx.load_cert_chain(cert_path, key_path)

        import os
        os.unlink(cert_path)
        os.unlink(key_path)

        return ctx

    def _get(self, base_url: str, command: str, arguments: str | None = None,
             timeout: int = REQUEST_TIMEOUT) -> str:
        url = self._build_url(base_url, command, arguments)
        use_https = base_url.startswith("https")

        if use_https:
            from requests.adapters import HTTPAdapter

            class _PinnedAdapter(HTTPAdapter):
                def __init__(self, ssl_ctx: ssl.SSLContext, **kwargs: Any):
                    self._ssl_ctx = ssl_ctx
                    super().__init__(**kwargs)

                def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
                    kwargs["ssl_context"] = self._ssl_ctx
                    super().init_poolmanager(*args, **kwargs)

            session = requests.Session()
            session.mount("https://", _PinnedAdapter(self._make_ssl_context()))
            resp = session.get(url, timeout=timeout, verify=False)
        else:
            resp = self._session.get(url, timeout=timeout)

        resp.raise_for_status()
        return resp.text

    def open_http(self, command: str, arguments: str | None = None,
                  timeout: int = REQUEST_TIMEOUT) -> str:
        return self._get(self.base_url_http, command, arguments, timeout)

    def open_https(self, command: str, arguments: str | None = None,
                   timeout: int = REQUEST_TIMEOUT) -> str:
        return self._get(self.base_url_https, command, arguments, timeout)

    # --- XML parsing helpers ---

    @staticmethod
    def get_xml_string(xml_text: str, tag_name: str) -> str | None:
        """Extract text content of the first matching XML tag."""
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return None
        elem = root.find(f".//{tag_name}")
        if elem is not None and elem.text:
            return elem.text
        return None

    @staticmethod
    def get_xml_string_from_hex(xml_text: str, tag_name: str) -> bytes | None:
        """Extract hex-encoded text content and decode to bytes."""
        hex_str = NvHTTP.get_xml_string(xml_text, tag_name)
        if hex_str is None:
            return None
        return bytes.fromhex(hex_str)

    @staticmethod
    def verify_response_status(xml_text: str) -> None:
        """Verify the response has status_code=200."""
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            raise HttpResponseError(-1, "Malformed XML")

        status_code_str = root.get("status_code", "-1")
        try:
            status_code = int(status_code_str)
        except ValueError:
            # Handle 0xFFFFFFFF case
            status_code = int(status_code_str, 0) if status_code_str.startswith("0x") else -1

        if status_code == 200:
            return

        status_message = root.get("status_message", "")
        raise HttpResponseError(status_code, status_message)

    # --- High-level operations ---

    def get_server_info(self, use_https: bool = False) -> str:
        """Fetch /serverinfo XML."""
        if use_https and self.server_cert_pem:
            try:
                xml = self.open_https("serverinfo")
                self.verify_response_status(xml)
                return xml
            except (HttpResponseError, requests.RequestException):
                pass

        xml = self.open_http("serverinfo")
        self.verify_response_status(xml)

        # Extract HTTPS port from response
        https_port_str = self.get_xml_string(xml, "HttpsPort")
        if https_port_str:
            port = int(https_port_str)
            if port > 0:
                self.https_port = port

        return xml

    def parse_server_info(self, xml: str) -> ServerInfo:
        """Parse /serverinfo XML into a ServerInfo dataclass."""
        get = lambda tag: self.get_xml_string(xml, tag) or ""

        state = get("state")
        current_game = 0
        if state.endswith("_SERVER_BUSY"):
            try:
                current_game = int(get("currentgame"))
            except ValueError:
                pass

        paired_str = get("PairStatus")
        paired = paired_str == "1"

        server_codec = 0
        codec_str = get("ServerCodecModeSupport")
        if codec_str:
            try:
                server_codec = int(codec_str)
            except ValueError:
                pass

        https_port = DEFAULT_HTTPS_PORT
        https_port_str = get("HttpsPort")
        if https_port_str:
            try:
                p = int(https_port_str)
                if p > 0:
                    https_port = p
            except ValueError:
                pass

        max_luma = 0
        max_luma_str = get("MaxLumaPixelsHEVC")
        if max_luma_str:
            try:
                max_luma = int(max_luma_str)
            except ValueError:
                pass

        gpu_type = get("gputype")
        is_nvidia = "true" if get("IsNvidia") else gpu_type.lower().startswith("nvidia") if gpu_type else False

        return ServerInfo(
            address=self.address,
            hostname=get("hostname"),
            mac=get("mac"),
            unique_id=get("uniqueid"),
            https_port=https_port,
            app_version=get("appversion"),
            gfe_version=get("GfeVersion"),
            server_codec_mode_support=server_codec,
            current_game=current_game,
            state=state,
            paired=paired,
            gpu_type=gpu_type,
            max_luma_pixels_hevc=max_luma,
            is_nvidia=bool(is_nvidia),
            server_cert_pem=self.server_cert_pem,
        )

    def get_app_list(self) -> list[AppInfo]:
        """Fetch and parse /applist."""
        xml = self.open_https("applist")
        self.verify_response_status(xml)

        apps: list[AppInfo] = []
        root = ElementTree.fromstring(xml)
        for app_elem in root.iter("App"):
            app_id_str = app_elem.findtext("ID", "0")
            name = app_elem.findtext("AppTitle", "")
            hdr = app_elem.findtext("IsHdrSupported", "0") == "1"
            collector = app_elem.findtext("IsAppCollectorGame", "0") == "1"
            apps.append(AppInfo(
                id=int(app_id_str),
                name=name,
                hdr_supported=hdr,
                is_app_collector_game=collector,
            ))
        return apps

    def launch_app(self, app_id: int, width: int, height: int, fps: int,
                   bitrate: int, supported_video_formats: int,
                   ri_aes_key: bytes, ri_aes_iv: bytes,
                   surroundaudioinfo: int, sops: bool = True,
                   local_audio: bool = False,
                   launch_query_params: str = "") -> str | None:
        """Launch or resume an app. Returns the RTSP session URL if provided."""
        ri_key_id = struct.unpack(">i", ri_aes_iv[:4])[0]

        args = (
            f"appid={app_id}"
            f"&mode={width}x{height}x{fps}"
            f"&additionalStates=1"
            f"&sops={1 if sops else 0}"
            f"&rikey={ri_aes_key.hex()}"
            f"&rikeyid={ri_key_id}"
            f"&localAudioPlayMode={1 if local_audio else 0}"
            f"&surroundAudioInfo={surroundaudioinfo}"
            f"&remoteControllersBitmap=0"
            f"&gcmap=0"
            f"&gcpersist=0"
        )
        if launch_query_params:
            args += "&" + launch_query_params

        xml = self.open_https("launch", args, timeout=LAUNCH_TIMEOUT)
        self.verify_response_status(xml)
        return self.get_xml_string(xml, "sessionUrl0")

    def resume_app(self, ri_aes_key: bytes, ri_aes_iv: bytes,
                   surroundaudioinfo: int,
                   launch_query_params: str = "") -> str | None:
        """Resume a running app. Returns the RTSP session URL if provided."""
        ri_key_id = struct.unpack(">i", ri_aes_iv[:4])[0]

        args = (
            f"rikey={ri_aes_key.hex()}"
            f"&rikeyid={ri_key_id}"
            f"&surroundAudioInfo={surroundaudioinfo}"
        )
        if launch_query_params:
            args += "&" + launch_query_params

        xml = self.open_https("resume", args, timeout=LAUNCH_TIMEOUT)
        self.verify_response_status(xml)
        return self.get_xml_string(xml, "sessionUrl0")

    def quit_app(self) -> None:
        """Quit the currently running app."""
        xml = self.open_https("cancel", timeout=30)
        self.verify_response_status(xml)
