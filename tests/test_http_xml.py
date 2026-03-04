"""Tests for HTTP client XML parsing."""

from moonlight_python.http_client import NvHTTP
from moonlight_python.exceptions import HttpResponseError

import pytest


SAMPLE_SERVERINFO = """\
<?xml version="1.0" encoding="utf-8"?>
<root status_code="200" status_message="OK">
    <hostname>DESKTOP-TEST</hostname>
    <appversion>7.1.431.0</appversion>
    <GfeVersion>3.23.0.74</GfeVersion>
    <uniqueid>ABCDEF123456</uniqueid>
    <HttpsPort>47984</HttpsPort>
    <mac>AA:BB:CC:DD:EE:FF</mac>
    <PairStatus>1</PairStatus>
    <currentgame>0</currentgame>
    <state>SUNSHINE_SERVER_FREE</state>
    <ServerCodecModeSupport>3</ServerCodecModeSupport>
    <gputype>NVIDIA GeForce RTX 3080</gputype>
    <MaxLumaPixelsHEVC>35389440</MaxLumaPixelsHEVC>
</root>
"""

SAMPLE_APPLIST = """\
<?xml version="1.0" encoding="utf-8"?>
<root status_code="200">
    <App>
        <AppTitle>Desktop</AppTitle>
        <ID>1</ID>
        <IsHdrSupported>0</IsHdrSupported>
        <IsAppCollectorGame>0</IsAppCollectorGame>
    </App>
    <App>
        <AppTitle>Steam</AppTitle>
        <ID>2</ID>
        <IsHdrSupported>1</IsHdrSupported>
        <IsAppCollectorGame>0</IsAppCollectorGame>
    </App>
</root>
"""

SAMPLE_ERROR = """\
<?xml version="1.0" encoding="utf-8"?>
<root status_code="401" status_message="Unauthorized" />
"""


def test_get_xml_string():
    assert NvHTTP.get_xml_string(SAMPLE_SERVERINFO, "hostname") == "DESKTOP-TEST"
    assert NvHTTP.get_xml_string(SAMPLE_SERVERINFO, "appversion") == "7.1.431.0"
    assert NvHTTP.get_xml_string(SAMPLE_SERVERINFO, "nonexistent") is None


def test_get_xml_string_from_hex():
    xml = '<root status_code="200"><plaincert>48656c6c6f</plaincert></root>'
    result = NvHTTP.get_xml_string_from_hex(xml, "plaincert")
    assert result == b"Hello"


def test_verify_response_status_ok():
    NvHTTP.verify_response_status(SAMPLE_SERVERINFO)  # Should not raise


def test_verify_response_status_error():
    with pytest.raises(HttpResponseError) as exc_info:
        NvHTTP.verify_response_status(SAMPLE_ERROR)
    assert exc_info.value.status_code == 401


def test_verify_response_status_malformed():
    with pytest.raises(HttpResponseError):
        NvHTTP.verify_response_status("not xml at all")


def test_parse_server_info():
    """parse_server_info extracts all expected fields."""
    import tempfile
    from moonlight_python.identity import Identity

    with tempfile.TemporaryDirectory() as tmpdir:
        ident = Identity(tmpdir)
        http = NvHTTP("192.168.1.100", ident)
        info = http.parse_server_info(SAMPLE_SERVERINFO)

        assert info.hostname == "DESKTOP-TEST"
        assert info.app_version == "7.1.431.0"
        assert info.gfe_version == "3.23.0.74"
        assert info.https_port == 47984
        assert info.paired is True
        assert info.current_game == 0  # state is _SERVER_FREE
        assert info.server_codec_mode_support == 3
        assert info.max_luma_pixels_hevc == 35389440
