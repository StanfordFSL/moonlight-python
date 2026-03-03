"""mDNS + manual server discovery.

Reference: moonlight-qt/app/backend/computermanager.cpp:369-381
"""

from __future__ import annotations

import socket
import time
from typing import Callable

from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

from .identity import Identity
from .http_client import NvHTTP
from .server import ServerInfo

MDNS_SERVICE_TYPE = "_nvstream._tcp.local."


class _Listener(ServiceListener):
    def __init__(self) -> None:
        self.addresses: list[str] = []

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            for addr in info.parsed_addresses():
                if addr not in self.addresses:
                    self.addresses.append(addr)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass


def discover_mdns(timeout: float = 5.0) -> list[str]:
    """Discover Moonlight/Sunshine hosts via mDNS. Returns list of IP addresses."""
    zc = Zeroconf()
    listener = _Listener()
    browser = ServiceBrowser(zc, MDNS_SERVICE_TYPE, listener)
    time.sleep(timeout)
    browser.cancel()
    zc.close()
    return listener.addresses


def discover_servers(identity: Identity, timeout: float = 5.0) -> list[ServerInfo]:
    """Discover and query all servers via mDNS."""
    addresses = discover_mdns(timeout)
    servers: list[ServerInfo] = []
    for addr in addresses:
        try:
            server = connect_to_server(addr, identity)
            servers.append(server)
        except Exception:
            pass
    return servers


def connect_to_server(host: str, identity: Identity, port: int = 47989) -> ServerInfo:
    """Connect to a specific host and fetch its server info."""
    http = NvHTTP(host, identity, http_port=port)
    xml = http.get_server_info()
    return http.parse_server_info(xml)
