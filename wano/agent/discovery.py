import socket
import time

from zeroconf import ServiceBrowser, ServiceListener, Zeroconf


class ControlPlaneListener(ServiceListener):
    def __init__(self):
        self.control_plane_url = None

    def add_service(self, zeroconf, service_type, name):
        info = zeroconf.get_service_info(service_type, name)
        if info:
            self.control_plane_url = f"http://{socket.inet_ntoa(info.addresses[0])}:{info.port}"

    def remove_service(self, zeroconf, service_type, name):
        pass

    def update_service(self, zeroconf, service_type, name):
        self.add_service(zeroconf, service_type, name)


def discover_control_plane(timeout: float = 5.0) -> str | None:
    zeroconf = Zeroconf()
    listener = ControlPlaneListener()
    ServiceBrowser(zeroconf, "_wano._tcp.local.", listener)
    time.sleep(timeout)
    zeroconf.close()
    return listener.control_plane_url
