import contextlib
import os
import socket
import warnings

import ray


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return str(local_ip)
    except OSError:
        with contextlib.suppress(socket.gaierror):
            result = socket.gethostbyname(socket.gethostname())
            return str(result) if result else "127.0.0.1"
    return "127.0.0.1"


class RayManager:
    def __init__(self, address: str | None = None):
        self.address = address
        self.is_running = False

    def start(self, port: int = 10001):
        if self.is_running:
            return
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        if "RAY_ADDRESS" in os.environ:
            del os.environ["RAY_ADDRESS"]
        node_ip = get_local_ip()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*RAY_ACCEL.*")
            try:
                try:
                    ray.init(
                        address=None,
                        ignore_reinit_error=True,
                        _node_ip_address=node_ip,
                        _temp_dir="/tmp/ray",
                        include_dashboard=False,
                        _gcs_server_port=port,
                    )
                except TypeError:
                    ray.init(
                        address=None,
                        ignore_reinit_error=True,
                        _node_ip_address=node_ip,
                        _temp_dir="/tmp/ray",
                        include_dashboard=False,
                    )
            except RuntimeError as e:
                if "already been initialized" not in str(e).lower():
                    raise
        self.is_running = True
        self.address = self.get_address()

    def stop(self):
        if self.is_running:
            ray.shutdown()
            self.is_running = False

    def get_address(self) -> str | None:
        if not self.is_running:
            return None
        address = ray.get_runtime_context().gcs_address
        return address if isinstance(address, str) else None
