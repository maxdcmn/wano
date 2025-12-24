import os
import warnings

import ray


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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*RAY_ACCEL.*")
            try:
                ray.init(
                    address=None,
                    ignore_reinit_error=True,
                    _node_ip_address="127.0.0.1",
                    _temp_dir="/tmp/ray",
                    include_dashboard=False,
                )
            except Exception as e:
                if "already been initialized" not in str(e).lower():
                    raise
        self.is_running = True

    def stop(self):
        if self.is_running:
            ray.shutdown()
            self.is_running = False

    def get_address(self) -> str | None:
        if not self.is_running:
            return None
        address = ray.get_runtime_context().gcs_address
        return address if isinstance(address, str) else None
