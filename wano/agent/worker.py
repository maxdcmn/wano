import contextlib
import os
import time
import warnings

import ray
import requests

from wano.agent.detector import detect_all
from wano.agent.discovery import discover_control_plane
from wano.models.compute import NodeCapabilities


class NodeAgent:
    def __init__(self, control_plane_url: str | None = None):
        self.control_plane_url = control_plane_url
        self.capabilities: NodeCapabilities | None = None
        self.running = False

    def _get_ray_node_id(self) -> str | None:
        if not ray.is_initialized():
            return None
        node_id = getattr(ray.get_runtime_context(), "node_id", None)
        if not node_id:
            return None
        with contextlib.suppress(AttributeError):
            return str(node_id.hex())
        return str(node_id)

    def _refresh_capabilities(self):
        self.capabilities = detect_all()
        if self.capabilities:
            self.capabilities.ray_node_id = self._get_ray_node_id()

    def start(self):
        self._refresh_capabilities()
        if not self.control_plane_url:
            self.control_plane_url = discover_control_plane()
            if not self.control_plane_url:
                raise RuntimeError("Control plane not found. Start it with 'wano up'")
        self.register_node()
        self.start_ray_worker()
        if self.capabilities:
            self.capabilities.ray_node_id = self._get_ray_node_id()
            self.register_node()
        self.running = True
        self.heartbeat_loop()

    def register_node(self):
        if not self.capabilities:
            raise RuntimeError("Capabilities not detected")
        try:
            requests.post(
                f"{self.control_plane_url}/register", json=self.capabilities.to_dict(), timeout=5
            ).raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError("Control plane not running. Start it with 'wano up'") from e

    def start_ray_worker(self):
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        try:
            response = requests.get(f"{self.control_plane_url}/ray-address", timeout=5)
            response.raise_for_status()
            ray_address = response.json()["ray_address"]
        except (requests.exceptions.RequestException, KeyError) as e:
            print(
                f"Warning: Could not get Ray address from control plane: {e}\nRay worker will not be available, but node registration succeeded"
            )
            return
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*RAY_ACCEL.*")
                ray.init(address=ray_address, ignore_reinit_error=True)
        except (RuntimeError, ConnectionError) as e:
            print(
                f"Warning: Could not connect to Ray cluster at {ray_address}: {e}\nRay worker will not be available, but node registration succeeded\nMake sure the control plane Ray head is running and accessible"
            )

    def heartbeat_loop(self):
        while self.running:
            try:
                self._refresh_capabilities()
                if not self.capabilities:
                    break
                requests.post(
                    f"{self.control_plane_url}/heartbeat",
                    json=self.capabilities.to_dict(),
                    timeout=5,
                ).raise_for_status()
            except (requests.exceptions.RequestException, RuntimeError) as e:
                print(f"Heartbeat failed: {e}")
            except KeyboardInterrupt:
                break
            time.sleep(10)

    def stop(self):
        self.running = False
        with contextlib.suppress(Exception):
            ray.shutdown()
