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

    def start(self):
        self.capabilities = detect_all()
        if not self.control_plane_url:
            self.control_plane_url = discover_control_plane()
            if not self.control_plane_url:
                raise RuntimeError("Control plane not found. Start it with 'wano up'")
        self.register_node()
        self.start_ray_worker()
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
        except Exception as e:
            print(
                f"Warning: Could not get Ray address from control plane: {e}\nRay worker will not be available, but node registration succeeded"
            )
            return
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*RAY_ACCEL.*")
                ray.init(address=ray_address, ignore_reinit_error=True)
        except Exception as e:
            print(
                f"Warning: Could not connect to Ray cluster at {ray_address}: {e}\nRay worker will not be available, but node registration succeeded\nMake sure the control plane Ray head is running and accessible"
            )

    def heartbeat_loop(self):
        while self.running:
            try:
                if not self.capabilities:
                    break
                self.capabilities = detect_all()
                requests.post(
                    f"{self.control_plane_url}/heartbeat",
                    json=self.capabilities.to_dict(),
                ).raise_for_status()
            except Exception as e:
                print(f"Heartbeat failed: {e}")
            time.sleep(10)

    def stop(self):
        self.running = False
        ray.shutdown()
